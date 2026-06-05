import os
import re
import json
import time
from itertools import islice
from typing import Optional, List, Dict
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# In-memory fetch from HF without saving to disk
import requests
from huggingface_hub import hf_hub_url

# Configuration
MODEL = "gpt-5"  # or "openai/gpt-4o-mini" for cost optimization
BATCH_SIZE = 5  # Process 5 entries at a time

# HF Repo, Dataset and format configuration
REPO_ID = "MOMINAAHSAN296/vtb-dataset"
DATASETS = ["feverous", "hybridqa", "sqa", "tabfact", "totto"]
FORMATS = ["html", "markdown", "latex"]
EXTENSIONS = {"html": ".html", "markdown": ".md", "latex": ".tex"}

# --------- Prompt config ---------
TASK_PROMPTS = {
    "table_partition": """What is the **first cell value** (not including headers) of the given table? What is the **last cell value** (not including headers) of the given table? Answer questions one by one and use | to split the answer. Answer the question without having any introduction or explanations.""",
    "table_first_cell": """What is the **first cell value** (not including headers) of the given table? Answer the question without having any introduction or explanations.""",
    "table_last_cell": """What is the **last cell value** (not including headers) of the given table? Answer the question without having any introduction or explanations.""",
    "size_detection": """How many rows in the table? How many columns in the table? Answer the questions one by one and use | to split the answer. Answer the question without having any introduction or explanations.""",
    "number_of_rows": """How many rows in the table? Answer the question without having any introduction or explanations.""",
    "number_of_columns": """How many columns in the table? Answer the question without having any introduction or explanations.""",
    "cell_lookup": """Row/column indices start at 0 (top‑left is 0|0). What is the position of the cell value {cell_value}? Use row index and column index to answer. Use | to split the answer. Answer the question without having any introduction or explanations.""",
    "reverse_lookup": """Row/column indices start at 0 (top‑left is 0|0). What is the cell value of row index {reverse_lookup_row}, column index {reverse_lookup_col} ? Only output the cell value without other information. Answer the question without having any introduction or explanations.""",
    "column_retrieval": """Row/column indices start at 0 (top‑left is 0|0). What is the column name with the index {column_idx} of the given table? Only give the column name without any explanation. Answer the question without having any introduction or explanations.""",
    "row_retrieval": """Row/column indices start at 0 (top‑left is 0|0). What are the cell values of the {row_idx} row in following table? Only list the cell values one by one using | to split the answers. Answer the question without having any introduction or explanations.""",
}

# OpenRouter client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
)


def read_table_content_from_hf(hf_token, dataset_name, format_type, image_id):
    ext = EXTENSIONS[format_type]
    filename = f"{dataset_name}/{format_type}/{image_id}{ext}"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    try:
        url = hf_hub_url(repo_id=REPO_ID, filename=filename, repo_type="dataset")
        r = requests.get(url, headers=headers, timeout=60)
        if r.status_code // 100 != 2:
            raise Exception(f"HTTP {r.status_code}: {r.reason}")
        return r.text
    except Exception as e:
        raise FileNotFoundError(
            f"{image_id}{ext} not found at {filename} in repo {REPO_ID}: {e}"
        )


def query_vlm_batch(
    model: str,
    batch_requests: List[Dict],
    retries: int = 5,
    wait_time: int = 8,
) -> List[str]:
    """
    Process multiple text requests in batch.
    batch_requests: List of dicts with 'table_content' and 'prompt' keys
    """
    results = []

    for request in batch_requests:
        table_content = request["table_content"]
        prompt = request["prompt"]

        messages = [
            {"role": "user", "content": f"{prompt}\n\nTable content:\n{table_content}"}
        ]

        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    top_p=1.0,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    extra_body={
                        "reasoning": {"effort": "minimal"}  # Cost optimization
                    },
                )
                result = (response.choices[0].message.content or "").strip()
                results.append(result)
                break
            except Exception as e:
                print(f"[Retry {attempt+1}/{retries}] Error: {type(e).__name__} - {e}")
                if attempt == retries - 1:
                    results.append("CONNECTION_FAILED")
                else:
                    time.sleep(wait_time)

    return results


def query_vlm(
    model: str,
    table_content: str,
    prompt: str,
    retries: int = 5,
    wait_time: int = 8,
) -> str:
    messages = [
        {"role": "user", "content": f"{prompt}\n\nTable content:\n{table_content}"}
    ]

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                extra_body={"reasoning": {"effort": "minimal"}},  # Cost optimization
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[Retry {attempt+1}/{retries}] Error: {type(e).__name__} - {e}")
            time.sleep(wait_time)
    return "CONNECTION_FAILED"


# --------- Core pipeline with batch processing ---------
def run_pipeline_for_dataset_format(
    dataset_name: str,
    format_type: str,
    out_path: str,
    max_samples: Optional[int],
):
    # Resume-safe load
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf8") as f:
            try:
                result_map = {row["id"]: row for row in json.load(f)}
            except Exception:
                result_map = {}
    else:
        result_map = {}

    # Load ground truth
    try:
        print(f"Loading ground truth for {dataset_name} from JSON file")
        gt_dataset = load_dataset(
            REPO_ID,
            data_files=f"data/3-suc/{dataset_name}.json",
            token=os.getenv("HF_TOKEN"),
            streaming=False,
        )["train"]
        print(f"Total samples in {dataset_name}: {len(gt_dataset)}")

        if max_samples and max_samples > 0:
            gt_samples = list(islice(gt_dataset, max_samples))
        else:
            gt_samples = list(gt_dataset)

        if not gt_samples:
            raise ValueError(f"No ground truth samples found for {dataset_name}.")

    except Exception as e:
        raise ValueError(f"Error loading ground truth for {dataset_name}: {e}")

    # Process entries in batches
    for batch_start in range(0, len(gt_samples), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(gt_samples))
        batch_entries = gt_samples[batch_start:batch_end]

        print(
            f"Processing batch {batch_start//BATCH_SIZE + 1}: entries {batch_start+1}-{batch_end}"
        )

        # Prepare batch data
        batch_data = []
        for entry in batch_entries:
            seq_id = entry["id"]
            image_id = entry["image_id"]
            existing_row = result_map.get(seq_id, entry.copy())
            if format_type not in existing_row:
                existing_row[format_type] = {}

            # task-specific fields
            gt = entry.get("suc", {})
            row = entry.copy()
            row["cell_value"] = gt.get("cell_value", "")
            rlk = gt.get("reverse_lookup_indices", "0|0").split("|")
            row["reverse_lookup_row"] = rlk[0]
            row["reverse_lookup_col"] = rlk[1]
            row["column_idx"] = gt.get("column_idx", "")
            row["row_idx"] = gt.get("row_idx", "")

            # fetch table content once per entry
            try:
                table_content = read_table_content_from_hf(
                    os.getenv("HF_TOKEN"), dataset_name, format_type, image_id
                )
                batch_data.append(
                    {
                        "entry": entry,
                        "existing_row": existing_row,
                        "row": row,
                        "table_content": table_content,
                        "seq_id": seq_id,
                    }
                )
            except Exception as e:
                print(
                    f"Table content for ID {image_id} not found in {format_type} format: {e}"
                )
                existing_row[format_type]["_file_missing"] = True
                result_map[seq_id] = existing_row

        # Process each task for the current batch
        for task_key, prompt_template in TASK_PROMPTS.items():
            batch_requests = []
            batch_indices = []

            # Prepare batch requests for current task
            for i, item in enumerate(batch_data):
                if task_key not in item["existing_row"][format_type]:
                    prompt = prompt_template.format(**item["row"])
                    batch_requests.append(
                        {"table_content": item["table_content"], "prompt": prompt}
                    )
                    batch_indices.append(i)

            # Execute batch requests if there are any
            if batch_requests:
                print(f"  Running task '{task_key}' for {len(batch_requests)} entries")
                batch_results = query_vlm_batch(MODEL, batch_requests)

                # Assign results back to batch data - RAW responses only
                for batch_idx, result in zip(batch_indices, batch_results):
                    print(f"Task: {task_key}")
                    print(f"Raw Response: {result}")
                    print("---")

                    # Store raw response without any post-processing
                    batch_data[batch_idx]["existing_row"][format_type][
                        task_key
                    ] = result

        # Update result map and clean up temp fields
        for item in batch_data:
            # cleanup temp fields
            for k in [
                "cell_value",
                "reverse_lookup_row",
                "reverse_lookup_col",
                "column_idx",
                "row_idx",
            ]:
                item["row"].pop(k, None)

            result_map[item["seq_id"]] = item["existing_row"]

        # Save progress after each batch
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(list(result_map.values()), f, ensure_ascii=False, indent=4)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GPT-5 Text-only SUC pipeline with batch processing"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max samples per dataset/format"
    )

    args = parser.parse_args()

    print("\n=== GPT-5 Text-Only SUC Pipeline Configuration ===")
    print(f"  Model           : {MODEL}")
    print(f"  Datasets        : {', '.join(DATASETS)}")
    print(f"  Formats         : {', '.join(FORMATS)}")
    print(f"  Max Samples     : {args.max_samples or 'All'}")
    print(f"  Batch Size      : {BATCH_SIZE}")
    print("=======================================================\n")

    out_dir = f"results/vlmpipeline-text/{MODEL}/suc"
    os.makedirs(out_dir, exist_ok=True)

    for dataset_name in DATASETS:
        print(f"\nProcessing dataset: {dataset_name}")
        out_path = os.path.join(out_dir, f"{dataset_name}.json")

        for format_type in FORMATS:
            try:
                print(f"  Format: {format_type}")
                run_pipeline_for_dataset_format(
                    dataset_name,
                    format_type,
                    out_path,
                    args.max_samples,
                )
            except Exception as e:
                print(f"Error processing {dataset_name}/{format_type}: {e}")
                continue

    print("\nGPT-5 Text-Only SUC Pipeline completed!")


if __name__ == "__main__":
    main()
