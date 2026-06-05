import os
import re
import json
import base64
import argparse
import time
import io
from itertools import islice
from typing import Optional, List, Dict
from PIL import Image
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# In-memory fetch from HF without saving to disk
import requests
from io import BytesIO
from huggingface_hub import hf_hub_url


MODEL = "gpt-5"
BATCH_SIZE = 5  # Process 5 entries at a time

# Hugging Face repo / datasets / formats
REPO_ID = "MOMINAAHSAN296/vtb-dataset"
DATASETS = ["feverous", "hybridqa", "sqa", "tabfact", "totto"]
FORMATS = ["html", "markdown", "latex"]

# --------- Prompts ---------
TASK_PROMPTS = {
    "table_partition": """What is the **first cell value** (not including headers) of the given table? What is the **last cell value** (not including headers) of the given table? Answer questions one by one and use | to split the answer. Answer the question without having any introduction or explanations.""",
    "table_first_cell": """What is the **first cell value** (not including headers) of the given table? Answer the question without having any introduction or explanations.""",
    "table_last_cell": """What is the **last cell value** (not including headers) of the given table? Answer the question without having any introduction or explanations.""",
    "size_detection": """How many rows in the table? How many columns in the table? Answer the questions one by one and use | to split the answer. Answer the question without having any introduction or explanations.""",
    "number_of_rows": """How many rows in the table? Answer the question without having any introduction or explanations.""",
    "number_of_columns": """How many columns in the table? Answer the question without having any introduction or explanations.""",
    "cell_lookup": """Row/column indices start at 0 (top-left is 0|0). What is the position of the cell value {cell_value}? Use row index and column index to answer. Use | to split the answer. Answer the question without having any introduction or explanations.""",
    "reverse_lookup": """Row/column indices start at 0 (top-left is 0|0). What is the cell value of row index {reverse_lookup_row}, column index {reverse_lookup_col}? Only output the cell value without other information. Answer the question without having any introduction or explanations.""",
    "column_retrieval": """Row/column indices start at 0 (top-left is 0|0). What is the column name with the index {column_idx} of the given table image? Only give the column name without any explanation. Answer the question without having any introduction or explanations.""",
    "row_retrieval": """Row/column indices start at 0 (top-left is 0|0). What are the cell values of the {row_idx} row in following table? Only list the cell values one by one using | to split the answers. Answer the question without having any introduction or explanations.""",
}


# --------- Image Helpers ---------
def _pil_from_bytes(b: bytes) -> Image.Image:
    im = Image.open(BytesIO(b))
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def fetch_image_by_id(
    hf_token: str, dataset_name: str, fmt: str, image_id: str
) -> Image.Image:
    filename = f"{dataset_name}/{fmt}/{image_id}.png"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    url = hf_hub_url(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    r = requests.get(url, headers=headers, stream=True, timeout=60)
    if r.status_code // 100 != 2:
        raise FileNotFoundError(f"{filename} not found in repo {REPO_ID}")
    return _pil_from_bytes(r.content)


def image_to_base64_from_pil(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def query_vlm_batch(
    model: str,
    batch_requests: List[Dict],
    retries: int = 5,
    wait_time: int = 8,
) -> List[str]:
    """
    Process multiple VLM requests in batch.
    batch_requests: List of dicts with 'image' and 'prompt' keys
    """
    results = []

    for request in batch_requests:
        image = request["image"]
        prompt = request["prompt"]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_base64_from_pil(image),
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    top_p=1.0,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    extra_body={"reasoning": {"effort": "minimal"}},
                )
                result = (resp.choices[0].message.content or "").strip()
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
    image: Image.Image,
    prompt: str,
    retries: int = 5,
    wait_time: int = 8,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_base64_from_pil(image),
                        "detail": "high",  # works with chat.completions image inputs
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                # max_tokens=1000,
                temperature=0.0,
                top_p=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                extra_body={"reasoning": {"effort": "minimal"}},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[Retry {attempt+1}/{retries}] Error: {type(e).__name__} - {e}")
            time.sleep(wait_time)
    return "CONNECTION_FAILED"


# --------- Core pipeline ---------
def run_pipeline_for_dataset_format(
    dataset_name: str,
    format_type: str,
    out_path: str,
    max_samples: Optional[int],
):
    # resume-safe load
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf8") as f:
            try:
                result_map = {row["id"]: row for row in json.load(f)}
            except Exception:
                result_map = {}
    else:
        result_map = {}

    # Load ground truth
    gt_dataset = load_dataset(
        REPO_ID,
        data_files=f"data/3-suc/{dataset_name}.json",
        token=os.getenv("HF_TOKEN"),
        streaming=False,
    )["train"]

    # Limit samples if requested
    if max_samples and max_samples > 0:
        gt_samples = list(islice(gt_dataset, max_samples))
    else:
        gt_samples = list(gt_dataset)

    # Process entries in batches of 5
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

            # fetch image once per entry
            try:
                image = fetch_image_by_id(
                    os.getenv("HF_TOKEN"), dataset_name, format_type, image_id
                )
                batch_data.append(
                    {
                        "entry": entry,
                        "existing_row": existing_row,
                        "row": row,
                        "image": image,
                        "seq_id": seq_id,
                    }
                )
            except Exception as e:
                print(f"Image missing for {seq_id}/{image_id} ({format_type}): {e}")
                existing_row[format_type]["_image_missing"] = True
                result_map[seq_id] = existing_row

        # Process each task for the current batch
        for task_key, prompt_template in TASK_PROMPTS.items():
            batch_requests = []
            batch_indices = []

            # Prepare batch requests for current task
            for i, item in enumerate(batch_data):
                if task_key not in item["existing_row"][format_type]:
                    prompt = prompt_template.format(**item["row"])
                    batch_requests.append({"image": item["image"], "prompt": prompt})
                    batch_indices.append(i)

            # Execute batch requests if there are any
            if batch_requests:
                print(f"  Running task '{task_key}' for {len(batch_requests)} entries")
                batch_results = query_vlm_batch(MODEL, batch_requests)

                # Assign results back to batch data
                for batch_idx, result in zip(batch_indices, batch_results):
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

        print(f"  Batch {batch_start//BATCH_SIZE + 1} completed and saved")


def main():
    parser = argparse.ArgumentParser(
        description="GPT-5 VLM SUC pipeline (chat completions) with batching"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per dataset/format (default: all)",
    )
    args = parser.parse_args()

    global client
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
    )

    print("\n=== VLM Pipeline Configuration ===")
    print(f"  Model        : {MODEL}")
    print(f"  Datasets     : {', '.join(DATASETS)}")
    print(f"  Formats      : {', '.join(FORMATS)}")
    print(f"  Max Samples  : {args.max_samples if args.max_samples else 'All'}")
    print(f"  Batch Size   : {BATCH_SIZE}")
    print(f"  Detail       : high")
    print("=================================\n")

    out_dir = f"results/vlmpipeline/{MODEL}/suc"
    os.makedirs(out_dir, exist_ok=True)

    for dataset in DATASETS:
        print(f"\nProcessing dataset: {dataset}")
        out_path = os.path.join(out_dir, f"{dataset}.json")
        for fmt in FORMATS:
            print(f"  Format: {fmt}")
            run_pipeline_for_dataset_format(dataset, fmt, out_path, args.max_samples)

    print("\nVLM Pipeline completed!")
    print("Results saved in:", out_dir)


if __name__ == "__main__":
    main()
