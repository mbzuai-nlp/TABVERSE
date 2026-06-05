import os
import re
import json
import base64
import argparse
import time
import io
from itertools import islice
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# NEW: fetch text files in-memory from HF without saving to disk
import requests
from io import BytesIO
from huggingface_hub import hf_hub_url

# HF Repo, Dataset and format configuration
REPO_ID = "MOMINAAHSAN296/vtb-dataset"
DATASETS = ["feverous", "hybridqa", "sqa", "tabfact", "totto"]
FORMATS = ["html", "markdown", "latex"]
EXTENSIONS = {"html": ".html", "markdown": ".md", "latex": ".tex"}

# --------- Prompt & Regex config ---------

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

GUIDED_REGEX = {
    "table_partition": r".+\|.+",
    "table_first_cell": r".+",
    "table_last_cell": r".+",
    "size_detection": r"\d+\|\d+",
    "number_of_rows": r"\d+",
    "number_of_columns": r"\d+",
    "cell_lookup": r"\d+\|\d+",
    "reverse_lookup": r".+",
    "column_retrieval": r".+",
    "row_retrieval": r".+(\|.+)+",
}

# vLLM client is set in main() after parsing args
client = None


def regex_transform(task, response):
    response = response.strip().lower()

    if task == "size_detection":
        nums = re.findall(r"\b(\d+)\b", response)
        if len(nums) >= 2:
            return f"{nums[0]}|{nums[1]}"
        return "EXTRACTION_FAILED"

    if task in {"number_of_rows", "number_of_columns"}:
        nums = re.findall(r"\b\d+\b", response)
        return nums[0] if nums else "EXTRACTION_FAILED"

    if task == "table_partition":
        parts = [p.strip() for p in re.split(r"[|]", response) if p.strip()]
        return f"{parts[0]}|{parts[-1]}" if len(parts) >= 2 else "EXTRACTION_FAILED"

    if task == "row_retrieval":
        response = re.sub(r"(answer:|row\s*\d+:)", "", response, flags=re.IGNORECASE)
        parts = [
            p.strip(' .,"') for p in re.split(r"[|,]| and ", response) if p.strip()
        ]
        return "|".join(parts) if len(parts) > 1 else "EXTRACTION_FAILED"

    if task == "cell_lookup":
        patterns = [
            r"\b(\d+)\s*\|\s*(\d+)\b",
            r"\[(\d+)\s*,\s*(\d+)\]",
            r"\((\d+)\s*,\s*(\d+)\)",
            r"row\s*(\d+).*?col(?:umn)?\s*(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return f"{match.group(1)}|{match.group(2)}"
        nums = re.findall(r"\b\d+\b", response)
        return f"{nums[0]}|{nums[1]}" if len(nums) >= 2 else "EXTRACTION_FAILED"

    if task in ["table_first_cell", "table_last_cell"]:
        val = re.sub(
            r"(answer:|the answer is:)", "", response, flags=re.IGNORECASE
        ).strip(" .,\"'")
        return val if val else "EXTRACTION_FAILED"

    if task == "column_retrieval":
        val = re.sub(
            r"(answer:|the answer is:|column|name|index|is)",
            "",
            response,
            flags=re.IGNORECASE,
        ).strip(" .,\"'")
        return val if val else "EXTRACTION_FAILED"

    if task == "reverse_lookup_value":
        val = re.sub(
            r"(answer:|the answer is:)", "", response, flags=re.IGNORECASE
        ).strip(" .,\"'")
        return val if val else "EXTRACTION_FAILED"

    if task == "task_prediction":
        val = re.sub(
            r"(answer:|the answer is:)", "", response, flags=re.IGNORECASE
        ).strip(" .,\"'")
        return val

    return response.strip(" .,\"'")


def _normalize_suc_prediction(task, raw_response):
    transformed = regex_transform(task, raw_response)
    if transformed == "EXTRACTION_FAILED":
        return raw_response
    return transformed


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


# --------- vLLM query (text-only, no images) ---------


def query_vlm(model_path, table_content, prompt, retries=3, wait_time=8):
    messages = [
        {"role": "user", "content": f"{prompt}\n\nTable content:\n{table_content}"}
    ]
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_path,
                messages=messages,
                max_tokens=256,
                temperature=0.0,
                top_p=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                extra_body={"best_of": 1, "top_k": -1},
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Retry {attempt+1}/{retries}] Error: {type(e).__name__} - {e}")
            time.sleep(wait_time)
    return "CONNECTION_FAILED"


# --------- Core pipeline (ONE JSON PER DATASET) ---------


def run_pipeline_for_dataset_format(
    dataset_name,
    format_type,
    model_path,
    hf_token,
    max_samples,
    out_path,
):
    # resume-safe per DATASET
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf8") as f:
            try:
                result_map = {row["id"]: row for row in json.load(f)}
            except Exception:
                result_map = {}
    else:
        result_map = {}

    # Load ground truth rows
    try:
        print(f"Loading ground truth for {dataset_name} from JSON file")
        gt_dataset = load_dataset(
            REPO_ID,
            data_files=f"data/3-suc/{dataset_name}.json",
            token=hf_token,
            streaming=False,
        )["train"]
        print(f"Total samples in {dataset_name}: {len(gt_dataset)}")
        max_samples = len(gt_dataset)
        gt_samples = list(islice(gt_dataset, max_samples))
        if not gt_samples:
            raise ValueError(f"No ground truth samples found for {dataset_name}.")

    except Exception as e:
        raise ValueError(f"Error loading ground truth for {dataset_name}: {e}")

    for entry in gt_samples:
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

        # fetch table content just once for all tasks
        try:
            table_content = read_table_content_from_hf(
                hf_token, dataset_name, format_type, image_id
            )
        except Exception as e:
            print(
                f"Table content for ID {image_id} not found in {format_type} format: {e}"
            )
            existing_row[format_type]["_file_missing"] = True
            result_map[seq_id] = existing_row
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(list(result_map.values()), f, ensure_ascii=False, indent=4)
            continue

        for task_key, prompt_template in TASK_PROMPTS.items():
            try:
                if task_key in existing_row[format_type]:
                    continue  # resume-skip

                prompt = prompt_template.format(**row)
                raw = query_vlm(model_path, table_content, prompt)

                print(f"Task: {task_key}")
                print(f"Prompt: {prompt}")
                print(f"Raw Response: {raw}")
                print("---")

                final_resp = _normalize_suc_prediction(task_key, raw)
                existing_row[format_type][task_key] = final_resp

            except Exception as e:
                print(f"Error for {seq_id} - {task_key}: {e}")
                existing_row[format_type][task_key] = "ERROR"

        # clean temp fields from the row we persist
        for k in [
            "cell_value",
            "reverse_lookup_row",
            "reverse_lookup_col",
            "column_idx",
            "row_idx",
        ]:
            existing_row.pop(k, None)

        # write/merge
        result_map[seq_id] = existing_row
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(list(result_map.values()), f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Text-only VLM SUC pipeline with HF streaming (no local downloads)"
    )
    parser.add_argument(
        "--model_name", required=True, help="Model name for output organization"
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to model as served by vLLM"
    )
    parser.add_argument(
        "--port", type=int, default=8044, help="vLLM server port (for base_url)"
    )
    parser.add_argument(
        "--hf_token",
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace token for private dataset access (default: from HF_TOKEN env var)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=1000, help="Max samples per dataset/format"
    )

    args = parser.parse_args()

    # vLLM client
    global client
    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{args.port}/v1")

    print("\n=== VLM Text-Only Pipeline Configuration ===")
    print(f"  Model Name      : {args.model_name}")
    print(f"  Model Path      : {args.model_path}")
    print(f"  Datasets        : {', '.join(DATASETS)}")
    print(f"  Formats         : {', '.join(FORMATS)}")
    print(f"  Max Samples     : {args.max_samples if args.max_samples else 'All'}")
    print("=================================\n")

    out_dir = f"results/vlmpipeline-text/{args.model_name}/suc"
    os.makedirs(out_dir, exist_ok=True)

    for dataset_name in DATASETS:
        out_path = os.path.join(out_dir, f"{dataset_name}.json")
        for format_type in FORMATS:
            try:
                run_pipeline_for_dataset_format(
                    dataset_name,
                    format_type,
                    args.model_name,
                    args.hf_token,
                    args.max_samples,
                    out_path,
                )
            except Exception as e:
                print(f"Error processing {dataset_name}/{format_type}: {e}")
                continue

    print("\nText-only VLM pipeline completed!")


if __name__ == "__main__":
    main()
