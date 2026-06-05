import os
import re
import json
import argparse
import time
from itertools import islice
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# HF Repo, Dataset and format configuration
REPO_ID = "MOMINAAHSAN296/vtb-dataset"
FORMATS = ["html", "markdown", "latex"]
EXTENSIONS = {"html": ".html", "markdown": ".md", "latex": ".tex"}

# --------- Prompt ---------

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

client = None


# -------------------- Cleaning & Postprocessing -------------------- #
def clean_response_text(text):
    if not text:
        return ""
    text = re.sub(r'^["\']|["\']$', "", str(text).strip())
    prefixes_to_remove = [
        r"^the answer is:?\s*",
        r"^answer:?\s*",
        r"^final answer:?\s*",
        r"^here is the\s*",
        r"^the final answer is:?\s*",
        r"^based on the provided table,?\s*",
        r"^the result is:?\s*",
        r"^response:?\s*",
        r"^output:?\s*",
        r"^value:?\s*",
        r"^cell value:?\s*",
        r"^first cell:?\s*",
        r"^last cell:?\s*",
        r"^the first cell value.*?is:?\s*",
        r"^the last cell value.*?is:?\s*",
        r"^first cell value.*?is:?\s*",
        r"^last cell value.*?is:?\s*",
    ]
    for p in prefixes_to_remove:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    instruction_patterns = [
        r"answer.*?without.*?introduction",
        r"only output.*?without.*?information",
        r"answer.*?question.*?without.*?explanation",
        r"give.*?without.*?explanation",
        r"list.*?using.*?split",
        r"one by one using.*?split",
    ]
    for p in instruction_patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[.,:;-]+\s*", "", text)
    text = re.sub(r"\s*[.,:;-]+\s*$", "", text)
    return text.strip()


def post_process_response(task, response):
    if not response or response in ["CONNECTION_FAILED", "ERROR"]:
        return "ERROR"

    response = response.strip()
    response = clean_response_text(response)

    if task == "table_partition":
        parts = None
        if "|" in response:
            parts = [clean_response_text(p) for p in response.split("|") if p.strip()]
        if not parts or len(parts) < 2:
            first_patterns = [
                r"first.*?(?:cell|value).*?[:is]\s*([^,\n\|]+)",
                r"([^,\n\|]+).*?(?:and|,|\|)",
                r"^([^,\n\|]+)",
            ]
            last_patterns = [
                r"last.*?(?:cell|value).*?[:is]\s*([^,\n\|]+)",
                r"(?:and|,|\|).*?([^,\n\|]+)$",
                r"([^,\n\|]+)$",
            ]
            first_val = last_val = None
            for pat in first_patterns:
                m = re.search(pat, response, re.IGNORECASE)
                if m:
                    first_val = clean_response_text(m.group(1))
                    break
            for pat in last_patterns:
                m = re.search(pat, response, re.IGNORECASE)
                if m:
                    last_val = clean_response_text(m.group(1))
                    break
            if first_val and last_val and first_val != last_val:
                parts = [first_val, last_val]
        if not parts or len(parts) < 2:
            for sep in ["\n", ",", " and ", ";", "\t"]:
                if sep in response:
                    temp = [clean_response_text(p) for p in response.split(sep)]
                    temp = [p for p in temp if p]
                    if len(temp) >= 2:
                        parts = [temp[0], temp[-1]]
                        break
        if parts and len(parts) >= 2:
            return f"{parts[0].strip()}|{parts[1].strip()}"
        return "ERROR"

    elif task in ["table_first_cell", "table_last_cell"]:
        for phrase in [
            r"first cell value.*?(?:is|of).*?table.*?(?:is)?",
            r"last cell value.*?(?:is|of).*?table.*?(?:is)?",
            r"(?:not including headers).*?(?:is)?",
            r"given table.*?(?:is)?",
        ]:
            response = re.sub(phrase, "", response, flags=re.IGNORECASE)
        response = re.sub(
            r'\s*["“”]?\s*in the\s*["“”]?.*?column.*$',
            "",
            response,
            flags=re.IGNORECASE,
        )
        response = re.sub(r'^["\']|["\']$', "", response)
        response = clean_response_text(response)
        return response.strip() if response.strip() else "ERROR"

    elif task == "size_detection":
        nums = re.findall(r"\d+", response)
        return f"{nums[0]}|{nums[1]}" if len(nums) >= 2 else "ERROR"

    elif task in ["number_of_rows", "number_of_columns"]:
        nums = re.findall(r"\d+", response)
        return nums[0] if nums else "ERROR"

    elif task == "cell_lookup":
        patterns = [
            r"(\d+)\s*\|\s*(\d+)",
            r"\[(\d+)\s*,\s*(\d+)\]",
            r"\((\d+)\s*,\s*(\d+)\)",
            r"row\s*(\d+).*?(?:col(?:umn)?)?\s*(\d+)",
            r"(\d+)\s*,\s*(\d+)",
        ]
        for pat in patterns:
            m = re.search(pat, response)
            if m:
                return f"{m.group(1)}|{m.group(2)}"
        nums = re.findall(r"\d+", response)
        return f"{nums[0]}|{nums[1]}" if len(nums) >= 2 else "EXTRACTION_FAILED"

    elif task == "reverse_lookup_value":
        response = re.sub(
            r"(?:row|column|index)\s*\d+", "", response, flags=re.IGNORECASE
        )
        response = re.sub(r"cell value.*?(?:is|of)", "", response, flags=re.IGNORECASE)
        response = clean_response_text(response)
        return response if response else "ERROR"

    elif task == "column_retrieval":
        for phrase in [
            r"column name.*?(?:is|of).*?table.*?(?:is)?",
            r"(?:with|at) index.*?\d+.*?(?:is)?",
            r"following table.*?(?:is)?",
        ]:
            response = re.sub(phrase, "", response, flags=re.IGNORECASE)
        response = clean_response_text(response)
        return response if response else "ERROR"

    elif task == "row_retrieval":
        response = re.sub(r"row\s+\d+\s*:", "", response, flags=re.IGNORECASE)
        response = re.sub(
            r"cell values.*?(?:are|of)", "", response, flags=re.IGNORECASE
        )
        parts = [
            clean_response_text(p)
            for p in (response.split("|") if "|" in response else response.split(","))
        ]
        parts = [p for p in parts if p]
        return "|".join(parts) if parts else "ERROR"

    elif task == "task_prediction":
        response = clean_response_text(response)
        return response if response else "ERROR"

    return clean_response_text(response) or "ERROR"


def read_table_content_from_hf(hf_token, dataset, fmt, ex_id):
    from huggingface_hub import hf_hub_url
    import requests

    ext = EXTENSIONS[fmt]
    filename = f"{dataset}/{fmt}/{ex_id}{ext}"
    url = hf_hub_url(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    r = requests.get(url, headers=headers, timeout=60)
    if r.status_code // 100 != 2:
        raise FileNotFoundError(f"{filename} not found in repo {REPO_ID}")
    return r.text


def query_llm(model_id, prompt, table_content, task_key, retries=5, wait_time=8):
    messages = [
        {
            "role": "user",
            "content": f"{prompt}\n\nHere is the table content:\n{table_content}",
        }
    ]
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=256,
                temperature=0.0,
                top_p=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                extra_body={"best_of": 1, "top_k": -1},
            )
            raw = response.choices[0].message.content.strip()

            return raw
        except Exception as e:
            print(f"[Retry {attempt+1}/{retries}] Error: {type(e).__name__} - {e}")
            time.sleep(wait_time)
    return "CONNECTION_FAILED"


# --------- Core pipeline (ONE JSON PER DATASET) ---------


def run_pipeline_for_dataset_format(
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
        print(f"[INFO] Loading ground truth with streaming...")
        gt_dataset = load_dataset(
            REPO_ID,
            data_files=f"data/3-suc/suc_generation.json",
            token=hf_token,
            streaming=False,
        )["train"]

        print(f"[INFO] Total samples in suc_generation: {len(gt_dataset)}")

        gt_samples = list(islice(gt_dataset, max_samples))
        print(f"[INFO] Total selected samples: {len(gt_samples)}")
        if not gt_samples:
            raise ValueError(
                f"[ERROR] No ground truth samples found for suc_generation."
            )
    except Exception as e:
        raise ValueError(f"Error loading ground truth for suc_generation: {e}")

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

        # Read table content from HF
        try:
            table_content = read_table_content_from_hf(
                hf_token, "representations", format_type, image_id
            )
        except Exception as e:
            print(
                f"Image for idx {seq_id} and image id {image_id} not found in {format_type} format: {e}"
            )
            existing_row[format_type]["_format_missing"] = True
            result_map[seq_id] = existing_row
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(list(result_map.values()), f, ensure_ascii=False, indent=4)
            continue

        for task_key, prompt_template in TASK_PROMPTS.items():
            try:
                if task_key in existing_row[format_type]:
                    continue  # resume-skip

                prompt = prompt_template.format(**row)
                raw_resp = query_llm(model_path, prompt, table_content, task_key)

                processed_resp = post_process_response(task_key, raw_resp)

                existing_row[format_type][task_key] = raw_resp
                existing_row[format_type][f"{task_key}_processed"] = processed_resp

            except Exception as e:
                print(f"Error for {seq_id} - {task_key}: {e}")
                existing_row[format_type][task_key] = "ERROR"
                existing_row[format_type][f"{task_key}_processed"] = "ERROR"

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
        description="LLM text-only pipeline (HF streaming, no local fallback)."
    )
    parser.add_argument(
        "--model_name", required=True, help="Model name for output organization"
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to model as served by vLLM"
    )
    parser.add_argument(
        "--port", type=int, default=8033, help="vLLM server port (for base_url)"
    )
    parser.add_argument(
        "--hf_token",
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace token for private dataset access (default: from HF_TOKEN env var)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max samples per dataset/format"
    )

    args = parser.parse_args()

    # vLLM client
    global client
    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{args.port}/v1")

    print("\n=== LLM Pipeline Configuration ===")
    print(f"  Model Name      : {args.model_name}")
    print(f"  Model Path      : {args.model_path}")
    print(f"  Formats         : {', '.join(FORMATS)}")
    print(f"  Max Samples     : {args.limit if args.limit else 'All'}")
    print("=================================\n")

    out_dir = f"results/llmpipeline/{args.model_name}"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "suc.json")
    for format_type in FORMATS:
        try:
            run_pipeline_for_dataset_format(
                format_type,
                args.model_name,
                args.hf_token,
                args.limit,
                out_path,
            )
        except Exception as e:
            print(f"Error processing {format_type}: {e}")
            continue

    print("\nLLM pipeline completed!")


if __name__ == "__main__":
    main()
