import os
import re
import json
import base64
import argparse
import time
import io
from itertools import islice
from PIL import Image
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# NEW: fetch images in-memory from HF without saving to disk
import requests
from io import BytesIO
from huggingface_hub import hf_hub_url

# HF Repo, Dataset and format configuration
REPO_ID = "MOMINAAHSAN296/vtb-dataset"
FORMATS = ["html", "markdown", "latex"]

# --------- SUC Prompt ---------
TASK_PROMPTS = {
    "table_partition": """What is the **first cell value** (not including headers) of the given table? What is the **last cell value** (not including headers) of the given table? Answer questions one by one and use | to split the answer. Answer the question without having any introduction or explanations.""",
    "table_first_cell": """What is the **first cell value** (not including headers) of the given table? Answer the question without having any introduction or explanations.""",
    "table_last_cell": """What is the **last cell value** (not including headers) of the given table? Answer the question without having any introduction or explanations.""",
    "size_detection": """How many rows in the table? How many columns in the table? Answer the questions one by one and use | to split the answer. Answer the question without having any introduction or explanations.""",
    "number_of_rows": """How many rows in the table? Answer the question without having any introduction or explanations.""",
    "number_of_columns": """How many columns in the table? Answer the question without having any introduction or explanations.""",
    "cell_lookup": """Row/column indices start at 0 (top-left is 0|0). What is the position of the cell value {cell_value}? Use row index and column index to answer. Use | to split the answer. Answer the question without having any introduction or explanations.""",
    "reverse_lookup": """Row/column indices start at 0 (top-left is 0|0). What is the cell value of row index {reverse_lookup_row}, column index {reverse_lookup_col} ? Only output the cell value without other information. Answer the question without having any introduction or explanations.""",
    "column_retrieval": """Row/column indices start at 0 (top-left is 0|0). What is the column name with the index {column_idx} of the given table image? Only give the column name without any explanation. Answer the question without having any introduction or explanations.""",
    "row_retrieval": """Row/column indices start at 0 (top-left is 0|0). What are the cell values of the {row_idx} row in following table? Only list the cell values one by one using | to split the answers. Answer the question without having any introduction or explanations.""",
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


# --------- Helpers: regex transform & normalization ---------
def strip_wrappers(s: str) -> str:
    return s.strip().strip("[](){}").strip().strip(" .,'\"")


def regex_transform(task, response):
    response = (response or "").strip().lower()

    if task == "size_detection":
        nums = re.findall(r"\b(\d+)\b(?:\s*(?:rows?|columns?|cols?))?", response)
        if len(nums) >= 2:
            return f"{nums[0]}|{nums[1]}"
        m = re.search(r"(\d+)\s*rows?.*?(\d+)\s*(?:columns?|cols?)", response)
        if m:
            return f"{m.group(1)}|{m.group(2)}"
        return "EXTRACTION_FAILED"

    if task in {"number_of_rows", "number_of_columns"}:
        patt = (
            r"(\d+)\s*(?:row|rows)"
            if "row" in task
            else r"(\d+)\s*(?:column|columns|col|cols)"
        )

        m = re.search(patt, response, re.IGNORECASE)
        if m:
            return m.group(1)
        nums = re.findall(r"\b\d+\b", response)
        return nums[0] if nums else "EXTRACTION_FAILED"

    if task == "table_partition":
        # 1) quick path: already piped
        txt = response.replace(" | ", "|").replace(" |", "|").replace("| ", "|")
        parts = [p.strip() for p in txt.split("|") if p.strip()]
        if len(parts) >= 2:
            return f"{strip_wrappers(parts[0])}|{strip_wrappers(parts[-1])}"

        # 2) normalize list-like outputs: remove brackets; split by commas/newlines/semicolons
        cleaned = strip_wrappers(response)
        # collapse multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned)
        # split by common separators if no pipe was found
        cand = [
            strip_wrappers(p)
            for p in re.split(r"[,\n;]+", cleaned)
            if strip_wrappers(p)
        ]

        if len(cand) >= 2:
            return f"{cand[0]}|{cand[-1]}"

        # 3) look for explicit "first ...", "last ..." mentions
        first = re.search(
            r"first(?:\s*cell)?(?:\s*value)?(?:\s*is)?\s*[\"']?([^|\"'\]\[\.]+)",
            response,
        )
        last = re.search(
            r"last(?:\s*cell)?(?:\s*value)?(?:\s*is)?\s*[\"']?([^|\"'\]\[\.]+)",
            response,
        )

        if first and last:
            return f"{strip_wrappers(first.group(1))}|{strip_wrappers(last.group(1))}"

        # 4) extreme fallback: grab two longest alphanumeric-ish spans
        spans = [strip_wrappers(s) for s in re.findall(r"[a-z0-9\+\-\.,%/ ]+", cleaned)]
        spans = [s for s in spans if s]
        if len(spans) >= 2:
            return f"{spans[0]}|{spans[-1]}"

        return "EXTRACTION_FAILED"

    if task == "row_retrieval":
        txt = re.sub(
            r"(?:answer:|the answer is:|row\s*\d+\s*:)",
            "",
            response,
            flags=re.IGNORECASE,
        )

        txt = txt.replace(",", "|").replace(" and ", "|")
        parts = [p.strip() for p in txt.split("|") if p.strip()]
        noise = {"the", "is", "are", "value", "values", "cell", "cells"}
        parts = [p for p in parts if p.lower() not in noise]
        return "|".join(parts) if parts else "EXTRACTION_FAILED"

    if task == "cell_lookup":
        patterns = [
            r"(?:row|r)\s*(\d+)\s*(?:,|and|&|\|)\s*(?:col(?:umn)?|c)?\s*(\d+)",
            r"\b(\d+)\s*\|\s*(\d+)\b",
            r"\[(\d+)\s*,\s*(\d+)\]",
            r"\((\d+)\s*,\s*(\d+)\)",
            r"row\s*(\d+).*?col(?:umn)?\s*(\d+)",
            r"(\d+)(?:th|st|nd|rd)?\s*row.*?(\d+)(?:th|st|nd|rd)?\s*col",
        ]
        for pat in patterns:
            m = re.search(pat, response, re.IGNORECASE)
            if m:
                return f"{m.group(1)}|{m.group(2)}"
        nums = re.findall(r"\b\d+\b", response)
        return f"{nums[0]}|{nums[1]}" if len(nums) >= 2 else "EXTRACTION_FAILED"

    if task in ["table_first_cell", "table_last_cell"]:
        txt = re.sub(
            r"(?:answer:|the answer is:|first|last|cell|value|is|:)",
            "",
            response,
            flags=re.IGNORECASE,
        )

        val = strip_wrappers(txt)
        return val if val else "EXTRACTION_FAILED"

    if task == "column_retrieval":
        txt = re.sub(
            r"(?:answer:|the answer is:|the|column|name|index|is|with)",
            "",
            response,
            flags=re.IGNORECASE,
        )

        txt = re.sub(r"\b\d+\b", "", txt)
        val = strip_wrappers(txt)
        return val if val else "EXTRACTION_FAILED"

    if task == "reverse_lookup":
        val = re.sub(r"(answer:|the answer is:)", "", response, flags=re.IGNORECASE)
        val = strip_wrappers(val.lstrip("."))
        return val if val else "EXTRACTION_FAILED"

    return strip_wrappers(response)


def _normalize_suc_prediction(task, raw_response):
    transformed = regex_transform(task, raw_response)
    if transformed == "EXTRACTION_FAILED":
        return raw_response
    return transformed


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


def query_vlm(
    model_path, image: Image.Image, prompt: str, retries=5, wait_time=8
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64_from_pil(image)},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_path,
                messages=messages,
                max_tokens=256,
                temperature=0.0,
                top_p=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                extra_body={"best_of": 1, "top_k": -1},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[Retry {attempt+1}/{retries}] Error: {type(e).__name__} - {e}")
            time.sleep(wait_time)
    return "CONNECTION_FAILED"


# --------- Core pipeline (ONE JSON PER DATASET) ---------


def run_pipeline_for_dataset_format(
    format_type,
    model_path,
    hf_token,
    limit,
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
        print(f"Loading ground truth for suc from JSON file")
        gt_dataset = load_dataset(
            REPO_ID,
            data_files=f"data/3-suc/suc_generation.json",
            token=hf_token,
            streaming=False,
        )["train"]
        print(f"Total samples in suc_generation: {len(gt_dataset)}")

        gt_samples = list(islice(gt_dataset, limit))
        if not gt_samples:
            raise ValueError(f"No ground truth samples found for suc_generation.")

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

        # fetch image just once for all tasks
        try:
            image = fetch_image_by_id(hf_token, "suc_generation", format_type, image_id)
        except Exception as e:
            print(
                f"Image for ID {seq_id} and image {image_id} not found in {format_type} format: {e}"
            )
            existing_row[format_type]["_image_missing"] = True
            result_map[seq_id] = existing_row
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(list(result_map.values()), f, ensure_ascii=False, indent=4)
            continue

        for task_key, prompt_template in TASK_PROMPTS.items():
            try:
                if task_key in existing_row[format_type]:
                    continue  # resume-skip

                prompt = prompt_template.format(**row)
                raw = query_vlm(model_path, image, prompt)

                final_resp = _normalize_suc_prediction(task_key, raw)
                existing_row[format_type][task_key] = raw
                existing_row[format_type][f"{task_key}_processed"] = final_resp

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
        description="VLM SUC pipeline with HF streaming images (no local downloads)"
    )
    parser.add_argument(
        "--model_name", required=True, help="Model name for output organization"
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to model as served by vLLM"
    )
    parser.add_argument(
        "--port", type=int, default=8023, help="vLLM server port (for base_url)"
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

    print("\n=== VLM Pipeline Configuration ===")
    print(f"  Model Name      : {args.model_name}")
    print(f"  Model Path      : {args.model_path}")
    print(f"  Formats         : {', '.join(FORMATS)}")
    print(f"  Max Samples     : {args.limit if args.limit else 'All'}")
    print("=================================\n")

    out_dir = f"results/vlmpipeline/{args.model_name}"
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

    print("\nVLM Pipeline completed!\n")


if __name__ == "__main__":
    main()
