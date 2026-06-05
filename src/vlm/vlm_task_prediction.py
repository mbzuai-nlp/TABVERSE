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

# ---- Prompts ----
TASK_PROMPT_GENERAL = {
    "task_prediction": (
        "Look at the given image and answer the following question directly. "
        "Do not include introductions, explanations, or extra text. "
        "Provide only the exact, precise final answer.\n{query}"
    )
}

TASK_PROMPT_BINARY = {
    "task_prediction": (
        "Look at the given image and answer the following question with only a single digit: "
        "1 if the statement is true, 0 if the statement is false. "
        "Do not include any explanations or extra text.\n{query}"
    )
}

# vLLM client is set in main() after parsing args
client = None


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


# --------- Minimal post-processing for task_prediction ---------

_WORD_TO_BIN_ONE = {"true", "yes", "y", "correct", "right", "1", "positive"}
_WORD_TO_BIN_ZERO = {"false", "no", "n", "incorrect", "wrong", "0", "negative"}


def _strip_wrappers_min(s: str) -> str:
    # quick, allocation-light cleanup
    return s.strip().strip("[](){}").strip(" .,'\"")


def _normalize_task_prediction(raw: str, binary: bool) -> str:
    """
    For binary datasets (feverous/tabfact), try to coerce output to '0' or '1'.
    For others, just trim wrappers and common prefixes.
    """
    s = (raw or "").strip()
    if not s:
        return s
    # remove common prefixes like "Answer:", "Final answer:", etc.
    s = re.sub(
        r"(?i)\b(?:final\s+answer|the\s+answer\s+is|answer)\s*:\s*", "", s
    ).strip()
    s = _strip_wrappers_min(s)

    if not binary:
        return s

    # 1) exact 0/1 token
    m = re.search(r"\b([01])\b", s)
    if m:
        return m.group(1)

    # 2) map first decisive word to 0/1
    first_tok = re.match(r"[A-Za-z0-9]+", s)
    if first_tok:
        tok = first_tok.group(0).lower()
        if tok in _WORD_TO_BIN_ONE:
            return "1"
        if tok in _WORD_TO_BIN_ZERO:
            return "0"

    # 3) look for any decisive word anywhere
    low = s.lower()
    if any(w in low.split() for w in _WORD_TO_BIN_ONE):
        return "1"
    if any(w in low.split() for w in _WORD_TO_BIN_ZERO):
        return "0"

    # 4) fallback: if there's any 0/1 char, return the first occurrence
    m2 = re.search(r"[01]", s)
    if m2:
        return m2.group(0)

    # last resort: return cleaned string (evaluation will mark if mismatched)
    return s


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
        print(f"Loading ground truth for task from JSON file")
        gt_dataset = load_dataset(
            REPO_ID,
            data_files=f"data/2-task/task.json",
            token=hf_token,
            streaming=False,
        )["train"]

        gt_samples = list(islice(gt_dataset, limit))
        if not gt_samples:
            raise ValueError(f"No ground truth samples found for task.")

    except Exception as e:
        raise ValueError(f"Error loading ground truth for task: {e}")

    for entry in gt_samples:
        seq_id = entry["id"]
        image_id = entry["image_id"]
        dataset_name = entry["dataset"]
        existing_row = result_map.get(seq_id, entry.copy())
        if format_type not in existing_row:
            existing_row[format_type] = {}

        row = entry.copy()
        row["query"] = row.get("query", "")

        # fetch image just once
        try:
            image = fetch_image_by_id(hf_token, "task", format_type, image_id)
        except Exception as e:
            print(
                f"Image for ID {seq_id} and image {image_id} not found in {format_type} format: {e}"
            )
            existing_row[format_type]["_image_missing"] = True
            result_map[seq_id] = existing_row
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(list(result_map.values()), f, ensure_ascii=False, indent=4)
            continue

        if "task_prediction" in existing_row[format_type]:
            continue  # resume-skip

        # Choose prompt set: binary for feverous/tabfact, general otherwise
        binary = dataset_name.lower() in {"feverous", "tabfact"}
        print(
            f"Using {'binary' if binary else 'general'} prompt set for {dataset_name}"
        )
        TASK_PROMPTS = TASK_PROMPT_BINARY if binary else TASK_PROMPT_GENERAL

        prompt = TASK_PROMPTS["task_prediction"].format(**row)
        raw = query_vlm(model_path, image, prompt)

        # Minimal, efficient post-processing
        final_resp = _normalize_task_prediction(raw, binary=binary)
        existing_row[format_type]["task_prediction"] = raw
        existing_row[format_type]["task_prediction_processed"] = final_resp

        # write/merge
        result_map[seq_id] = existing_row
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(list(result_map.values()), f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="VLM Task Prediction pipeline with HF streaming images (no local downloads)"
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

    out_path = os.path.join(out_dir, "task.json")
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
            print(f"Error processing format_type {format_type}: {e}")
            continue

    print("\nVLM Pipeline completed!")


if __name__ == "__main__":
    main()
