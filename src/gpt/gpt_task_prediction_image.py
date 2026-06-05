import os
import re
import json
import base64
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

# Configuration
MODEL = "gpt-5"  # or "openai/gpt-4o-mini" for cost optimization
BATCH_SIZE = 5  # Process 5 entries at a time

# Hugging Face repo / datasets / formats
REPO_ID = "MOMINAAHSAN296/vtb-dataset"
DATASETS = ["feverous", "hybridqa", "sqa", "tabfact", "totto"]
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

# OpenRouter client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
)


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
                    # max_tokens=256,
                    temperature=0.0,
                    top_p=1.0,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    extra_body={
                        "reasoning": {"effort": "minimal"}  # Cost optimization
                    },
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
                # max_tokens=256,
                temperature=0.0,
                top_p=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                extra_body={"reasoning": {"effort": "minimal"}},  # Cost optimization
            )
            return (resp.choices[0].message.content or "").strip()
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
            data_files=f"data/2-task/{dataset_name}.json",
            token=os.getenv("HF_TOKEN"),
            streaming=False,
        )["train"]

        if max_samples and max_samples > 0:
            gt_samples = list(islice(gt_dataset, max_samples))
        else:
            gt_samples = list(gt_dataset)

        if not gt_samples:
            raise ValueError(f"No ground truth samples found for {dataset_name}.")

    except Exception as e:
        raise ValueError(f"Error loading ground truth for {dataset_name}: {e}")

    # Choose prompt set: binary for feverous/tabfact, general otherwise
    binary = dataset_name.lower() in {"feverous", "tabfact"}
    print(f"Using {'binary' if binary else 'general'} prompt set for {dataset_name}")
    TASK_PROMPTS = TASK_PROMPT_BINARY if binary else TASK_PROMPT_GENERAL

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

            # Skip if already processed
            if "task_prediction" in existing_row[format_type]:
                continue

            row = entry.copy()
            row["query"] = row.get("query", "")

            # Fetch image
            try:
                image = fetch_image_by_id(
                    os.getenv("HF_TOKEN"), dataset_name, format_type, image_id
                )
                prompt = TASK_PROMPTS["task_prediction"].format(**row)

                batch_data.append(
                    {
                        "existing_row": existing_row,
                        "image": image,
                        "prompt": prompt,
                        "seq_id": seq_id,
                    }
                )

            except Exception as e:
                print(
                    f"Image for ID {seq_id} and image {image_id} not found in {format_type} format: {e}"
                )
                existing_row[format_type]["_image_missing"] = True
                result_map[seq_id] = existing_row
                continue

        if not batch_data:
            continue

        # Process batch
        batch_requests = [
            {"image": item["image"], "prompt": item["prompt"]} for item in batch_data
        ]
        batch_results = query_vlm_batch(MODEL, batch_requests)

        # Process results
        for batch_idx, result in enumerate(batch_results):
            if batch_idx >= len(batch_data):
                break

            item = batch_data[batch_idx]
            seq_id = item["seq_id"]

            print(f"Prompt: {item['prompt']}")
            print(f"Raw Response: {result}")
            print("---")

            item["existing_row"][format_type]["task_prediction"] = result

            # Update result map
            result_map[seq_id] = item["existing_row"]

        # Save progress after each batch
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(list(result_map.values()), f, ensure_ascii=False, indent=4)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GPT-5 Task Prediction pipeline with batch processing"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max samples per dataset/format"
    )

    args = parser.parse_args()

    print("\n=== GPT-5 Task Prediction Pipeline Configuration ===")
    print(f"  Model           : {MODEL}")
    print(f"  Datasets        : {', '.join(DATASETS)}")
    print(f"  Formats         : {', '.join(FORMATS)}")
    print(f"  Max Samples     : {args.max_samples or 'All'}")
    print(f"  Batch Size      : {BATCH_SIZE}")
    print(f"  Detail          : high")
    print("=================================================\n")

    out_dir = f"results/vlmpipeline/{MODEL}/task"
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

    print("\nGPT-5 Task Prediction Pipeline completed!")


if __name__ == "__main__":
    main()
