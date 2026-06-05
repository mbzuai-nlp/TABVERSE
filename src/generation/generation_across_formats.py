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
from difflib import SequenceMatcher
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# NEW: fetch images in-memory from HF without saving to disk
import requests
from io import BytesIO
from huggingface_hub import hf_hub_url

# HF Repo, Dataset and format configuration
REPO_ID = "MOMINAAHSAN296/vtb-dataset"
DATASETS = ["feverous", "hybridqa", "sqa", "tabfact", "totto"]
FORMATS = ["html", "markdown", "latex"]

# --------- Generation Prompts ---------

GENERATION_PROMPTS = {
    "html": "Generate the complete HTML code that exactly represents this image. Provide only the code without any explanations.",
    "latex": "Generate the complete LaTeX code that exactly represents this image. Provide only the code without any explanations.",
    "markdown": "Generate the complete Markdown code that exactly represents this image. Provide only the code without any explanations.",
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
                max_tokens=5000,
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


# --------- Core pipeline (FORMAT-BASED GENERATION) ---------


def run_generation_pipeline_for_source_format(
    source_format,
    target_formats,
    model_path,
    hf_token,
    max_samples,
    out_path,
):
    # resume-safe per source format
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf8") as f:
            try:
                result_map = {row["sample_id"]: row for row in json.load(f)}
            except Exception:
                result_map = {}
    else:
        result_map = {}

    # Load combined subset data from HuggingFace
    try:
        print(f"Loading combined subset data from HuggingFace")
        gt_dataset = load_dataset(
            REPO_ID,
            data_files="data/4-subset/combined_subset.json",
            token=hf_token,
            streaming=False,
        )["train"]
        print(f"Total samples in subset: {len(gt_dataset)}")
        max_samples = len(gt_dataset)
        gt_samples = list(islice(gt_dataset, max_samples))
        if not gt_samples:
            raise ValueError(f"No samples found in combined subset.")

    except Exception as e:
        raise ValueError(f"Error loading combined subset from HuggingFace: {e}")

    for entry in gt_samples:
        sample_id = entry["sample_id"]
        dataset_name = entry["dataset"]
        image_id = entry["image_id"]

        existing_row = result_map.get(sample_id, entry.copy())

        # Initialize source format data if not exists
        if source_format not in existing_row:
            existing_row[source_format] = {}

        # fetch source image
        try:
            image = fetch_image_by_id(hf_token, dataset_name, source_format, image_id)
        except Exception as e:
            print(
                f"Source image for sample_id {sample_id} and image {image_id} not found in {source_format} format: {e}"
            )
            existing_row[source_format]["_image_missing"] = True
            result_map[sample_id] = existing_row
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(list(result_map.values()), f, ensure_ascii=False, indent=4)
            continue

        # Generate code for each target format
        for target_format in target_formats:
            if target_format not in existing_row[source_format]:
                existing_row[source_format][target_format] = {}

            if "generated" not in existing_row[source_format][target_format]:
                try:
                    prompt = GENERATION_PROMPTS[target_format]
                    generated_code = query_vlm(model_path, image, prompt)

                    print(f"Sample ID: {sample_id}, Dataset: {dataset_name}")
                    print(f"Source: {source_format} -> Target: {target_format}")
                    print(
                        f"Generated: {generated_code[:100]}..."
                    )  # Show first 100 chars
                    print("---")

                    existing_row[source_format][target_format][
                        "generated"
                    ] = generated_code

                except Exception as e:
                    print(
                        f"Error generating {target_format} from {source_format} for sample_id {sample_id}: {e}"
                    )
                    existing_row[source_format][target_format]["generated"] = "ERROR"

        # write/merge
        result_map[sample_id] = existing_row
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(list(result_map.values()), f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="VLM Generation pipeline with HF streaming images (no local downloads)"
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
        "--max_samples", type=int, default=100, help="Max samples per dataset/format"
    )

    args = parser.parse_args()

    # vLLM client
    global client
    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{args.port}/v1")

    print("\n=== Generation Pipeline Configuration ===")
    print(f"  Model Name      : {args.model_name}")
    print(f"  Model Path      : {args.model_path}")
    print(f"  Datasets        : {', '.join(DATASETS)}")
    print(f"  Formats         : {', '.join(FORMATS)}")
    print(f"  Max Samples     : {args.max_samples if args.max_samples else 'All'}")
    print("=================================\n")

    out_dir = f"results/vlmpipeline/{args.model_name}/generation_format"
    os.makedirs(out_dir, exist_ok=True)

    # Define target formats for each source format
    format_mappings = {
        "html": ["latex", "markdown"],  # HTML images ->   LaTeX + Markdown
        "markdown": ["latex", "html"],  # Markdown images ->   LaTeX + HTML
        "latex": ["html", "markdown"],  # LaTeX images ->  HTML + Markdown
    }

    for source_format in FORMATS:
        target_formats = format_mappings[source_format]
        out_path = os.path.join(out_dir, f"{source_format}.json")

        try:
            print(f"\n=== Processing {source_format.upper()} as source format ===")
            print(f"Target formats: {', '.join(target_formats)}")

            run_generation_pipeline_for_source_format(
                source_format,
                target_formats,
                args.model_name,  # NAME!!
                args.hf_token,
                args.max_samples,
                out_path,
            )
        except Exception as e:
            print(f"Error processing {source_format}: {e}")
            continue

    print("\nGeneration pipeline completed!")


if __name__ == "__main__":
    main()
