import os
import json
import base64
import argparse
import io
import asyncio
from itertools import islice

from PIL import Image
from datasets import load_dataset
from openai import AsyncOpenAI
from dotenv import load_dotenv
from huggingface_hub import hf_hub_url
import requests

load_dotenv()

# ===================== CONFIG ===================== #

ERROR_MESSAGE = "CONNECTION_FAILED"

REPO_ID = "MOMINAAHSAN296/tabverse"
FORMATS = ["html", "markdown", "latex"]
BATCH_SIZE = 10

# Generation system prompt (image -> code). Keep it strict like your VLM script.
SYSTEM_PROMPT = (
    "You are a precise visual table transcription assistant.\n"
    "You must convert the given table image into the requested structured format.\n"
    "Do not use external knowledge.\n"
    "Do not guess or infer missing information.\n"
    "Output ONLY the requested code/text in the target format.\n"
    "Do not add explanations, prefixes, or extra text."
)

# Target-format prompts
GEN_PROMPTS = {
    "html": (
        "Generate the complete HTML code that exactly represents the table in this image. "
        "Provide only the HTML code without any explanations."
    ),
    "markdown": (
        "Generate the complete Markdown table that exactly represents the table in this image. "
        "Provide only the Markdown without any explanations."
    ),
    "latex": (
        "Generate the complete LaTeX tabular code that exactly represents the table in this image. "
        "Provide only the LaTeX code without any explanations."
    ),
}

async_client = None

# ===================== HELPERS ===================== #

def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)

def load_results(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f)}

# ===================== IMAGE HELPERS (UNCHANGED) ===================== #

def _pil_from_bytes(b: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(b))
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im

def fetch_image_by_id(hf_token, dataset, fmt, image_id):
    """
    Matches your VLM script style.
    Assumes HF path: {dataset}/{fmt}/{image_id}.png
    In your QA/SUC VLM script, dataset="representations".
    """
    filename = f"{dataset}/{fmt}/{image_id}.png"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    url = hf_hub_url(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return _pil_from_bytes(r.content)

def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

# ===================== ASYNC VLM QUERY (MINIMAL CHANGE: max_tokens bigger) ===================== #

async def async_query_vlm_generation(model_name, image, prompt, semaphore, retries=5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_base64(image)}},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    async with semaphore:
        for _ in range(retries):
            try:
                resp = await async_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=5000,  # generation needs more room
                    temperature=0.0,
                    top_p=1.0,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    extra_body={"best_of": 1, "top_k": -1},
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                await asyncio.sleep(3)

    return ERROR_MESSAGE

# ===================== INTER + INTRA GENERATION PIPELINE ===================== #

async def run_vlm_inter_intra_generation_pipeline(model_name, hf_token, limit, out_path):
    """
    Loads subset list and performs 3x3:
      source image format (html/md/tex) -> target text format (html/md/tex).
    Output JSON style mirrors your VLM script:
      results[sid] = full entry once
      results[sid][src_fmt]["generation"][tgt_fmt] = generated_text
    """

    # IMPORTANT: dataset file path
    # If your subset file is different, change ONLY this string.
    dataset = load_dataset(
        REPO_ID,
        data_files="data/3-suc/suc_generation.json",
        token=hf_token,
    )["train"]

    samples = list(islice(dataset, limit))
    results = load_results(out_path)
    semaphore = asyncio.Semaphore(BATCH_SIZE)

    jobs = []
    for src_fmt in FORMATS:
        for tgt_fmt in FORMATS:
            for entry in samples:
                sid = entry["id"] if "id" in entry else entry.get("sample_id")

                # Safety: ensure we have an id field
                if sid is None:
                    # if the subset uses a different key, you can replace this logic
                    continue

                # Resume check
                if (
                    sid in results
                    and src_fmt in results[sid]
                    and "generation" in results[sid][src_fmt]
                    and tgt_fmt in results[sid][src_fmt]["generation"]
                ):
                    continue

                jobs.append((entry, sid, src_fmt, tgt_fmt))

    print(f"[QUEUE] VLM Inter+Intra Generation: {len(jobs)} pending jobs")

    for i in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[i : i + BATCH_SIZE]
        tasks = []
        kept = []

        for entry, sid, src_fmt, tgt_fmt in batch:
            # STORE FULL INPUT ENTRY ONCE (same change you made in VLM QA/SUC script)
            results.setdefault(sid, {k: v for k, v in entry.items() if k != "suc"})

            # Initialize nesting like your VLM script style
            results[sid].setdefault(src_fmt, {})
            results[sid][src_fmt].setdefault("generation", {})

            # Fetch source image (from the same place as QA/SUC VLM script)
            # If your generation images are elsewhere, change only "representations".
            try:
                image = fetch_image_by_id(hf_token, "representations", src_fmt, entry["image_id"])
            except Exception:
                results[sid][src_fmt]["_image_missing"] = True
                continue

            prompt = GEN_PROMPTS[tgt_fmt]
            tasks.append(async_query_vlm_generation(model_name, image, prompt, semaphore))
            kept.append((sid, src_fmt, tgt_fmt))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for (sid, src_fmt, tgt_fmt), resp in zip(kept, responses):
            
            if resp != ERROR_MESSAGE:
                results[sid][src_fmt]["generation"][tgt_fmt] = str(resp)

        atomic_write_json(out_path, list(results.values()))
        print(f"[FLUSH] Generation batch {i//BATCH_SIZE + 1}")

# ===================== MAIN ===================== #

async def main():
    parser = argparse.ArgumentParser("VLM pipeline (Inter+Intra Generation)")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--port", type=int, default=8023)
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    global async_client
    async_client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{args.port}/v1",
    )

    out_dir = f"results/vlmpipeline/{args.model_name}"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "inter_intra_generation.json")

    print("[INFO] Running Inter + Intra Generation (3x3)")
    await run_vlm_inter_intra_generation_pipeline(
        args.model_name,
        args.hf_token,
        args.limit,
        out_path,
    )

    print("[INFO] Generation pipeline completed")

if __name__ == "__main__":
    asyncio.run(main())