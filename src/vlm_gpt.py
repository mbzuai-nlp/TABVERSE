import os
import re
import json
import base64
import argparse
import time
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

ERROR_MESSAGE = "ERROR_OCCURRED"

REPO_ID = "MOMINAAHSAN296/tabverse"
FORMATS = ["html", "markdown", "latex"]
BATCH_SIZE = 10

SYSTEM_PROMPT = (
    "You are a precise visual table reasoning assistant.\n"
    "You must answer questions strictly based on the given image content.\n"
    "or convert the given table image into the requested structured format.\n"
    "Do not use external knowledge.\n"
    "Do not guess or infer missing information.\n"
    "Do not add explanations, prefixes, or extra text.\n"
    "Follow the output format exactly as requested."
)

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


# ===================== IMAGE HELPERS ===================== #


def _pil_from_bytes(b: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(b))
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def fetch_image_by_id(hf_token, dataset, fmt, image_id):
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


# ===================== ASYNC VLM QUERY ===================== #


async def async_query_vlm(
    model_name,
    image,
    prompt,
    semaphore,
    retries=5,
    wait_time=8,
):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64(image), "detail": "high"},
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    async with semaphore:
        for attempt in range(retries):
            try:
                resp = await async_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_completion_tokens=5000,  # generation needs more room
                    top_p=1.0,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    reasoning_effort="low",
                )
                return (resp.choices[0].message.content or "").strip()

            except Exception as e:
                await asyncio.sleep(wait_time)
                
    return ERROR_MESSAGE


# ===================== TASK PREDICTION ===================== #

TASK_PROMPT_GENERAL = (
    "Look at the given image and answer the following question directly.\n{query}"
)
TASK_PROMPT_BINARY = (
    "Look at the given image and answer with 1 (true) or 0 (false) only.\n{query}"
)


async def run_vlm_task_pipeline(model_name, hf_token, limit, out_path):
    dataset = load_dataset(
        REPO_ID,
        data_files="data/2-task/task.json",
        token=hf_token,
    )["train"]

    samples = list(islice(dataset, limit))
    results = load_results(out_path)

    semaphore = asyncio.Semaphore(BATCH_SIZE)

    jobs = []
    for fmt in FORMATS:
        for entry in samples:
            sid = entry["id"]
            if (
                sid in results
                and fmt in results[sid]
                and "task_prediction" in results[sid][fmt]
            ):
                continue
            jobs.append((entry, fmt))

    print(f"[QUEUE] VLM Task: {len(jobs)} pending jobs")

    for i in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[i : i + BATCH_SIZE]
        tasks = []

        for entry, fmt in batch:
            sid = entry["id"]

            # STORE FULL INPUT ENTRY ONCE
            results.setdefault(sid, dict(entry))
            results[sid].setdefault(fmt, {})

            try:
                image = fetch_image_by_id(
                    hf_token, "representations", fmt, entry["image_id"]
                )
            except Exception:
                results[sid][fmt]["_image_missing"] = True
                continue

            binary = entry["dataset"].lower() in {"feverous", "tabfact"}
            prompt = (TASK_PROMPT_BINARY if binary else TASK_PROMPT_GENERAL).format(
                query=entry["query"]
            )

            tasks.append(async_query_vlm(model_name, image, prompt, semaphore))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        idx = 0
        for entry, fmt in batch:
            sid = entry["id"]
            resp = responses[idx]
            
            if resp != ERROR_MESSAGE:        
                results[sid][fmt]["task_prediction"] = str(resp)
                
            idx += 1

        atomic_write_json(out_path, list(results.values()))
        print(f"[FLUSH] Task batch {i//BATCH_SIZE + 1}")


# ===================== SUC ===================== #

TASK_PROMPTS_SUC = {
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


async def run_vlm_suc_pipeline(model_name, hf_token, limit, out_path):
    dataset = load_dataset(
        REPO_ID,
        data_files="data/3-suc/suc_generation.json",
        token=hf_token,
    )["train"]

    samples = list(islice(dataset, limit))
    results = load_results(out_path)

    semaphore = asyncio.Semaphore(BATCH_SIZE)

    jobs = []
    for fmt in FORMATS:
        for task in TASK_PROMPTS_SUC:
            for entry in samples:
                sid = entry["id"]
                if sid in results and fmt in results[sid] and task in results[sid][fmt]:
                    continue
                jobs.append((entry, fmt, task))

    print(f"[QUEUE] VLM SUC: {len(jobs)} pending jobs")

    for i in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[i : i + BATCH_SIZE]
        tasks = []

        for entry, fmt, task in batch:
            sid = entry["id"]

            # STORE FULL INPUT ENTRY ONCE
            results.setdefault(sid, dict(entry))
            results[sid].setdefault(fmt, {})

            try:
                image = fetch_image_by_id(
                    hf_token, "representations", fmt, entry["image_id"]
                )
            except Exception:
                results[sid][fmt]["_image_missing"] = True
                continue

            gt = entry.get("suc", {})
            rlk = gt.get("reverse_lookup_indices", "0|0").split("|")
            vars_ = {
                "cell_value": gt.get("cell_value", ""),
                "reverse_lookup_row": rlk[0],
                "reverse_lookup_col": rlk[1],
                "column_idx": gt.get("column_idx", ""),
                "row_idx": gt.get("row_idx", ""),
            }

            prompt = TASK_PROMPTS_SUC[task].format(**vars_)
            tasks.append(async_query_vlm(model_name, image, prompt, semaphore))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        idx = 0
        for entry, fmt, task in batch:
            sid = entry["id"]
            resp = responses[idx]
            
            if resp != ERROR_MESSAGE:        
                results[sid][fmt][task] = str(resp)
            
            idx += 1

        atomic_write_json(out_path, list(results.values()))
        print(f"[FLUSH] SUC batch {i//BATCH_SIZE + 1}")


# ===================== INTER + INTRA GENERATION PIPELINE ===================== #
# Generation system prompt (image -> code). Keep it strict like your VLM script.
GEN_SYSTEM_PROMPT = (
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


async def run_vlm_generation_pipeline(model_name, hf_token, limit, out_path):
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
                image = fetch_image_by_id(
                    hf_token, "representations", src_fmt, entry["image_id"]
                )
            except Exception:
                results[sid][src_fmt]["_image_missing"] = True
                continue

            prompt = GEN_PROMPTS[tgt_fmt]
            tasks.append(async_query_vlm(model_name, image, prompt, semaphore))
            kept.append((sid, src_fmt, tgt_fmt))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for (sid, src_fmt, tgt_fmt), resp in zip(kept, responses):
            if tgt_fmt in results[sid][src_fmt]["generation"]:
                continue
            results[sid][src_fmt]["generation"][tgt_fmt] = (
                str(resp) if isinstance(resp, Exception) else resp
            )

        atomic_write_json(out_path, list(results.values()))
        print(f"[FLUSH] Generation batch {i//BATCH_SIZE + 1}")


# ===================== MAIN ===================== #


async def main():
    parser = argparse.ArgumentParser("Unified VLM pipeline (Task + SUC)")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    global async_client
    async_client = AsyncOpenAI(api_key=args.api_key)
    out_dir = f"results/vlmpipeline/{args.model_name}"
    os.makedirs(out_dir, exist_ok=True)

    print("[INFO] Running VLM Task Prediction")
    await run_vlm_task_pipeline(
        args.model_name,
        args.hf_token,
        args.limit,
        os.path.join(out_dir, "task.json"),
    )

    print("[INFO] Running VLM SUC")
    await run_vlm_suc_pipeline(
        args.model_name,
        args.hf_token,
        args.limit,
        os.path.join(out_dir, "suc.json"),
    )

    print("[INFO] Running Inter + Intra Generation (3x3)")
    await run_vlm_generation_pipeline(
        args.model_name,
        args.hf_token,
        args.limit,
        os.path.join(out_dir, "generation.json"),
    )

    print("[INFO] VLM pipeline completed")


if __name__ == "__main__":
    asyncio.run(main())
