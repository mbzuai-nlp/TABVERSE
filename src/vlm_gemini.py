import os
import re
import json
import base64
import argparse
import time
import io
import asyncio
import functools
from threading import Lock
from itertools import islice
from typing import Dict, Any

from PIL import Image
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_url
import requests

import google.generativeai as genai
from google.generativeai import types

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


# ===================== ASYNC + CHECKPOINT ===================== #

semaphore = asyncio.Semaphore(BATCH_SIZE)
checkpoint_lock = Lock()


async def run_in_thread(func, *args, **kwargs):
    async with semaphore:
        return await asyncio.to_thread(functools.partial(func, *args, **kwargs))


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


# ===================== GEMINI VLM QUERY ===================== #


async def async_query_gemini_vlm(
    model,
    image: Image.Image,
    prompt: str,
    retries: int = 5,
    wait_time: int = 8,
):
    for attempt in range(retries):
        try:
            resp = await run_in_thread(
                model.generate_content,
                [image, prompt],
            )

            if getattr(resp, "text", None):
                return resp.text.strip()

            return f"ERROR: finish_reason={resp.candidates[0].finish_reason}"

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


async def run_vlm_task_pipeline(model, hf_token, limit, out_path):
    dataset = load_dataset(
        REPO_ID,
        data_files="data/2-task/task.json",
        token=hf_token,
    )["train"]

    samples = list(islice(dataset, limit))
    results = load_results(out_path)

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

    print(f"[QUEUE] Task jobs: {len(jobs)}")

    for i in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[i : i + BATCH_SIZE]
        tasks, kept = [], []

        for entry, fmt in batch:
            sid = entry["id"]
            results.setdefault(sid, dict(entry))
            results[sid].setdefault(fmt, {})

            try:
                image = fetch_image_by_id(
                    hf_token, "representations", fmt, entry["image_id"]
                )
            except Exception as e:
                print(f"[WARN] Image fetch failed for {sid} {fmt}: {e}")
                results[sid][fmt]["_image_missing"] = True
                continue

            binary = entry["dataset"].lower() in {"feverous", "tabfact"}
            prompt = (TASK_PROMPT_BINARY if binary else TASK_PROMPT_GENERAL).format(
                query=entry["query"]
            )

            tasks.append(async_query_gemini_vlm(model, image, prompt))
            kept.append((sid, fmt))

        responses = await asyncio.gather(*tasks)

        for (sid, fmt), resp in zip(kept, responses):
            
            if resp != ERROR_MESSAGE:
                results[sid][fmt]["task_prediction"] = resp

        atomic_write_json(out_path, list(results.values()))
        print(f"[FLUSH] Task batch {i // BATCH_SIZE + 1}")


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


async def run_vlm_suc_pipeline(model, hf_token, limit, out_path):
    dataset = load_dataset(
        REPO_ID,
        data_files="data/3-suc/suc_generation.json",
        token=hf_token,
    )["train"]

    samples = list(islice(dataset, limit))
    results = load_results(out_path)

    jobs = []
    for fmt in FORMATS:
        for task in TASK_PROMPTS_SUC:
            for entry in samples:
                sid = entry["id"]
                if sid in results and fmt in results[sid] and task in results[sid][fmt]:
                    continue
                jobs.append((entry, fmt, task))

    print(f"[QUEUE] SUC jobs: {len(jobs)}")

    for i in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[i : i + BATCH_SIZE]
        tasks, kept = [], []

        for entry, fmt, task in batch:
            sid = entry["id"]
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
            tasks.append(async_query_gemini_vlm(model, image, prompt))
            kept.append((sid, fmt, task))

        responses = await asyncio.gather(*tasks)

        for (sid, fmt, task), resp in zip(kept, responses):
    
            if resp != ERROR_MESSAGE:
                results[sid][fmt][task] = resp

        atomic_write_json(out_path, list(results.values()))
        print(f"[FLUSH] SUC batch {i // BATCH_SIZE + 1}")


# ===================== GENERATION ===================== #
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


async def run_vlm_generation_pipeline(model, hf_token, limit, out_path):
    dataset = load_dataset(
        REPO_ID,
        data_files="data/3-suc/suc_generation.json",
        token=hf_token,
    )["train"]

    samples = list(islice(dataset, limit))
    results = load_results(out_path)

    jobs = []
    for src_fmt in FORMATS:
        for tgt_fmt in FORMATS:
            for entry in samples:
                sid = entry["id"]
                if (
                    sid in results
                    and src_fmt in results[sid]
                    and "generation" in results[sid][src_fmt]
                    and tgt_fmt in results[sid][src_fmt]["generation"]
                ):
                    continue
                jobs.append((entry, sid, src_fmt, tgt_fmt))

    print(f"[QUEUE] Generation jobs: {len(jobs)}")

    for i in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[i : i + BATCH_SIZE]
        tasks, kept = [], []

        for entry, sid, src_fmt, tgt_fmt in batch:
            results.setdefault(sid, dict(entry))
            results[sid].setdefault(src_fmt, {})
            results[sid][src_fmt].setdefault("generation", {})

            try:
                image = fetch_image_by_id(
                    hf_token, "representations", src_fmt, entry["image_id"]
                )
            except Exception:
                results[sid][src_fmt]["_image_missing"] = True
                continue

            prompt = GEN_PROMPTS[tgt_fmt]
            tasks.append(async_query_gemini_vlm(model, image, prompt))
            kept.append((sid, src_fmt, tgt_fmt))

        responses = await asyncio.gather(*tasks)

        for (sid, src_fmt, tgt_fmt), resp in zip(kept, responses):
            results[sid][src_fmt]["generation"][tgt_fmt] = resp

        atomic_write_json(out_path, list(results.values()))
        print(f"[FLUSH] Generation batch {i // BATCH_SIZE + 1}")


# ===================== MAIN ===================== #


async def main():
    parser = argparse.ArgumentParser("Gemini VLM Pipeline")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        args.model_name,
        system_instruction=SYSTEM_PROMPT,
        generation_config=types.GenerationConfig(
            temperature=0,
            top_p=1,
            max_output_tokens=5000,
        ),
    )

    out_dir = f"results/vlmpipeline/{args.model_name}"
    os.makedirs(out_dir, exist_ok=True)

    await run_vlm_task_pipeline(
        model, args.hf_token, args.limit, os.path.join(out_dir, "task.json")
    )

    await run_vlm_suc_pipeline(
        model, args.hf_token, args.limit, os.path.join(out_dir, "suc.json")
    )

    await run_vlm_generation_pipeline(
        model, args.hf_token, args.limit, os.path.join(out_dir, "generation.json")
    )

    print("[DONE] Gemini VLM pipeline completed")


if __name__ == "__main__":
    asyncio.run(main())
