import os
import json
import argparse
import asyncio
from itertools import islice

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ===================== CONFIG ===================== #

ERROR_MESSAGE = "ERROR_OCCURRED"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
FORMATS = ["html", "markdown", "latex"]
BATCH_SIZE = 10
EXTENSIONS = {"html": ".html", "markdown": ".md", "latex": ".tex"}

SYSTEM_PROMPT = (
    "You are a precise table reasoning assistant.\n"
    "You must answer questions strictly based on the given table content.\n"
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

def read_table_content_from_hf(hf_token, dataset, fmt, ex_id):
    ext = EXTENSIONS[fmt]
    path = os.path.join(DATA_DIR, "4-representations", fmt, f"{ex_id}{ext}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ===================== ASYNC TEXT QUERY (NO IMAGE) ===================== #

async def async_query_vlm_text(model_name, table_text, prompt, semaphore, retries=5):
    # NOTE: Text-only message. No image parts at all.
    user_text = f"{prompt}\n\nHere is the table content:\n{table_text}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    async with semaphore:
        for _ in range(retries):
            try:
                resp = await async_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=256,
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

# ===================== TASK PREDICTION ===================== #

TASK_PROMPT_GENERAL = (
    "Look at the given table and answer the following question directly.\n{query}"
)
TASK_PROMPT_BINARY = (
    "Look at the given table and answer with 1 (true) or 0 (false) only.\n{query}"
)

async def run_vlm_task_pipeline_text(model_name, hf_token, limit, out_path):
    with open(os.path.join(DATA_DIR, "2-task", "task.json"), "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = list(islice(dataset, limit))
    results = load_results(out_path)
    semaphore = asyncio.Semaphore(BATCH_SIZE)

    jobs = []
    for fmt in FORMATS:
        for entry in samples:
            sid = entry["id"]
            if sid in results and fmt in results[sid] and "task_prediction" in results[sid][fmt]:
                continue
            jobs.append((entry, fmt))

    print(f"[QUEUE] VLM_TEXT Task: {len(jobs)} pending jobs")

    for i in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[i : i + BATCH_SIZE]
        tasks = []
        kept = []

        for entry, fmt in batch:
            sid = entry["id"]

            # OUTPUT JSON STRUCTURE MATCHES YOUR UPDATED VLM SCRIPT:
            # Store full entry once at the top-level.
            results.setdefault(sid, dict(entry))
            results[sid].setdefault(fmt, {})

            try:
                table_text = read_table_content_from_hf(
                    hf_token, "representations", fmt, entry["image_id"]
                )
            except Exception:
                results[sid][fmt]["_table_text_missing"] = True
                continue

            binary = entry["dataset"].lower() in {"feverous", "tabfact"}
            prompt = (TASK_PROMPT_BINARY if binary else TASK_PROMPT_GENERAL).format(
                query=entry["query"]
            )

            tasks.append(async_query_vlm_text(model_name, table_text, prompt, semaphore))
            kept.append((sid, fmt))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for (sid, fmt), resp in zip(kept, responses):
            if resp == ERROR_MESSAGE:
                continue
            
            results[sid][fmt]["task_prediction"] = str(resp)

        atomic_write_json(out_path, list(results.values()))
        print(f"[FLUSH] Task batch {i//BATCH_SIZE + 1}")

# ===================== SUC ===================== #

# TASK_PROMPTS_SUC = {
#     "table_partition": """What is the **first cell value** (not including headers) of the given table? What is the **last cell value** (not including headers) of the given table? Answer questions one by one and use | to split the answer. Answer the question without having any introduction or explanations.""",
#     "table_first_cell": """What is the **first cell value** (not including headers) of the given table? Answer the question without having any introduction or explanations.""",
#     "table_last_cell": """What is the **last cell value** (not including headers) of the given table? Answer the question without having any introduction or explanations.""",
#     "size_detection": """How many rows in the table? How many columns in the table? Answer the questions one by one and use | to split the answer. Answer the question without having any introduction or explanations.""",
#     "number_of_rows": """How many rows in the table? Answer the question without having any introduction or explanations.""",
#     "number_of_columns": """How many columns in the table? Answer the question without having any introduction or explanations.""",
#     "cell_lookup": """Row/column indices start at 0 (top-left is 0|0). What is the position of the cell value {cell_value}? Use row index and column index to answer. Use | to split the answer. Answer the question without having any introduction or explanations.""",
#     "reverse_lookup": """Row/column indices start at 0 (top-left is 0|0). What is the cell value of row index {reverse_lookup_row}, column index {reverse_lookup_col} ? Only output the cell value without other information. Answer the question without having any introduction or explanations.""",
#     "column_retrieval": """Row/column indices start at 0 (top-left is 0|0). What is the column name with the index {column_idx} of the given table? Only give the column name without any explanation. Answer the question without having any introduction or explanations.""",
#     "row_retrieval": """Row/column indices start at 0 (top-left is 0|0). What are the cell values of the {row_idx} row in following table? Only list the cell values one by one using | to split the answers. Answer the question without having any introduction or explanations.""",
# }

TASK_PROMPTS_SUC = {
    "table_partition": """What is the **first cell value** of the given table? What is the **last cell value** of the given table? Answer questions one by one and use | to split the answer. Answer the question without having any introduction or explanations.""",
    "table_first_cell": """What is the **first cell value** of the given table? Answer the question without having any introduction or explanations.""",
    "table_last_cell": """What is the **last cell value** of the given table? Answer the question without having any introduction or explanations.""",
    "size_detection": """How many rows in the table? How many columns in the table? Answer the questions one by one and use | to split the answer. Answer the question without having any introduction or explanations.""",
    "number_of_rows": """How many rows in the table? Answer the question without having any introduction or explanations.""",
    "number_of_columns": """How many columns in the table? Answer the question without having any introduction or explanations.""",
    "cell_lookup": """What is the position of the cell value {cell_value}? Use row index and column index to answer. Use | to split the answer. Answer the question without having any introduction or explanations.""",
    "reverse_lookup": """What is the cell value of row index {reverse_lookup_row}, column index {reverse_lookup_col} ? Only output the cell value without other information. Answer the question without having any introduction or explanations.""",
    "column_retrieval": """What is the column name with the index {column_idx} of the given table image? Only give the column name without any explanation. Answer the question without having any introduction or explanations.""",
    "row_retrieval": """What are the cell values of the {row_idx} row in following table? Only list the cell values one by one using | to split the answers. Answer the question without having any introduction or explanations.""",
}

async def run_vlm_suc_pipeline_text(model_name, hf_token, limit, out_path):
    with open(os.path.join(DATA_DIR, "3-suc", "suc_generation.json"), "r", encoding="utf-8") as f:
        dataset = json.load(f)

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

    print(f"[QUEUE] VLM_TEXT SUC: {len(jobs)} pending jobs")

    for i in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[i : i + BATCH_SIZE]
        tasks = []
        kept = []

        for entry, fmt, task in batch:
            sid = entry["id"]

            # OUTPUT JSON STRUCTURE MATCHES YOUR UPDATED VLM SCRIPT
            results.setdefault(sid, dict(entry))
            results[sid].setdefault(fmt, {})

            try:
                table_text = read_table_content_from_hf(
                    hf_token, "representations", fmt, entry["image_id"]
                )
            except Exception:
                results[sid][fmt]["_table_text_missing"] = True
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
            tasks.append(async_query_vlm_text(model_name, table_text, prompt, semaphore))
            kept.append((sid, fmt, task))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for (sid, fmt, task), resp in zip(kept, responses):
            if resp == ERROR_MESSAGE:
                continue
            results[sid][fmt][task] = str(resp)

        atomic_write_json(out_path, list(results.values()))
        print(f"[FLUSH] SUC batch {i//BATCH_SIZE + 1}")

# ===================== MAIN ===================== #

async def main():
    parser = argparse.ArgumentParser("Unified VLM_TEXT pipeline (Task + SUC)")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--port", type=int, default=8023)
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--suc_only", action="store_true")
    args = parser.parse_args()

    global async_client
    async_client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{args.port}/v1",
    )

    base = "results/prompt2_suc/vlmpipeline_text" if args.suc_only else "results/vlmpipeline_text"
    out_dir = f"{base}/{args.model_name}"
    os.makedirs(out_dir, exist_ok=True)

    if not args.suc_only:
        print("[INFO] Running VLM_TEXT Task Prediction")
        await run_vlm_task_pipeline_text(
            args.model_name,
            args.hf_token,
            args.limit,
            os.path.join(out_dir, "task.json"),
        )

    print("[INFO] Running VLM_TEXT SUC")
    await run_vlm_suc_pipeline_text(
        args.model_name,
        args.hf_token,
        args.limit,
        os.path.join(out_dir, "suc.json"),
    )

    print("[INFO] VLM_TEXT pipeline completed")

if __name__ == "__main__":
    asyncio.run(main())