import os
import json
import asyncio
import functools
import re
from threading import Lock
from typing import Optional, Dict
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv

# ===================== CONFIG ===================== #

load_dotenv()

MODEL = "gemini-3-flash-preview"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment/.env")

genai.configure(api_key=GEMINI_API_KEY)

# Concurrency / batching
BATCH_SIZE = 10  # concurrent requests (tune based on RPM limits)
semaphore = asyncio.Semaphore(BATCH_SIZE)
checkpoint_lock = Lock()

# Dataset configuration
INPUT_DIR = "data_full/2-task"
DATASETS = ["feverous", "sqa", "tabfact", "wikitq"]  # adjust as needed

# Output directory
OUT_DIR = f"results/question_complexity/{MODEL}"
os.makedirs(OUT_DIR, exist_ok=True)

ERROR_PREFIX = "ERROR_OCCURRED:"
ERROR_MESSAGE = "ERROR_OCCURRED"

# ===================== PROMPT ===================== #

SYSTEM_PROMPT = (
    "You are an expert at classifying table-question pairs into Question Categories.\n"
    "You must follow the rules exactly and output exactly ONE line in the required format."
)

TAGGING_PROMPT_TEMPLATE = """
You are given:

TABLE (as JSON with header + rows):
{table_json}

QUESTION:
{query}

TASK:
Assign exactly ONE Question Category using the definitions below.

===========================================================
CRITICAL RULE (TABLE-REQUIREDNESS)
===========================================================

A question can be assigned a predefined Question Category 
ONLY IF the table is REQUIRED to produce the final answer.

External knowledge policy:

You MAY use external knowledge to:
- Map a description or condition to an entity present in the table
  (e.g., "country with population 67.02 million" → "France")
- Resolve aliases, real-world facts, or descriptive constraints 
  ONLY for the purpose of identifying the correct row(s)

You MUST NOT:
- Output a final answer that does not come directly from table cells
- Compute the final answer using non-table facts
- Label as answerable if the table is not necessary

If the final answer can be obtained without using the table at all,
output:
None of the above — Not table-required

If the required answer value does not exist in or cannot be computed 
from the table cells, output:
None of the above — Missing answer attribute in table


===========================================================
PROCEDURE
===========================================================

Step A — Table-Requiredness Check

1) Identify what the question asks for:
   - single value
   - list/set
   - number
   - yes/no

2) Identify which table cell(s) must be read or aggregated 
   to produce the final answer.

3) Verify that the final answer is directly read from or computed 
   using ONLY table cells.

4) If the table is not needed to produce the final answer →
   None of the above — Not table-required

5) If the answer value is not present or cannot be computed 
   from table cells →
   None of the above — Missing answer attribute in table

Otherwise, proceed to classification.


===========================================================
QUESTION CATEGORIES (Choose EXACTLY ONE)
===========================================================

- Simple Lookup  
  Identify ONE row and read ONE cell.  
  (No filtering beyond locating that row.)

- Conditional Lookup  
  Apply one or more conditions to select row(s), 
  then read ONE resulting value.

- Multi-Item Lookup  
  Return multiple values/rows from the table (a list/set).

- Aggregation / Counting / Arithmetic  
  Compute a number from table values 
  (count/sum/avg/difference/ratio/percent/etc.).

- Comparison & Extremum  
  Choose max/min/earliest/latest by comparing table values.

- Single-step Binary Verification  
  Verify one statement directly using the table.

- Multi-hop Binary Verification  
  Verify a statement requiring multiple reasoning steps 
  across the table.


===========================================================
OUTPUT FORMAT (STRICT)
===========================================================

Output exactly ONE line and nothing else:

Question Category: <CATEGORY NAME>

OR (if not answerable under rules):

Question Category: None of the above
"""

# ===================== IO HELPERS ===================== #

def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)


def load_results(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            rows = json.load(f)
            return {r["id"]: r for r in rows}
        except Exception:
            return {}


def safe_table_json(entry: Dict, max_cell_len: int = 800) -> str:
    """Compact + truncate to avoid huge payloads."""
    table = entry.get("table", {}) or {}
    header = table.get("header", []) or []
    rows = table.get("rows", []) or []

    def trunc(x):
        s = str(x)
        return s if len(s) <= max_cell_len else (s[:max_cell_len] + "…")

    compact = {
        "header": [trunc(h) for h in header],
        "rows": [[trunc(c) for c in r] for r in rows],
    }
    if "caption" in table and table["caption"]:
        compact["caption"] = trunc(table["caption"])

    return json.dumps(compact, ensure_ascii=False, indent=2)

# ===================== PARSING HELPERS ===================== #

QCAT_RE = re.compile(r"^Question Category:\s*(.+)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_tagging_response(text: str) -> Dict[str, Optional[str]]:
    out = {"question_category": None}
    if not text:
        return out
    m = QCAT_RE.search(text)
    if m:
        out["question_category"] = m.group(1).strip()
    return out

# ===================== GEMINI CALL ===================== #

async def run_in_thread(func, *args, **kwargs):
    """Run a blocking Gemini call in a thread pool."""
    async with semaphore:
        return await asyncio.to_thread(functools.partial(func, *args, **kwargs))


async def async_tag_one(
    model,
    entry: Dict,
    retries: int = 5,
    wait_time: int = 8,
) -> str:
    """Async classify one table+query with retries."""
    table_json = safe_table_json(entry)
    query = entry.get("query", "")

    user_prompt = TAGGING_PROMPT_TEMPLATE.format(
        table_json=table_json,
        query=query,
    )

    last_error = None
    for attempt in range(retries):
        try:
            resp = await run_in_thread(
                model.generate_content,
                user_prompt,
            )
            if getattr(resp, "text", None):
                return resp.text.strip()
            return f"{ERROR_PREFIX} finish_reason={resp.candidates[0].finish_reason}"

        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            print(f"[Retry {attempt+1}/{retries}] Error: {last_error}")
            await asyncio.sleep(wait_time)

    return f"{ERROR_PREFIX} {last_error}" if last_error else ERROR_MESSAGE

# ===================== PIPELINE ===================== #

async def run_pipeline_for_dataset(
    model,
    dataset_name: str,
    max_samples: Optional[int],
    out_path: str,
    retries: int = 5,
):
    failed_log_path = out_path.replace(".json", "_FAILED.jsonl")

    input_file = os.path.join(INPUT_DIR, f"{dataset_name}.json")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Dataset file not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if max_samples and max_samples > 0:
        samples = samples[:max_samples]

    results = load_results(out_path)

    stats = {"total": 0, "successful": 0, "failed": 0, "skipped": 0}

    jobs = []
    for entry in samples:
        sid = entry["id"]
        if sid in results and results[sid].get("question_category"):
            stats["skipped"] += 1
            continue
        jobs.append(entry)

    print(f"[QUEUE] {dataset_name}: {len(jobs)} items to tag (skipped={stats['skipped']})")

    for i in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[i:i + BATCH_SIZE]

        tasks = [async_tag_one(model, entry, retries=retries) for entry in batch]
        responses = await asyncio.gather(*tasks)

        for entry, resp in zip(batch, responses):
            sid = entry["id"]
            stats["total"] += 1

            results.setdefault(sid, dict(entry))

            if resp.startswith(ERROR_PREFIX) or resp == ERROR_MESSAGE:
                stats["failed"] += 1
                err = resp.replace(ERROR_PREFIX, "").strip() if resp.startswith(ERROR_PREFIX) else resp
                failed_entry = {
                    "id": sid,
                    "dataset": dataset_name,
                    "query": entry.get("query", ""),
                    "error": err,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                with checkpoint_lock:
                    with open(failed_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(failed_entry, ensure_ascii=False) + "\n")
                results[sid]["question_category"] = ""
                continue

            parsed = parse_tagging_response(resp)

            if not parsed["question_category"]:
                stats["failed"] += 1
                failed_entry = {
                    "id": sid,
                    "dataset": dataset_name,
                    "query": entry.get("query", ""),
                    "error": f"Unparseable output: {resp[:200]}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                with checkpoint_lock:
                    with open(failed_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(failed_entry, ensure_ascii=False) + "\n")
                results[sid]["question_category"] = ""
                continue

            results[sid]["question_category"] = parsed["question_category"]
            stats["successful"] += 1

        with checkpoint_lock:
            atomic_write_json(out_path, list(results.values()))

        print(f"[FLUSH] {dataset_name} batch {i // BATCH_SIZE + 1}/{(len(jobs) + BATCH_SIZE - 1) // BATCH_SIZE}")

    print(f"\n📊 Stats for {dataset_name}: {stats}")
    if stats["failed"] > 0:
        print(f"⚠️  Failures logged to: {failed_log_path}")
    return stats

# ===================== MAIN ===================== #

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Async table-question tagging with Gemini 3 Flash Preview")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per dataset")
    parser.add_argument("--retries", type=int, default=5, help="Retries per request")
    args = parser.parse_args()

    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=200,
        ),
    )

    print("\n=== Tagging Pipeline Configuration ===")
    print(f"Model: {MODEL}")
    print(f"Datasets: {DATASETS}")
    print(f"Input dir: {INPUT_DIR}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Batch size (concurrency): {BATCH_SIZE}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print("====================================\n")

    overall = {"total": 0, "successful": 0, "failed": 0, "skipped": 0}

    for dataset_name in DATASETS:
        out_path = os.path.join(OUT_DIR, f"{dataset_name}.json")
        print(f"\n{'='*70}\nProcessing dataset: {dataset_name}\n{'='*70}")
        stats = await run_pipeline_for_dataset(
            model=model,
            dataset_name=dataset_name,
            max_samples=args.max_samples,
            out_path=out_path,
            retries=args.retries,
        )
        for k in overall:
            overall[k] += stats.get(k, 0)

    print("\n" + "="*70)
    print("📊 OVERALL STATS")
    print("="*70)
    print(overall)
    print("✅ Done.")


if __name__ == "__main__":
    asyncio.run(main())
