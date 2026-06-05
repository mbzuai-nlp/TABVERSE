import os
import re
import json
import argparse
from itertools import islice
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
from huggingface_hub import hf_hub_url
import requests

load_dotenv()

REPO_ID = "MOMINAAHSAN296/tabverse"
FORMATS = ["html", "markdown", "latex"]
EXTENSIONS = {"html": ".html", "markdown": ".md", "latex": ".tex"}

SYSTEM_PROMPT = (
    "You are a precise table reasoning assistant.\n"
    "You must answer questions strictly based on the given table content.\n"
    "Do not use external knowledge.\n"
    "Do not guess or infer missing information.\n"
    "Do not add explanations, prefixes, or extra text.\n"
    "Follow the output format exactly as requested by user."
)

# ===================== Shared Utilities ===================== #


def read_table_content_from_hf(hf_token, dataset, fmt, ex_id):
    ext = EXTENSIONS[fmt]
    filename = f"{dataset}/{fmt}/{ex_id}{ext}"
    url = hf_hub_url(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.text


def batch_query_llm(client, model_name, batch, max_tokens=256):
    """
    batch: list of {meta..., messages}
    NOTE: The OpenAI chat completions API expects `messages` to be a list of dicts
    for a single completion call. The previous code attempted to pass a list of
    lists (one messages list per item) in a single call which causes a 400.
    This function calls the API once per item and returns outputs in the same order.
    """
    outputs = []
    for item in batch:
        resp = client.chat.completions.create(
            model=model_name,
            messages=item["messages"],
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            extra_body={"best_of": 1, "top_k": -1},
        )
        content = (resp.choices[0].message.content or "").strip()
        outputs.append({**item, "raw_response": content})
    return outputs


# ===================== TASK PREDICTION ===================== #

TASK_PROMPT_GENERAL = (
    "Look at the given table and answer the following question directly. "
    "Do not include introductions or explanations.\n{query}"
)

TASK_PROMPT_BINARY = (
    "Look at the given table and answer with 1 (true) or 0 (false) only.\n{query}"
)

# ============================ SUC ============================ #

TASK_PROMPTS_SUC = {
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


def normalize_task_prediction(raw, binary):
    s = re.sub(r"(?i)\b(answer|final answer)\s*:\s*", "", raw).strip()
    if not binary:
        return s
    if "1" in s:
        return "1"
    if "0" in s:
        return "0"
    return s


def run_task_prediction_pipeline(
    client, model_name, hf_token, max_samples, batch_size, out_path
):
    dataset = load_dataset(
        REPO_ID,
        data_files="data/2-task/task.json",
        token=hf_token,
        streaming=False,
    )["train"]

    samples = list(islice(dataset, max_samples))

    # ---------- SAFE RESUME ----------
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            existing = json.load(f)
        results = {row["id"]: row for row in existing}
    else:
        results = {}

    # ensure all current samples exist in results
    for entry in samples:
        if entry["id"] not in results:
            results[entry["id"]] = entry.copy()

    # ---------- DEBUG RESUME INFO ----------
    total = len(samples)
    done = sum(
        1
        for row in results.values()
        if any(
            fmt in row and "task_prediction_processed" in row.get(fmt, {})
            for fmt in FORMATS
        )
    )
    print(f"[RESUME] Task Prediction: {done}/{total} samples already processed")

    for fmt in FORMATS:
        batch = []

        for entry in samples:
            row_existing = results[entry["id"]]
            if fmt in row_existing and "task_prediction_processed" in row_existing.get(
                fmt, {}
            ):
                continue

            table = read_table_content_from_hf(
                hf_token, "representations", fmt, entry["image_id"]
            )

            binary = entry["dataset"].lower() in {"feverous", "tabfact"}
            prompt = (TASK_PROMPT_BINARY if binary else TASK_PROMPT_GENERAL).format(
                query=entry["query"]
            )

            batch.append(
                {
                    "id": entry["id"],
                    "format": fmt,
                    "binary": binary,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"{prompt}\n\nHere is the table content:\n{table}",
                        },
                    ],
                }
            )

            if len(batch) == batch_size:
                outputs = batch_query_llm(client, model_name, batch)
                for o in outputs:
                    row = results[o["id"]]
                    row.setdefault(fmt, {})
                    row[fmt]["task_prediction"] = o["raw_response"]
                    row[fmt]["task_prediction_processed"] = normalize_task_prediction(
                        o["raw_response"], o["binary"]
                    )
                batch.clear()

                with open(out_path, "w") as f:
                    json.dump(list(results.values()), f, indent=4)

        if batch:
            outputs = batch_query_llm(client, model_name, batch)
            for o in outputs:
                row = results[o["id"]]
                row.setdefault(fmt, {})
                row[fmt]["task_prediction"] = o["raw_response"]
                row[fmt]["task_prediction_processed"] = normalize_task_prediction(
                    o["raw_response"], o["binary"]
                )

            with open(out_path, "w") as f:
                json.dump(list(results.values()), f, indent=4)


# ===================== SUC PIPELINE ===================== #
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


def run_suc_pipeline(client, model_name, hf_token, max_samples, batch_size, out_path):
    dataset = load_dataset(
        REPO_ID,
        data_files="data/3-suc/suc_generation.json",
        token=hf_token,
        streaming=False,
    )["train"]

    samples = list(islice(dataset, max_samples))

    # ---------- SAFE RESUME + LIMIT-INCREASE FIX ----------
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            existing = json.load(f)
        results = {row["id"]: row for row in existing}
    else:
        results = {}

    for entry in samples:
        if entry["id"] not in results:
            results[entry["id"]] = entry.copy()

    # ---------- DEBUG RESUME INFO ----------
    total = len(samples)
    done = sum(
        1 for row in results.values() if any(fmt in row and row[fmt] for fmt in FORMATS)
    )
    print(f"[RESUME] SUC: {done}/{total} samples already have partial or full outputs")

    for fmt in FORMATS:
        for task_key, prompt_template in TASK_PROMPTS_SUC.items():
            batch = []

            for entry in samples:
                row_existing = results[entry["id"]]
                if fmt in row_existing and task_key in row_existing.get(fmt, {}):
                    continue

                table = read_table_content_from_hf(
                    hf_token, "representations", fmt, entry["image_id"]
                )

                gt = entry.get("suc", {})
                rlk = gt.get("reverse_lookup_indices", "0|0").split("|")

                prompt = prompt_template.format(
                    cell_value=gt.get("cell_value", ""),
                    reverse_lookup_row=rlk[0],
                    reverse_lookup_col=rlk[1],
                    column_idx=gt.get("column_idx", ""),
                    row_idx=gt.get("row_idx", ""),
                )

                batch.append(
                    {
                        "id": entry["id"],
                        "format": fmt,
                        "task": task_key,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": f"{prompt}\n\nHere is the table content:\n{table}",
                            },
                        ],
                    }
                )

                if len(batch) == batch_size:
                    outputs = batch_query_llm(client, model_name, batch)
                    for o in outputs:
                        row = results[o["id"]]
                        row.setdefault(fmt, {})
                        row[fmt][task_key] = o["raw_response"]
                        row[fmt][f"{task_key}_processed"] = post_process_response(
                            task_key, o["raw_response"]
                        )
                    batch.clear()

                    with open(out_path, "w") as f:
                        json.dump(list(results.values()), f, indent=4)

            if batch:
                outputs = batch_query_llm(client, model_name, batch)
                for o in outputs:
                    row = results[o["id"]]
                    row.setdefault(fmt, {})
                    row[fmt][task_key] = o["raw_response"]
                    row[fmt][f"{task_key}_processed"] = post_process_response(
                        task_key, o["raw_response"]
                    )

                with open(out_path, "w") as f:
                    json.dump(list(results.values()), f, indent=4)

    # clean temp fields
    for row in results.values():
        for k in [
            "cell_value",
            "reverse_lookup_row",
            "reverse_lookup_col",
            "column_idx",
            "row_idx",
        ]:
            row.pop(k, None)


# ===================== MAIN ===================== #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--port", type=int, default=8033)
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{args.port}/v1")

    out_dir = f"results/llmpipeline/{args.model_name}"
    os.makedirs(out_dir, exist_ok=True)

    print("Running Task Prediction")
    run_task_prediction_pipeline(
        client,
        args.model_name,
        args.hf_token,
        args.limit,
        args.batch_size,
        os.path.join(out_dir, "task.json"),
    )

    print("Running SUC Tasks")
    run_suc_pipeline(
        client,
        args.model_name,
        args.hf_token,
        args.limit,
        args.batch_size,
        os.path.join(out_dir, "suc.json"),
    )

    print("All pipelines completed")


if __name__ == "__main__":
    main()
