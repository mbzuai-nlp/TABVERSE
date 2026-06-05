# Two new metric added ; Field Accuracy gives partial credit for pipe-separated answers by comparing each gold field with the prediction at the same position, e.g. 10|4 vs 10|5 gets 0.5.

# Relaxed Accuracy checks whether the gold answer fields appear anywhere in the prediction, even if the output format is wrong, e.g. 10|4 vs row: 10 col: 4 gets 1.0.

import os
import json
import re
import csv
import argparse
from collections import defaultdict

TASK_KEYS = [
    "table_partition",
    "table_first_cell",
    "table_last_cell",
    "size_detection",
    "number_of_rows",
    "number_of_columns",
    "cell_lookup",
    "reverse_lookup",
    "column_retrieval",
    "row_retrieval",
]

FORMATS = ["html", "markdown", "latex"]

# Abbreviated task column names (match paper table header)
TASK_ABBREV = {
    "table_partition":   "T.P.",
    "table_first_cell":  "F.C.",
    "table_last_cell":   "L.C.",
    "size_detection":    "S.D.",
    "number_of_rows":    "#Rows",
    "number_of_columns": "#Cols",
    "cell_lookup":       "C.Lu.",
    "reverse_lookup":    "R.Lu.",
    "column_retrieval":  "Co.Rt.",
    "row_retrieval":     "Ro.Rt.",
}

WIDE_HEADER = ["model", "format"] + list(TASK_ABBREV.values()) + ["OVERALL"]

# ── Pipeline configs ────────────────────────────────────────────────────────
PIPELINE_CONFIGS = {
    "vlm": {
        "label":           "VLM",
        "results_dir":     "results/vlmpipeline",
        "pred_key":        lambda task: task,
        "prompt_version":  "original",
    },
    "vlm-text": {
        "label":           "VLM-TEXT",
        "results_dir":     "results/vlmpipeline_text",
        "pred_key":        lambda task: task,
        "prompt_version":  "original",
    },
    "llm": {
        "label":           "LLM",
        "results_dir":     "results/llmpipeline",
        "pred_key":        lambda task: task,
        "prompt_version":  "original",
    },
    "vlm_prompt2": {
        "label":           "VLM",
        "results_dir":     "results/prompt2_suc/vlmpipeline",
        "pred_key":        lambda task: task,
        "prompt_version":  "prompt2",
    },
    "vlm-text_prompt2": {
        "label":           "VLM-TEXT",
        "results_dir":     "results/prompt2_suc/vlmpipeline_text",
        "pred_key":        lambda task: task,
        "prompt_version":  "prompt2",
    },
    "llm_prompt2": {
        "label":           "LLM",
        "results_dir":     "results/prompt2_suc/llmpipeline",
        "pred_key":        lambda task: task,
        "prompt_version":  "prompt2",
    },
}

FINAL_CSV_DIR = os.path.join("results", "scores", "suc")
SUMMARY_CSV   = os.path.join(FINAL_CSV_DIR, "suc.csv")
SUMMARY_CSV_ORIGINAL = os.path.join(FINAL_CSV_DIR, "suc_original.csv")
SUMMARY_CSV_PROMPT2 = os.path.join(FINAL_CSV_DIR, "prompt2", "suc.csv")

SUMMARY_HEADER = [
    "prompt_version",
    "pipeline",
    "model",
    "format",
    "task",
    "correct",
    "total",
    "accuracy_pct",
    "field_accuracy_pct",
    "relaxed_accuracy_pct",
]


# ── Post-processing ─────────────────────────────────────────────────────────
def strip_think(text):
    text = str(text or "")

    # If model output contains a completed reasoning block,
    # evaluate only the text after </think>.
    m = re.search(r"</think>", text, flags=re.IGNORECASE)
    if m:
        text = text[m.end():]

    return text.strip()

def clean_prediction(text):
    """Unified post-processing: strip Answer: prefix, normalize whitespace,
    pipe separators, case, quotes and backslashes."""
    if text is None:
        return ""

    text = strip_think(text)

    text = re.sub(r'^answer:\s*', '', text, flags=re.IGNORECASE)
    text = text.replace('\n', ' ')
    text = text.replace('"', '').replace('\\', '')
    text = re.sub(r'\s*\|\s*', '|', text)   # normalize spaces around |
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text


def clean_gold(text):
    """Normalize ground-truth answer the same way (pipe + whitespace)."""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r'\s*\|\s*', '|', text)
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text


def is_error_like(s):
    s = (s or "").strip().lower()
    return s.startswith("error:") or s in {"connection_failed", "extraction_failed"}


# ── Added metrics ───────────────────────────────────────────────────────────

def split_fields(text):
    """Split normalized answer into pipe-separated fields."""
    text = clean_gold(text)
    if not text:
        return []
    return [p.strip() for p in text.split("|") if p.strip()]


def field_accuracy_score(gold, pred):
    """
    Position-wise field accuracy.

    Example:
        gold = "10|4"
        pred = "10|5"
        score = 1/2 = 0.5

    This requires the model to use the expected pipe-separated structure.
    """
    gold_fields = split_fields(gold)
    pred_fields = split_fields(pred)

    if not gold_fields:
        return 0.0

    correct = 0
    for i, gold_field in enumerate(gold_fields):
        if i < len(pred_fields) and pred_fields[i] == gold_field:
            correct += 1

    return correct / len(gold_fields)


def field_appears(gold_field, pred_text):
    """
    Boundary-aware check for whether a gold field appears in the prediction.

    Numeric fields are matched with number boundaries, so:
        gold "4" does not match "14"
        gold "10" does not match "100"

    Text fields are matched by normalized substring.
    """
    g = clean_gold(gold_field)
    p = clean_prediction(pred_text)

    if not g or not p:
        return False

    # Numeric field: allow commas, decimals, signs, percentages.
    # Match as a complete numeric-like unit, not as a substring of another number.
    if re.fullmatch(r"[-+−]?\d[\d,]*(?:\.\d+)?%?", g):
        pattern = rf"(?<![\d,.\-+−]){re.escape(g)}(?![\d,.\-+−%])"
        return re.search(pattern, p) is not None

    # Text field: normalized phrase containment.
    return g in p


def relaxed_accuracy_score(task, gold, pred):
    """
    Relaxed accuracy / relaxed field recall.

    Score = fraction of gold fields that appear somewhere in prediction.

    Special handling for cell_lookup:
        gold row|col should not get full credit if row/column are swapped.
        If prediction contains explicit row/column wording, check row near row
        and column near column. Otherwise fall back to ordered first-two-number check.
    """
    gold_fields = split_fields(gold)
    pred_norm = clean_prediction(pred)

    if not gold_fields or not pred_norm:
        return 0.0

    # Special case: row|column lookup needs order/role awareness.
    if task == "cell_lookup" and len(gold_fields) == 2:
        row_g, col_g = gold_fields

        row_match = re.search(
            rf"\brow\b[^0-9\-+−]*{re.escape(row_g)}\b",
            pred_norm,
            flags=re.IGNORECASE,
        )
        col_match = re.search(
            rf"\bcol(?:umn)?\b[^0-9\-+−]*{re.escape(col_g)}\b",
            pred_norm,
            flags=re.IGNORECASE,
        )

        if row_match or col_match:
            return (1.0 if row_match else 0.0) * 0.5 + (1.0 if col_match else 0.0) * 0.5

        nums = re.findall(r"[-+−]?\d[\d,]*(?:\.\d+)?%?", pred_norm)
        if len(nums) >= 2:
            correct = 0
            if nums[0] == row_g:
                correct += 1
            if nums[1] == col_g:
                correct += 1
            return correct / 2.0

        return 0.0

    matched = 0
    for gold_field in gold_fields:
        if field_appears(gold_field, pred_norm):
            matched += 1

    return matched / len(gold_fields)


# ── Core evaluation ─────────────────────────────────────────────────────────

def evaluate(rows, fmt, pred_key_fn):
    """Return EM counts, totals, field accuracy sums, and relaxed accuracy sums."""
    task_correct = defaultdict(int)
    task_total   = defaultdict(int)
    task_field_sum = defaultdict(float)
    task_relaxed_sum = defaultdict(float)

    for row in rows:
        gt         = row.get("suc", {}) or {}
        pred_block = row.get(fmt, {}) or {}

        for task in TASK_KEYS:
            if task not in gt:
                continue
            pred_key = pred_key_fn(task)
            if pred_key not in pred_block:
                continue

            gold = clean_gold(str(gt[task]))
            pred = clean_prediction(pred_block[pred_key])

            task_total[task] += 1

            if not is_error_like(pred):
                if gold == pred:
                    task_correct[task] += 1

                task_field_sum[task] += field_accuracy_score(gold, pred)
                task_relaxed_sum[task] += relaxed_accuracy_score(task, gold, pred)

    return task_correct, task_total, task_field_sum, task_relaxed_sum


# ── I/O helpers ─────────────────────────────────────────────────────────────

def load_suc_file(results_dir, model_name):
    path = os.path.join(results_dir, model_name, "suc.json")
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to parse {path}: {e}")
            return None


def open_csv_append(path, header):
    """Open CSV in append mode; write header only if the file is new/empty."""
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(header)
    return f, writer


def discover_models(pipeline_key):
    """List model directories that contain suc.json for the given pipeline."""
    results_dir = PIPELINE_CONFIGS[pipeline_key]["results_dir"]
    if not os.path.isdir(results_dir):
        return []
    return sorted(
        m for m in os.listdir(results_dir)
        if os.path.exists(os.path.join(results_dir, m, "suc.json"))
    )


def get_pipeline_type(pipeline_key):
    """Extract pipeline type from pipeline_key.
    E.g., 'vlm_prompt2' -> 'vlm', 'vlm-text' -> 'vlm-text'
    """
    if pipeline_key.endswith("_prompt2"):
        return pipeline_key.replace("_prompt2", "")
    return pipeline_key


def get_wide_csv_path(pipeline_key, prompt_version):
    """Get the wide-format CSV path for a pipeline."""
    pipeline_type = get_pipeline_type(pipeline_key)
    base_dir = os.path.join("results", "scores", "suc")
    
    if prompt_version == "prompt2":
        base_dir = os.path.join(base_dir, "prompt2")
    
    if pipeline_type == "vlm":
        return os.path.join(base_dir, "suc_vlm.csv")
    elif pipeline_type == "vlm-text":
        return os.path.join(base_dir, "suc_vlm_text.csv")
    elif pipeline_type == "llm":
        return os.path.join(base_dir, "suc_llm.csv")


# ── Per-model runner ─────────────────────────────────────────────────────────

def run_one(pipeline_key, model_name):
    cfg         = PIPELINE_CONFIGS[pipeline_key]
    label       = cfg["label"]
    pred_key_fn = cfg["pred_key"]
    prompt_version = cfg["prompt_version"]

    rows = load_suc_file(cfg["results_dir"], model_name)
    if rows is None:
        return

    print(f"\n{'=' * 65}")
    print(f"  {label} | {model_name} | {prompt_version}")
    print(f"{'=' * 65}")

    # Open combined summary CSV
    combined_f, combined_writer = open_csv_append(SUMMARY_CSV, SUMMARY_HEADER)
    
    # Open version-specific summary CSV
    if prompt_version == "original":
        version_csv = SUMMARY_CSV_ORIGINAL
    else:
        version_csv = SUMMARY_CSV_PROMPT2
    os.makedirs(os.path.dirname(version_csv), exist_ok=True)
    version_f, version_writer = open_csv_append(version_csv, SUMMARY_HEADER)
    
    # Open wide CSV
    wide_csv = get_wide_csv_path(pipeline_key, prompt_version)
    os.makedirs(os.path.dirname(wide_csv), exist_ok=True)
    wide_f, wide_writer = open_csv_append(wide_csv, WIDE_HEADER)
    
    try:
        for fmt in FORMATS:
            task_correct, task_total, task_field_sum, task_relaxed_sum = evaluate(
                rows, fmt, pred_key_fn
            )

            if not any(task_total.get(t, 0) > 0 for t in TASK_KEYS):
                continue  # format not present for this model

            print(f"\n  [{fmt}]")
            overall_c = overall_t = 0
            overall_field_sum = 0.0
            overall_relaxed_sum = 0.0
            wide_row = [model_name, fmt]

            for task in TASK_KEYS:
                c = task_correct.get(task, 0)
                t = task_total.get(task, 0)
                if t == 0:
                    wide_row.append("")
                    continue

                acc = c / t * 100.0
                field_acc = task_field_sum.get(task, 0.0) / t * 100.0
                relaxed_acc = task_relaxed_sum.get(task, 0.0) / t * 100.0

                overall_c += c
                overall_t += t
                overall_field_sum += task_field_sum.get(task, 0.0)
                overall_relaxed_sum += task_relaxed_sum.get(task, 0.0)

                print(
                    f"    {task:22s}  "
                    f"EM {c:4d}/{t:<4d} {acc:6.1f}%  "
                    f"Field {field_acc:6.1f}%  "
                    f"Relaxed {relaxed_acc:6.1f}%"
                )

                row_data = [
                    prompt_version,
                    label,
                    model_name,
                    fmt,
                    task,
                    c,
                    t,
                    round(acc, 1),
                    round(field_acc, 1),
                    round(relaxed_acc, 1),
                ]
                
                combined_writer.writerow(row_data)
                version_writer.writerow(row_data)
                wide_row.append(round(acc, 1))

            if overall_t > 0:
                ov_acc = overall_c / overall_t * 100.0
                ov_field_acc = overall_field_sum / overall_t * 100.0
                ov_relaxed_acc = overall_relaxed_sum / overall_t * 100.0

                print(
                    f"    {'OVERALL':22s}  "
                    f"EM {overall_c:4d}/{overall_t:<4d} {ov_acc:6.1f}%  "
                    f"Field {ov_field_acc:6.1f}%  "
                    f"Relaxed {ov_relaxed_acc:6.1f}%"
                )

                row_data = [
                    prompt_version,
                    label,
                    model_name,
                    fmt,
                    "OVERALL",
                    overall_c,
                    overall_t,
                    round(ov_acc, 1),
                    round(ov_field_acc, 1),
                    round(ov_relaxed_acc, 1),
                ]
                
                combined_writer.writerow(row_data)
                version_writer.writerow(row_data)
                wide_row.append(round(ov_acc, 1))
            else:
                wide_row.append("")

            wide_writer.writerow(wide_row)

        combined_f.flush()
        version_f.flush()
        wide_f.flush()
    finally:
        combined_f.close()
        version_f.close()
        wide_f.close()

    print(f"\n  -> Appended to {SUMMARY_CSV}")
    print(f"  -> Appended to {version_csv}")
    print(f"  -> Appended to {wide_csv}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SUC Evaluation — per-task accuracy across all pipelines + prompt variants."
    )
    parser.add_argument(
        "--pipeline", choices=list(PIPELINE_CONFIGS), default=None,
        help="Pipeline type (omit to run all pipelines)"
    )
    parser.add_argument(
        "--model_name", default=None,
        help="Model name folder (omit to run all models in the pipeline)"
    )
    args = parser.parse_args()

    os.makedirs(FINAL_CSV_DIR, exist_ok=True)

    pipelines = [args.pipeline] if args.pipeline else list(PIPELINE_CONFIGS.keys())

    for pipeline_key in pipelines:
        models = ([args.model_name] if args.model_name
                  else discover_models(pipeline_key))
        for model_name in models:
            run_one(pipeline_key, model_name)

    print(f"\nDone. Output CSVs:")
    print(f"  Combined:  {SUMMARY_CSV}")
    print(f"  Original:  {SUMMARY_CSV_ORIGINAL}")
    print(f"  Prompt2:   {SUMMARY_CSV_PROMPT2}")


if __name__ == "__main__":
    main()