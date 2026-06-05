import os
import re
import json
import csv
import argparse

FORMATS = ["html", "markdown", "latex"]

PIPELINE_CONFIGS = {
    "vlm": {
        "label":       "VLM",
        "results_dir": "results/vlmpipeline",
        "task_file":   lambda model: f"results/vlmpipeline/{model}/task.json",
        "pred_key":    "task_prediction",
    },
    "vlm-text": {
        "label":       "VLM-TEXT",
        "results_dir": "results/vlmpipeline_text",
        "task_file":   lambda model: f"results/vlmpipeline_text/{model}/task.json",
        "pred_key":    "task_prediction",
    },
    "llm": {
        "label":       "LLM",
        "results_dir": "results/llmpipeline",
        "task_file":   lambda model: f"results/llmpipeline/{model}/task.json",
        "pred_key":    "task_prediction",
    },
}


# ===================== HELPERS ===================== #

def strip_think(text):
    text = str(text or "")

    # If model output contains a completed reasoning block,
    # evaluate only the text after </think>.
    m = re.search(r"</think>", text, flags=re.IGNORECASE)
    if m:
        text = text[m.end():]

    return text.strip()


def clean_text(text):
    text = strip_think(text)
    return re.sub(r"\s+", " ", str(text).strip().lower())


def is_error(s):
    s = (s or "").strip().lower()
    return s.startswith("error") or s in {"connection_failed", "extraction_failed", ""}


def gold_list(row):
    g = row["label"]
    return g if isinstance(g, list) else [g]


def split_multi_items(text):
    """Split a multi-item prediction string by , ; or newline."""
    parts = re.split(r"[,;\n]+", str(text))
    return sorted([clean_text(p) for p in parts if clean_text(p)])


def soft_match(guess, gold):
    """
    Existing EM-style match used in the original code.

    True if:
      1. guess exactly matches gold after clean_text normalization.
      2. gold is a leading substring of guess followed by a non-alphanumeric char.
    """
    if guess == gold:
        return True

    if guess.startswith(gold) and len(guess) > len(gold):
        next_char = guess[len(gold)]
        if not next_char.isalnum():
            return True

    return False


def normalize_number_string(s):
    """Normalize numeric strings by removing commas."""
    return str(s).replace(",", "").strip()


def is_numeric_answer(s):
    """Detect simple numeric answers, including commas, decimals, signs, and percentages."""
    s = clean_text(s)
    return re.fullmatch(r"[-+−]?\d[\d,]*(?:\.\d+)?%?", s) is not None


def contains_number_answer(pred, gold):
    """
    Boundary-aware numeric containment.
    Example:
      gold = "84,298"
      pred = "The population is 84,298."
      -> True

    Also supports comma-normalized comparison:
      gold = "84298"
      pred = "84,298"
      -> True
    """
    pred_norm = clean_text(pred)
    gold_norm = clean_text(gold)

    gold_num = normalize_number_string(gold_norm)

    nums = re.findall(r"[-+−]?\d[\d,]*(?:\.\d+)?%?", pred_norm)
    nums_norm = [normalize_number_string(n) for n in nums]

    return gold_num in nums_norm


def contains_text_answer(pred, gold):
    """
    Boundary-aware text containment.
    This is more relaxed than EM, but avoids obvious substring false positives.
    """
    pred_norm = clean_text(pred)
    gold_norm = clean_text(gold)

    if not pred_norm or not gold_norm:
        return False

    # Escape the gold answer and require non-alphanumeric boundaries.
    pattern = rf"(?<![a-z0-9]){re.escape(gold_norm)}(?![a-z0-9])"
    return re.search(pattern, pred_norm) is not None


def relaxed_single_match(pred, gold):
    """
    Relaxed QA match.

    Counts as correct if the normalized gold answer appears as a complete span
    in the prediction. Numeric answers are matched using extracted numbers with
    comma normalization.

    For yes/no answers, avoids giving credit when the prediction contains both.
    """
    pred_norm = clean_text(pred)
    gold_norm = clean_text(gold)

    if is_error(pred_norm):
        return False

    if not pred_norm or not gold_norm:
        return False

    # Exact / existing soft match still counts.
    if soft_match(pred_norm, gold_norm):
        return True

    # Conservative yes/no handling.
    if gold_norm in {"yes", "no"}:
        has_yes = re.search(r"(?<![a-z0-9])yes(?![a-z0-9])", pred_norm) is not None
        has_no = re.search(r"(?<![a-z0-9])no(?![a-z0-9])", pred_norm) is not None

        if has_yes and has_no:
            return False
        return (gold_norm == "yes" and has_yes) or (gold_norm == "no" and has_no)

    # Numeric answer containment.
    if is_numeric_answer(gold_norm):
        return contains_number_answer(pred_norm, gold_norm)

    # Text answer containment.
    return contains_text_answer(pred_norm, gold_norm)


def relaxed_multi_item_match(pred, golds):
    """
    Relaxed matching for multi-item answers.

    Counts as correct only if every gold item appears somewhere in the prediction.
    This avoids requiring exact comma-separated formatting, but still requires
    all listed gold items to be present.
    """
    pred_norm = clean_text(pred)
    if is_error(pred_norm):
        return False

    golds = [clean_text(g) for g in golds if g is not None and clean_text(g)]
    if not golds:
        return False

    return all(relaxed_single_match(pred_norm, g) for g in golds)


def load_task_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Task file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===================== CORE EVAL ===================== #

def evaluate_em(rows, fmt, pred_key):
    """
    Returns (correct, total) for the original exact-match-style evaluation.

    Multi-Item Lookup:
        prediction is split on comma/semicolon/newline and compared as a
        sorted set against the gold label list.

    Other categories:
        single-string match using the existing soft_match rule.
    """
    correct = total = 0

    for row in rows:
        pred_block = row.get(fmt, {}) or {}
        if "label" not in row or pred_key not in pred_block:
            continue

        golds = [clean_text(g) for g in gold_list(row) if g is not None]
        raw_pred = pred_block[pred_key]
        total += 1

        category = row.get("question_category", "")

        if category == "Multi-Item Lookup":
            pred_items = split_multi_items(raw_pred)
            if pred_items == sorted(golds):
                correct += 1
        else:
            guess = clean_text(raw_pred)
            if not is_error(guess) and any(soft_match(guess, g) for g in golds):
                correct += 1

    return correct, total


def evaluate_relaxed(rows, fmt, pred_key):
    """
    Returns (correct, total) for relaxed QA accuracy.

    This diagnostic metric counts a prediction as correct if the gold answer
    appears as a complete normalized span inside the prediction.

    For multi-item answers, all gold items must appear somewhere in the prediction.
    """
    correct = total = 0

    for row in rows:
        pred_block = row.get(fmt, {}) or {}
        if "label" not in row or pred_key not in pred_block:
            continue

        golds = gold_list(row)
        raw_pred = pred_block[pred_key]
        total += 1

        category = row.get("question_category", "")

        if category == "Multi-Item Lookup":
            if relaxed_multi_item_match(raw_pred, golds):
                correct += 1
        else:
            if any(relaxed_single_match(raw_pred, g) for g in golds):
                correct += 1

    return correct, total


# ===================== CSV HELPERS ===================== #

FINAL_CSV_DIR = os.path.join("results", "scores", "task")
SUMMARY_CSV = os.path.join(FINAL_CSV_DIR, "task.csv")

SUMMARY_HEADER = [
    "pipeline",
    "model",
    "html_em",
    "latex_em",
    "markdown_em",
    "html_relaxed",
    "latex_relaxed",
    "markdown_relaxed",
]


def open_csv_append(path, header):
    """Open CSV in append mode; write header only if file is new/empty."""
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(header)
    return f, writer


# ===================== REPORT FUNCTIONS ===================== #

def report_summary(rows, model_name, pipeline_label, pred_key):
    sep = "=" * 86
    print(f"\n{sep}")
    print(f"  {pipeline_label}")
    print(f"  Model : {model_name}")
    print(f"  Summary  (EM + relaxed QA accuracy per format)")
    print(sep)
    print(
        f"{'Format':<12} "
        f"{'EM Correct':>10} {'Total':>7} {'EM %':>8} "
        f"{'Relaxed Correct':>16} {'Relaxed %':>11}"
    )
    print("-" * 86)

    em_accs = {}
    relaxed_accs = {}

    for fmt in FORMATS:
        em_correct, em_total = evaluate_em(rows, fmt, pred_key)
        relaxed_correct, relaxed_total = evaluate_relaxed(rows, fmt, pred_key)

        em_acc = f"{em_correct / em_total * 100:.2f}" if em_total > 0 else "N/A"
        relaxed_acc = (
            f"{relaxed_correct / relaxed_total * 100:.2f}"
            if relaxed_total > 0
            else "N/A"
        )

        em_accs[fmt] = em_acc
        relaxed_accs[fmt] = relaxed_acc

        print(
            f"{fmt:<12} "
            f"{em_correct:>10} {em_total:>7} {em_acc:>8} "
            f"{relaxed_correct:>16} {relaxed_acc:>11}"
        )

    f, writer = open_csv_append(SUMMARY_CSV, SUMMARY_HEADER)
    try:
        writer.writerow([
            pipeline_label,
            model_name,
            em_accs["html"],
            em_accs["latex"],
            em_accs["markdown"],
            relaxed_accs["html"],
            relaxed_accs["latex"],
            relaxed_accs["markdown"],
        ])
        f.flush()
    finally:
        f.close()

    print(f"  -> Appended to {SUMMARY_CSV}")


# ===================== MAIN ===================== #

def discover_models(pipeline_key):
    """Return list of model names that have a task.json for the given pipeline."""
    cfg = PIPELINE_CONFIGS[pipeline_key]
    results_dir = cfg["results_dir"]

    if not os.path.isdir(results_dir):
        return []

    models = []
    for name in sorted(os.listdir(results_dir)):
        task_path = cfg["task_file"](name)
        if os.path.exists(task_path):
            models.append(name)

    return models


def run_one(pipeline_key, model_name):
    cfg = PIPELINE_CONFIGS[pipeline_key]
    task_path = cfg["task_file"](model_name)
    pred_key = cfg["pred_key"]
    label = cfg["label"]

    try:
        rows = load_task_file(task_path)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return

    report_summary(rows, model_name, label, pred_key)


def main():
    parser = argparse.ArgumentParser(
        description="Task Evaluation — EM and relaxed QA accuracy"
    )
    parser.add_argument(
        "--pipeline",
        choices=["vlm", "vlm-text", "llm"],
        default=None,
        help="Pipeline type (omit to run all)",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Model name (omit to run all models in the pipeline)",
    )
    args = parser.parse_args()

    os.makedirs(FINAL_CSV_DIR, exist_ok=True)

    pipelines = [args.pipeline] if args.pipeline else list(PIPELINE_CONFIGS.keys())

    first_pipeline = True

    for pipeline_key in pipelines:
        if args.model_name:
            models = [args.model_name]
        else:
            models = discover_models(pipeline_key)
            if not models:
                print(f"[INFO] No models with task.json found for pipeline: {pipeline_key}")
                continue

        # Blank row separator between pipeline groups.
        if not first_pipeline:
            f, _ = open_csv_append(SUMMARY_CSV, SUMMARY_HEADER)
            f.write("\n")
            f.close()

        first_pipeline = False

        for model_name in models:
            run_one(pipeline_key, model_name)


if __name__ == "__main__":
    main()