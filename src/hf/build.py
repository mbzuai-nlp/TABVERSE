"""
Build and push the TABVERSE dataset to Hugging Face Hub.

Two configs are uploaded:
  - qa  (700 rows): Task Prediction — one row per Q-Table pair
  - suc (629 rows): SUC + SR — one row per unique table

Each row embeds three Image() columns (html / markdown / latex renderings)
plus the raw source code for each format.  Storing images as datasets.Image()
causes the HF Dataset Viewer to show them inline.

Usage:
    python src/hf/build.py --repo MBZUAI/TABVERSE [--data-dir data/5-hf] [--private]

The dataset card lives at data/5-hf/README.md and is uploaded separately by the
GitHub Action (upload_hf.yml) after this script finishes.
"""

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value


# ── helpers ──────────────────────────────────────────────────────────────────

def load_image_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def table_to_str(table: dict) -> str:
    """Flatten the table dict to a compact JSON string for the HF column."""
    return json.dumps(table, ensure_ascii=False)


# ── config builders ──────────────────────────────────────────────────────────

def build_qa(data_dir: Path) -> Dataset:
    """700-row Task Prediction config."""
    records = json.loads((data_dir / "task.json").read_text())

    html_dir = data_dir / "html"
    md_dir   = data_dir / "markdown"
    tex_dir  = data_dir / "latex"

    rows = []
    for r in records:
        iid = r["image_id"]
        rows.append({
            "id":               r["id"],
            "image_id":         iid,
            # images ── stored as raw bytes; datasets.Image() handles encoding
            "html_image":       load_image_bytes(html_dir / f"{iid}.png"),
            "markdown_image":   load_image_bytes(md_dir   / f"{iid}.png"),
            "latex_image":      load_image_bytes(tex_dir  / f"{iid}.png"),
            # source code
            "html_code":        load_text(html_dir / f"{iid}.html"),
            "markdown_code":    load_text(md_dir   / f"{iid}.md"),
            "latex_code":       load_text(tex_dir  / f"{iid}.tex"),
            # table structure
            "table":            table_to_str(r["table"]),
            # task fields
            "query":            r["query"],
            "label":            r["label"],
            "question_category": r["question_category"],
            "question_difficulty": r["question_difficulty"],
            "dataset":          r["dataset"],
            "score":            r["score"],
        })

    features = Features({
        "id":                   Value("int32"),
        "image_id":             Value("string"),
        "html_image":           Image(),
        "markdown_image":       Image(),
        "latex_image":          Image(),
        "html_code":            Value("string"),
        "markdown_code":        Value("string"),
        "latex_code":           Value("string"),
        "table":                Value("string"),
        "query":                Value("string"),
        "label":                Sequence(Value("string")),
        "question_category":    Value("string"),
        "question_difficulty":  Value("string"),
        "dataset":              Value("string"),
        "score":                Value("int32"),
    })

    return Dataset.from_list(rows, features=features)


def build_suc(data_dir: Path) -> Dataset:
    """629-row SUC + SR config (one row per unique table)."""
    records = json.loads((data_dir / "suc_generation.json").read_text())

    html_dir = data_dir / "html"
    md_dir   = data_dir / "markdown"
    tex_dir  = data_dir / "latex"

    rows = []
    for r in records:
        iid = r["image_id"]
        suc = r["suc"]
        rows.append({
            "id":               r["id"],
            "image_id":         iid,
            # images
            "html_image":       load_image_bytes(html_dir / f"{iid}.png"),
            "markdown_image":   load_image_bytes(md_dir   / f"{iid}.png"),
            "latex_image":      load_image_bytes(tex_dir  / f"{iid}.png"),
            # source code (also used for SR / format generation)
            "html_code":        load_text(html_dir / f"{iid}.html"),
            "markdown_code":    load_text(md_dir   / f"{iid}.md"),
            "latex_code":       load_text(tex_dir  / f"{iid}.tex"),
            # table structure
            "table":            table_to_str(r["table"]),
            "dataset":          r["dataset"],
            # SUC fields (flattened)
            "table_partition":          suc.get("table_partition", ""),
            "size_detection":           suc.get("size_detection", ""),
            "cell_value":               suc.get("cell_value", ""),
            "cell_lookup":              suc.get("cell_lookup", ""),
            "reverse_lookup_indices":   suc.get("reverse_lookup_indices", ""),
            "reverse_lookup":           suc.get("reverse_lookup", ""),
            "column_idx":               suc.get("column_idx", -1),
            "column_retrieval":         suc.get("column_retrieval", ""),
            "row_idx":                  suc.get("row_idx", -1),
            "row_retrieval":            suc.get("row_retrieval", ""),
            "table_first_cell":         suc.get("table_first_cell", ""),
            "table_last_cell":          suc.get("table_last_cell", ""),
            "number_of_rows":           suc.get("number_of_rows", 0),
            "number_of_columns":        suc.get("number_of_columns", 0),
        })

    features = Features({
        "id":                       Value("int32"),
        "image_id":                 Value("string"),
        "html_image":               Image(),
        "markdown_image":           Image(),
        "latex_image":              Image(),
        "html_code":                Value("string"),
        "markdown_code":            Value("string"),
        "latex_code":               Value("string"),
        "table":                    Value("string"),
        "dataset":                  Value("string"),
        # SUC
        "table_partition":          Value("string"),
        "size_detection":           Value("string"),
        "cell_value":               Value("string"),
        "cell_lookup":              Value("string"),
        "reverse_lookup_indices":   Value("string"),
        "reverse_lookup":           Value("string"),
        "column_idx":               Value("int32"),
        "column_retrieval":         Value("string"),
        "row_idx":                  Value("int32"),
        "row_retrieval":            Value("string"),
        "table_first_cell":         Value("string"),
        "table_last_cell":          Value("string"),
        "number_of_rows":           Value("int32"),
        "number_of_columns":        Value("int32"),
    })

    return Dataset.from_list(rows, features=features)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build & push TABVERSE to HF Hub")
    parser.add_argument("--repo",     required=True, help="HF repo id, e.g. MBZUAI/TABVERSE")
    parser.add_argument("--data-dir", default="data/5-hf", help="Path to 5-hf directory")
    parser.add_argument("--private",  action="store_true", help="Upload as private repo")
    parser.add_argument("--token",    default=os.getenv("HF_TOKEN"), help="HF token (default: $HF_TOKEN)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assert data_dir.exists(), f"data dir not found: {data_dir}"

    print("Building qa config  (700 rows)...")
    qa_ds = build_qa(data_dir)
    print(qa_ds)

    print("\nBuilding suc config (629 rows)...")
    suc_ds = build_suc(data_dir)
    print(suc_ds)

    print(f"\nPushing to {args.repo} ...")

    # qa config — use split="test" to match the benchmark convention
    qa_ds.push_to_hub(
        args.repo,
        config_name="qa",
        split="test",
        private=args.private,
        token=args.token,
    )
    print("  qa config pushed.")

    # suc config
    suc_ds.push_to_hub(
        args.repo,
        config_name="suc",
        split="test",
        private=args.private,
        token=args.token,
    )
    print("  suc config pushed.")
    print("\nDone. Visit: https://huggingface.co/datasets/" + args.repo)


if __name__ == "__main__":
    main()
