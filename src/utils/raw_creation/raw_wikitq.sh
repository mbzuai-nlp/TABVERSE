#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/mominaahsan/Desktop/VisualTableBench"
OUT="${ROOT}/data_full/1-raw/wikitq.jsonl"

python - <<'PY'
import json
from datasets import load_dataset

ROOT="/Users/mominaahsan/Desktop/VisualTableBench"
OUT=f"{ROOT}/data_full/1-raw/wikitq.jsonl"

# Uses your local dataset script (scripts/dataset_collection/wikitq.py)
ds = load_dataset(
    f"{ROOT}/scripts/dataset_collection/wikitq.py",
    name="random-split-1",
    split="test",
)

print("[INFO] Loaded WikiTQ test samples:", len(ds))

with open(OUT, "w", encoding="utf-8") as f:
    for ex in ds:
        out = {
            "table": {
                "header": ex["table_header"],
                "rows": ex["table_data"],
            },
            "query": ex["question"],
            "label": ex["answer_text"],   # WTQ answers are lists; keep as-is
            "table_file": ex.get("table_file", ""),
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

print("[OK] Wrote:", OUT)
PY