#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/mominaahsan/Desktop/VisualTableBench"
OUT_DIR="$ROOT/data_full/1-raw"
OUT_PATH="$OUT_DIR/feverous.jsonl"

mkdir -p "$OUT_DIR"

python - <<'PY'
import json
from datasets import load_dataset

ROOT="/Users/mominaahsan/Desktop/VisualTableBench"
OUT_PATH=f"{ROOT}/data_full/1-raw/feverous.jsonl"

# FEVEROUS has train + dev (validation). Public labeled eval is dev.
ds = load_dataset(
    f"{ROOT}/scripts/dataset_collection/feverous.py",
    split="validation",
)

print(f"[INFO] Loaded FEVEROUS dev/validation samples (TABLE-ONLY evidence filter applied in feverous.py): {len(ds)}")

LABEL_MAP = {
    "SUPPORTS": "1",
    "REFUTES": "0",
    "NOT ENOUGH INFO": "2",
}

bad_label = 0

with open(OUT_PATH, "w", encoding="utf-8") as fout:
    for ex in ds:
        lab = LABEL_MAP.get(ex.get("label", ""), None)
        if lab is None:
            bad_label += 1
            lab = str(ex.get("label", ""))

        out = {
            "table": {
                "header": ex["table"]["header"],
                "rows": ex["table"]["rows"]
            },
            "query": ex["statement"],
            "label": [lab],
            "table_file": ex.get("id",""),
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"[INFO] unmapped labels: {bad_label}")
print(f"[OK] Wrote: {OUT_PATH}")
PY