#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/mominaahsan/Desktop/VisualTableBench"
OUT_DIR="$ROOT/data_full/1-raw"
OUT_PATH="$OUT_DIR/tabfact.jsonl"

mkdir -p "$OUT_DIR"

python - <<'PY'
import json
from datasets import load_dataset

ROOT="/Users/mominaahsan/Desktop/VisualTableBench"
OUT_PATH=f"{ROOT}/data_full/1-raw/tabfact.jsonl"

# Load ONLY test split
ds = load_dataset(f"{ROOT}/scripts/dataset_collection/tabfact.py", split="test")
print(f"[INFO] Loaded TabFact test samples: {len(ds)}")

with open(OUT_PATH, "w", encoding="utf8") as fout:
    for ex in ds:
        try:
            label_int = int(ex["label"])  # 1=entailed, 0=refuted
        except Exception:
            label_int = 2  # fallback (should not happen)

        out = {
            "table": {
                "header": ex["table"]["header"],
                "rows": ex["table"]["rows"]
            },
            "query": ex["statement"],
            "label": [str(label_int)],   # ✅ normalized to ["1"] / ["0"]
            "table_file": ex["table"]["id"]
        }

        fout.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"[OK] Wrote: {OUT_PATH}")
PY