#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/mominaahsan/Desktop/VisualTableBench"
OUT_DIR="$ROOT/data_full/1-raw"
OUT_PATH="$OUT_DIR/hybridqa.jsonl"

mkdir -p "$OUT_DIR"

python - <<'PY'
import json
import re
from datasets import load_dataset

ROOT="/Users/mominaahsan/Desktop/VisualTableBench"
OUT_PATH=f"{ROOT}/data_full/1-raw/hybridqa.jsonl"

def norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u00A0", " ").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def numish(s: str) -> str:
    s = norm(s)
    if not s:
        return ""
    if re.search(r"\d+:\d+", s):
        return re.findall(r"\d+:\d+", s)[0]
    s2 = s.replace(",", "")
    digs = re.findall(r"\d+", s2)
    if digs:
        return " ".join(digs)
    return ""

def table_answerable(labels, rows) -> bool:
    cells = [norm(c) for r in rows for c in r]
    cellset = set([c for c in cells if c])
    table_text = " ".join(cells)

    for a in labels:
        a = norm(a)
        if not a:
            continue
        if a in cellset:
            return True
        if len(a) >= 2 and a in table_text:
            return True
        na = numish(a)
        if na:
            for c in cells:
                nc = numish(c)
                if nc and (nc == na or na in nc or nc in na):
                    return True
    return False

ds = load_dataset(f"{ROOT}/scripts/dataset_collection/hybridqa.py", split="validation")
print(f"[INFO] Loaded HybridQA validation samples: {len(ds)}")

kept = 0
dropped = 0

with open(OUT_PATH, "w", encoding="utf8") as fout:
    for ex in ds:
        header = ex["table"]["header"]
        rows = ex["table"]["rows"]

        ans = ex.get("answer_text", "")
        labels = [ans] if isinstance(ans, str) else (ans or [])
        labels = [str(x) for x in labels]

        # table-answerable filter
        if not table_answerable(labels, rows):
            dropped += 1
            continue

        out = {
            "table": {
                "header": header,
                "rows": rows,
            },
            "query": ex["question"],
            "label": labels,
            "table_file": ex.get("table_id", "")
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        kept += 1

print(f"[INFO] Kept (table-answerable, loose): {kept}")
print(f"[INFO] Dropped (likely needs passage/context): {dropped}")
print(f"[OK] Wrote: {OUT_PATH}")
PY