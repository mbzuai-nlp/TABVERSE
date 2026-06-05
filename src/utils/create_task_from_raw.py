"""
Convert data/1-raw/data.json → data/2-task/task.json

Output fields per entry:
  id                  – sequential integer starting from 0
  image_id            – from raw data
  table               – from raw data
  query               – from raw data
  label               – from raw data
  question_category   – from raw data
  score               – from raw data
  question_difficulty – from raw data
  dataset             – from raw data
"""

import json
from pathlib import Path

INPUT_FILE = Path("data/1-raw/data.json")
OUTPUT_DIR = Path("data/2-task")
OUTPUT_FILE = OUTPUT_DIR / "task.json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} entries from {INPUT_FILE}")

    task_data = []
    for idx, item in enumerate(raw_data):
        entry = {
            "id": idx,
            "image_id": item.get("image_id", ""),
            "table": item.get("table", {}),
            "query": item.get("query", ""),
            "label": item.get("label", []),
            "question_category": item.get("question_category", ""),
            "score": item.get("score", None),
            "question_difficulty": item.get("question_difficulty", ""),
            "dataset": item.get("dataset", ""),
        }
        task_data.append(entry)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(task_data, f, ensure_ascii=False, indent=4)

    print(f"✓ Written {len(task_data)} entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
