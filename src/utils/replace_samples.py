#!/usr/bin/env python3
"""Replace one entry in data/1-raw/data.json.

Usage:
  python scripts/replace_samples.py --new-id 42
  python scripts/replace_samples.py --new-id 42 --dry-run

The script:
  1. Finds the entry with the given new_id and reads its question_difficulty
     and question_category.
  2. Scans results/question_difficulty_complexity/<dataset>.json files for a
     candidate that matches those two fields AND whose (image_id, query) pair
     is NOT already present in data.json.  The same image_id is allowed as
     long as the query is different.
  3. Swaps the entry in-place, preserving the original new_id.
  4. Appends a record to the replacement log (data/1-raw/replacement_log.json)
     so that every (image_id, query) pair that was ever replaced OUT or IN is
     excluded from future replacement runs — preventing infinite loops of
     resampling the same bad entries.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_PATH    = Path("data/1-raw/data.json")
POOL_DIR     = Path("results/question_difficulty_complexity")
LOG_PATH     = Path("data/1-raw/replacement_log.json")
RANDOM_SEED  = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_log(log_path: Path) -> list[dict]:
    """Load the replacement log, or return an empty list if it doesn't exist yet."""
    if not log_path.exists():
        return []
    try:
        log = json.loads(log_path.read_text(encoding="utf-8"))
        if isinstance(log, list):
            return log
    except Exception as exc:
        print(f"  [log] Could not load {log_path}: {exc} — starting fresh.")
    return []


def samples_from_log(log: list[dict]) -> set[tuple[str, str]]:
    """
    Return every (image_id, query) pair that has ever appeared in the log —
    both the sample replaced OUT and the sample brought IN.
    This prevents both sides from ever being selected again in future runs,
    breaking any replacement loop.
    """
    pairs: set[tuple[str, str]] = set()
    for record in log:
        for img_field, q_field in (
            ("old_image_id", "old_query"),
            ("new_image_id", "new_query"),
        ):
            img = record.get(img_field)
            qry = (record.get(q_field) or "").strip()
            if img:
                pairs.add((img, qry))
    return pairs


def build_pool(
    pool_dir: Path,
    used_samples: set[tuple[str, str]],
) -> dict[tuple[str, str], list[dict]]:
    """
    Load all candidate samples from the pool directory, grouped by
    (question_difficulty, question_category).  Samples whose (image_id, query)
    pair already exists in data.json are excluded up-front.  The same
    image_id is allowed when the query differs.
    """
    pool: dict[tuple[str, str], list[dict]] = {}

    for json_file in sorted(pool_dir.glob("*.json")):
        try:
            rows = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [pool] Could not load {json_file.name}: {exc}")
            continue

        for item in rows:
            if not isinstance(item, dict):
                continue
            img_id = item.get("image_id")
            if not img_id:
                continue
            query = (item.get("query") or "").strip()
            if (img_id, query) in used_samples:
                continue
            diff  = (item.get("question_difficulty") or "").strip()
            cat   = (item.get("question_category")   or "").strip()
            if not diff or not cat:
                continue
            key = (diff, cat)
            pool.setdefault(key, []).append(
                {**item, "dataset": json_file.stem}
            )

    return pool


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-id", type=int, required=True, nargs='+',
        help="One or more new_id values of entries in data.json to replace.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned replacement without writing the file.",
    )
    parser.add_argument(
        "--data-path", type=Path, default=DATA_PATH,
        help=f"Path to data.json (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--pool-dir", type=Path, default=POOL_DIR,
        help=f"Directory with pool JSON files (default: {POOL_DIR})",
    )
    parser.add_argument(
        "--log-path", type=Path, default=LOG_PATH,
        help=f"Path to replacement log JSON (default: {LOG_PATH})",
    )
    args = parser.parse_args()
    TARGET_NEW_IDS = args.new_id

    random.seed(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Load data.json
    # ------------------------------------------------------------------
    if not args.data_path.exists():
        raise SystemExit(f"File not found: {args.data_path}")

    data: list[dict] = load_json(args.data_path)
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON list, got {type(data).__name__}")

    # Index by new_id for fast lookup
    idx_by_new_id: dict[int, int] = {
        entry["new_id"]: pos
        for pos, entry in enumerate(data)
        if isinstance(entry.get("new_id"), int)
    }

    # Collect all (image_id, query) pairs currently in the dataset.
    # Same image_id is fine as long as the query differs.
    used_samples: set[tuple[str, str]] = {
        (entry.get("image_id", ""), (entry.get("query") or "").strip())
        for entry in data
        if entry.get("image_id")
    }

    # ------------------------------------------------------------------
    # Load replacement log — add historically used pairs to exclusion set
    # ------------------------------------------------------------------
    log_path = args.log_path
    log: list[dict] = load_log(log_path)
    log_samples = samples_from_log(log)
    print(f"Replacement log: {len(log)} prior record(s), "
          f"{len(log_samples)} historically used (image_id, query) pair(s).")
    used_samples |= log_samples

    # ------------------------------------------------------------------
    # Build pool (candidates whose (image_id, query) pair is not in data.json)
    # ------------------------------------------------------------------
    print("Building replacement pool …")
    pool = build_pool(args.pool_dir, used_samples)
    total_candidates = sum(len(v) for v in pool.values())
    print(f"  {total_candidates} candidates across {len(pool)} (difficulty, category) buckets.\n")

    # ------------------------------------------------------------------
    # Process each target new_id
    # ------------------------------------------------------------------
    replacements_made = 0
    warnings: list[str] = []

    for new_id in TARGET_NEW_IDS:
        pos = idx_by_new_id.get(new_id)
        if pos is None:
            warnings.append(f"  new_id={new_id}: NOT FOUND in data.json — skipping.")
            continue

        old_entry = data[pos]
        diff = (old_entry.get("question_difficulty") or "").strip()
        cat  = (old_entry.get("question_category")   or "").strip()
        key  = (diff, cat)

        candidates = pool.get(key, [])
        if not candidates:
            warnings.append(
                f"  new_id={new_id}: no replacement found for "
                f"difficulty='{diff}' category='{cat}'."
            )
            continue

        # Pick a random candidate and remove it from the pool so it can't be
        # used again for another replacement in the same run.
        chosen_idx = random.randrange(len(candidates))
        candidate  = candidates.pop(chosen_idx)
        if not candidates:
            del pool[key]

        # Build the replacement entry: keep new_id, copy everything else.
        new_entry: dict = {"new_id": new_id}
        for k, v in candidate.items():
            if k != "new_id":
                new_entry[k] = v

        log_record = {
            "timestamp":           datetime.now(timezone.utc).isoformat(),
            "new_id":              new_id,
            "old_image_id":        old_entry.get("image_id"),
            "old_query":           (old_entry.get("query") or "").strip(),
            "old_dataset":         old_entry.get("dataset"),
            "new_image_id":        new_entry.get("image_id"),
            "new_query":           (new_entry.get("query") or "").strip(),
            "new_dataset":         new_entry.get("dataset"),
            "question_difficulty": diff,
            "question_category":   cat,
        }

        if args.dry_run:
            print(
                f"  [DRY-RUN] new_id={new_id}: replace image_id={old_entry.get('image_id')!r} "
                f"({diff} / {cat})"
                f"\n            → image_id={new_entry.get('image_id')!r}  "
                f"dataset={new_entry.get('dataset')!r}  "
                f"query={str(new_entry.get('query',''))[:80]!r}"
            )
        else:
            data[pos] = new_entry
            # Exclude this (image_id, query) pair for subsequent iterations
            used_samples.add(
                (new_entry.get("image_id", ""), (new_entry.get("query") or "").strip())
            )
            log.append(log_record)

        replacements_made += 1

    # ------------------------------------------------------------------
    # Report warnings
    # ------------------------------------------------------------------
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(w)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if not args.dry_run:
        dump_json(args.data_path, data)
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(args.log_path, log)
        print(f"\n✓ {replacements_made} replacement(s) applied → {args.data_path}")
        print(f"✓ Replacement log updated   → {args.log_path}")
    else:
        print(f"\n(dry-run) {replacements_made} replacement(s) would be applied."
              f" Log would grow to {len(log) + replacements_made} record(s).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
