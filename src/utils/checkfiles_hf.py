#!/usr/bin/env python3
"""
Count how many format files and PNGs exist in each subfolder of the
'totto' directory in a Hugging Face repo (remote).

Subfolders checked: html, markdown, latex
Counts reported per subfolder:
  - number of format files (.html / .md / .tex)
  - number of .png files

Usage:
  python count_hf_totto_files.py
  # or set REPO_ID and (optionally) PATH_PREFIX via env vars:
  REPO_ID="MOMINAAHSAN296/vtb-dataset" python count_hf_totto_files.py

If the repo is private, set HF_TOKEN in your environment.
"""

import os
import sys
from collections import defaultdict
from typing import Dict, Tuple

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Please install huggingface_hub first: pip install huggingface_hub")
    sys.exit(1)

# === Config ===
REPO_ID = os.getenv("REPO_ID", "MOMINAAHSAN296/vtb-dataset")
REPO_TYPE = "dataset"  # change if this repo is a 'model' or 'space'
PATH_PREFIX = os.getenv("PATH_PREFIX", "totto")  # folder inside the repo
SUBFOLDERS = {
    "html": ".html",
    "markdown": ".md",
    "latex": ".tex",
}


def main():
    api = HfApi()  # will pick up HF_TOKEN automatically if set for private repos
    print(f"[INFO] Listing files for repo: {REPO_ID} (type={REPO_TYPE})")
    try:
        files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    except Exception as e:
        print(f"[ERR] Could not list files: {e}")
        sys.exit(1)

    # Filter to totto/ subtree only
    prefix = PATH_PREFIX.strip("/").rstrip("/") + "/"
    totto_files = [p for p in files if p.startswith(prefix)]

    if not totto_files:
        print(
            f"[WARN] No files found under '{prefix}'. "
            f"Double-check PATH_PREFIX and repo layout."
        )
        sys.exit(0)

    # Prepare counters
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"format": 0, "png": 0})

    for sub, ext in SUBFOLDERS.items():
        base = f"{prefix}{sub}/"
        # format files
        counts[sub]["format"] = sum(
            1 for p in totto_files if p.startswith(base) and p.endswith(ext)
        )
        # png files
        counts[sub]["png"] = sum(
            1 for p in totto_files if p.startswith(base) and p.endswith(".png")
        )

    # Pretty print
    print("\n=== totto Remote File Counts (Hugging Face) ===")
    print(f"Repo: {REPO_ID}")
    print(f"Path: /{PATH_PREFIX}\n")
    for sub, ext in SUBFOLDERS.items():
        fmt = counts[sub]["format"]
        png = counts[sub]["png"]
        print(f"{sub.upper():<10} → {ext} files: {fmt:<6} | PNG files: {png:<6}")


if __name__ == "__main__":
    main()
