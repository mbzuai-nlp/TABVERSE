"""
Build data/4-representations/ from data/3-suc/suc_generation.json.

For each unique image_id in suc_generation.json, copy:
  representations/{dataset}/html/{image_id}.html  + .png  → data/4-representations/html/
  representations/{dataset}/markdown/{image_id}.md + .png  → data/4-representations/markdown/
  representations/{dataset}/latex/{image_id}.tex  + .png  → data/4-representations/latex/
"""

import json
import shutil
from pathlib import Path

SUC_FILE        = Path("data/3-suc/suc_generation.json")
REPR_ROOT       = Path("representations")
OUTPUT_ROOT     = Path("data/4-representations")

FORMAT_EXT = {
    "html":     "html",
    "markdown": "md",
    "latex":    "tex",
}


def main():
    with open(SUC_FILE, "r", encoding="utf-8") as f:
        suc_data = json.load(f)

    # Collect unique (image_id, dataset) pairs
    seen = {}
    for entry in suc_data:
        iid = entry["image_id"]
        if iid not in seen:
            seen[iid] = entry["dataset"]

    print(f"Unique image_ids: {len(seen)}")

    # Create output dirs
    for fmt in FORMAT_EXT:
        (OUTPUT_ROOT / fmt).mkdir(parents=True, exist_ok=True)

    copied = {fmt: 0 for fmt in FORMAT_EXT}
    missing = []

    for image_id, dataset in seen.items():
        for fmt, ext in FORMAT_EXT.items():
            src_dir = REPR_ROOT / dataset / fmt
            dst_dir = OUTPUT_ROOT / fmt

            # Copy the format file
            fmt_file = src_dir / f"{image_id}.{ext}"
            if fmt_file.exists():
                shutil.copy2(fmt_file, dst_dir / fmt_file.name)
                copied[fmt] += 1
            else:
                missing.append(str(fmt_file))

            # Copy the png (same png lives in each format subfolder)
            png_file = src_dir / f"{image_id}.png"
            dst_png  = dst_dir / f"{image_id}.png"
            if png_file.exists() and not dst_png.exists():
                shutil.copy2(png_file, dst_png)

    print("\nFiles copied per format:")
    for fmt, count in copied.items():
        print(f"  {fmt}: {count} format files")

    if missing:
        print(f"\nMissing files ({len(missing)}):")
        for m in missing:
            print(f"  {m}")
    else:
        print("\n✓ All files copied successfully.")


if __name__ == "__main__":
    main()
