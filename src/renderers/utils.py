from pathlib import Path

def get_all_unrendered_files(folder: Path, ext: str):
    files = sorted(folder.glob(f"*{ext}"))
    return [f for f in files if not f.with_suffix(".png").exists()]