# upload_single_dataset.py
from pathlib import Path
from huggingface_hub import create_repo, upload_folder

REPO_ID = "MOMINAAHSAN296/vtb-dataset"  # HF dataset repo
PRIVATE = True
LOCAL_ROOT = Path("/mnt/data1/momina/VisualTableBench")


def upload_one(ds: str):
    src = LOCAL_ROOT / ds
    if not src.exists():
        raise FileNotFoundError(f"Local folder not found: {src}")
    if not any(src.rglob("*")):
        raise RuntimeError(f"{src} is empty.")

    # Create the repo if needed (does not clone)
    create_repo(REPO_ID, repo_type="dataset", private=PRIVATE, exist_ok=True)

    # Upload ONLY this dataset’s folder to the path ds/ in the repo
    upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(src),  # points to your local sqa/ (or tabfact/, totto/)
        path_in_repo=ds,  # ensures files land under ds/ on the Hub
        commit_message=f"Upload {ds} (html/markdown/latex)",
        ignore_patterns=["*.DS_Store", "__pycache__/**", ".ipynb_checkpoints/**"],
    )
    print(f" Uploaded {ds} to https://huggingface.co/datasets/{REPO_ID}/tree/main/{ds}")


if __name__ == "__main__":
    # change to "tabfact" or "totto" etc. and re-run to upload another folder
    upload_one("data")
