import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
import requests
from huggingface_hub import hf_hub_url
from tqdm import tqdm

# HF Repo and Dataset configuration
REPO_ID = "MOMINAAHSAN296/vtb-dataset"
DATASETS = ["feverous", "hybridqa", "sqa", "tabfact", "totto"]
FORMATS = ["html", "markdown", "latex"]
EXTENSIONS = {"html": ".html", "markdown": ".md", "latex": ".tex"}


def download_ground_truth_file(
    hf_token: str, dataset_name: str, format_type: str, image_id: str, local_dir: str
) -> bool:
    """
    Download a single ground truth file to local directory.
    """
    ext = EXTENSIONS[format_type]
    filename = f"{dataset_name}/{format_type}/{image_id}{ext}"
    url = hf_hub_url(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    # Create local file path
    local_file_dir = Path(local_dir) / dataset_name / format_type
    local_file_dir.mkdir(parents=True, exist_ok=True)
    local_file_path = local_file_dir / f"{image_id}{ext}"

    # Skip if file already exists
    if local_file_path.exists():
        return True

    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            with open(local_file_path, "w", encoding="utf-8") as f:
                f.write(r.text)
            return True
        else:
            print(f"Failed to download {filename}: HTTP {r.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False


def get_dataset_image_ids(hf_token: str, dataset_name: str) -> list:
    """
    Get list of image IDs for a dataset from the JSON file.
    """
    try:
        gt_dataset = load_dataset(
            REPO_ID,
            data_files=f"data/3-suc/{dataset_name}.json",
            token=hf_token,
            streaming=False,
        )["train"]

        return [entry["image_id"] for entry in gt_dataset]
    except Exception as e:
        print(f"Error loading image IDs for {dataset_name}: {e}")
        return []


def download_all_ground_truth(hf_token: str, local_dir: str):
    """
    Download all ground truth files for all datasets and formats.
    """
    total_files = 0
    downloaded_files = 0

    for dataset_name in DATASETS:
        print(f"\nProcessing {dataset_name}...")

        # Get image IDs for this dataset
        image_ids = get_dataset_image_ids(hf_token, dataset_name)
        print(f"Found {len(image_ids)} images in {dataset_name}")

        if not image_ids:
            continue

        for format_type in FORMATS:
            print(f"  Downloading {format_type} files...")

            format_downloaded = 0
            format_total = len(image_ids)

            # Use tqdm for progress bar
            for image_id in tqdm(image_ids, desc=f"  {format_type}", leave=False):
                total_files += 1
                if download_ground_truth_file(
                    hf_token, dataset_name, format_type, image_id, local_dir
                ):
                    downloaded_files += 1
                    format_downloaded += 1

            print(
                f"    Downloaded {format_downloaded}/{format_total} {format_type} files"
            )

    print(f"\nDownload complete!")
    print(f"Total files downloaded: {downloaded_files}/{total_files}")
    print(f"Files saved to: {local_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download ground truth files locally")
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="Hugging Face token for accessing the dataset",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./ground_truth",
        help="Local directory to save ground truth files",
    )

    args = parser.parse_args()

    print("Starting ground truth download...")
    print(f"Will save files to: {args.local_dir}")

    # Create local directory
    Path(args.local_dir).mkdir(parents=True, exist_ok=True)

    # Download all files
    download_all_ground_truth(args.hf_token, args.local_dir)


if __name__ == "__main__":
    main()
