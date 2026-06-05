import json
import uuid
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("../../data/1-raw")
OUTPUT_DIR = Path("../../data/1-task")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_unique_uuid(existing_uuids):
    while True:
        new_uuid = uuid.uuid4().hex[:8]
        if new_uuid not in existing_uuids:
            existing_uuids.add(new_uuid)
            return new_uuid


def add_uuid_to_file(file_path):
    out_path = OUTPUT_DIR / file_path.name
    existing_uuids = set()
    with open(file_path, "r", encoding="utf-8") as f_in, open(
        out_path, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            try:
                sample = json.loads(line.strip())
                sample_uuid = generate_unique_uuid(existing_uuids)
                updated_sample = {"id": sample_uuid}
                updated_sample.update(sample)
                f_out.write(json.dumps(updated_sample, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[ERROR] in {file_path.name}: {e}")
                continue


def main():
    all_files = list(INPUT_DIR.glob("*.jsonl"))
    for file in tqdm(all_files, desc="Adding UUIDs"):
        add_uuid_to_file(file)


if __name__ == "__main__":
    main()
