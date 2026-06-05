import json
import os
from pathlib import Path
from tqdm import tqdm
from src.utils.structured_data_linearize import StructuredDataLinearize

# Config
FORMATS = {
    "markdown": "md",
    "html": "html",
    "latex": "tex",
}

INPUT_DIR = Path("data/3-suc")  # now contains .json files
OUTPUT_ROOT = Path("structured_representations")
USE_STRUCTURE_MARK = True
ADD_GRAMMAR = False
CHANGE_ORDER = False


def clean_caption(caption):
    return "".join(caption) if isinstance(caption, list) else caption


def iter_samples_from_json(file_path: Path):
    """Yield samples from a JSON file.
    Supports:
      - A list of samples
      - A dict with key 'data' containing a list of samples
      - A single-sample dict
    """
    with open(file_path, "r", encoding="utf-8") as f_in:
        obj = json.load(f_in)

    if isinstance(obj, list):
        for s in obj:
            yield s
    elif isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            for s in obj["data"]:
                yield s
        else:
            # Assume single sample dict
            yield obj
    else:
        raise ValueError(f"Unsupported JSON structure in {file_path}")


def generate_representations(file_path: Path):
    dataset = file_path.stem  # e.g., "feverous"
    for sample in tqdm(
        list(iter_samples_from_json(file_path)), desc=f"Processing {dataset}"
    ):
        try:
            sample_id = sample["image_id"]

            context_val = sample.get("context", "")
            context_str = (
                "".join(context_val)
                if isinstance(context_val, list)
                else (context_val or "")
            )
            passage = sample.get("passage", "")
            caption = clean_caption(sample.get("table", {}).get("caption", ""))

            structured_data_dict = {
                "title": sample.get("title", ""),
                "context": context_str + passage,
                "table": {
                    "header": sample["table"]["header"],
                    "rows": sample["table"]["rows"],
                    "caption": caption,
                },
            }

            linearizer = StructuredDataLinearize()

            for fmt, ext in FORMATS.items():
                content = linearizer.retrieve_linear_function(
                    func=fmt,
                    use_structure_mark=USE_STRUCTURE_MARK,
                    add_grammar=ADD_GRAMMAR,
                    change_order=CHANGE_ORDER,
                    structured_data_dict=structured_data_dict,
                )

                out_dir = OUTPUT_ROOT / dataset / fmt
                out_dir.mkdir(parents=True, exist_ok=True)

                out_path = out_dir / f"{sample_id}.{ext}"
                with open(out_path, "w", encoding="utf-8") as f_out:
                    f_out.write(content)

        except Exception as e:
            print(f"[ERROR] in file {file_path.name}, sample: {e}")
            continue


def main():
    for file_path in INPUT_DIR.glob("*.json"):
        generate_representations(file_path)


if __name__ == "__main__":
    main()
