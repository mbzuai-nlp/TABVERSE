import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import pandas as pd
from datasets import load_dataset
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings("ignore")

REPO_ID = "MOMINAAHSAN296/vtb-dataset"
DATASETS = ["feverous", "hybridqa", "sqa", "tabfact", "totto"]
FORMATS = ["html", "markdown", "latex"]
EXTENSIONS = {"html": ".html", "markdown": ".md", "latex": ".tex"}


def clean_content(content: str) -> str:
    if not content:
        return ""
    content = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*\n", "", content)
    return re.sub(r"\n```(\s*)$", "", content).strip()


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()) if s else ""


def get_image_ids(
    hf_token: str, dataset: str, pipeline: str = "generation"
) -> List[str]:
    try:
        if pipeline == "generation_format":
            with open(
                "/mnt/data1/momina/VisualTableBench/data/4-subset/combined_subset.json",
                "r",
            ) as f:
                return [
                    item["image_id"]
                    for item in json.load(f)
                    if item.get("dataset") == dataset
                ]
        else:
            ds = load_dataset(
                REPO_ID,
                data_files=f"data/3-suc/{dataset}.json",
                token=hf_token,
                streaming=False,
            )["train"]
            return [row["image_id"] for row in ds]
    except:
        return []


def read_gt(local_dir: str, dataset: str, fmt: str, image_id: str) -> Optional[str]:
    path = Path(local_dir) / dataset / fmt / f"{image_id}{EXTENSIONS[fmt]}"
    try:
        return (
            clean_content(path.read_text(encoding="utf-8")) if path.exists() else None
        )
    except:
        return None


def parse_html(html: str) -> List[List[str]]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if not table:
        return []

    rows, occupied = table.find_all("tr"), defaultdict(set)
    grid = []

    for r, tr in enumerate(rows):
        c = 0
        row_data = []
        for cell in tr.find_all(["td", "th"], recursive=False):
            while c in occupied[r]:
                c += 1
            rs, cs = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
            txt = normalize_text(cell.get_text(" ", strip=True) if cell.text else "")

            for rr in range(r, r + rs):
                for cc in range(c, c + cs):
                    occupied[rr].add(cc)
                    if rr == r:
                        row_data.append((cc, txt))
            c += cs

        row_data.sort()
        if r >= len(grid):
            grid.extend([[] for _ in range(r - len(grid) + 1)])
        grid[r] = [txt for _, txt in row_data]

    return grid


def parse_markdown(md: str) -> List[List[str]]:
    lines = [ln.strip() for ln in md.strip().splitlines()]
    rows = [
        ln
        for ln in lines
        if ln.startswith("|")
        and ln.count("|") >= 2
        and not re.match(r"^\|[\s\-\|:]+\|$", ln)
    ]
    if not rows:
        return []

    grid = []
    for ln in rows:
        cells = [normalize_text(p.strip()) for p in ln.strip("|").split("|")]
        grid.append(cells)

    if grid:
        max_cols = max(len(row) for row in grid)
        grid = [row + [""] * (max_cols - len(row)) for row in grid]

    return grid


def parse_latex(tex: str) -> List[List[str]]:
    m = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", tex, re.S)
    if not m:
        return []

    body = re.sub(r"\\(toprule|midrule|bottomrule|hline)", "", m.group(1))
    raw_rows = [ln.strip() for ln in body.split("\\\\") if ln.strip()]

    grid = []
    for row_text in raw_rows:
        row_safe = row_text.replace(r"\&", "__AMP__")
        cells = [
            normalize_text(c.strip().replace("__AMP__", "&"))
            for c in row_safe.split("&")
        ]
        cells = [re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", c) for c in cells]
        cells = [re.sub(r"\\[a-zA-Z]+", "", c).strip() for c in cells]
        grid.append(cells)

    if grid:
        max_cols = max(len(row) for row in grid)
        grid = [row + [""] * (max_cols - len(row)) for row in grid]

    return grid


def content_to_grid(content: str, fmt: str) -> List[List[str]]:
    parsers = {"html": parse_html, "markdown": parse_markdown, "latex": parse_latex}
    return parsers.get(fmt, lambda x: [])(content)


def calculate_accuracies(
    true_grid: List[List[str]], pred_grid: List[List[str]]
) -> Dict[str, float]:
    if not true_grid or not pred_grid:
        return {"cell_acc": 0.0, "row_acc": 0.0, "column_acc": 0.0}

    # Pad grids to same size
    max_rows = max(len(true_grid), len(pred_grid))
    max_cols = max(
        max(len(row) for row in true_grid) if true_grid else 0,
        max(len(row) for row in pred_grid) if pred_grid else 0,
    )

    true_padded = [row + [""] * (max_cols - len(row)) for row in true_grid] + [
        [""] * max_cols for _ in range(max_rows - len(true_grid))
    ]
    pred_padded = [row + [""] * (max_cols - len(row)) for row in pred_grid] + [
        [""] * max_cols for _ in range(max_rows - len(pred_grid))
    ]

    # Cell accuracy
    total_cells = max_rows * max_cols
    correct_cells = sum(
        1
        for i in range(max_rows)
        for j in range(max_cols)
        if true_padded[i][j] == pred_padded[i][j]
    )
    cell_acc = correct_cells / total_cells if total_cells > 0 else 0.0

    # Row accuracy
    correct_rows = sum(1 for i in range(max_rows) if true_padded[i] == pred_padded[i])
    row_acc = correct_rows / max_rows if max_rows > 0 else 0.0

    # Column accuracy
    correct_cols = sum(
        1
        for j in range(max_cols)
        if [true_padded[i][j] for i in range(max_rows)]
        == [pred_padded[i][j] for i in range(max_rows)]
    )
    column_acc = correct_cols / max_cols if max_cols > 0 else 0.0

    return {"cell_acc": cell_acc, "row_acc": row_acc, "column_acc": column_acc}


def load_results_dataset(path: str) -> Dict:
    results, generation_dir = {}, Path(path)
    if not generation_dir.exists():
        raise FileNotFoundError(f"Path {path} does not exist")

    for jf in generation_dir.glob("*.json"):
        dataset_name = jf.stem
        results[dataset_name] = {}
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            for entry in data:
                image_id = entry.get("image_id")
                if not image_id:
                    continue
                if image_id not in results[dataset_name]:
                    results[dataset_name][image_id] = {}
                for fmt in FORMATS:
                    block = entry.get(fmt)
                    if isinstance(block, dict):
                        gen = block.get("generated")
                        if gen and gen != "ERROR":
                            results[dataset_name][image_id][fmt] = clean_content(gen)
        except:
            continue
    return results


def load_results_format(path: str, model_name: str = "") -> Dict:
    results = {fmt: {} for fmt in FORMATS}

    # Special handling for GPT-5 model
    if "gpt-5" in model_name.lower() or "gpt5" in model_name.lower():
        return load_gpt5_results_format(path)

    for input_fmt in FORMATS:
        json_file = Path(path) / f"{input_fmt}.json"
        if json_file.exists():
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                for entry in data:
                    image_id = entry.get("image_id")
                    if image_id and input_fmt in entry:
                        if image_id not in results[input_fmt]:
                            results[input_fmt][image_id] = {}
                        input_block = entry[input_fmt]
                        for output_fmt in FORMATS:
                            if output_fmt != input_fmt and output_fmt in input_block:
                                output_block = input_block[output_fmt]
                                if isinstance(output_block, dict):
                                    gen_content = output_block.get("generated")
                                    if gen_content and gen_content != "ERROR":
                                        results[input_fmt][image_id][output_fmt] = (
                                            clean_content(gen_content)
                                        )
            except:
                continue
    return results


def load_gpt5_results_format(path: str) -> Dict:
    """Special loader for GPT-5 format where all formats are nested under each format key"""
    results = {fmt: {} for fmt in FORMATS}

    for input_fmt in FORMATS:
        json_file = Path(path) / f"{input_fmt}.json"
        if json_file.exists():
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                for entry in data:
                    image_id = entry.get("image_id")
                    if image_id and input_fmt in entry:
                        if image_id not in results[input_fmt]:
                            results[input_fmt][image_id] = {}
                        input_block = entry[input_fmt]

                        # For GPT-5: input_block contains all formats (html, markdown, latex)
                        # We need to extract all formats, not just cross-format ones
                        for output_fmt in FORMATS:
                            if output_fmt in input_block:
                                output_block = input_block[output_fmt]
                                if isinstance(output_block, dict):
                                    gen_content = output_block.get("generated")
                                    if gen_content and gen_content != "ERROR":
                                        results[input_fmt][image_id][output_fmt] = (
                                            clean_content(gen_content)
                                        )
            except:
                continue
    return results


class AccuracyEvaluator:
    def __init__(self, local_gt_dir: str, hf_token: str):
        self.local_gt_dir, self.hf_token = local_gt_dir, hf_token

    def evaluate_model(
        self,
        generation_path: str,
        pipeline_type: str = "generation",
        model_name: str = "",
    ) -> Dict[str, Dict[str, float]]:
        if pipeline_type == "generation_format":
            return self._evaluate_format_pipeline(generation_path, model_name)
        return self._evaluate_dataset_pipeline(generation_path)

    def _evaluate_format_pipeline(self, path: str, model_name: str = "") -> Dict:
        generated_data, results = load_results_format(path, model_name), {}
        is_gpt5 = "gpt-5" in model_name.lower() or "gpt5" in model_name.lower()

        for input_fmt in FORMATS:
            for output_fmt in FORMATS:
                # For GPT-5, include identity transformations (same format to same format)
                # For other models, skip identity transformations
                if not is_gpt5 and input_fmt == output_fmt:
                    continue

                for dataset_name in DATASETS:
                    image_ids = get_image_ids(
                        self.hf_token, dataset_name, "generation_format"
                    )
                    scores = self._eval_format_combo(
                        generated_data, input_fmt, output_fmt, dataset_name, image_ids
                    )
                    results[f"{dataset_name}-{input_fmt}-{output_fmt}"] = scores
        return results

    def _evaluate_dataset_pipeline(self, path: str) -> Dict:
        generated_data, results = load_results_dataset(path), {}
        for dataset_name in DATASETS:
            image_ids = get_image_ids(self.hf_token, dataset_name, "generation")
            for fmt in FORMATS:
                scores = self._eval_dataset_fmt(
                    generated_data, dataset_name, fmt, image_ids
                )
                results[f"{dataset_name}-{fmt}"] = scores
        return results

    def _eval_format_combo(
        self, data: Dict, in_fmt: str, out_fmt: str, dataset: str, ids: List[str]
    ) -> Dict:
        scores = {"cell_acc": [], "row_acc": [], "column_acc": []}
        processed = 0
        if in_fmt in data:
            for img_id in ids:
                if img_id in data[in_fmt] and out_fmt in data[in_fmt][img_id]:
                    gt_content = read_gt(self.local_gt_dir, dataset, out_fmt, img_id)
                    if gt_content is None:
                        continue
                    pred_content = data[in_fmt][img_id][out_fmt]
                    try:
                        true_grid = content_to_grid(gt_content, out_fmt)
                        pred_grid = content_to_grid(pred_content, out_fmt)
                        metrics = calculate_accuracies(true_grid, pred_grid)
                        for key in scores:
                            scores[key].append(metrics[key])
                        processed += 1
                    except:
                        for key in scores:
                            scores[key].append(0.0)
        result = {
            key: sum(vals) / len(vals) if vals else 0.0 for key, vals in scores.items()
        }
        result["count"] = processed
        return result

    def _eval_dataset_fmt(
        self, data: Dict, dataset: str, fmt: str, ids: List[str]
    ) -> Dict:
        scores = {"cell_acc": [], "row_acc": [], "column_acc": []}
        processed = 0
        if dataset in data:
            for img_id in ids:
                if img_id in data[dataset] and fmt in data[dataset][img_id]:
                    gt_content = read_gt(self.local_gt_dir, dataset, fmt, img_id)
                    if gt_content is None:
                        continue
                    pred_content = data[dataset][img_id][fmt]
                    try:
                        true_grid = content_to_grid(gt_content, fmt)
                        pred_grid = content_to_grid(pred_content, fmt)
                        metrics = calculate_accuracies(true_grid, pred_grid)
                        for key in scores:
                            scores[key].append(metrics[key])
                        processed += 1
                    except:
                        for key in scores:
                            scores[key].append(0.0)
        result = {
            key: sum(vals) / len(vals) if vals else 0.0 for key, vals in scores.items()
        }
        result["count"] = processed
        return result


def save_csv(model_name: str, results: Dict, path: str, pipeline_type: str) -> str:
    rows = []
    if pipeline_type == "generation_format":
        is_gpt5 = "gpt-5" in model_name.lower() or "gpt5" in model_name.lower()

        for in_fmt in FORMATS:
            for out_fmt in FORMATS:
                # For GPT-5, include identity transformations; for others, skip them
                if not is_gpt5 and in_fmt == out_fmt:
                    continue

                for dataset in DATASETS:
                    key = f"{dataset}-{in_fmt}-{out_fmt}"
                    m = results.get(
                        key,
                        {
                            "cell_acc": 0.0,
                            "row_acc": 0.0,
                            "column_acc": 0.0,
                            "count": 0,
                        },
                    )
                    rows.append(
                        {
                            "Dataset": dataset,
                            "Image_Format": in_fmt,
                            "Structure_Format": out_fmt,
                            "Cell_Accuracy": m["cell_acc"],
                            "Row_Accuracy": m["row_acc"],
                            "Column_Accuracy": m["column_acc"],
                            "Count": m["count"],
                        }
                    )
    else:
        for dataset in DATASETS:
            for fmt in FORMATS:
                key = f"{dataset}-{fmt}"
                m = results.get(
                    key,
                    {"cell_acc": 0.0, "row_acc": 0.0, "column_acc": 0.0, "count": 0},
                )
                rows.append(
                    {
                        "Dataset": dataset,
                        "Format": fmt,
                        "Cell_Accuracy": m["cell_acc"],
                        "Row_Accuracy": m["row_acc"],
                        "Column_Accuracy": m["column_acc"],
                        "Count": m["count"],
                    }
                )

    out_csv = os.path.join(path, f"{model_name}_word_accuracy_scores.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved accuracy results to {out_csv}")
    return out_csv


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate table content using word-level accuracy metrics"
    )
    parser.add_argument(
        "--generation_paths",
        nargs="+",
        required=True,
        help="Paths to generation directories",
    )
    parser.add_argument("--hf_token", required=True, help="Hugging Face token")
    parser.add_argument(
        "--local_gt_dir",
        default="/mnt/data1/momina/VisualTableBench/ground_truth",
        help="Ground truth directory",
    )
    parser.add_argument(
        "--pipeline_type",
        choices=["generation", "generation_format"],
        default="generation",
        help="Pipeline type",
    )
    args = parser.parse_args()

    print(f"Starting Word Accuracy evaluation (Pipeline: {args.pipeline_type})")
    evaluator = AccuracyEvaluator(args.local_gt_dir, args.hf_token)
    all_results = {}

    for gen_path in args.generation_paths:
        model_name = os.path.basename(os.path.dirname(gen_path)) or os.path.basename(
            gen_path.rstrip("/")
        )
        print(f"\nEvaluating {model_name}...")
        try:
            results = evaluator.evaluate_model(gen_path, args.pipeline_type, model_name)
            all_results[model_name] = results
            save_csv(model_name, results, gen_path, args.pipeline_type)
            print(f"✓ Completed {model_name}")
        except Exception as e:
            print(f"✗ Error evaluating {model_name}: {e}")

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    for model_name, results in all_results.items():
        print(f"\n{model_name}: {len(results)} evaluations completed")


if __name__ == "__main__":
    main()
