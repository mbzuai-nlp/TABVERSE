import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
from datasets import load_dataset
from lxml import html
from apted import APTED, Config
from apted.helpers import Tree

# ============================== CONFIG ==============================
REPO_ID = "MOMINAAHSAN296/vtb-dataset"
DATASETS = ["hybridqa"]
FORMATS = ["html", "markdown", "latex"]
EXTENSIONS = {"html": ".html", "markdown": ".md", "latex": ".tex"}

# ground-truth root
LOCAL_GT_DIR = "/mnt/data1/momina/VisualTableBench/ground_truth"


# ============================== I/O HELPERS ==============================
def clean_generated_content(content: str) -> str:
    """Strip Markdown fences; keep inner text."""
    if not content:
        return ""
    content = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*\n", "", content)
    content = re.sub(r"\n```[\s\S]*$", "", content)
    return content.strip()


def ensure_html_document(s: str) -> str:
    """Wrap fragment into <html><body>...</body></html> so XPath works."""
    s = (s or "").strip()
    if "<html" in s.lower() and "<body" in s.lower():
        return s
    return f"<html><body>{s}</body></html>"


def has_table_substring(s: str, fmt: str) -> bool:
    """Cheap pre-check for diagnostics; not a real parser."""
    if not s:
        return False
    low = s.lower()
    if fmt == "html":
        return ("<table" in low) and ("</table>" in low)
    if fmt == "markdown":
        return "|" in s and bool(
            re.search(
                r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", s, flags=re.M
            )
        )
    if fmt == "latex":
        return "\\begin{tabular}" in s or "\\begin{table}" in s
    return False


def read_ground_truth_from_local(
    local_dir: str, dataset_name: str, format_type: str, image_id: str
) -> Optional[str]:
    ext = EXTENSIONS[format_type]
    p = Path(local_dir) / dataset_name / format_type / f"{image_id}{ext}"
    try:
        if p.exists():
            return clean_generated_content(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def get_dataset_image_ids(
    hf_token: str,
    dataset_name: str,
    pipeline_type: str = "generation",
    max_samples: Optional[int] = None,
) -> List[str]:
    """Get canonical image_id list from HF manifest or combined subset."""
    try:
        if pipeline_type == "generation_format":
            subset_path = (
                "/mnt/data1/momina/VisualTableBench/data/4-subset/combined_subset.json"
            )
            with open(subset_path, "r") as f:
                data = json.load(f)
            image_ids = [
                item["image_id"] for item in data if item.get("dataset") == dataset_name
            ]
            return image_ids[:max_samples] if max_samples is not None else image_ids
        else:
            ds = load_dataset(
                REPO_ID,
                data_files=f"data/3-suc/{dataset_name}.json",
                token=hf_token,
                streaming=False,
            )["train"]
            image_ids = [row["image_id"] for row in ds]
            return image_ids[:max_samples] if max_samples is not None else image_ids
    except Exception:
        return []


def load_generated_results_by_dataset(
    generation_path: str, fmt: str
) -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}
    generation_dir = Path(generation_path)
    if not generation_dir.exists():
        raise FileNotFoundError(f"Generation path {generation_path} does not exist")

    for jf in generation_dir.glob("*.json"):
        dataset_name = jf.stem
        results.setdefault(dataset_name, {})
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            for entry in data:
                image_id = entry.get("image_id")
                if not image_id:
                    continue
                text = None
                if isinstance(entry.get(fmt), dict):
                    text = entry[fmt].get("generated")
                if text in (None, "ERROR"):
                    text = entry.get(f"{fmt}_generated")
                if text in (None, "ERROR"):
                    text = entry.get("generated")
                if text and text != "ERROR":
                    results[dataset_name][image_id] = clean_generated_content(text)
        except Exception:
            continue
    return results


def load_generated_results_by_format(
    generation_path: str,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Load results for generation_format pipeline where each format has its own JSON file.
    Returns: Dict[input_format][image_id][output_format] = generated_content
    """
    results: Dict[str, Dict[str, Dict[str, str]]] = {f: {} for f in FORMATS}
    generation_dir = Path(generation_path)
    if not generation_dir.exists():
        raise FileNotFoundError(f"Generation path {generation_path} does not exist")
    for input_fmt in FORMATS:
        json_file = generation_dir / f"{input_fmt}.json"
        if not json_file.exists():
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            for entry in data:
                image_id = entry.get("image_id")
                if not image_id:
                    continue
                if image_id not in results[input_fmt]:
                    results[input_fmt][image_id] = {}
                block = entry.get(input_fmt)
                if not isinstance(block, dict):
                    continue
                for output_fmt in FORMATS:
                    if output_fmt == input_fmt:
                        continue
                    out = block.get(output_fmt)
                    if isinstance(out, dict):
                        gen = out.get("generated")
                        if gen and gen != "ERROR":
                            results[input_fmt][image_id][output_fmt] = (
                                clean_generated_content(gen)
                            )
        except Exception:
            continue
    return results


# ============================== TEDS CORE (Exact string per cell) ==============================
def _collapse_ws(s: str) -> str:
    """Only collapse whitespace. No Unicode/dash/quote/number/date normalization."""
    if not s:
        return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


class TableTree(Tree):
    """Minimal node for APTED. 'content' is a single string per cell."""

    def __init__(
        self, tag, colspan=None, rowspan=None, content: Optional[str] = None, *children
    ):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content  # string for 'td'
        self.children = list(children)


class ExactConfig(Config):
    def rename(self, n1, n2):
        # structural mismatch = full cost
        if (
            (n1.tag != n2.tag)
            or (n1.colspan != n2.colspan)
            or (n1.rowspan != n2.rowspan)
        ):
            return 1.0
        # cells: exact string equality (already whitespace-collapsed at parse time)
        if n1.tag == "td":
            a = n1.content or ""
            b = n2.content or ""
            return 0.0 if a == b else 1.0
        return 0.0


def _count_nodes(root: TableTree) -> int:
    n = 1
    for c in getattr(root, "children", []):
        n += _count_nodes(c)
    return n


def teds_similarity_from_trees(tree_pred: TableTree, tree_true: TableTree) -> float:
    n_nodes = max(_count_nodes(tree_pred), _count_nodes(tree_true))
    if n_nodes == 0:
        return 1.0
    ed = APTED(tree_pred, tree_true, ExactConfig()).compute_edit_distance()
    return 1.0 - (float(ed) / n_nodes)


# ============================== HTML ADAPTER (header text included) ==============================
def parse_html_table_to_tree(html_doc: str) -> TableTree:
    parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
    doc = html.fromstring(html_doc, parser=parser)
    tables = doc.xpath("body//table") or doc.xpath(".//table")
    if not tables:
        return TableTree("table", None, None, None, *deque())
    chosen = tables[0]

    # Collect rows in the order: thead -> tbody -> direct tr (avoid duplicates)
    rows: List = []
    rows.extend(chosen.xpath("./thead/tr"))
    rows.extend(chosen.xpath("./tbody/tr"))
    # add direct <tr> that are not already in thead/tbody
    direct_tr = [tr for tr in chosen.xpath("./tr") if tr not in rows]
    rows.extend(direct_tr)

    table = TableTree("table", None, None, None, *deque())
    tbody = TableTree("tbody", None, None, None, *deque())
    table.children.append(tbody)

    for tr_el in rows:
        tr = TableTree("tr", None, None, None, *deque())
        tbody.children.append(tr)
        for cell in tr_el.xpath("./th|./td"):
            colspan = int(cell.attrib.get("colspan", "1"))
            rowspan = int(cell.attrib.get("rowspan", "1"))
            txt = _collapse_ws("".join(cell.itertext()))
            tr.children.append(TableTree("td", colspan, rowspan, txt, *deque()))
    return table


# ============================== MARKDOWN ADAPTER (header text included) ==============================
_MD_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _md_split_row(ln: str) -> List[str]:
    ln = ln.strip()
    if ln.startswith("|"):
        ln = ln[1:]
    if ln.endswith("|"):
        ln = ln[:-1]
    return [c.strip() for c in ln.split("|")]


def _extract_md_tables(md: str) -> List[Tuple[int, int, List[str]]]:
    if not md:
        return []
    lines = md.splitlines()
    tables = []
    i, n = 0, len(lines)
    while i < n:
        if "|" not in lines[i]:
            i += 1
            continue
        j = i
        block = []
        while j < n and "|" in lines[j]:
            block.append(lines[j])
            j += 1
        sep_at = None
        for k in range(1, min(4, len(block))):
            if _MD_SEP_RE.match(block[k]):
                sep_at = k
                break
        if sep_at is not None:
            tables.append((i, j - 1, block))
        i = j
    return tables


def parse_md_table(md: str) -> TableTree:
    if not md:
        return TableTree("table", None, None, None, *deque())

    blocks = _extract_md_tables(md)
    if not blocks:
        return TableTree("table", None, None, None, *deque())
    _, _, block = blocks[0]

    sep_idx = None
    for k, ln in enumerate(block):
        if _MD_SEP_RE.match(ln):
            sep_idx = k
            break
    if sep_idx is None or sep_idx == 0:
        return TableTree("table", None, None, None, *deque())

    header_cells = _md_split_row(block[sep_idx - 1])
    data_rows = [_md_split_row(ln) for ln in block[sep_idx + 1 :] if "|" in ln]

    table = TableTree("table", None, None, None, *deque())
    tbody = TableTree("tbody", None, None, None, *deque())
    table.children.append(tbody)

    # header row WITH TEXT
    tr = TableTree("tr", None, None, None, *deque())
    tbody.children.append(tr)
    for cell in header_cells:
        tr.children.append(TableTree("td", 1, 1, _collapse_ws(cell), *deque()))

    # body rows (no merges)
    for r in data_rows:
        tr = TableTree("tr", None, None, None, *deque())
        tbody.children.append(tr)
        for cell in r:
            tr.children.append(TableTree("td", 1, 1, _collapse_ws(cell), *deque()))
    return table


# ============================== LaTeX ADAPTER (header text included) ==============================
_LATEX_TABLE_ENV_RE = re.compile(
    r"\\begin\{table\}[\s\S]*?\\begin\{tabular\}\{(?P<cols>[^}]*)\}(?P<body>[\s\S]*?)\\end\{tabular\}[\s\S]*?\\end\{table\}",
    re.MULTILINE,
)
_LATEX_TABULAR_RE = re.compile(
    r"\\begin\{tabular\}\{(?P<cols>[^}]*)\}(?P<body>[\s\S]*?)\\end\{tabular\}",
    re.MULTILINE,
)


def _split_unescaped(text: str, sep: str) -> List[str]:
    return re.split(r"(?<!\\)" + re.escape(sep), text)


def _strip_booktabs(body: str) -> str:
    return re.sub(r"\\(toprule|midrule|bottomrule)\s*", "", body)


def _strip_comments(body: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", body)


def _tex_unescape_and_collapse(s: str) -> str:
    """Unescape a few common LaTeX escapes, collapse whitespace only."""
    if not s:
        return ""
    s = (
        s.replace(r"\&", "&")
        .replace(r"\%", "%")
        .replace(r"\#", "#")
        .replace(r"\_", "_")
        .replace(r"\{", "{")
        .replace(r"\}", "}")
    )
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_rows_from_body(body: str) -> List[List[Tuple[str, int, int]]]:
    """Return rows → list of (text, colspan, rowspan) with multicolumn/multirow expanded."""
    body = _strip_comments(body)
    body = _strip_booktabs(body)
    body = body.strip()

    raw_rows = [r for r in _split_unescaped(body, r"\\") if r.strip()]
    rows: List[List[Tuple[str, int, int]]] = []
    active_multi = {}  # col_idx -> (remaining_rows, text)

    for raw in raw_rows:
        if re.search(r"\\hline|\\cline\{", raw):
            continue
        parts = [p.strip() for p in _split_unescaped(raw, "&")]

        expanded: List[Tuple[str, int, int]] = []
        for p in parts:
            mc = re.match(r"^\\multicolumn\{(\d+)\}\{[^}]*\}\{([\s\S]*)\}\s*$", p)
            if mc:
                expanded.append((mc.group(2).strip(), int(mc.group(1)), 1))
                continue
            mr = re.match(r"^\\multirow\{(\d+)\}\{[^}]*\}\{([\s\S]*)\}\s*$", p)
            if mr:
                expanded.append((mr.group(2).strip(), 1, int(mr.group(1))))
                continue
            expanded.append((p, 1, 1))

        laid: List[Tuple[str, int, int, str, bool]] = []
        col = 0

        while col in active_multi:
            remaining, txt = active_multi[col]
            laid.append(("", 1, 1, txt, True))
            remaining -= 1
            if remaining <= 0:
                del active_multi[col]
            else:
                active_multi[col] = (remaining, txt)
            col += 1

        for text, cspan, rspan in expanded:
            while col in active_multi:
                remaining, txt = active_multi[col]
                laid.append(("", 1, 1, txt, True))
                remaining -= 1
                if remaining <= 0:
                    del active_multi[col]
                else:
                    active_multi[col] = (remaining, txt)
                col += 1

            txt_norm = _tex_unescape_and_collapse(text)
            laid.append((text, cspan, rspan, txt_norm, False))
            if rspan > 1:
                active_multi[col] = (rspan - 1, txt_norm)
            col += max(1, cspan)

        row_cells: List[Tuple[str, int, int]] = []
        for _orig, cspan, rspan, txt, _ph in laid:
            row_cells.append((txt, cspan, rspan))
        rows.append(row_cells)

    return rows


def parse_latex_table(
    tex: str, prefer_tabular_inside_table_env: bool = True
) -> TableTree:
    if not tex:
        return TableTree("table", None, None, None, *deque())

    m = _LATEX_TABLE_ENV_RE.search(tex) if prefer_tabular_inside_table_env else None
    if not m:
        m = _LATEX_TABULAR_RE.search(tex)
    if not m:
        return TableTree("table", None, None, None, *deque())

    body = m.group("body")
    rows_with_spans = _extract_rows_from_body(body)
    if not rows_with_spans:
        return TableTree("table", None, None, None, *deque())

    table = TableTree("table", None, None, None, *deque())
    tbody = TableTree("tbody", None, None, None, *deque())
    table.children.append(tbody)

    # IMPORTANT: include header row WITH TEXT (no structure-only)
    for row in rows_with_spans:
        tr = TableTree("tr", None, None, None, *deque())
        tbody.children.append(tr)
        for txt, cspan, rspan in row:
            tr.children.append(TableTree("td", int(cspan), int(rspan), txt, *deque()))
    return table


# ============================== FRONT DOOR ==============================
def parse_to_TableTree(pred: str, true: str, fmt: str):
    if fmt == "html":
        return (
            parse_html_table_to_tree(ensure_html_document(pred or "")),
            parse_html_table_to_tree(ensure_html_document(true or "")),
        )
    if fmt == "markdown":
        return (parse_md_table(pred or ""), parse_md_table(true or ""))
    if fmt == "latex":
        return (parse_latex_table(pred or ""), parse_latex_table(true or ""))
    raise ValueError(f"Unsupported format: {fmt}")


def teds_similarity_generic(pred: str, true: str, fmt: str) -> float:
    """Exact-match TEDS: cell strings (incl. headers) must match for TEDS=1.0."""
    try:
        tree_pred, tree_true = parse_to_TableTree(pred, true, fmt)
        return teds_similarity_from_trees(tree_pred, tree_true)
    except Exception as e:
        print(f"[error] TEDS computation failed for {fmt}: {e}")
        return 0.0


# ============================== EVALUATION ==============================
def evaluate_model_teds_generic(
    generation_path: str,
    hf_token: str,
    fmt: str,
    pipeline_type: str = "generation",
    diag_first_k: int = 5,
    max_samples: int = None,
) -> Dict[str, Dict[str, float]]:
    if pipeline_type == "generation_format":
        generated_results_by_format = load_generated_results_by_format(generation_path)
        results: Dict[str, Dict[str, float]] = {}
        for input_fmt in FORMATS:
            for output_fmt in FORMATS:
                if input_fmt == output_fmt or fmt != output_fmt:
                    continue
                for dataset_name in DATASETS:
                    dataset_image_ids = get_dataset_image_ids(
                        hf_token, dataset_name, pipeline_type, max_samples
                    )

                    common_images = []
                    if input_fmt in generated_results_by_format:
                        for img_id in dataset_image_ids:
                            if (
                                img_id in generated_results_by_format[input_fmt]
                                and output_fmt
                                in generated_results_by_format[input_fmt][img_id]
                            ):
                                common_images.append(img_id)

                    sims = []
                    processed = 0
                    for i, image_id in enumerate(common_images):
                        gt_content = read_ground_truth_from_local(
                            LOCAL_GT_DIR, dataset_name, output_fmt, image_id
                        )
                        if gt_content is None:
                            continue
                        pred_content = generated_results_by_format[input_fmt][image_id][
                            output_fmt
                        ]
                        sim = teds_similarity_generic(
                            pred_content, gt_content, fmt=output_fmt
                        )
                        sims.append(sim)
                        processed += 1
                        if i < diag_first_k:
                            print(
                                f"[{dataset_name} | {input_fmt}->{output_fmt}] image_id={image_id} TEDS={sim:.4f}"
                            )

                    key = f"{dataset_name}-{input_fmt}-{output_fmt}"
                    results[key] = {
                        "teds": (sum(sims) / len(sims)) if sims else 0.0,
                        "count": processed,
                    }
                    print(
                        f"[avg] {key}: TEDS={results[key]['teds']:.3f} (n={processed})"
                    )
        return results

    else:
        preds = load_generated_results_by_dataset(generation_path, fmt)
        results: Dict[str, Dict[str, float]] = {}

        for dataset_name in DATASETS:
            image_ids = get_dataset_image_ids(
                hf_token, dataset_name, pipeline_type, max_samples
            )
            sims = []
            if dataset_name not in preds:
                results[dataset_name] = {"teds": 0.0, "count": 0}
                print(f"[avg] {dataset_name} [{fmt}]: TEDS=0.000 (n=0)")
                continue

            common = [iid for iid in image_ids if iid in preds[dataset_name]]
            processed = 0
            for i, image_id in enumerate(common):
                gt_text = read_ground_truth_from_local(
                    LOCAL_GT_DIR, dataset_name, fmt, image_id
                )
                if gt_text is None:
                    continue
                pred_text = preds[dataset_name][image_id]
                sim = teds_similarity_generic(pred_text, gt_text, fmt=fmt)
                sims.append(sim)
                processed += 1
                if i < diag_first_k:
                    print(
                        f"[{dataset_name} | {fmt}] image_id={image_id} TEDS={sim:.4f}"
                    )

            avg_sim = (sum(sims) / len(sims)) if sims else 0.0
            results[dataset_name] = {"teds": avg_sim, "count": processed}
            print(f"[avg] {dataset_name} [{fmt}]: TEDS={avg_sim:.3f} (n={processed})")
        return results


def evaluate_all_formats(
    generation_path: str,
    hf_token: str,
    pipeline_type: str = "generation",
    diag_first_k: int = 5,
    max_samples: Optional[int] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """results_by_fmt[fmt][dataset] = {'teds': float, 'count': int} or results_by_fmt[fmt][conversion_key] = {'teds': float, 'count': int}"""
    results_by_fmt = {}
    for fmt in FORMATS:
        results_by_fmt[fmt] = evaluate_model_teds_generic(
            generation_path=generation_path,
            hf_token=hf_token,
            fmt=fmt,
            pipeline_type=pipeline_type,
            diag_first_k=diag_first_k,
            max_samples=max_samples,
        )
    return results_by_fmt


def save_multi_format_csv(
    model_name: str,
    results_by_fmt: Dict[str, Dict[str, Dict[str, float]]],
    generation_path: str,
    pipeline_type: str = "generation",
) -> str:
    """One CSV aggregating all formats."""
    rows = []
    if pipeline_type == "generation_format":
        for fmt, per_ds in results_by_fmt.items():
            for key, m in per_ds.items():
                parts = key.split("-")
                if len(parts) >= 3:
                    dataset_name, input_fmt, output_fmt = parts[0], parts[1], parts[2]
                    rows.append(
                        {
                            "Dataset": dataset_name,
                            "Image_Format": input_fmt,
                            "Structure_Format": output_fmt,
                            "TEDS": m.get("teds", 0.0),
                            "Count": m.get("count", 0),
                        }
                    )
    else:
        for fmt, per_ds in results_by_fmt.items():
            for dataset_name, m in per_ds.items():
                rows.append(
                    {
                        "Dataset": dataset_name,
                        "Format": fmt,
                        "TEDS": m.get("teds", 0.0),
                        "Count": m.get("count", 0),
                    }
                )
    df = pd.DataFrame(rows)
    out_csv = os.path.join(
        generation_path, f"{model_name}_teds_evaluation_scores_QA.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"[csv] Saved TEDS results to {out_csv}")
    return out_csv


# ============================== CLI ==============================
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate generated tables (HTML/MD/LaTeX) with TEDS"
    )
    ap.add_argument(
        "--generation_paths",
        nargs="+",
        required=True,
        help="Folders with <dataset>.json prediction files",
    )
    ap.add_argument("--hf_token", required=True, help="Hugging Face token")
    ap.add_argument(
        "--pipeline_type",
        type=str,
        choices=["generation", "generation_format"],
        default="generation",
        help="Pipeline type: 'generation' or 'generation_format'",
    )
    ap.add_argument(
        "--diag_first_k",
        type=int,
        default=5,
        help="Diagnostics for first K items (per dataset/format)",
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate per dataset",
    )

    args = ap.parse_args()

    print(f"[cfg] GT root: {LOCAL_GT_DIR}")
    print(f"[cfg] Formats: {FORMATS}")
    print(f"[cfg] Pipeline: {args.pipeline_type}")

    all_results = {}
    csv_files = []

    for gen_path in args.generation_paths:
        model_name = os.path.basename(os.path.dirname(gen_path)) or os.path.basename(
            gen_path.rstrip("/")
        )
        print(f"\n=== Evaluating {model_name} ===")
        try:
            res_by_fmt = evaluate_all_formats(
                gen_path,
                args.hf_token,
                args.pipeline_type,
                args.diag_first_k,
                args.max_samples,
            )
            all_results[model_name] = res_by_fmt
            csv_file = save_multi_format_csv(
                model_name, res_by_fmt, gen_path, args.pipeline_type
            )
            csv_files.append(csv_file)
            print(f"[done] {model_name}")
        except Exception as e:
            print(f"[error] evaluating {model_name}: {e}")

    # Summary section
    print("\n=== TEDS EVALUATION SUMMARY ===")
    for model_name, model_results in all_results.items():
        print(f"\n{model_name}:")
        if args.pipeline_type == "generation_format":
            print(
                f"{'Dataset':<12} {'Image_Fmt':<11} {'Struct_Fmt':<11} {'TEDS':<8} {'Count':<6}"
            )
            print("-" * 52)
            for fmt, per_ds in model_results.items():
                for key, m in per_ds.items():
                    parts = key.split("-")
                    if len(parts) >= 3:
                        dataset_name, input_fmt, output_fmt = (
                            parts[0],
                            parts[1],
                            parts[2],
                        )
                        print(
                            f"{dataset_name:<12} {input_fmt:<11} {output_fmt:<11} {m['teds']:<8.3f} {m['count']:<6}"
                        )
        else:
            print(f"{'Dataset':<12} {'Format':<10} {'TEDS':<8} {'Count':<6}")
            print("-" * 40)
            for fmt, per_ds in model_results.items():
                for dataset_name, m in per_ds.items():
                    print(
                        f"{dataset_name:<12} {fmt:<10} {m['teds']:<8.3f} {m['count']:<6}"
                    )

    print("\nCSV files:")
    for f in csv_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
