# teds_multi_eval.py
# End-to-end TEDS for HTML / Markdown / LaTeX
# Policy:
# - Word-level tokens
# - Exact match after light normalization (lowercase, numbers, single-token dates)
# - Headers included for structure-only scoring (header text ignored)
# - Preserve real rowspan/colspan where available
# - Index columns are ALWAYS KEPT (no dropping)

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

from lxml import etree, html
from apted import APTED, Config
from apted.helpers import Tree

from decimal import Decimal, InvalidOperation
from datetime import datetime


# ============================== CONFIG ==============================
REPO_ID = "MOMINAAHSAN296/vtb-dataset"
DATASETS = ["feverous", "hybridqa", "sqa", "tabfact", "totto"]
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
    hf_token: str, dataset_name: str, pipeline_type: str = "generation"
) -> List[str]:
    """Get canonical image_id list from HF manifest or combined subset."""
    try:
        if pipeline_type == "generation_format":
            # Load from combined subset file
            subset_path = (
                "/mnt/data1/momina/VisualTableBench/data/4-subset/combined_subset.json"
            )
            with open(subset_path, "r") as f:
                data = json.load(f)
            # Filter by dataset and return image_ids
            return [
                item["image_id"] for item in data if item.get("dataset") == dataset_name
            ]
        else:
            # Original generation pipeline
            ds = load_dataset(
                REPO_ID,
                data_files=f"data/3-suc/{dataset_name}.json",
                token=hf_token,
                streaming=False,
            )["train"]
            return [row["image_id"] for row in ds]
    except Exception:
        return []


def load_generated_results_by_dataset(
    generation_path: str, fmt: str
) -> Dict[str, Dict[str, str]]:
    print(f"Loading [{fmt}] predictions from: {generation_path}")
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
                # prefer format-specific nested dict
                if isinstance(entry.get(fmt), dict):
                    text = entry[fmt].get("generated")
                # explicit key like "html_generated"
                if text in (None, "ERROR"):
                    text = entry.get(f"{fmt}_generated")
                # plain "generated"
                if text in (None, "ERROR"):
                    text = entry.get("generated")

                if text and text != "ERROR":
                    results[dataset_name][image_id] = clean_generated_content(text)
        except Exception as e:
            print(f"Error processing {jf}: {e}")
            continue

    return results


def load_generated_results_by_format(
    generation_path: str,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Load results for generation_format pipeline where each format has its own JSON file.
    Returns: Dict[input_format][image_id][output_format] = generated_content
    """
    print(f"Loading generated results from: {generation_path}")
    results: Dict[str, Dict[str, Dict[str, str]]] = {}
    generation_dir = Path(generation_path)
    if not generation_dir.exists():
        raise FileNotFoundError(f"Generation path {generation_path} does not exist")

    # Initialize results structure
    for input_fmt in FORMATS:
        results[input_fmt] = {}

    # Look for format-specific JSON files (html.json, markdown.json, latex.json)
    for input_fmt in FORMATS:
        json_file = generation_dir / f"{input_fmt}.json"
        print(f"  Looking for {json_file}")
        if json_file.exists():
            print(f"  Found {json_file}, loading...")
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                print(f"    Loaded {len(data)} entries from {input_fmt}.json")

                for i, entry in enumerate(data):
                    if i < 3:  # Debug first 3 entries
                        print(f"    Entry {i}: keys = {list(entry.keys())}")

                    image_id = entry.get("image_id")
                    if image_id:
                        if image_id not in results[input_fmt]:
                            results[input_fmt][image_id] = {}

                        # Look for the input format block (e.g., "markdown", "latex", "html")
                        input_format_block = entry.get(input_fmt)
                        if isinstance(input_format_block, dict):
                            if i < 3:  # Debug first 3 entries
                                print(
                                    f"    Input format block keys: {list(input_format_block.keys())}"
                                )

                            # Within the input format block, look for output format blocks
                            for output_fmt in FORMATS:
                                if (
                                    output_fmt != input_fmt
                                ):  # Skip same format (no cross-conversion)
                                    output_block = input_format_block.get(output_fmt)
                                    if isinstance(output_block, dict):
                                        gen_content = output_block.get("generated")
                                        if gen_content and gen_content != "ERROR":
                                            results[input_fmt][image_id][output_fmt] = (
                                                clean_generated_content(gen_content)
                                            )
                                            if i < 3:  # Debug first 3 entries
                                                print(
                                                    f"    Successfully added {output_fmt} for image {image_id} (from {input_fmt})"
                                                )
                                        else:
                                            if i < 3:  # Debug first 3 entries
                                                print(
                                                    f"    Skipping {output_fmt} for image {image_id}: generated = {gen_content}"
                                                )
                                    else:
                                        if i < 3:  # Debug first 3 entries
                                            print(
                                                f"    No {output_fmt} block found in {input_fmt} for image {image_id}"
                                            )
                        else:
                            if i < 3:  # Debug first 3 entries
                                print(
                                    f"    No {input_fmt} block found for image {image_id}"
                                )

            except Exception as e:
                print(f"  Error loading {input_fmt}.json: {e}")
                continue
        else:
            print(f"  {json_file} not found")

    formats_found = []
    for input_fmt in results:
        for img_id, img_formats in results[input_fmt].items():
            for fmt in img_formats.keys():
                conversion = f"{input_fmt}->{fmt}"
                if conversion not in formats_found:
                    formats_found.append(conversion)

    print(f"Loaded cross-format results: {formats_found}")
    return results


# ============================== TEDS CORE ==============================
class TableTree(Tree):
    """Minimal node for APTED."""

    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content  # list of tokens for 'td' (word-level)
        self.children = list(children)

    def bracket(self):
        # Optional debug representation
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                self.tag,
                self.colspan,
                self.rowspan,
                self.content,
            )
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


# Tokenization & light normalization
_TOKEN_RE = re.compile(
    r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"  # numbers like 95,644 or -12.5
    r"|(?:\d{1,4}[./-]\d{1,2}[./-]\d{1,4})"  # dates like 2012-10-31 or 31/10/2012
    r"|\w+"  # words (unicode)
    r"|[^\w\s]",  # punctuation as separate tokens
    re.UNICODE,
)
_NUM_PAT = re.compile(r"^[+-]?[\d,]+(?:\.\d+)?$")
_DATE_SEP_PAT = re.compile(r"^\d{1,4}([./-])\d{1,2}\1\d{1,4}$")
_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y.%m.%d",
    "%d.%m.%Y",
    "%m.%d.%Y",
]


def tokenize_text(s: str) -> List[str]:
    """Word-level tokens: words/numbers & punctuation as separate tokens."""
    if not s:
        return []
    return _TOKEN_RE.findall(s)


def _norm_number(tok: str) -> str:
    raw = tok.replace(",", "")
    try:
        d = Decimal(raw)
        return format(d.normalize(), "f")  # drop trailing zeros, avoid sci
    except InvalidOperation:
        return tok


def _norm_date_token(tok: str) -> str:
    if not _DATE_SEP_PAT.match(tok):
        return tok
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(tok, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return tok


def _norm_token(tok: str) -> str:
    t = tok.lower().strip()
    if _NUM_PAT.match(t):
        return _norm_number(t)
    if _DATE_SEP_PAT.match(t):
        return _norm_date_token(t)
    return t


class CustomConfig(Config):
    def rename(self, n1, n2):
        # structural mismatch = full cost
        if (
            (n1.tag != n2.tag)
            or (n1.colspan != n2.colspan)
            or (n1.rowspan != n2.rowspan)
        ):
            return 1.0
        # cells: exact word-level equality after light normalization
        if n1.tag == "td":
            a = [_norm_token(t) for t in (n1.content or [])]
            b = [_norm_token(t) for t in (n2.content or [])]
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
    ed = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
    return 1.0 - (float(ed) / n_nodes)


# ============================== HTML ADAPTER ==============================
def parse_html_table_to_tree(
    html_doc: str,
) -> TableTree:
    def _norm(s: str) -> str:
        if not s:
            return ""
        s = s.replace("\u00a0", " ")
        s = s.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
    doc = html.fromstring(html_doc, parser=parser)
    tables = doc.xpath("body//table") or doc.xpath(".//table")
    if not tables:
        return TableTree("table", None, None, None, *deque())

    chosen = tables[0]  # one table per entry

    # Build tree
    tt = TableTree("table", None, None, None, *deque())
    tbody = TableTree("tbody", None, None, None, *deque())
    tt.children.append(tbody)

    # headers: include, but ignore header text (structure-only)
    for tr_el in chosen.xpath(".//thead/tr"):
        tr = TableTree("tr", None, None, None, *deque())
        tbody.children.append(tr)
        for cell in tr_el.xpath("./th|./td"):
            colspan = int(cell.attrib.get("colspan", "1"))
            rowspan = int(cell.attrib.get("rowspan", "1"))
            tr.children.append(TableTree("td", colspan, rowspan, [], *deque()))

    # body: include all cells; keep index; content tokenized
    for tr_el in chosen.xpath(".//tbody/tr") or chosen.xpath(".//tr"):
        tr = TableTree("tr", None, None, None, *deque())
        tbody.children.append(tr)
        for cell in tr_el.xpath("./th|./td"):
            colspan = int(cell.attrib.get("colspan", "1"))
            rowspan = int(cell.attrib.get("rowspan", "1"))
            txt = re.sub(
                r"\s+", " ", "".join(cell.itertext()).replace("\u00a0", " ")
            ).strip()
            tokens = tokenize_text(txt)
            tr.children.append(TableTree("td", colspan, rowspan, tokens, *deque()))
    return tt


# ============================== MARKDOWN ADAPTER ==============================
_MD_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _md_norm_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ")
    s = s.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


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


def parse_md_table(
    md: str,  # "none"|"structure"|"text"
) -> TableTree:
    if not md:
        return TableTree("table", None, None, None, *deque())

    blocks = _extract_md_tables(md)
    if not blocks:
        return TableTree("table", None, None, None, *deque())
    _, _, block = blocks[0]  # first valid pipe table

    # find header separator
    sep_idx = None
    for k, ln in enumerate(block):
        if _MD_SEP_RE.match(ln):
            sep_idx = k
            break
    if sep_idx is None or sep_idx == 0:
        return TableTree("table", None, None, None, *deque())

    header_cells = _md_split_row(block[sep_idx - 1])
    data_rows = [_md_split_row(ln) for ln in block[sep_idx + 1 :] if "|" in ln]

    # ... find first pipe table & split header/data ...
    table = TableTree("table", None, None, None, *deque())
    tbody = TableTree("tbody", None, None, None, *deque())
    table.children.append(tbody)

    # header row as structure-only
    tr = TableTree("tr", None, None, None, *deque())
    tbody.children.append(tr)
    for cell in header_cells:
        tr.children.append(TableTree("td", 1, 1, [], *deque()))

    # body rows (no merges)
    for r in data_rows:
        tr = TableTree("tr", None, None, None, *deque())
        tbody.children.append(tr)
        for cell in r:
            text = _md_norm_text(cell)
            tr.children.append(TableTree("td", 1, 1, tokenize_text(text), *deque()))
    return table


# ============================== LaTeX ADAPTER ==============================
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


def _tex_norm_text(s: str) -> str:
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
    s = s.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
    s = s.replace("–", "-").replace("—", "-")
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

        # place active multirows carried from previous row
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
            # skip through already occupied columns by multirow
            while col in active_multi:
                remaining, txt = active_multi[col]
                laid.append(("", 1, 1, txt, True))
                remaining -= 1
                if remaining <= 0:
                    del active_multi[col]
                else:
                    active_multi[col] = (remaining, txt)
                col += 1

            txt_norm = _tex_norm_text(text)
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
    tex: str,
    prefer_tabular_inside_table_env: bool = True,
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

    header = rows_with_spans[0] if rows_with_spans else []
    data_rows = rows_with_spans[1:] if len(rows_with_spans) > 1 else []

    table = TableTree("table", None, None, None, *deque())
    tbody = TableTree("tbody", None, None, None, *deque())
    table.children.append(tbody)

    # header (structure-only)
    tr = TableTree("tr", None, None, None, *deque())
    tbody.children.append(tr)
    for txt, cspan, rspan in header:
        tr.children.append(TableTree("td", int(cspan), int(rspan), [], *deque()))

    # body
    for row in data_rows:
        tr = TableTree("tr", None, None, None, *deque())
        tbody.children.append(tr)
        for txt, cspan, rspan in row:
            tr.children.append(
                TableTree("td", int(cspan), int(rspan), tokenize_text(txt), *deque())
            )
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


def validate_parser_output(
    pred_content: str, true_content: str, fmt: str
) -> Dict[str, bool]:
    """Debug helper to identify parser failures."""

    def has_table_markers(content: str, format_type: str) -> bool:
        if not content:
            return False
        content_lower = content.lower()
        if format_type == "html":
            return "<table" in content_lower
        elif format_type == "markdown":
            return "|" in content and content.count("|") >= 2
        elif format_type == "latex":
            return "\\begin{tabular}" in content or "\\begin{table}" in content
        return False

    pred_has_table = has_table_markers(pred_content, fmt)
    true_has_table = has_table_markers(true_content, fmt)

    try:
        pred_tree, _ = parse_to_TableTree(pred_content, true_content, fmt)
        pred_parsed = _count_nodes(pred_tree) > 1
    except:
        pred_parsed = False

    try:
        _, true_tree = parse_to_TableTree(pred_content, true_content, fmt)
        true_parsed = _count_nodes(true_tree) > 1
    except:
        true_parsed = False

    return {
        "pred_detected": pred_has_table,
        "true_detected": true_has_table,
        "pred_parsed": pred_parsed,
        "true_parsed": true_parsed,
        "parsing_reasonable": pred_parsed and true_parsed,
    }


def teds_similarity_generic(pred: str, true: str, fmt: str) -> float:
    """Enhanced TEDS with validation diagnostics."""
    validation = validate_parser_output(pred, true, fmt)

    if not validation["parsing_reasonable"]:
        print(
            f"Warning: Parser issues detected for {fmt} - pred_parsed: {validation['pred_parsed']}, true_parsed: {validation['true_parsed']}"
        )

    try:
        tree_pred, tree_true = parse_to_TableTree(pred, true, fmt)
        return teds_similarity_from_trees(tree_pred, tree_true)
    except Exception as e:
        print(f"Error in TEDS computation for {fmt}: {e}")
        return 0.0


# ============================== EVALUATION ==============================
def evaluate_model_teds_generic(
    generation_path: str,
    hf_token: str,
    fmt: str,
    pipeline_type: str = "generation",
    diag_first_k: int = 5,
) -> Dict[str, Dict[str, float]]:
    if pipeline_type == "generation_format":
        # New pipeline: load format-specific files
        generated_results_by_format = load_generated_results_by_format(generation_path)
        results: Dict[str, Dict[str, float]] = {}

        # Process each input format -> output format combination
        for input_fmt in FORMATS:
            for output_fmt in FORMATS:
                if input_fmt == output_fmt:
                    continue  # Skip same format conversions

                if (
                    fmt == output_fmt
                ):  # Only evaluate when the current format matches the output format
                    print(f"\n--- Processing {input_fmt} -> {output_fmt} ---")

                    for dataset_name in DATASETS:
                        print(f"  Dataset: {dataset_name}")
                        dataset_image_ids = get_dataset_image_ids(
                            hf_token, dataset_name, pipeline_type
                        )
                        print(
                            f"  Found {len(dataset_image_ids)} image IDs for {dataset_name}"
                        )

                        # Filter images that have this conversion and are in this dataset
                        common_images = []
                        if input_fmt in generated_results_by_format:
                            for img_id in dataset_image_ids:
                                if (
                                    img_id in generated_results_by_format[input_fmt]
                                    and output_fmt
                                    in generated_results_by_format[input_fmt][img_id]
                                ):
                                    common_images.append(img_id)

                        print(
                            f"    Found {len(common_images)} common images to evaluate"
                        )

                        sims = []
                        processed = 0
                        for i, image_id in enumerate(common_images):
                            if i and (i % 100 == 0):
                                print(
                                    f"    Processed {i}/{len(common_images)} images..."
                                )

                            # Ground truth is in the OUTPUT format (what we're trying to generate)
                            gt_content = read_ground_truth_from_local(
                                LOCAL_GT_DIR, dataset_name, output_fmt, image_id
                            )
                            if gt_content is None:
                                continue

                            # Generated content is also in the OUTPUT format
                            pred_content = generated_results_by_format[input_fmt][
                                image_id
                            ][output_fmt]

                            if (
                                not has_table_substring(gt_content, output_fmt)
                                and processed < diag_first_k
                            ):
                                print(
                                    f"[diag] GT not table-ish: {dataset_name}/{image_id} :: {repr(gt_content[:160])}"
                                )
                            if (
                                not has_table_substring(pred_content, output_fmt)
                                and processed < diag_first_k
                            ):
                                print(
                                    f"[diag] Pred not table-ish: {dataset_name}/{image_id} :: {repr(pred_content[:160])}"
                                )

                            try:
                                sim = teds_similarity_generic(
                                    pred_content, gt_content, fmt=output_fmt
                                )
                                sims.append(sim)
                                processed += 1
                                if processed <= diag_first_k:
                                    print(
                                        f"    [{processed}/{len(common_images)}] {image_id}: TEDS={sim:.4f}"
                                    )
                            except Exception as e:
                                print(
                                    f"    [{processed+1}/{len(common_images)}] {image_id}: ERROR - {type(e).__name__}: {e}"
                                )
                                sims.append(0.0)
                                processed += 1

                        print(
                            f"    Successfully processed {processed} images for {dataset_name}"
                        )

                        # Create key in the format: dataset-inputformat-outputformat
                        key = f"{dataset_name}-{input_fmt}-{output_fmt}"
                        if processed > 0:
                            avg_sim = sum(sims) / len(sims)
                            results[key] = {"teds": avg_sim, "count": processed}
                            print(
                                f"    Average TEDS ({input_fmt}->{output_fmt}): {avg_sim:.3f}"
                            )
                        else:
                            results[key] = {"teds": 0.0, "count": 0}

        return results

    else:
        # Original generation pipeline
        preds = load_generated_results_by_dataset(generation_path, fmt)
        results: Dict[str, Dict[str, float]] = {}

        for dataset_name in DATASETS:
            print(f"\n--- {dataset_name} [{fmt}] ---")
            image_ids = get_dataset_image_ids(hf_token, dataset_name, pipeline_type)
            sims = []
            if dataset_name not in preds:
                print(f"    No predictions for {dataset_name} [{fmt}]")
                results[dataset_name] = {"teds": 0.0, "count": 0}
                continue

            common = [iid for iid in image_ids if iid in preds[dataset_name]]
            print(f"    {len(common)} common image IDs")
            processed = 0
            for i, image_id in enumerate(common):
                gt_text = read_ground_truth_from_local(
                    LOCAL_GT_DIR, dataset_name, fmt, image_id
                )
                if gt_text is None:
                    print(f"    [{i+1}/{len(common)}] {image_id}: SKIPPED (no GT)")
                    continue
                pred_text = preds[dataset_name][image_id]

                if not has_table_substring(gt_text, fmt) and processed < diag_first_k:
                    print(
                        f"[diag] GT not table-ish: {dataset_name}/{image_id} :: {repr(gt_text[:160])}"
                    )
                if not has_table_substring(pred_text, fmt) and processed < diag_first_k:
                    print(
                        f"[diag] Pred not table-ish: {dataset_name}/{image_id} :: {repr(pred_text[:160])}"
                    )

                try:
                    sim = teds_similarity_generic(pred_text, gt_text, fmt=fmt)
                    sims.append(sim)
                    processed += 1
                    print(f"    [{processed}/{len(common)}] {image_id}: TEDS={sim:.4f}")
                except Exception as e:
                    print(
                        f"    [{processed+1}/{len(common)}] {image_id}: ERROR - {type(e).__name__}: {e}"
                    )
                    sims.append(0.0)
                    processed += 1

            avg_sim = (sum(sims) / len(sims)) if sims else 0.0
            results[dataset_name] = {"teds": avg_sim, "count": processed}
            print(f"    Average [{fmt}]: {avg_sim:.3f}")

        return results


def evaluate_all_formats(
    generation_path: str,
    hf_token: str,
    pipeline_type: str = "generation",
    diag_first_k: int = 5,
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
        # Create rows for cross-format evaluation
        for fmt, per_ds in results_by_fmt.items():
            for key, m in per_ds.items():
                # Parse key format: dataset-inputformat-outputformat
                parts = key.split("-")
                if len(parts) >= 3:
                    dataset_name = parts[0]
                    input_fmt = parts[1]
                    output_fmt = parts[2]
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
        # Original format for regular generation pipeline
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
    out_csv = os.path.join(generation_path, f"{model_name}_teds_evaluation_scores.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved TEDS results for {model_name} to {out_csv}")
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
        "--diag_first_k", type=int, default=5, help="Diagnostics for first K items"
    )
    args = ap.parse_args()

    print("Ground-truth root:", LOCAL_GT_DIR)
    print("Will evaluate formats:", FORMATS)
    print(f"Pipeline type: {args.pipeline_type}")

    all_results = {}
    csv_files = []

    for gen_path in args.generation_paths:
        model_name = os.path.basename(os.path.dirname(gen_path)) or os.path.basename(
            gen_path.rstrip("/")
        )
        print(f"\n{'='*60}\nEvaluating {model_name} (all formats)…\n{'='*60}")
        try:
            res_by_fmt = evaluate_all_formats(
                gen_path, args.hf_token, args.pipeline_type, args.diag_first_k
            )
            all_results[model_name] = res_by_fmt
            csv_file = save_multi_format_csv(
                model_name, res_by_fmt, gen_path, args.pipeline_type
            )
            csv_files.append(csv_file)
            print(f"✓ Completed evaluation for {model_name}")
        except Exception as e:
            print(f"✗ Error evaluating {model_name}: {e}")

    # Summary section
    print("\n" + "=" * 80)
    print("TEDS EVALUATION SUMMARY")
    print("=" * 80)

    for model_name, model_results in all_results.items():
        print(f"\n{model_name}:")
        if args.pipeline_type == "generation_format":
            print(
                f"{'Dataset':<10} {'Image_Fmt':<10} {'Struct_Fmt':<10} {'TEDS':<10} {'Count':<6}"
            )
            print("-" * 60)
            for fmt, per_ds in model_results.items():
                for key, m in per_ds.items():
                    # Parse key format: dataset-inputformat-outputformat
                    parts = key.split("-")
                    if len(parts) >= 3:
                        dataset_name = parts[0]
                        input_fmt = parts[1]
                        output_fmt = parts[2]
                        print(
                            f"{dataset_name:<10} {input_fmt:<10} {output_fmt:<10} "
                            f"{m['teds']:<10.3f} {m['count']:<6}"
                        )
        else:
            print(f"{'Dataset':<10} {'Format':<10} {'TEDS':<10} {'Count':<6}")
            print("-" * 40)
            for fmt, per_ds in model_results.items():
                for dataset_name, m in per_ds.items():
                    print(
                        f"{dataset_name:<10} {fmt:<10} {m['teds']:<10.3f} {m['count']:<6}"
                    )

    print("\nCSV files saved:")
    for f in csv_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
