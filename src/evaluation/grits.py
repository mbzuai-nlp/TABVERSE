import os
import re
import json
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from bs4 import BeautifulSoup
import itertools
import warnings

warnings.filterwarnings("ignore")

FORMATS   = ["html", "markdown", "latex"]
DATASETS  = ["wikitq", "hybridqa", "sqa", "tabfact", "feverous"]
EXTENSIONS = {"html": ".html", "markdown": ".md", "latex": ".tex"}
FMT_SHORT  = {"html": "html", "markdown": "md", "latex": "ltx"}
GAP_1D, GAP_OUT = 0.2, 0.2

# ── Path constants ─────────────────────────────────────────────────────────────
GT_DIR = os.path.join("data", "4-representations")

PIPELINE_CONFIGS = {
    "vlm": {
        "label":       "VLM",
        "results_dir": "results/vlmpipeline",
        "gen_file":    "inter_intra_generation.json",
    },
}

FINAL_CSV_DIR  = os.path.join("results", "final_csv", "generation")
SUMMARY_CSV    = os.path.join(FINAL_CSV_DIR, "generation.csv")
SUMMARY_HEADER = [
    "pipeline", "model", "dataset", "input_format", "output_format",
    "grits_top", "grits_con",
    "grits_precision_top", "grits_recall_top",
    "grits_precision_con", "grits_recall_con",
    "count",
]

PIPELINE_CSV = {
    "vlm": os.path.join(FINAL_CSV_DIR, "generation_vlm.csv"),
}

# Wide CSV: one row per model; columns = all 9 format-pair combinations × 2 metrics
_PAIRS       = [(i, o) for i in FORMATS for o in FORMATS]
_PAIR_LABELS = [f"{FMT_SHORT[i]}->{FMT_SHORT[o]}" for i, o in _PAIRS]
WIDE_HEADER  = (
    ["model"]
    + [f"{p}_top" for p in _PAIR_LABELS]
    + [f"{p}_con" for p in _PAIR_LABELS]
    + ["OVERALL_top", "OVERALL_con"]
)


def clean_generated_content(content: str) -> str:
    if not content:
        return ""
    content = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*\n", "", content)
    return re.sub(r"\n```(\s*)$", "", content).strip()


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()) if s else ""


def tokenize_words(s: str) -> List[str]:
    return re.findall(r"[a-z0-9%+\-/.]+", s)


def token_exact_match_similarity(a: str, b: str) -> float:
    ta, tb = tokenize_words(normalize_text(a)), tokenize_words(normalize_text(b))
    return 1.0 if ta == tb else 0.0


def iou_relspan(a, b) -> float:
    if (
        not (hasattr(a, "__len__") and hasattr(b, "__len__"))
        or len(a) != 4
        or len(b) != 4
    ):
        return 0.0
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    A1, A2 = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return inter / (A1 + A2 - inter) if A1 + A2 - inter > 0 else 0.0


# ── Table parsers ──────────────────────────────────────────────────────────────

def html_to_cells(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    rows, occupied, cells = table.find_all("tr"), defaultdict(set), []
    for r, tr in enumerate(rows):
        c = 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while c in occupied[r]:
                c += 1
            rs, cs = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
            txt = cell.get_text(" ", strip=True) if cell.text is not None else ""
            row_nums, col_nums = list(range(r, r + rs)), list(range(c, c + cs))
            for rr in row_nums:
                occupied[rr].update(col_nums)
            cells.append(
                {
                    "row_nums": row_nums,
                    "column_nums": col_nums,
                    "is_column_header": cell.name == "th"
                    or tr.find_parent("thead") is not None,
                    "cell_text": txt,
                }
            )
            c += cs
    return cells


def markdown_to_cells(md: str) -> List[Dict]:
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

    split = [[p.strip() for p in ln.strip("|").split("|")] for ln in rows]
    if not split:
        return []

    C = max(len(r) for r in split)
    cells = []
    for r, row in enumerate(split):
        row = row + [""] * (C - len(row))
        for k, txt in enumerate(row):
            cells.append(
                {
                    "row_nums": [r],
                    "column_nums": [k],
                    "is_column_header": r == 0,
                    "cell_text": txt,
                }
            )
    return cells


def latex_to_cells(tex: str) -> List[Dict]:
    m = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", tex, re.S)
    if not m:
        return []

    body = re.sub(r"\\(toprule|midrule|bottomrule|hline)", "", m.group(1))
    raw_rows = [ln.strip() for ln in body.split("\\\\") if ln.strip()]

    def parse_spans(cell: str) -> Tuple[str, int, int]:
        txt, cs, rs = cell.strip(), 1, 1
        mc_match = re.match(r"^\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$", txt, re.S)
        if mc_match:
            cs, txt = int(mc_match.group(1)), mc_match.group(2).strip()
        mr_match = re.match(r"^\\multirow\{(\d+)\}\{[^}]*\}\{(.*)\}$", txt, re.S)
        if mr_match:
            rs, txt = int(mr_match.group(1)), mr_match.group(2).strip()
        txt = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", txt)
        return re.sub(r"\\[a-zA-Z]+", "", txt).strip(), cs, rs

    occupied, cells = defaultdict(set), []
    SAFE_AMP = "__LATEX_ESCAPED_AMP__"

    for r, row_text in enumerate(raw_rows):
        row_safe = row_text.replace(r"\&", SAFE_AMP)
        cols, c = [c.strip() for c in row_safe.split("&")], 0
        for col_text in cols:
            while c in occupied[r]:
                c += 1
            txt, cs, rs = parse_spans(col_text)
            txt = txt.replace(SAFE_AMP, "&")
            row_nums, col_nums = list(range(r, r + rs)), list(range(c, c + cs))
            for rr in row_nums:
                for cc in col_nums:
                    occupied[rr].add(cc)
            cells.append(
                {
                    "row_nums": row_nums,
                    "column_nums": col_nums,
                    "is_column_header": r == 0,
                    "cell_text": txt,
                }
            )
            c += cs
    return cells


def content_to_cells(content: str, fmt: str) -> List[Dict]:
    parsers = {
        "html": html_to_cells,
        "markdown": markdown_to_cells,
        "latex": latex_to_cells,
    }
    return parsers.get(fmt, lambda x: [])(content)


def cells_to_grid(cells: List[Dict], key: str = "cell_text") -> np.ndarray:
    if not cells:
        return np.zeros((0, 0), dtype=object)
    R = max(max(c["row_nums"]) for c in cells) + 1
    C = max(max(c["column_nums"]) for c in cells) + 1
    grid = [["" for _ in range(C)] for _ in range(R)]

    for c in cells:
        for r in c["row_nums"]:
            for k in c["column_nums"]:
                if key == "relspan":
                    r0, r1 = min(c["row_nums"]), max(c["row_nums"]) + 1
                    c0, c1 = min(c["column_nums"]), max(c["column_nums"]) + 1
                    grid[r][k] = [c0 - k, r0 - r, c1 - k, r1 - r]
                else:
                    grid[r][k] = c.get(key, "")
    return np.array(grid, dtype=object)


def compute_fscore(
    tp: float, num_true: int, num_pos: int
) -> Tuple[float, float, float]:
    precision = (tp / num_pos) if num_pos > 0 else 1.0
    recall = (tp / num_true) if num_true > 0 else 1.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return f1, precision, recall


def _init_dp(n1: int, n2: int):
    scores = np.zeros((n1 + 1, n2 + 1), dtype=float)
    ptrs = np.zeros((n1 + 1, n2 + 1), dtype=int)
    ptrs[1:, 0], ptrs[0, 1:] = -1, 1
    return scores, ptrs


def _trace(ptrs: np.ndarray):
    i, j, s1, s2 = ptrs.shape[0] - 1, ptrs.shape[1] - 1, [], []
    while not (i == 0 and j == 0):
        p = ptrs[i, j]
        if p == 0:
            i -= 1
            j -= 1
            s1.append(i)
            s2.append(j)
        elif p == -1:
            i -= 1
        else:
            j -= 1
    s1.reverse()
    s2.reverse()
    return s1, s2


def align_1d(
    seq1: List[Any],
    seq2: List[Any],
    reward_lu: Dict[Tuple, float],
    return_alignment=False,
):
    n1, n2 = len(seq1), len(seq2)
    scores, ptrs = _init_dp(n1, n2)
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            r = reward_lu[seq1[i - 1] + seq2[j - 1]]
            d, up, lf = (
                scores[i - 1, j - 1] + r,
                scores[i - 1, j] - GAP_1D,
                scores[i, j - 1] - GAP_1D,
            )
            m = max(d, up, lf)
            scores[i, j] = m
            ptrs[i, j] = 0 if d == m else (-1 if up == m else 1)
    score = scores[-1, -1]
    if not return_alignment:
        return score
    a1, a2 = _trace(ptrs)
    return a1, a2, score


def align_2d_outer(
    true_shape: Tuple[int, int],
    pred_shape: Tuple[int, int],
    reward_lu: Dict[Tuple, float],
):
    R1, C1, R2, C2 = true_shape[0], true_shape[1], pred_shape[0], pred_shape[1]
    scores, ptrs = _init_dp(R1, R2)
    for r1 in range(1, R1 + 1):
        for r2 in range(1, R2 + 1):
            seq1 = [(r1 - 1, c) for c in range(C1)]
            seq2 = [(r2 - 1, c) for c in range(C2)]
            rw = align_1d(seq1, seq2, reward_lu)
            d, up, lf = (
                scores[r1 - 1, r2 - 1] + rw,
                scores[r1 - 1, r2] - GAP_OUT,
                scores[r1, r2 - 1] - GAP_OUT,
            )
            m = max(d, up, lf)
            scores[r1, r2] = m
            ptrs[r1, r2] = 0 if d == m else (-1 if up == m else 1)
    a_r1, a_r2 = _trace(ptrs)
    return a_r1, a_r2, scores[-1, -1]


def factored_2dmss(A: np.ndarray, B: np.ndarray, reward_fn):
    pre, trn = {}, {}
    R1, C1, R2, C2 = A.shape[0], A.shape[1], B.shape[0], B.shape[1]
    for trow, tcol, prow, pcol in itertools.product(
        range(R1), range(C1), range(R2), range(C2)
    ):
        r = reward_fn(A[trow, tcol], B[prow, pcol])
        pre[(trow, tcol, prow, pcol)] = r
        trn[(tcol, trow, pcol, prow)] = r

    num_true, num_pos = R1 * C1, R2 * C2
    tru_rows, pr_rows, row_score = align_2d_outer((R1, C1), (R2, C2), pre)
    tru_cols, pr_cols, col_score = align_2d_outer((C1, R1), (C2, R2), trn)
    ub_score = min(row_score, col_score)
    ub_f1, _, _ = compute_fscore(ub_score, num_true, num_pos)

    pos_score = sum(
        pre[(r1, c1, r2, c2)]
        for r1, r2 in zip(tru_rows, pr_rows)
        for c1, c2 in zip(tru_cols, pr_cols)
    )
    f1, p, r = compute_fscore(pos_score, num_true, num_pos)
    return f1, p, r, ub_f1


def grits_from_content(
    true_content: str, pred_content: str, fmt: str
) -> Dict[str, float]:
    gt_cells, pr_cells = (
        content_to_cells(true_content, fmt) or [],
        content_to_cells(pred_content, fmt) or [],
    )
    if not gt_cells or not pr_cells:
        return {
            f"grits_{k}": 0.0
            for k in [
                "top",
                "con",
                "precision_top",
                "recall_top",
                "precision_con",
                "recall_con",
                "top_upper_bound",
                "con_upper_bound",
            ]
        }

    T_top, P_top = cells_to_grid(gt_cells, "relspan"), cells_to_grid(
        pr_cells, "relspan"
    )
    top_f1, top_p, top_r, top_ub = factored_2dmss(T_top, P_top, iou_relspan)
    T_con, P_con = cells_to_grid(gt_cells, "cell_text"), cells_to_grid(
        pr_cells, "cell_text"
    )
    con_f1, con_p, con_r, con_ub = factored_2dmss(
        T_con, P_con, token_exact_match_similarity
    )

    return {
        "grits_top": top_f1,
        "grits_precision_top": top_p,
        "grits_recall_top": top_r,
        "grits_top_upper_bound": top_ub,
        "grits_con": con_f1,
        "grits_precision_con": con_p,
        "grits_recall_con": con_r,
        "grits_con_upper_bound": con_ub,
    }


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_generation_file(results_dir: str, model_name: str) -> Optional[List]:
    path = os.path.join(results_dir, model_name, "inter_intra_generation.json")
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to parse {path}: {e}")
            return None


def load_ground_truth(output_fmt: str, image_id: str) -> Optional[str]:
    path = os.path.join(GT_DIR, output_fmt, f"{image_id}{EXTENSIONS[output_fmt]}")
    if not os.path.exists(path):
        return None
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return None


def open_csv_append(path, header):
    """Open CSV in append mode; write header only if the file is new/empty."""
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(header)
    return f, writer


def discover_models(pipeline_key: str) -> List[str]:
    cfg = PIPELINE_CONFIGS[pipeline_key]
    results_dir = cfg["results_dir"]
    gen_file    = cfg["gen_file"]
    if not os.path.isdir(results_dir):
        return []
    return sorted(
        m for m in os.listdir(results_dir)
        if os.path.exists(os.path.join(results_dir, m, gen_file))
    )


# ── Core evaluation ─────────────────────────────────────────────────────────

def _score_entry(entry: Dict) -> List[Tuple]:
    """Top-level worker: scores one JSON entry across all 9 format pairs.
    Returns a list of (dataset, in_fmt, out_fmt, metric, value) tuples.
    Must be top-level (not nested) to be picklable by multiprocessing.
    """
    results = []
    image_id = entry.get("image_id")
    dataset  = entry.get("dataset", "unknown")
    for in_fmt in FORMATS:
        in_block = entry.get(in_fmt)
        if not isinstance(in_block, dict):
            continue
        gen_block = in_block.get("generation")
        if not isinstance(gen_block, dict):
            continue
        for out_fmt in FORMATS:
            raw_pred = gen_block.get(out_fmt)
            if not raw_pred or raw_pred == "ERROR":
                continue
            gt = load_ground_truth(out_fmt, image_id)
            if gt is None:
                continue
            pred = clean_generated_content(raw_pred)
            try:
                m = grits_from_content(gt, pred, out_fmt)
            except Exception:
                m = {k: 0.0 for k in [
                    "grits_top", "grits_precision_top", "grits_recall_top",
                    "grits_top_upper_bound",
                    "grits_con", "grits_precision_con", "grits_recall_con",
                    "grits_con_upper_bound",
                ]}
            for metric, value in m.items():
                results.append((dataset, in_fmt, out_fmt, metric, value))
    return results


def evaluate_model_data(rows: List[Dict], n_workers: int = 32) -> Dict:
    """Return per-(dataset, input_fmt, output_fmt) score lists and counts."""
    scores: Dict = defaultdict(lambda: defaultdict(list))
    counts: Dict = defaultdict(int)
    total = len(rows)
    done  = 0
    t0    = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_score_entry, row): i for i, row in enumerate(rows)}
        for fut in as_completed(futures):
            done += 1
            # Collect all metric tuples for this entry, then count each
            # (dataset, in_fmt, out_fmt) key exactly once (not once per metric).
            entry_keys: set = set()
            for (dataset, in_fmt, out_fmt, metric, value) in fut.result():
                scores[(dataset, in_fmt, out_fmt)][metric].append(value)
                entry_keys.add((dataset, in_fmt, out_fmt))
            for key in entry_keys:
                counts[key] += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (total - done) / rate if rate > 0 else 0
                print(
                    f"  [entry {done:4d}/{total}]  "
                    f"pairs scored: {sum(counts.values())}  "
                    f"elapsed: {elapsed/60:.1f}m  "
                    f"ETA: {eta/60:.1f}m",
                    flush=True,
                )

    return {"scores": dict(scores), "counts": dict(counts)}


# ── Per-model runner ─────────────────────────────────────────────────────────

def run_one(pipeline_key: str, model_name: str):
    cfg   = PIPELINE_CONFIGS[pipeline_key]
    label = cfg["label"]
    rows  = load_generation_file(cfg["results_dir"], model_name)
    if rows is None:
        return

    print(f"\n{'=' * 65}")
    print(f"  {label} | {model_name}")
    print(f"{'=' * 65}")

    result = evaluate_model_data(rows)
    scores = result["scores"]
    counts = result["counts"]

    long_f, long_writer = open_csv_append(SUMMARY_CSV,              SUMMARY_HEADER)
    wide_f, wide_writer = open_csv_append(PIPELINE_CSV[pipeline_key], WIDE_HEADER)
    try:
        pair_top: Dict = {}
        pair_con: Dict = {}
        all_top:  List = []
        all_con:  List = []

        for in_fmt in FORMATS:
            for out_fmt in FORMATS:
                pair_all_top: List = []
                pair_all_con: List = []
                for dataset in DATASETS:
                    key = (dataset, in_fmt, out_fmt)
                    if key not in scores or counts.get(key, 0) == 0:
                        continue
                    vals = scores[key]
                    c    = counts[key]
                    top  = sum(vals["grits_top"]) / c
                    con  = sum(vals["grits_con"]) / c
                    pt   = sum(vals["grits_precision_top"]) / c
                    rt   = sum(vals["grits_recall_top"]) / c
                    pc   = sum(vals["grits_precision_con"]) / c
                    rc   = sum(vals["grits_recall_con"]) / c
                    long_writer.writerow([
                        label, model_name, dataset, in_fmt, out_fmt,
                        round(top, 4), round(con, 4),
                        round(pt, 4), round(rt, 4),
                        round(pc, 4), round(rc, 4),
                        c,
                    ])
                    pair_all_top.extend(vals["grits_top"])
                    pair_all_con.extend(vals["grits_con"])

                lbl = f"{FMT_SHORT[in_fmt]}->{FMT_SHORT[out_fmt]}"
                if pair_all_top:
                    avg_top = sum(pair_all_top) / len(pair_all_top)
                    avg_con = sum(pair_all_con) / len(pair_all_con)
                    pair_top[lbl] = round(avg_top, 4)
                    pair_con[lbl] = round(avg_con, 4)
                    all_top.extend(pair_all_top)
                    all_con.extend(pair_all_con)
                    print(f"  {lbl:12s}  top={avg_top:.4f}  con={avg_con:.4f}  n={len(pair_all_top)}")
                else:
                    pair_top[lbl] = ""
                    pair_con[lbl] = ""

        overall_top = round(sum(all_top) / len(all_top), 4) if all_top else ""
        overall_con = round(sum(all_con) / len(all_con), 4) if all_con else ""
        wide_row = (
            [model_name]
            + [pair_top.get(p, "") for p in _PAIR_LABELS]
            + [pair_con.get(p, "") for p in _PAIR_LABELS]
            + [overall_top, overall_con]
        )
        wide_writer.writerow(wide_row)
        long_f.flush()
        wide_f.flush()
    finally:
        long_f.close()
        wide_f.close()

    print(f"\n  -> Appended to {SUMMARY_CSV}")
    print(f"  -> Appended to {PIPELINE_CSV[pipeline_key]}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generation Evaluation — GriTS_Top and GriTS_Con per format pair."
    )
    parser.add_argument(
        "--pipeline", choices=list(PIPELINE_CONFIGS), default=None,
        help="Pipeline type (omit to run all pipelines)",
    )
    parser.add_argument(
        "--model_name", default=None,
        help="Model name folder (omit to run all models in the pipeline)",
    )
    args = parser.parse_args()

    os.makedirs(FINAL_CSV_DIR, exist_ok=True)

    pipelines = [args.pipeline] if args.pipeline else list(PIPELINE_CONFIGS.keys())
    for pipeline_key in pipelines:
        models = (
            [args.model_name] if args.model_name
            else discover_models(pipeline_key)
        )
        for model_name in models:
            run_one(pipeline_key, model_name)

    print(f"\nDone. CSV → {SUMMARY_CSV}")


if __name__ == "__main__":
    main()