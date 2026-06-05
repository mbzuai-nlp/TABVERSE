---
license: cc-by-4.0
task_categories:
  - table-question-answering
  - visual-question-answering
language:
  - en
tags:
  - table-understanding
  - multimodal
  - benchmark
  - vlm
  - llm
  - html
  - markdown
  - latex
size_categories:
  - n<1K
configs:
  - config_name: qa
    data_files:
      - split: test
        path: qa/test-*.parquet
    default: true
  - config_name: suc
    data_files:
      - split: test
        path: suc/test-*.parquet
---

<p align="center">
  <img src="https://huggingface.co/datasets/MBZUAI/TABVERSE/resolve/main/raw/assets/logo-light.png" alt="TABVERSE" width="280">
</p>

<h1 align="center">Benchmarking Cross-Format Table Understanding in LLMs and VLMs</h1>

<!-- <p align="center">
  <a href="https://mbzuai-nlp.github.io/TABVERSE/">🌐 Website</a> &nbsp;|&nbsp;
  <a href="https://github.com/mbzuai-nlp/TABVERSE">💻 GitHub</a>
</p> -->

<p align="center">
  <b>Momina Ahsan</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Sarfraz Ahmad</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Ming Shan Hee</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Roy Ka-Wei Lee</b><sup>2</sup> &nbsp;·&nbsp;
  <b>Preslav Nakov</b><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)
</p>
<p align="center">
  <sup>2</sup> Singapore University of Technology and Design (SUTD)
</p>

---

A controlled multimodal table benchmark that aligns **HTML, Markdown, and LaTeX** table
representations with rendered PNG images — enabling systematic evaluation of how format
and modality interact, with table content held fixed across all three views.

|                              |                                                         |
| ---------------------------- | ------------------------------------------------------- |
| Q–Table pairs (`qa` config)  | **700** (350 Easy · 350 Hard)                           |
| Unique tables (`suc` config) | **629**                                                 |
| Formats per table            | 3 (HTML · Markdown · LaTeX) + PNG rendering             |
| Tasks                        | SUC · QA (Task Prediction) · SR (Format Generation)     |
| Source datasets              | FEVEROUS · HybridQA · SQA · TabFact · ToTTo             |
| Models evaluated             | 17 (open-weight VLMs, open-weight LLMs, GPT-4o, Gemini) |

---

## Dataset configs

### `qa` — Task Prediction (700 rows)

One row per question–table pair. Each row contains the question, gold label, difficulty,
and category — plus the full table rendered in all three formats as images and source code.

| Column                | Type         | Description                                                         |
| --------------------- | ------------ | ------------------------------------------------------------------- |
| `id`                  | int          | Row index                                                           |
| `image_id`            | string       | Unique table identifier                                             |
| `html_image`          | Image        | PNG rendering of the HTML table                                     |
| `markdown_image`      | Image        | PNG rendering of the Markdown table                                 |
| `latex_image`         | Image        | PNG rendering of the LaTeX table                                    |
| `html_code`           | string       | Raw HTML source                                                     |
| `markdown_code`       | string       | Raw Markdown source                                                 |
| `latex_code`          | string       | Raw LaTeX source                                                    |
| `table`               | string       | JSON-encoded `{header, rows}`                                       |
| `query`               | string       | Natural-language question                                           |
| `label`               | list[string] | Gold answer(s)                                                      |
| `question_category`   | string       | One of 7 reasoning categories (see below)                           |
| `question_difficulty` | string       | `Easy` or `Hard`                                                    |
| `dataset`             | string       | Source dataset (`wikitq`, `feverous`, `sqa`, `hybridqa`, `tabfact`) |
| `score`               | int          | Annotated complexity score                                          |

**Question categories:** Simple Lookup · Conditional Lookup · Multi-Item Lookup ·
Single-step Binary Verification · Multi-hop Binary Verification ·
Comparison & Extremum · Aggregation / Counting / Arithmetic

### `suc` — Structured Understanding & Comprehension / Format Generation (629 rows)

One row per unique table. SUC fields supply ground-truth answers for structural
probing tasks. Because all three format code strings are present, the `suc` config
also covers **Structure Reconstruction** (SR) — any format can serve as input and
any other as the generation target.

| Column                   | Type   | Description                              |
| ------------------------ | ------ | ---------------------------------------- |
| `id`                     | int    | Row index                                |
| `image_id`               | string | Unique table identifier                  |
| `html_image`             | Image  | PNG rendering of the HTML table          |
| `markdown_image`         | Image  | PNG rendering of the Markdown table      |
| `latex_image`            | Image  | PNG rendering of the LaTeX table         |
| `html_code`              | string | Raw HTML source                          |
| `markdown_code`          | string | Raw Markdown source                      |
| `latex_code`             | string | Raw LaTeX source                         |
| `table`                  | string | JSON-encoded `{header, rows}`            |
| `dataset`                | string | Source dataset                           |
| `table_partition`        | string | Which partition the table belongs to     |
| `size_detection`         | string | Gold answer: `{rows}\|{cols}`            |
| `cell_value`             | string | Gold answer: value at a sampled cell     |
| `cell_lookup`            | string | Sampled cell coordinates `{row}\|{col}`  |
| `reverse_lookup_indices` | string | Row/col indices for reverse lookup       |
| `reverse_lookup`         | string | Gold answer: value for reverse lookup    |
| `column_idx`             | int    | Sampled column index                     |
| `column_retrieval`       | string | Gold answer: column header at that index |
| `row_idx`                | int    | Sampled row index                        |
| `row_retrieval`          | string | Gold answer: full row at that index      |
| `table_first_cell`       | string | Value of cell (0, 0)                     |
| `table_last_cell`        | string | Value of last cell                       |
| `number_of_rows`         | int    | Table row count                          |
| `number_of_columns`      | int    | Table column count                       |

---

## Quickstart

```python
from datasets import load_dataset

# Task Prediction (QA) — shows images in Dataset Viewer
qa = load_dataset("MBZUAI/TABVERSE", name="qa", split="test")
print(qa[0]["query"])          # natural-language question
qa[0]["html_image"]            # PIL Image of the HTML-rendered table
qa[0]["html_code"]             # raw HTML source

# SUC / Format Generation
suc = load_dataset("MBZUAI/TABVERSE", name="suc", split="test")
print(suc[0]["size_detection"])   # e.g. "118|9"
print(suc[0]["cell_value"])       # gold cell value
suc[0]["markdown_image"]          # PIL Image of the Markdown-rendered table
```

### Format generation (SR) with the `suc` config

```python
# HTML → Markdown generation
for row in suc:
    source = row["html_code"]    # input
    target = row["markdown_code"] # generation target
```

---

## Tasks

### SUC — Structured Understanding & Comprehension

Structural probing tasks that isolate table-parsing ability from content knowledge.
All answers are derived from the table structure itself.

| Sub-task             | Input                             | Gold answer column                    |
| -------------------- | --------------------------------- | ------------------------------------- |
| Size detection       | table image / code                | `size_detection`                      |
| Cell value retrieval | table + `cell_lookup` coordinates | `cell_value`                          |
| Reverse lookup       | table + `reverse_lookup` value    | `reverse_lookup_indices`              |
| Column retrieval     | table + `column_idx`              | `column_retrieval`                    |
| Row retrieval        | table + `row_idx`                 | `row_retrieval`                       |
| First / last cell    | table                             | `table_first_cell`, `table_last_cell` |

### QA — Task Prediction

Free-form natural-language question answering over tables. Uses the `qa` config.
Evaluation metric: **exact match** (after normalisation).

### SR — Structure Reconstruction (Format Generation)

Given a table in one format, generate the table in another format.
Six conversion directions are possible from the `suc` config:
HTML↔Markdown, HTML↔LaTeX, Markdown↔LaTeX.
Evaluation: BLEU + structural similarity.

---

## Evaluation

We evaluate **17 models** in three pipeline modes:

| Mode      | Input                                      | Models                                          |
| --------- | ------------------------------------------ | ----------------------------------------------- |
| LLM       | plain text (one of the three code formats) | Qwen2.5-3B/7B, SmolLM2-1.7B, GPT-4o, Gemini     |
| VLM-Image | rendered PNG image                         | Qwen-VL-2.5-3B/7B, SmolVLM-1.7B, GPT-4o, Gemini |
| VLM-Text  | code string fed to a VLM                   | same VLMs in text-only mode                     |

Primary metric: **exact-match accuracy** per task and per format.

---

## Source datasets

| Dataset           | Description                               |
| ----------------- | ----------------------------------------- |
| WikiTQ / HybridQA | Open-domain QA over Wikipedia tables      |
| SQA               | Sequential question answering over tables |
| TabFact           | Fact verification over Wikipedia tables   |
| FEVEROUS          | Fact extraction and verification          |

All tables come from held-out splits to prevent contamination.

---

## Citation

```bibtex
@misc{ahsan2025tabverse,
  title   = {{TABVERSE}: Benchmarking Cross-Format Table Understanding in {LLMs} and {VLMs}},
  author  = {Ahsan, Momina and Ahmad, Sarfraz and Hee, Ming Shan and
             Lee, Roy Ka-Wei and Nakov, Preslav},
  year    = {2025},
  url     = {https://huggingface.co/datasets/MBZUAI/TABVERSE}
}
```
