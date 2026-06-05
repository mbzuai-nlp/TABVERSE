# coding=utf-8
"""
WikiTableQuestions (WTQ) wrapper adapted for VisualTableBench / Table-Meets-LLM.

This follows the official HF loader approach (downloads the WTQ zip)
but re-exports examples using the same naming as your other dataset scripts:
 - question -> question
 - answers  -> answer_text (Sequence[str])
 - table: name/header/rows -> table_file, table_header, table_data
"""

import os
import datasets

_CITATION = """\
@inproceedings{pasupat-liang-2015-compositional,
  title = "Compositional Semantic Parsing on Semi-Structured Tables",
  author = "Pasupat, Panupong and Liang, Percy",
  booktitle = "Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics",
  year = "2015"
}
"""

_DESCRIPTION = """\
This WikiTableQuestions dataset is a large-scale dataset for the task of question answering on semi-structured tables.
"""

_HOMEPAGE = "https://nlp.stanford.edu/software/sempre/wikitable"
_LICENSE = "Creative Commons Attribution Share Alike 4.0 International"

# Official release zip that contains the TSVs
_DATA_URL = (
    "https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip"
)
_WIKITQ_VERSION = datasets.Version("1.0.2")

class WikiTQ(datasets.GeneratorBasedBuilder):
    VERSION = _WIKITQ_VERSION

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=f"random-split-{i}",
            version=_WIKITQ_VERSION,
            description=f"random-split-{i}",
        )
        for i in range(1, 6)
    ]

    DEFAULT_CONFIG_NAME = "random-split-1"

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "question": datasets.Value("string"),
                "answer_text": datasets.features.Sequence(datasets.Value("string")),
                "table_file": datasets.Value("string"),
                "table_header": datasets.features.Sequence(datasets.Value("string")),
                "table_data": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # The official repo stores train/dev as <config>-train.tsv, <config>-dev.tsv
        train_file = f"{self.config.name}-train.tsv"
        dev_file = f"{self.config.name}-dev.tsv"
        test_file = "pristine-unseen-tables.tsv"  # official test file name in the zip

        root_dir = os.path.join(dl_manager.download_and_extract(_DATA_URL), "WikiTableQuestions")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"main_filepath": os.path.join(root_dir, "data", train_file), "root_dir": root_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"main_filepath": os.path.join(root_dir, "data", test_file), "root_dir": root_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"main_filepath": os.path.join(root_dir, "data", dev_file), "root_dir": root_dir},
            ),
        ]

    def _read_table_from_file(self, table_name: str, root_dir: str):
        # WTQ stores tables as tsv files inside the archive; some table names end with .csv in older scripts,
        # normalize to .tsv and read.
        def _extract_table_content(_line: str):
            # split on tab and strip newline
            _vals = [_.replace("\n", " ").strip() for _ in _line.strip("\n").split("\t")]
            return _vals

        rows = []
        table_name = table_name.replace(".csv", ".tsv")
        tbl_path = os.path.join(root_dir, table_name)
        # Some repos place tables under a subfolder; accept either direct or in 'data' subdir.
        if not os.path.exists(tbl_path):
            tbl_path = os.path.join(root_dir, "data", table_name)
        with open(tbl_path, "r", encoding="utf8") as table_f:
            lines = table_f.readlines()
            header = _extract_table_content(lines[0])
            for line in lines[1:]:
                rows.append(_extract_table_content(line))
        return {"header": header, "rows": rows, "name": table_name}

    def _generate_examples(self, main_filepath, root_dir):
        # main_filepath e.g. /.../data/random-split-1-train.tsv or pristine-unseen-tables.tsv
        with open(main_filepath, encoding="utf-8") as f:
            # skip header if present (official files have header line)
            first = f.readline()
            # If first line contains tab-separated header words like "id\tquestion\t..." skip; else rewind.
            if not ("\t" in first and ("question" in first.lower() or "id" in first.lower())):
                # first line was actual data; reset
                f.seek(0)
            for idx, line in enumerate(f):
                # official format: example_id \t question \t table_name \t answer(s)
                parts = line.strip("\n").split("\t")
                if len(parts) < 4:
                    # skip malformed line
                    continue
                example_id, question, table_name, answer = parts[0], parts[1], parts[2], parts[3]
                answer_list = answer.split("|") if answer != "" else []
                # read table content
                table_content = self._read_table_from_file(table_name, root_dir)
                yield idx, {
                    "id": example_id,
                    "question": question,
                    "answer_text": answer_list,
                    "table_file": table_content.get("name", table_name),
                    "table_header": table_content["header"],
                    "table_data": table_content["rows"],
                }