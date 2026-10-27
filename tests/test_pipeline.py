import csv
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from evaluation.api import evaluate_prediction_directory
from pipeline.live_runner import parse_methods, select_records
from run_all import build_parser, print_hard_case_details


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


class PipelineSelectionTests(unittest.TestCase):
    def test_member_aliases_expand_in_canonical_order(self):
        self.assertEqual(parse_methods("C,A"), ["tfidf_cosine", "tfidf_lr", "qwen_lora"])
        self.assertEqual(
            parse_methods("members"),
            ["tfidf_lr", "qwen_few_shot", "qwen_lora"],
        )
        self.assertEqual(
            parse_methods("all"),
            ["tfidf_cosine", "tfidf_lr", "qwen_zero_shot", "qwen_few_shot", "qwen_lora"],
        )

    def test_random_subset_is_reproducible_and_keeps_file_order(self):
        rows = [{"id": index} for index in range(20)]
        first = select_records(rows, 7, "random", 42)
        second = select_records(rows, 7, "random", 42)
        self.assertEqual(first, second)
        self.assertEqual([row["id"] for row in first], sorted(row["id"] for row in first))

    def test_default_is_full_evaluation_and_quick_is_explicit(self):
        defaults = build_parser().parse_args([])
        self.assertEqual(defaults.split, "evaluation")
        self.assertFalse(defaults.quick)
        self.assertIsNone(defaults.limit)
        quick = build_parser().parse_args(["--quick"])
        self.assertTrue(quick.quick)


class DynamicEvaluationTests(unittest.TestCase):
    def test_fresh_prediction_directory_uses_d_evaluation(self):
        gold = [
            {"id": 1, "sentence1": "甲", "sentence2": "甲", "label": 1},
            {"id": 2, "sentence1": "甲", "sentence2": "乙", "label": 0},
        ]
        predictions = [
            {**gold[0], "prediction": 1, "method": "tfidf_lr", "latency_ms": 1.0},
            {**gold[1], "prediction": 0, "method": "tfidf_lr", "latency_ms": 2.0},
        ]
        temp_root = Path(__file__).resolve().parents[1] / "_tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            prediction_dir = root / "predictions"
            prediction_dir.mkdir()
            write_jsonl(root / "gold.jsonl", gold)
            write_jsonl(prediction_dir / "tfidf_lr.jsonl", predictions)
            result = evaluate_prediction_directory(
                split_name="validation",
                gold_path=root / "gold.jsonl",
                prediction_dir=prediction_dir,
                output_dir=root / "evaluation",
                make_plots=False,
            )
            self.assertEqual(result["models"], 1)
            with Path(result["comparison_csv"]).open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["accuracy"], "1.0")
            self.assertTrue(Path(result["summary_html"]).is_file())
            self.assertTrue((root / "evaluation" / "errors" / "validation").is_dir())

    def test_three_required_case_patterns_are_selected_from_current_predictions(self):
        gold = [
            {"id": 1, "sentence1": "甲", "sentence2": "甲", "label": 1},
            {"id": 2, "sentence1": "甲", "sentence2": "乙", "label": 1},
            {"id": 3, "sentence1": "丙", "sentence2": "丁", "label": 1},
        ]
        method_predictions = {
            "tfidf_lr": [1, 0, 0],
            "qwen_few_shot": [1, 1, 0],
            "qwen_lora": [1, 1, 1],
        }
        temp_root = Path(__file__).resolve().parents[1] / "_tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            prediction_dir = root / "predictions"
            prediction_dir.mkdir()
            write_jsonl(root / "gold.jsonl", gold)
            for method, values in method_predictions.items():
                write_jsonl(
                    prediction_dir / f"{method}.jsonl",
                    [
                        {**row, "prediction": prediction, "method": method, "latency_ms": 1.0}
                        for row, prediction in zip(gold, values)
                    ],
                )
            result = evaluate_prediction_directory(
                split_name="quick",
                gold_path=root / "gold.jsonl",
                prediction_dir=prediction_dir,
                output_dir=root / "evaluation",
                make_plots=False,
            )
            hard_cases = result["hard_cases"]
            self.assertEqual(hard_cases["status"], "complete")
            self.assertEqual(
                [case["selected_case"]["id"] for case in hard_cases["cases"]],
                [1, 2, 3],
            )
            self.assertTrue((root / "evaluation" / "hard_case_output.json").is_file())
            terminal = StringIO()
            with redirect_stdout(terminal):
                print_hard_case_details(hard_cases)
            rendered = terminal.getvalue()
            self.assertIn("句子1：甲", rendered)
            self.assertIn("真实标签：1（相似）", rendered)
            self.assertIn("B · Qwen Few-shot", rendered)
            self.assertIn("错误", rendered)


if __name__ == "__main__":
    unittest.main()
