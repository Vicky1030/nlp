"""Frozen char_1_2 TF-IDF cosine inference for member-D evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np

from data_loader import load_data


METHOD = "tfidf_cosine"
REPRESENTATION = "char_1_2"
THRESHOLD = 0.28
MODEL_SHA256 = "bf0e9d5ade8cc13cb2d619a3ec226014f472e8b307b64a39aad90d011ff10db5"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cosine(left, right):
    left_zero = np.asarray(left.getnnz(axis=1) == 0)
    right_zero = np.asarray(right.getnnz(axis=1) == 0)
    scores = np.asarray(left.multiply(right).sum(axis=1)).reshape(-1)
    scores[left_zero | right_zero] = 0.0
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite similarity")
    if ((scores < -1e-12) | (scores > 1 + 1e-12)).any():
        raise ValueError("similarity outside tolerance")
    return np.clip(scores, 0.0, 1.0)


class FrozenPredictor:
    """Load the frozen vectorizer once and serve ordered sentence-pair predictions."""

    def __init__(self, package_root=None):
        self.package_root = Path(package_root or Path(__file__).resolve().parent).resolve()
        self.config = json.loads((self.package_root / "frozen_config.json").read_text(encoding="utf-8"))
        if self.config != {"method": METHOD, "representation": REPRESENTATION, "threshold": THRESHOLD, "operator": ">=", "vectorizer": "model/vectorizer.joblib", "metadata": "model/metadata.json"}:
            raise ValueError("frozen configuration mismatch")
        self.model_path = self.package_root / self.config["vectorizer"]
        self.metadata_path = self.package_root / self.config["metadata"]
        if _sha(self.model_path) != MODEL_SHA256:
            raise ValueError("vectorizer SHA-256 mismatch")
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("scheme") != REPRESENTATION or self.metadata.get("model_sha256") != MODEL_SHA256:
            raise ValueError("vectorizer metadata mismatch")
        self.vectorizer = joblib.load(self.model_path)

    def predict_pairs(self, sentence_pairs):
        pairs = list(sentence_pairs)
        normalized = []
        for index, pair in enumerate(pairs):
            if isinstance(pair, str) or not isinstance(pair, (list, tuple)):
                raise TypeError(f"sentence_pairs[{index}] must be a list or tuple containing exactly two strings")
            if len(pair) != 2:
                raise ValueError(f"sentence_pairs[{index}] must contain exactly two items; got {len(pair)}")
            first, second = pair
            if not isinstance(first, str) or not isinstance(second, str):
                raise TypeError(f"sentence_pairs[{index}] must contain two strings")
            if not first.strip() or not second.strip():
                raise ValueError(f"sentence_pairs[{index}] contains an empty or whitespace-only sentence")
            normalized.append((first, second))
        if not normalized:
            return []
        left = [pair[0] for pair in normalized]
        right = [pair[1] for pair in normalized]
        scores = _cosine(self.vectorizer.transform(left), self.vectorizer.transform(right))
        return [{"prediction": int(score >= THRESHOLD), "similarity": float(score)} for score in scores]


def predict_pairs(sentence_pairs, package_root=None):
    """Return ordered prediction/similarity dictionaries; labels are not accepted or used."""
    return FrozenPredictor(package_root).predict_pairs(sentence_pairs)


def _targets(output_root):
    root = Path(output_root).resolve()
    return root, root / "evaluation_predictions.jsonl", root / "evaluation_similarity.jsonl", root / "inference_report.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 JSONL with id,sentence1,sentence2,label")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; existing targets are rejected")
    args = parser.parse_args()
    output_root, prediction_path, similarity_path, report_path = _targets(args.output_dir)
    conflicts = [path for path in (prediction_path, similarity_path, report_path) if path.exists()]
    if conflicts:
        raise SystemExit("Refusing to overwrite existing output files; choose a new --output-dir:\n" + "\n".join(map(str, conflicts)))
    rows = load_data(args.input)
    predictor = FrozenPredictor()
    for left, right in (("预热句子一", "预热句子二"), ("中文测试", "中文测试"), ("warmup", "warm up")):
        predictor.predict_pairs([(left, right)])
    pairs = [(row["sentence1"], row["sentence2"]) for row in rows]
    results = []
    for row, pair in zip(rows, pairs):
        started = time.perf_counter_ns()
        result = predictor.predict_pairs([pair])[0]
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError("invalid latency")
        results.append((row, result, elapsed_ms))
    output_root.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8", newline="\n") as pred_file, similarity_path.open("w", encoding="utf-8", newline="\n") as sim_file:
        for row, result, elapsed_ms in results:
            pred_file.write(json.dumps({"id": row["id"], "sentence1": row["sentence1"], "sentence2": row["sentence2"], "label": row["label"], "prediction": result["prediction"], "method": METHOD, "latency_ms": elapsed_ms}, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            sim_file.write(json.dumps({"id": row["id"], "label": row["label"], "similarity": result["similarity"]}, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    report = {
        "status": "passed", "method": METHOD, "representation": REPRESENTATION, "threshold": THRESHOLD, "operator": ">=",
        "input": {"path": str(args.input.resolve()), "sha256": _sha(args.input), "rows": len(rows)},
        "model": {"path": str(predictor.model_path), "sha256": _sha(predictor.model_path), "metadata": str(predictor.metadata_path)},
        "latency_scope": "warmed CPU single-record; text processing, two transforms, sparse cosine, threshold; excludes model load, warmup, I/O, scoring and record assembly",
        "outputs": {"predictions": str(prediction_path), "similarity": str(similarity_path)},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "passed", "rows": len(rows), "output_dir": str(output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
