"""Frozen TF-IDF + Logistic Regression inference package."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack


THRESHOLD = 0.53
METHOD = "tfidf_lr"


def _validate_pairs(sentence_pairs):
    if isinstance(sentence_pairs, (str, bytes)):
        raise TypeError("sentence_pairs must be an iterable of sentence pairs, not a string")
    pairs = []
    for index, pair in enumerate(sentence_pairs):
        location = f"sentence_pairs[{index}]"
        if not isinstance(pair, (list, tuple)):
            raise TypeError(f"{location} must be a list or tuple containing exactly two strings")
        if len(pair) != 2:
            raise ValueError(f"{location} must contain exactly two strings; got {len(pair)} items")
        if not isinstance(pair[0], str) or not isinstance(pair[1], str):
            raise TypeError(f"{location} must contain only strings")
        if not pair[0].strip() or not pair[1].strip():
            raise ValueError(f"{location} must not contain blank or whitespace-only sentences")
        pairs.append((pair[0], pair[1]))
    return pairs


class FrozenPredictor:
    def __init__(self, package_dir=None):
        root = Path(package_dir or __file__).resolve()
        self.root = root if root.is_dir() else root.parent
        self.config = json.loads((self.root / "frozen_config.json").read_text(encoding="utf-8"))
        if self.config.get("status") != "frozen" or self.config.get("config_id") != "char_1_2__C1p0__cw_balanced":
            raise ValueError("unexpected or unfrozen model configuration")
        self.threshold = self.config["threshold"]
        if self.threshold != THRESHOLD or self.config.get("threshold_operator") != ">=":
            raise ValueError("frozen threshold does not match package protocol")
        model_path = self.root / self.config["model_relative_path"]
        vectorizer_path = self.root / self.config["vectorizer_relative_path"]
        if _sha256(model_path) != self.config["source_sha256"]["copied_model"]:
            raise ValueError("frozen LR model hash mismatch")
        if _sha256(vectorizer_path) != self.config["source_sha256"]["copied_vectorizer"]:
            raise ValueError("frozen TF-IDF vectorizer hash mismatch")
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
        classes = list(self.model.classes_)
        if 1 not in classes:
            raise ValueError("frozen LR model has no positive class 1")
        self.positive_column = classes.index(1)
        if classes != self.config["classes"]:
            raise ValueError("frozen LR classes differ from metadata")
        if f"{type(self.model).__module__}.{type(self.model).__qualname__}" != self.config["model_class"]:
            raise ValueError("unexpected LR model class")
        if f"{type(self.vectorizer).__module__}.{type(self.vectorizer).__qualname__}" != self.config["vectorizer_class"]:
            raise ValueError("unexpected TF-IDF vectorizer class")
        if len(self.vectorizer.vocabulary_) != self.config["vocabulary_size"]:
            raise ValueError("unexpected frozen feature dimension")
        if self.model.n_features_in_ != self.config["feature_dimension"] or self.config["feature_dimension"] != 2 * self.config["vocabulary_size"]:
            raise ValueError("LR feature dimension does not equal 2V")
        vectorizer_params = self.vectorizer.get_params(deep=False)
        for key, expected in self.config["tfidf_parameters"].items():
            actual = list(vectorizer_params[key]) if key == "ngram_range" else vectorizer_params[key]
            if actual != expected:
                raise ValueError(f"frozen vectorizer parameter mismatch: {key}")
        model_params = self.model.get_params(deep=False)
        for key, expected in self.config["lr_parameters"].items():
            if model_params.get(key) != expected:
                raise ValueError(f"frozen LR parameter mismatch: {key}")

    def _predict_one(self, sentence1, sentence2):
        left = self.vectorizer.transform([sentence1])
        right = self.vectorizer.transform([sentence2])
        features = hstack((abs(left - right), left.multiply(right)), format="csr", dtype=np.float64)
        probability = float(self.model.predict_proba(features)[0, self.positive_column])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("model returned a probability outside [0, 1]")
        return probability, int(probability >= self.threshold)

    def predict_pairs(self, sentence_pairs):
        pairs = _validate_pairs(sentence_pairs)
        if not pairs:
            return []
        return [
            {"prediction": prediction, "probability": probability}
            for probability, prediction in (self._predict_one(a, b) for a, b in pairs)
        ]


def predict_pairs(sentence_pairs):
    """Convenience API; creates and loads a predictor for this call."""
    return FrozenPredictor().predict_pairs(sentence_pairs)


def _load_jsonl(path):
    path = Path(path).resolve()
    rows = []
    first_id_line = {}

    def reject_duplicate_keys(pairs):
        row = {}
        for key, value in pairs:
            if key in row:
                raise ValueError(f"{path}: duplicate JSON field {key!r}")
            row[key] = value
        return row

    with path.open("rb") as stream:
        for line_no, raw_line in enumerate(stream, 1):
            if line_no == 1 and raw_line.startswith(b"\xef\xbb\xbf"):
                raise ValueError(f"{path}:{line_no}: UTF-8 BOM is not allowed")
            try:
                line = raw_line.decode("utf-8")
                row = json.loads(line, object_pairs_hook=reject_duplicate_keys)
            except UnicodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid UTF-8: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            except ValueError as exc:
                message = str(exc)
                if message.startswith(f"{path}:"):
                    message = message.replace(f"{path}:", f"{path}:{line_no}:", 1)
                raise ValueError(message) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: each line must be a JSON object")
            required = {"id", "sentence1", "sentence2", "label"}
            if set(row) != required:
                raise ValueError(f"{path}:{line_no}: expected fields id,sentence1,sentence2,label")
            if type(row["id"]) is not int or type(row["label"]) is not int or row["label"] not in (0, 1):
                raise ValueError(f"{path}:{line_no}: id and binary label must be integers")
            if any(not isinstance(row[name], str) or not row[name].strip() for name in ("sentence1", "sentence2")):
                raise ValueError(f"{path}:{line_no}: sentences must be non-whitespace strings")
            if row["id"] in first_id_line:
                first = first_id_line[row["id"]]
                raise ValueError(f"{path}:{line_no}: duplicate id {row['id']}; first appeared at line {first}")
            first_id_line[row["id"]] = line_no
            rows.append(row)
    return rows


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Run frozen TF-IDF + LR inference")
    parser.add_argument("--input", required=True, type=Path, help="four-field JSONL input")
    parser.add_argument("--output-dir", required=True, type=Path, help="new output directory")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    rows = _load_jsonl(args.input)
    predictor = FrozenPredictor()
    for row in rows[:10]:
        predictor._predict_one(row["sentence1"], row["sentence2"])
    predictions = []
    probabilities = []
    output.mkdir(parents=True, exist_ok=False)
    for row in rows:
        started = time.perf_counter_ns()
        probability, prediction = predictor._predict_one(row["sentence1"], row["sentence2"])
        elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
        predictions.append({"id": row["id"], "sentence1": row["sentence1"], "sentence2": row["sentence2"],
                           "label": row["label"], "prediction": prediction, "method": METHOD,
                           "latency_ms": elapsed})
        probabilities.append({"id": row["id"], "probability": probability, "threshold": predictor.threshold})
    with (output / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (output / "probabilities.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in probabilities:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {"rows": len(rows), "threshold": THRESHOLD, "method": METHOD,
               "input_sha256": _sha256(args.input), "warmup_records": min(10, len(rows)),
               "timing_scope": "two TF-IDF transforms, sparse feature construction, predict_proba and threshold decision; excludes model loading, warmup, file I/O and scoring"}
    (output / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "rows": len(rows), "output_dir": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
