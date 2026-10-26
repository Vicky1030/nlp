import argparse
import json
from pathlib import Path


def divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def evaluate(records):
    valid = lambda value: type(value) is int and value in (0, 1)
    correct = sum(valid(row.get("prediction")) and row["prediction"] == row["label"] for row in records)
    invalid = sum(not valid(row.get("prediction")) for row in records)
    confusion = [[0, 0], [0, 0]]
    for row in records:
        prediction = row.get("prediction")
        if valid(prediction):
            confusion[row["label"]][prediction] += 1

    per_class = {}
    for label in (0, 1):
        true_positive = confusion[label][label]
        predicted = sum(valid(row.get("prediction")) and row["prediction"] == label for row in records)
        support = sum(row["label"] == label for row in records)
        precision = divide(true_positive, predicted)
        recall = divide(true_positive, support)
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": divide(2 * precision * recall, precision + recall),
            "support": support,
        }
    return {
        "total": len(records),
        "correct": correct,
        "accuracy": divide(correct, len(records)),
        "invalid_count": invalid,
        "confusion_matrix_valid_0_1": confusion,
        "per_class": per_class,
        "macro_f1": (per_class["0"]["f1"] + per_class["1"]["f1"]) / 2,
    }


def main():
    parser = argparse.ArgumentParser(description="Score a JSONL prediction file against its labels")
    parser.add_argument("predictions", type=Path, help="JSONL with integer label and prediction fields")
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        parser.error("prediction file is empty")
    if any(type(row.get("label")) is not int or row["label"] not in (0, 1) for row in records):
        parser.error("every row must have an integer label equal to 0 or 1")
    print(json.dumps(evaluate(records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
