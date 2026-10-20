"""Independently verify selected validation-threshold rows from the existing CSV."""
import argparse
import csv
import json
import math
from pathlib import Path


THRESHOLDS = (0.20, 0.28, 0.40)
TOLERANCE = 1e-14


def metrics_from_counts(tn, fp, fn, tp):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_positive = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_negative = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {
        "precision_1": precision,
        "recall_1": recall,
        "macro_f1": (f1_positive + f1_negative) / 2,
        "predicted_1": fp + tp,
        "total": tn + fp + fn + tp,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-table", type=Path, required=True)
    args = parser.parse_args()

    with args.threshold_table.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    results = []
    for threshold in THRESHOLDS:
        matches = [
            row
            for row in rows
            if row["scheme"] == "char_1_2"
            and float(row["threshold"]) == threshold
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"Expected one char_1_2 row for threshold {threshold}; found {len(matches)}"
            )

        row = matches[0]
        counts = {key: int(row[key]) for key in ("TN", "FP", "FN", "TP")}
        calculated = metrics_from_counts(**{key.lower(): value for key, value in counts.items()})
        differences = {
            key: abs(calculated[key] - float(row[key]))
            for key in ("precision_1", "recall_1", "macro_f1")
        }
        if any(not math.isfinite(value) or value > TOLERANCE for value in differences.values()):
            raise SystemExit(f"Metric/count mismatch at threshold {threshold}: {differences}")
        results.append(
            {
                "scheme": row["scheme"],
                "threshold": threshold,
                **counts,
                **calculated,
                "csv_metrics": {
                    key: float(row[key])
                    for key in ("precision_1", "recall_1", "macro_f1")
                },
                "absolute_differences": differences,
            }
        )

    print(json.dumps({"status": "passed", "source": str(args.threshold_table.resolve()), "matched_rows": len(results), "total_comparison_rows": len(rows), "tolerance": TOLERANCE, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
