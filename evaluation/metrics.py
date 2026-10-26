"""Metrics implementation using only the Python standard library."""
import math


def _prf(y_true, y_pred, label):
    tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
    fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
    fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    support = sum(t == label for t in y_true)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support,
            "tp": tp, "fp": fp, "fn": fn}


def _quantile_linear(values, q):
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    position = (len(xs) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return xs[low]
    return xs[low] + (xs[high] - xs[low]) * (position - low)


def calculate_metrics(rows, positive_label=1):
    if not rows:
        raise ValueError("No rows to evaluate")
    y_true = [int(r["label"]) for r in rows]
    raw_pred = [r.get("prediction") for r in rows]
    invalid = sum(p is None for p in raw_pred)
    # Null/invalid predictions count as incorrect and remain in the denominator.
    y_pred_valid = [int(p) if p is not None else -999 for p in raw_pred]
    n = len(rows)
    correct = sum(t == p for t, p in zip(y_true, y_pred_valid))
    per_class = {str(label): _prf(y_true, y_pred_valid, label) for label in (0, 1)}
    cm = [[sum(t == a and p == b for t, p in zip(y_true, y_pred_valid))
           for b in (0, 1)] for a in (0, 1)]
    latency = []
    for row in rows:
        val = row.get("latency_ms")
        if val is None:
            continue
        try:
            val = float(val)
            if math.isfinite(val) and val >= 0:
                latency.append(val)
        except (TypeError, ValueError):
            pass
    positive = per_class[str(positive_label)]
    result = {
        "n_samples": n,
        "accuracy": correct / n,
        "precision_positive": positive["precision"],
        "recall_positive": positive["recall"],
        "f1_positive": positive["f1"],
        "macro_f1": (per_class["0"]["f1"] + per_class["1"]["f1"]) / 2,
        "per_class": per_class,
        "confusion_matrix": {"labels": [0, 1], "rows_true_cols_pred": cm},
        "correct": correct,
        "errors": n - correct,
        "error_rate": (n - correct) / n,
        "invalid_predictions": invalid,
        "invalid_rate": invalid / n,
        "latency_count": len(latency),
        "latency_mean_ms": sum(latency) / len(latency) if latency else None,
        "latency_median_ms": _quantile_linear(latency, 0.5),
        "latency_p95_ms": _quantile_linear(latency, 0.95),
        "latency_min_ms": min(latency) if latency else None,
        "latency_max_ms": max(latency) if latency else None,
    }
    return result
