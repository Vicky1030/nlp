"""统一评价指标：不依赖 sklearn，避免不同成员环境不一致。"""
from collections import Counter


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _binary_prf(y_true, y_pred, label):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1,
            "support": sum(1 for t in y_true if t == label)}


def calculate_metrics(y_true, y_pred, positive_label=1):
    if len(y_true) != len(y_pred):
        raise ValueError(f"标签数与预测数不一致：{len(y_true)} != {len(y_pred)}")
    if not y_true:
        raise ValueError("没有可评价的数据。")
    labels = sorted(set(y_true) | set(y_pred))
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    per_class = {str(label): _binary_prf(y_true, y_pred, label) for label in labels}
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class) if per_class else 0.0
    positive = _binary_prf(y_true, y_pred, positive_label)

    # 二分类时按真实标签行、预测标签列组织混淆矩阵。
    confusion = {
        "labels": labels,
        "matrix": [[sum(1 for t, p in zip(y_true, y_pred) if t == a and p == b)
                    for b in labels] for a in labels]
    }
    result = {
        "n_samples": len(y_true),
        "accuracy": accuracy,
        "precision_positive": positive["precision"],
        "recall_positive": positive["recall"],
        "f1_positive": positive["f1"],
        "macro_f1": macro_f1,
        "positive_label": positive_label,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "correct": sum(t == p for t, p in zip(y_true, y_pred)),
        "errors": sum(t != p for t, p in zip(y_true, y_pred)),
        "error_rate": sum(t != p for t, p in zip(y_true, y_pred)) / len(y_true),
    }
    return result


def latency_summary(records):
    """汇总 latency_ms；只统计存在且可转成数字的记录。"""
    values = []
    for row in records:
        value = row.get("latency_ms")
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return {}
    values.sort()
    n = len(values)
    # 使用线性插值百分位数（与 numpy.percentile 默认方法一致）
    def percentile_linear(sorted_values, q):
        if len(sorted_values) == 1:
            return sorted_values[0]
        position = (len(sorted_values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = position - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    mid = n // 2
    median = values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2
    return {
        "latency_count": n,
        "latency_mean_ms": sum(values) / n,
        "latency_median_ms": median,
        "latency_p95_ms": percentile_linear(values, 0.95),
        "latency_min_ms": values[0],
        "latency_max_ms": values[-1],
    }
