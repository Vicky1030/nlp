"""从统一 JSONL 计算指标；无效预测计入分母且作为错误，不静默丢弃。"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('file')
    p.add_argument('--output')
    a = p.parse_args()
    with open(a.file, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise SystemExit('空预测文件')
    if any(type(r.get('label')) is not int or r['label'] not in (0, 1) for r in rows):
        raise SystemExit('所有样本必须包含 0/1 真实 label')
    required = {'id', 'sentence1', 'sentence2', 'label', 'prediction', 'method', 'latency_ms'}
    if any(set(row) != required for row in rows):
        raise SystemExit('预测文件必须严格使用统一七字段格式')
    ids = [row['id'] for row in rows]
    if any(type(item) is not int for item in ids) or len(ids) != len(set(ids)):
        raise SystemExit('样本 ID 必须是唯一整数')
    methods = {r.get('method') for r in rows}
    if len(methods) != 1:
        raise SystemExit(f'同一评估文件必须是同一 method，实际：{methods}')
    y_true = [r['label'] for r in rows]
    predictions = [r.get('prediction') for r in rows]
    invalid = sum(type(p) is not int or p not in (0, 1) for p in predictions)
    # 特殊值 -1 表示模型未提供 0/1 预测。sklearn labels=[0,1] 会把这类情况视为真实类的漏判。
    y_pred = [p if type(p) is int and p in (0, 1) else -1 for p in predictions]
    valid_latencies = [float(r['latency_ms']) for r in rows if isinstance(r.get('latency_ms'), (int, float))]
    precision_by_class = precision_score(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0,
    )
    recall_by_class = recall_score(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0,
    )
    f1_by_class = f1_score(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0,
    )
    extended_confusion = {
        str(label): {
            '0': sum(t == label and p == 0 for t, p in zip(y_true, y_pred)),
            '1': sum(t == label and p == 1 for t, p in zip(y_true, y_pred)),
            'invalid': sum(t == label and p == -1 for t, p in zip(y_true, y_pred)),
        }
        for label in (0, 1)
    }
    prediction_counts = Counter(
        str(p) if type(p) is int and p in (0, 1) else 'invalid'
        for p in predictions
    )
    summary = {
        'method': next(iter(methods)),
        'total': len(rows),
        'label_counts': {str(label): y_true.count(label) for label in (0, 1)},
        'prediction_counts': {
            '0': prediction_counts['0'],
            '1': prediction_counts['1'],
            'invalid': prediction_counts['invalid'],
        },
        'invalid_count': invalid,
        'invalid_rate': invalid / len(rows),
        'invalid_policy': 'prediction=null is retained and counted as incorrect',
        'accuracy_including_invalid_as_wrong': accuracy_score(y_true, y_pred),
        'precision_macro_including_invalid_as_wrong': precision_score(y_true, y_pred, labels=[0, 1], average='macro', zero_division=0),
        'recall_macro_including_invalid_as_wrong': recall_score(y_true, y_pred, labels=[0, 1], average='macro', zero_division=0),
        'f1_macro_including_invalid_as_wrong': f1_score(y_true, y_pred, labels=[0, 1], average='macro', zero_division=0),
        'per_class': {
            str(label): {
                'precision': float(precision_by_class[index]),
                'recall': float(recall_by_class[index]),
                'f1': float(f1_by_class[index]),
            }
            for index, label in enumerate((0, 1))
        },
        'confusion_matrix_valid_0_1': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        'confusion_matrix_with_invalid': extended_confusion,
        'latency_count': len(valid_latencies),
        'mean_latency_ms': mean(valid_latencies) if valid_latencies else None,
        'median_latency_ms': median(valid_latencies) if valid_latencies else None,
        'p95_latency_ms': percentile(valid_latencies, 0.95),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if a.output:
        output_path = Path(a.output).resolve()
        if output_path.exists():
            raise FileExistsError(f'评估输出已存在，拒绝覆盖：{output_path}')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print('评估 JSON：', output_path)


if __name__ == '__main__':
    main()
