"""从统一 JSONL 计算指标；无效预测计入分母且作为错误，不静默丢弃。"""
import argparse
import json
from statistics import mean, median
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument('file')
    a = p.parse_args()
    with open(a.file, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise SystemExit('空预测文件')
    if any(type(r.get('label')) is not int or r['label'] not in (0, 1) for r in rows):
        raise SystemExit('所有样本必须包含 0/1 真实 label')
    methods = {r.get('method') for r in rows}
    if len(methods) != 1:
        raise SystemExit(f'同一评估文件必须是同一 method，实际：{methods}')
    y_true = [r['label'] for r in rows]
    predictions = [r.get('prediction') for r in rows]
    invalid = sum(type(p) is not int or p not in (0, 1) for p in predictions)
    # 特殊值 -1 表示模型未提供 0/1 预测。sklearn labels=[0,1] 会把这类情况视为真实类的漏判。
    y_pred = [p if type(p) is int and p in (0, 1) else -1 for p in predictions]
    valid_latencies = [float(r['latency_ms']) for r in rows if isinstance(r.get('latency_ms'), (int, float))]
    summary = {
        'method': next(iter(methods)),
        'total': len(rows),
        'invalid_count': invalid,
        'invalid_rate': invalid / len(rows),
        'accuracy_including_invalid_as_wrong': accuracy_score(y_true, y_pred),
        'precision_macro_including_invalid_as_wrong': precision_score(y_true, y_pred, labels=[0, 1], average='macro', zero_division=0),
        'recall_macro_including_invalid_as_wrong': recall_score(y_true, y_pred, labels=[0, 1], average='macro', zero_division=0),
        'f1_macro_including_invalid_as_wrong': f1_score(y_true, y_pred, labels=[0, 1], average='macro', zero_division=0),
        'confusion_matrix_valid_0_1': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        'latency_count': len(valid_latencies),
        'mean_latency_ms': mean(valid_latencies) if valid_latencies else None,
        'median_latency_ms': median(valid_latencies) if valid_latencies else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if invalid:
        print('警告：混淆矩阵仅包含 0/1 预测；无效预测需在报告中单独解释。')


if __name__ == '__main__':
    main()
