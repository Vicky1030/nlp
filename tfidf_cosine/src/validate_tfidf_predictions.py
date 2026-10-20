"""Strictly validate a seven-field TF-IDF prediction JSONL against evaluation."""
import argparse
import json
import math
from pathlib import Path

FIELDS = ('id', 'sentence1', 'sentence2', 'label', 'prediction', 'method', 'latency_ms')


def load_json(path, fields):
    rows = []
    with Path(path).open('rb') as stream:
        for line_no, raw in enumerate(stream, 1):
            if line_no == 1 and raw.startswith(b'\xef\xbb\xbf'):
                raise ValueError(f'{path}:{line_no}: UTF-8 BOM is not allowed')
            try:
                obj = json.loads(raw.decode('utf-8'))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f'{path}:{line_no}: invalid JSON: {exc}') from exc
            if type(obj) is not dict or set(obj) != set(fields):
                raise ValueError(f'{path}:{line_no}: expected exactly fields {fields}')
            if type(obj['id']) is not int or isinstance(obj['id'], bool):
                raise ValueError(f'{path}:{line_no}: id must be integer, not bool')
            if type(obj['label']) is not int or isinstance(obj['label'], bool) or obj['label'] not in (0, 1):
                raise ValueError(f'{path}:{line_no}: label must be integer 0/1, not bool')
            if 'prediction' in obj and (type(obj['prediction']) is not int or isinstance(obj['prediction'], bool) or obj['prediction'] not in (0, 1)):
                raise ValueError(f'{path}:{line_no}: prediction must be integer 0/1, not bool')
            for field in ('sentence1', 'sentence2'):
                if type(obj[field]) is not str or not obj[field].strip():
                    raise ValueError(f'{path}:{line_no}: {field} must be non-whitespace string')
            if 'method' in obj and obj['method'] != 'tfidf_cosine':
                raise ValueError(f'{path}:{line_no}: method must be tfidf_cosine')
            if 'latency_ms' in obj and (isinstance(obj['latency_ms'], bool) or not isinstance(obj['latency_ms'], (int, float)) or not math.isfinite(float(obj['latency_ms'])) or obj['latency_ms'] < 0):
                raise ValueError(f'{path}:{line_no}: latency_ms must be finite and non-negative')
            rows.append(obj)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    args = p.parse_args()
    evaluation = load_json(args.evaluation, ('id', 'sentence1', 'sentence2', 'label'))
    predictions = load_json(args.predictions, FIELDS)
    if len(evaluation) != len(predictions):
        raise ValueError(f'count mismatch: evaluation={len(evaluation)}, predictions={len(predictions)}')
    for line_no, (source, result) in enumerate(zip(evaluation, predictions), 1):
        for field in ('id', 'sentence1', 'sentence2', 'label'):
            if source[field] != result[field]:
                raise ValueError(f'{args.predictions}:{line_no}: {field} differs from evaluation')
    if len({row['id'] for row in predictions}) != len(predictions):
        raise ValueError(f'{args.predictions}: duplicate prediction id')
    print(json.dumps({'passed': True, 'rows': len(predictions), 'strict_fields': list(FIELDS), 'aligned_to_evaluation': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
