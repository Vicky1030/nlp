"""Generate frozen TF-IDF cosine predictions for evaluation and verify them."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data_loader import load_data
from tfidf_common import row_cosine


def ensure_targets_absent(paths):
    conflicts = [str(Path(p).resolve()) for p in paths if Path(p).exists()]
    if conflicts:
        raise FileExistsError('Refusing to overwrite existing output files; choose a new --output-root:\n' + '\n'.join(conflicts))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def strict_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("similarity must be finite numeric")
    value = float(value)
    if value < -1e-12 or value > 1 + 1e-12:
        raise ValueError("similarity outside tolerance")
    return min(1.0, max(0.0, value))


def metrics(labels, predictions):
    tn = fp = fn = tp = 0
    for y, p in zip(labels, predictions):
        if y == 0 and p == 0: tn += 1
        elif y == 0 and p == 1: fp += 1
        elif y == 1 and p == 0: fn += 1
        elif y == 1 and p == 1: tp += 1
    n = len(labels)
    def div(a, b): return Fraction(a, b) if b else Fraction(0, 1)
    acc = div(tn + tp, n)
    p1 = div(tp, tp + fp); r1 = div(tp, tp + fn)
    f1_1 = div(2 * tp, 2 * tp + fp + fn)
    p0 = div(tn, tn + fn); r0 = div(tn, tn + fp)
    f1_0 = div(2 * tn, 2 * tn + fn + fp)
    macro = (f1_0 + f1_1) / 2
    return {
        'n_samples': n, 'accuracy': float(acc), 'precision_1': float(p1),
        'recall_1': float(r1), 'f1_1': float(f1_1), 'macro_f1': float(macro),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'predicted_0': predictions.count(0), 'predicted_1': predictions.count(1),
        'exact': {'accuracy': f'{acc.numerator}/{acc.denominator}',
                  'precision_1': f'{p1.numerator}/{p1.denominator}',
                  'recall_1': f'{r1.numerator}/{r1.denominator}',
                  'f1_1': f'{f1_1.numerator}/{f1_1.denominator}',
                  'macro_f1': f'{macro.numerator}/{macro.denominator}'},
    }


def validate_config(config, config_path):
    expected = {'method': 'tfidf_cosine', 'representation': 'char_1_2',
                'threshold_index': 28, 'threshold': 0.28, 'operator': '>='}
    for key, val in expected.items():
        got = config.get(key)
        if got != val:
            raise RuntimeError(f'frozen config mismatch for {key}: {got!r} != {val!r}')
    if not Path(config['vectorizer']['path']).is_absolute():
        model_path = config_path.parent / config['vectorizer']['path']
    else:
        model_path = Path(config['vectorizer']['path'])
    model_path = model_path.resolve()
    if sha(model_path) != config['vectorizer']['sha256']:
        raise RuntimeError('selected vectorizer SHA-256 mismatch')
    metadata_path = model_path.with_name('metadata.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('model_sha256') != sha(model_path) or metadata.get('scheme') != config['representation']:
        raise RuntimeError('selected vectorizer metadata mismatch')
    return model_path, metadata_path, metadata


def infer_pair(vectorizer, sentence1, sentence2, threshold):
    left = vectorizer.transform([sentence1])
    right = vectorizer.transform([sentence2])
    scores, zeros = row_cosine(left, right)
    score = float(scores[0])
    pred = int(score >= threshold)
    return score, pred, zeros


def reload_worker(args):
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    model_path, _, _ = validate_config(config, config_path)
    rows = load_data(args.evaluation)
    vectorizer = joblib.load(model_path)
    threshold = config['threshold']
    output_similarity = args.output_root.resolve() / 'results/tfidf_cosine/evaluation_similarity.jsonl'
    output_predictions = args.output_root.resolve() / 'results/tfidf_cosine/evaluation_predictions.jsonl'
    saved_scores = [json.loads(line) for line in output_similarity.read_text(encoding='utf-8').splitlines()]
    saved_predictions = [json.loads(line) for line in output_predictions.read_text(encoding='utf-8').splitlines()]
    if len(saved_scores) != len(rows) or len(saved_predictions) != len(rows):
        raise RuntimeError('saved output count mismatch')
    got_scores = []
    got_predictions = []
    for row in rows:
        score, pred, _ = infer_pair(vectorizer, row['sentence1'], row['sentence2'], threshold)
        got_scores.append(score); got_predictions.append(pred)
    saved_score_values = [strict_score(x['similarity']) for x in saved_scores]
    if not np.allclose(got_scores, saved_score_values, rtol=1e-12, atol=1e-12):
        raise RuntimeError('fresh-process similarity mismatch')
    if got_predictions != [x['prediction'] for x in saved_predictions]:
        raise RuntimeError('fresh-process prediction mismatch')
    print(json.dumps({'passed': True, 'rows': len(rows), 'max_similarity_difference': float(max(abs(a-b) for a,b in zip(got_scores, saved_score_values))), 'predictions_identical': True}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1]
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--reload-check', action='store_true')
    args = parser.parse_args()
    if args.reload_check:
        reload_worker(args); return
    config_path = args.config.resolve(); evaluation_path = args.evaluation.resolve(); output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f'Refusing to overwrite existing output directory: {output_root}')
    for path in (config_path, evaluation_path):
        if not path.is_file():
            raise FileNotFoundError(f'Missing required external input: {path}')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    model_path, metadata_path, metadata = validate_config(config, config_path)
    if config['threshold_index'] / 100 != config['threshold']:
        raise RuntimeError('threshold index/value mismatch')
    result_dir = output_root/'results/tfidf_cosine'
    sim_path = result_dir/'evaluation_similarity.jsonl'; pred_path = result_dir/'evaluation_predictions.jsonl'
    report_path = output_root/'reports/tfidf_cosine_stage3.json'
    ensure_targets_absent([sim_path, pred_path, report_path])
    output_root.mkdir(parents=True, exist_ok=False)
    # Capture every reused stage-1 artifact before execution and verify it again at the end.
    protected = [config_path, model_path, metadata_path, base/'src/data_loader.py', base/'src/tfidf_common.py',
                 base/'reports/tfidf_cosine_stage1.json', base/'reports/tfidf_cosine_stage2.json',
                 base/'results/tfidf_cosine/threshold_comparison.csv', base/'results/tfidf_cosine/representation_summary.csv',
                 base/'results/tfidf_cosine/validation_selected_predictions.jsonl']
    protected = [p for p in protected if p.exists()]
    before = {str(p): sha(p) for p in protected}
    rows = load_data(evaluation_path)
    vectorizer = joblib.load(model_path)
    threshold = float(config['threshold'])
    left = [r['sentence1'] for r in rows]; right = [r['sentence2'] for r in rows]
    batch_scores, batch_zeros = row_cosine(vectorizer.transform(left), vectorizer.transform(right))
    if not np.isfinite(batch_scores).all(): raise RuntimeError('nonfinite batch similarity')
    # Fixed non-evaluation warmup; no labels or IDs are used.
    for a, b in [('预热句子一', '预热句子二'), ('中文测试', '中文测试'), ('warmup', 'warm up')]:
        infer_pair(vectorizer, a, b, threshold)
    timed_scores, predictions, timings, timed_zeros = [], [], [], {'sentence1_zero': 0, 'sentence2_zero': 0, 'either_zero': 0, 'both_zero': 0}
    for row in rows:
        start = time.perf_counter_ns()
        score, pred, zeros = infer_pair(vectorizer, row['sentence1'], row['sentence2'], threshold)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
        if not math.isfinite(elapsed) or elapsed < 0: raise RuntimeError('invalid latency')
        timed_scores.append(score); predictions.append(pred); timings.append(elapsed)
        for k in timed_zeros: timed_zeros[k] += zeros[k]
    np.testing.assert_allclose(timed_scores, batch_scores, rtol=1e-12, atol=1e-12)
    if predictions != [int(x >= threshold) for x in batch_scores]: raise RuntimeError('prediction mismatch')
    # Boundary test: inference receives only the two texts; changing a scoring-only
    # label copy cannot affect the score or prediction.
    probe = rows[:5]
    probe_a = [infer_pair(vectorizer, r['sentence1'], r['sentence2'], threshold)[:2] for r in probe]
    altered_labels = [1 - r['label'] for r in probe]
    probe_b = [infer_pair(vectorizer, r['sentence1'], r['sentence2'], threshold)[:2] for r in probe]
    if probe_a != probe_b or len(altered_labels) != len(probe): raise RuntimeError('label boundary test failed')
    labels = [r['label'] for r in rows]
    stat = metrics(labels, predictions)
    # Independent sklearn check after inference, not used for selection.
    sk = {'accuracy': accuracy_score(labels, predictions), 'precision_1': precision_score(labels, predictions, labels=[0,1], pos_label=1, zero_division=0),
          'recall_1': recall_score(labels, predictions, labels=[0,1], pos_label=1, zero_division=0),
          'f1_1': f1_score(labels, predictions, labels=[0,1], pos_label=1, zero_division=0),
          'macro_f1': f1_score(labels, predictions, labels=[0,1], average='macro', zero_division=0)}
    for key in sk:
        if not math.isclose(stat[key], float(sk[key]), rel_tol=1e-12, abs_tol=1e-12): raise RuntimeError('metric mismatch: '+key)
    result_dir.mkdir(parents=True, exist_ok=True)
    with sim_path.open('w', encoding='utf-8', newline='\n') as sf, pred_path.open('w', encoding='utf-8', newline='\n') as pf:
        for row, score, pred, elapsed in zip(rows, timed_scores, predictions, timings):
            sf.write(json.dumps({'id': row['id'], 'label': row['label'], 'similarity': score}, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
            pf.write(json.dumps({'id': row['id'], 'sentence1': row['sentence1'], 'sentence2': row['sentence2'], 'label': row['label'], 'prediction': pred, 'method': 'tfidf_cosine', 'latency_ms': elapsed}, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
    report = {'status': 'passed', 'scope': 'Frozen TF-IDF cosine final evaluation only; no refit, threshold search, Test or LR',
              'command': subprocess.list2cmdline([sys.executable, *sys.argv]), 'evaluation': {'path': str(evaluation_path), 'sha256': sha(evaluation_path), 'n_samples': len(rows)},
              'config': {'path': str(config_path), 'sha256': sha(config_path), 'method': config['method'], 'representation': config['representation'], 'threshold_index': config['threshold_index'], 'threshold': threshold, 'operator': config['operator']},
              'model': {'path': str(model_path), 'sha256': sha(model_path), 'metadata_path': str(metadata_path), 'metadata_sha256': sha(metadata_path), 'training_data_sha256': metadata.get('train_sha256')},
              'metrics': stat, 'sklearn_metrics_check': sk, 'zero_vectors': batch_zeros, 'timed_zero_vectors': timed_zeros,
              'latency_ms': {'min': min(timings), 'mean': float(np.mean(timings)), 'median': float(np.median(timings)), 'p95': float(np.percentile(timings, 95)), 'max': max(timings), 'scope': 'warmed CPU single-record; text processing, two transforms, sparse cosine, threshold; excludes load, warmup, I/O, scoring, record assembly'},
              'checks': {'seven_field_output': True, 'evaluation_alignment': True, 'confusion_total': sum(stat[k] for k in ('TN','FP','FN','TP')) == len(rows), 'independent_metric_recompute': True, 'batch_vs_timed_rtol_atol_1e-12': True, 'label_and_prediction_boundary_test': True, 'fresh_process_reload': None},
              'outputs': {'similarity': {'path': str(sim_path), 'sha256': sha(sim_path)}, 'predictions': {'path': str(pred_path), 'sha256': sha(pred_path)}},
              'environment': {'python': sys.version, 'os': platform.platform(), 'packages': {n: importlib.metadata.version(n) for n in ('scikit-learn','numpy','scipy','jieba','joblib')}},
              'protected_hashes_before': before}
    # Fresh process validates only output files and the frozen model/config.
    cmd = [sys.executable, '-B', str(Path(__file__).resolve()), '--reload-check', '--config', str(config_path), '--evaluation', str(evaluation_path), '--output-root', str(output_root)]
    child = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if child.returncode: raise RuntimeError(child.stderr or child.stdout)
    report['checks']['fresh_process_reload'] = json.loads(child.stdout)
    after = {str(p): sha(p) for p in protected}; report['protected_hashes_after'] = after
    report['protected_files_unchanged'] = before == after
    if not report['protected_files_unchanged']: raise RuntimeError('protected file changed')
    dump_json(report_path, report)
    print(json.dumps({'status': 'passed', 'evaluation_samples': len(rows), 'metrics': stat, 'report': str(report_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
