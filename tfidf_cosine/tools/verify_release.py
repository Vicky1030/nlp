"""Capture real commands and check reproduction against an external formal package."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from check_processed import check

ROOT = Path(__file__).resolve().parents[1]


def read_rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, required=True,
                   help='External complete formal package containing historical artifacts')
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    out = a.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    runs = []

    def execute(name, args, success=True):
        cmd = [sys.executable, '-B', '-X', 'utf8', *map(str, args)]
        start = dt.datetime.now().astimezone().isoformat()
        result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True)
        end = dt.datetime.now().astimezone().isoformat()
        (out / (name + '.stdout.txt')).write_bytes(result.stdout)
        (out / (name + '.stderr.txt')).write_bytes(result.stderr)
        runs.append(dict(name=name, command=cmd, interpreter=sys.executable,
                         cwd=str(ROOT), started=start, ended=end, exit_code=result.returncode,
                         stdout=name + '.stdout.txt', stderr=name + '.stderr.txt'))
        (out / 'commands.json').write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding='utf-8')
        if (result.returncode == 0) != success:
            raise RuntimeError(f'{name}: unexpected exit {result.returncode}; see captured logs')
        print(f'{name}: exit={result.returncode}', flush=True)
        return result

    data = check(a.data_dir)
    execute('declared_dependencies', ['-m', 'pip', '--cache-dir', out / 'pip-cache',
                                      'install', '-r', ROOT / 'requirements.txt'])
    execute('pip_check', ['-m', 'pip', 'check'])
    execute('pip_freeze', ['-m', 'pip', 'freeze'])
    full_args = [ROOT / 'run_tfidf_cosine.py', '--data-dir', a.data_dir.resolve(),
                 '--output-root', out / 'full']
    execute('full', full_args)
    frozen_args = [ROOT / 'src/run_tfidf_stage3.py', '--config',
                   ROOT / 'artifacts/tfidf_cosine_full_run_final/stage2/artifacts/tfidf_cosine/selected_config.json',
                   '--evaluation', a.data_dir.resolve() / 'evaluation.jsonl',
                   '--output-root', out / 'quick']
    execute('quick', frozen_args)
    def hashes():
        return {str(f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest()
                for directory in ('full', 'quick') for f in (out / directory).rglob('*') if f.is_file()}
    before = hashes()
    execute('full_overwrite_refused', full_args, False)
    execute('quick_overwrite_refused', frozen_args, False)
    if before != hashes():
        raise RuntimeError('Overwrite refusal modified existing outputs')
    execute('missing_data_refused', [ROOT / 'run_tfidf_cosine.py', '--data-dir', out / 'missing',
                                    '--output-root', out / 'must_not_exist'], False)
    quick_out = out / 'quick_missing'
    missing_quick = [ROOT / 'src/run_tfidf_stage3.py', '--config',
                     ROOT / 'artifacts/tfidf_cosine_full_run_final/stage2/artifacts/tfidf_cosine/selected_config.json',
                     '--evaluation', out / 'missing' / 'evaluation.jsonl', '--output-root', quick_out]
    execute('missing_evaluation_refused', missing_quick, False)
    if quick_out.exists():
        raise RuntimeError('Missing-data quick inference created output')
    config = json.loads((out / 'full/stage2/artifacts/tfidf_cosine/selected_config.json').read_text(encoding='utf-8'))
    with (out / 'full/stage2/results/tfidf_cosine/threshold_comparison.csv').open(encoding='utf-8', newline='') as f:
        comparisons = list(csv.DictReader(f))
    assert len(comparisons) == 306
    assert all(sum(r['scheme'] == s for r in comparisons) == 102 for s in ('char_1','char_1_2','word_1'))
    assert (config['method'], config['representation'], config['threshold']) == ('tfidf_cosine','char_1_2',0.28)
    historical = a.reference_root.resolve() / 'artifacts/tfidf_cosine_full_run_final/stage3/results/tfidf_cosine'
    expected_preds = read_rows(historical / 'evaluation_predictions.jsonl')
    expected_scores = read_rows(historical / 'evaluation_similarity.jsonl')
    differences = {}
    for name, root in [('full', out / 'full/stage3'), ('quick', out / 'quick')]:
        results = root / 'results/tfidf_cosine'
        preds = read_rows(results / 'evaluation_predictions.jsonl')
        scores = read_rows(results / 'evaluation_similarity.jsonl')
        assert len(preds) == len(scores) == len(expected_preds) == len(expected_scores) == 4316
        keys = ('id','sentence1','sentence2','label','prediction','method')
        assert [[r[k] for k in keys] for r in preds] == [[r[k] for k in keys] for r in expected_preds]
        assert [(r['id'], r['label']) for r in scores] == [(r['id'], r['label']) for r in expected_scores]
        x = np.array([r['similarity'] for r in scores]); y = np.array([r['similarity'] for r in expected_scores])
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)
        differences[name] = float(np.max(np.abs(x-y)))
        report = json.loads((root / 'reports/tfidf_cosine_stage3.json').read_text(encoding='utf-8'))
        assert [report['metrics'][k] for k in ('TN','FP','FN','TP')] == [1994,984,729,609]
    summary = dict(status='passed', data=data, comparison_rows=len(comparisons),
                   selected={k:config[k] for k in ('method','representation','threshold')},
                   predictions=4316, labels_identical_to_formal=True,
                   similarity_rtol=1e-12, similarity_atol=1e-12, max_similarity_difference=differences,
                   metrics=report['metrics'], overwrite_refusal_preserved_outputs=True,
                   command_count=len(runs), python=sys.version)
    summary['full_output'] = str(out / 'full')
    summary['quick_output'] = str(out / 'quick')
    summary['commands_log'] = str(out / 'commands.json')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
