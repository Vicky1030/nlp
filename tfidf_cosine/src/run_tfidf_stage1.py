"""Fit Train only, then transform validation; no thresholds/classification."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import importlib.metadata
import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from data_loader import load_data
from tfidf_common import SCHEMES, dictionary_info, make_vectorizer, row_cosine, transform_pairs

BASE = Path(__file__).resolve().parents[1]


def ensure_targets_absent(paths):
    conflicts = [str(Path(p).resolve()) for p in paths if Path(p).exists()]
    if conflicts:
        raise FileExistsError('Refusing to overwrite existing output files; choose a new --output-root:\n' + '\n'.join(conflicts))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8', newline='\n')


def params_json(v):
    result = {}
    for k, x in v.get_params(deep=False).items():
        if k == 'dtype':
            x = 'numpy.float64'
        elif callable(x):
            x = x.__module__ + '.' + x.__qualname__
        elif isinstance(x, tuple):
            x = list(x)
        result[k] = x
    return result


def state(v):
    return dict(v.vocabulary_), v.idf_.copy()


def unchanged(v, snapshot):
    return v.vocabulary_ == snapshot[0] and np.array_equal(v.idf_, snapshot[1])


def train_only_fit(train_texts):
    fitted = {}
    for scheme in SCHEMES:
        vectorizer = make_vectorizer(scheme)
        vectorizer.fit(train_texts)
        fitted[scheme] = vectorizer
    return fitted


def check_cosine(v, left, right, scores):
    indices = sorted({0, 1, 2, len(left)//2, len(left)-1})
    errors = []
    for i in indices:
        standard = float(cosine_similarity(v.transform([left[i]]), v.transform([right[i]]))[0, 0])
        errors.append(abs(standard-scores[i]))
    # explicit sparse zero/nonzero combinations, including either side and both
    a = csr_matrix([[0., 0.], [1., 0.], [0., 0.], [1., 0.]])
    b = csr_matrix([[1., 0.], [0., 0.], [0., 0.], [1., 0.]])
    got, zeros = row_cosine(a, b)
    ref = np.array([cosine_similarity(a[i], b[i])[0, 0] for i in range(4)])
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
    assert zeros['either_zero'] == 3 and zeros['both_zero'] == 1
    assert max(errors) <= 1e-12
    # Fail, rather than hide, excessive or nonfinite values.
    for value in (2.0, float('nan')):
        try:
            row_cosine(csr_matrix([[value]]), csr_matrix([[1.0]]))
        except ValueError:
            pass
        else:
            raise AssertionError('bad numeric value accepted')
    return {'fixed_indices_0based': indices, 'max_absolute_error': float(max(errors)),
            'explicit_zero_combinations_passed': True, 'out_of_range_and_nonfinite_rejected': True}


def synthetic_check(scheme):
    v = make_vectorizer(scheme)
    v.fit(['甲甲共同 trainonly', '乙乙共同 trainonly'])
    snapshot = state(v)
    validation_text = '龘龘验证 exclusivevalidation'
    heldout = set(v.build_analyzer()(validation_text)) - set(v.vocabulary_)
    assert heldout
    v.transform([validation_text])
    assert unchanged(v, snapshot) and not heldout.intersection(v.vocabulary_)
    return {'validation_only_features': sorted(heldout), 'vocabulary_and_idf_unchanged': True}


def reload_worker(args):
    v = joblib.load(args.model)
    rows = load_data(args.validation)
    scores, zeros = transform_pairs(v, [r['sentence1'] for r in rows], [r['sentence2'] for r in rows])
    saved = [json.loads(s) for s in args.scores.read_text(encoding='utf8').splitlines()]
    assert [(r['id'], r['label']) for r in rows] == [(r['id'], r['label']) for r in saved]
    expected = np.array([r['similarity'] for r in saved])
    np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-12)
    print(json.dumps({'passed': True, 'max_absolute_error': float(np.max(np.abs(scores-expected))),
                      'rows': len(rows), 'tokenizer': params_json(v)['tokenizer'], 'zero_vectors': zeros}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--reload-check', action='store_true')
    p.add_argument('--model', type=Path)
    p.add_argument('--scores', type=Path)
    args = p.parse_args()
    if args.reload_check:
        reload_worker(args)
        return
    args.train = args.train.resolve(); args.validation = args.validation.resolve()
    dest = args.output_root.resolve()
    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=False)
    targets = []
    for scheme in SCHEMES:
        targets += [dest/'artifacts/tfidf_cosine'/scheme/'vectorizer.joblib',
                    dest/'artifacts/tfidf_cosine'/scheme/'metadata.json',
                    dest/'results/tfidf_cosine'/f'validation_similarity_{scheme}.jsonl']
    targets += [dest/'reports/tfidf_cosine_stage1.json',
                dest/'artifacts/tfidf_cosine/requirements.txt',
                dest/'artifacts/tfidf_cosine/environment_freeze.txt']
    ensure_targets_absent(targets)
    # Explicit allowlist: never enumerate/hash other dataset files.
    protected = [args.train, args.validation, Path(__file__).with_name('data_loader.py')]
    optional_docs = [BASE/'README.md', BASE/'TFIDF_STAGE1.md']
    protected.extend(path for path in optional_docs if path.is_file())
    before = {str(f): sha(f) for f in dict.fromkeys(protected)}
    environment = {'python': sys.version, 'executable': sys.executable, 'os': platform.platform(),
                   'packages': {n: importlib.metadata.version(n) for n in ('scikit-learn','numpy','scipy','jieba','joblib')}}
    code_hashes = {n: sha(Path(__file__).with_name(n)) for n in ('tfidf_common.py','run_tfidf_stage1.py','data_loader.py')}
    dictionary = dictionary_info()
    train_rows = load_data(args.train)
    documents = [r[field] for r in train_rows for field in ('sentence1','sentence2')]
    assert len(documents) == 2*len(train_rows)
    # Fit path completes before validation is read; only ordered sentence strings enter fit.
    fitted = train_only_fit(documents)
    validation = load_data(args.validation)
    left = [r['sentence1'] for r in validation]; right = [r['sentence2'] for r in validation]
    assert validation and len({r['id'] for r in validation}) == len(validation)
    report = {'scope': 'TF-IDF stage1 only; no thresholds, predictions, classification metrics or evaluation/Test access',
              'command': subprocess.list2cmdline([sys.executable, *sys.argv]), 'environment': environment,
              'train_pairs': len(train_rows), 'fit_documents': len(documents), 'validation_pairs': len(validation),
              'training_document_order': 'file row order, sentence1 then sentence2; repeated texts retained',
              'inputs': {'train': {'path': str(args.train), 'sha256': sha(args.train)},
                         'validation': {'path': str(args.validation), 'sha256': sha(args.validation)}},
              'dictionary': dictionary, 'code_sha256': code_hashes, 'scheme_order': list(SCHEMES), 'schemes': {}}
    for scheme in SCHEMES:
        v = fitted[scheme]; snapshot = state(v)
        scores, zeros = transform_pairs(v, left, right)
        assert unchanged(v, snapshot)
        checks = check_cosine(v, left, right, scores)
        synthetic = synthetic_check(scheme)
        folder = dest/'artifacts/tfidf_cosine'/scheme; folder.mkdir(parents=True, exist_ok=True)
        model = folder/'vectorizer.joblib'; joblib.dump(v, model)
        output = dest/'results/tfidf_cosine'/f'validation_similarity_{scheme}.jsonl'
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', encoding='utf8', newline='\n') as f:
            for row, value in zip(validation, scores):
                f.write(json.dumps({'id': row['id'], 'label': row['label'], 'similarity': float(value)}, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
        reread = [json.loads(s) for s in output.read_text(encoding='utf8').splitlines()]
        assert len(reread) == len(validation)
        assert all(set(r)=={'id','label','similarity'} for r in reread)
        assert [(r['id'],r['label']) for r in reread] == [(r['id'],r['label']) for r in validation]
        assert np.array_equal(np.array([r['similarity'] for r in reread]), scores)
        cmd = [sys.executable, '-B', str(Path(__file__).resolve()), '--reload-check', '--model', str(model), '--validation', str(args.validation), '--scores', str(output)]
        worker = subprocess.run(cmd, capture_output=True, text=True, encoding='utf8')
        if worker.returncode:
            raise RuntimeError(worker.stderr)
        reloaded = json.loads(worker.stdout)
        metadata = {'scheme': scheme, 'parameters': params_json(v), 'train_pairs': len(train_rows),
                    'fit_documents': len(documents), 'vocabulary_size': len(v.vocabulary_), 'train_sha256': sha(args.train),
                    'environment': environment, 'tokenization': dictionary if scheme=='word_1' else {'analyzer':'sklearn char',
                    'behavior':'decode and configured preprocessing (lowercase=False, strip_accents=None), then sklearn collapses consecutive whitespace (2+) to a single ASCII space before character n-grams; no extra normalization'},
                    'code_sha256': code_hashes, 'model_sha256': sha(model)}
        write_json(folder/'metadata.json', metadata)
        report['schemes'][scheme] = {'vocabulary_size': len(v.vocabulary_), 'zero_vectors': zeros,
            'min_similarity': float(scores.min()), 'max_similarity': float(scores.max()), 'all_finite': bool(np.isfinite(scores).all()),
            'standard_cosine_check': checks, 'synthetic_leakage_check': synthetic, 'validation_state_unchanged': unchanged(v,snapshot),
            'reload_process': {'command': subprocess.list2cmdline(cmd), 'exit_code': worker.returncode, 'result': reloaded},
            'files': {str(model): sha(model), str(output): sha(output), str(folder/'metadata.json'): sha(folder/'metadata.json')}}
        print(scheme, 'vocabulary=', len(v.vocabulary_), 'zeros=', zeros, flush=True)
    # Record sklearn's inspected implementation and a behavioral probe, not a raw-Unicode claim.
    source = inspect.getsource(TfidfVectorizer._char_ngrams)
    char_probe = fitted['char_1'].build_analyzer()('A  \tB')
    assert char_probe == ['A', ' ', 'B']
    report['sklearn_char_behavior'] = {'probe_input': 'A  \tB', 'char_1_tokens': char_probe,
                                     'implementation_source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'source': source}
    after = {str(f): sha(f) for f in dict.fromkeys(protected)}
    assert before == after, 'reused files changed'
    report['protected_hashes_before'] = before; report['protected_hashes_after'] = after
    report['protected_files_unchanged'] = True; report['status'] = 'passed'
    write_json(dest/'reports/tfidf_cosine_stage1.json', report)
    requirements = dest/'artifacts/tfidf_cosine/requirements.txt'
    requirements.write_text(''.join(n+'=='+v+'\n' for n,v in environment['packages'].items()), encoding='utf8', newline='\n')
    frozen = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (requirements.parent/'environment_freeze.txt').write_text(frozen.stdout, encoding='utf8', newline='\n')
    print('Completed all three schemes; report:', dest/'reports/tfidf_cosine_stage1.json')


if __name__ == '__main__':
    main()
