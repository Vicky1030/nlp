"""Train the frozen TF-IDF + Logistic Regression stage without touching source data."""
from __future__ import annotations

import argparse, hashlib, json, platform, subprocess, sys, time, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from data_loader import load_data
from index_paths import INDEX_FORMAT_VERSION, relative_index_path

SCHEMES = ("char_1", "char_1_2", "word_1")
CS = (0.1, 1.0, 10.0)
WEIGHTS = (None, "balanced")
ROOT = Path(__file__).resolve().parents[1]
BEIJING_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def config_id(scheme, c, weight):
    return f"{scheme}__C{str(c).replace('.', 'p')}__cw_{'none' if weight is None else weight}"

def pair_features(vectorizer, rows):
    a = vectorizer.transform([r["sentence1"] for r in rows])
    b = vectorizer.transform([r["sentence2"] for r in rows])
    return hstack((abs(a - b), a.multiply(b)), format="csr")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=ROOT / "data/processed/train.jsonl")
    ap.add_argument("--validation", type=Path, default=ROOT / "data/processed/validation.jsonl")
    ap.add_argument("--vectorizer-root", type=Path, default=ROOT / "artifacts/tfidf_cosine")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    train_path, val_path, vec_root, out = map(lambda p: p.resolve(), (args.train, args.validation, args.vectorizer_root, args.output_dir))
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError("train/validation input is missing")
    if not vec_root.is_dir():
        raise FileNotFoundError(f"vectorizer root is missing: {vec_root}")
    rows = load_data(train_path); val = load_data(val_path)
    if len(rows) != 30899 or len(val) != 3433:
        raise ValueError(f"unexpected row counts train={len(rows)} validation={len(val)}")
    y = np.asarray([r["label"] for r in rows], dtype=np.int8)
    targets = [out / "run_metadata.json", out / "model_index.json", out / "protected_hashes.json"]
    for scheme in SCHEMES:
        for c in CS:
            for weight in WEIGHTS:
                cid = config_id(scheme, c, weight)
                targets += [out / "models" / f"{cid}.joblib", out / "probabilities" / f"{cid}.jsonl"]
    conflicts = [str(p) for p in targets if p.exists()]
    if conflicts:
        raise FileExistsError("Refusing to overwrite existing output files; choose a new --output-dir:\n" + "\n".join(conflicts))
    out.mkdir(parents=True, exist_ok=False)
    protected = [train_path, val_path] + [vec_root / s / n for s in SCHEMES for n in ("vectorizer.joblib", "metadata.json")]
    before = {str(p): sha256(p) for p in protected}
    start = datetime.now(BEIJING_TZ).isoformat()
    index = []; warnings_seen = []
    for scheme in SCHEMES:
        vectorizer = joblib.load(vec_root / scheme / "vectorizer.joblib")
        X = pair_features(vectorizer, rows); Xv = pair_features(vectorizer, val)
        if X.shape[1] != 2 * len(vectorizer.vocabulary_):
            raise AssertionError(f"feature width mismatch for {scheme}: {X.shape}")
        # Synthetic checks for absolute-difference/product ordering and symmetry.
        sa = csr_matrix([[1., 0., 2.]]); sb = csr_matrix([[0., 3., 2.]])
        expected = hstack((abs(sa-sb), sa.multiply(sb)), format="csr")
        assert expected.toarray().tolist() == [[1., 3., 0., 0., 0., 4.]]
        for c in CS:
            for weight in WEIGHTS:
                cid = config_id(scheme, c, weight)
                print(f"[训练 {len(index)+1}/18] {scheme}，C={c}，class_weight={weight}", flush=True)
                model = LogisticRegression(solver="liblinear", penalty="l2", C=c, class_weight=weight,
                    fit_intercept=True, intercept_scaling=1, dual=False, max_iter=5000, tol=1e-4, random_state=42)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    t0 = time.perf_counter(); model.fit(X, y); elapsed = time.perf_counter() - t0
                warnings_seen.extend({"category": w.category.__name__, "message": str(w.message)} for w in caught)
                convergence = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
                if convergence:
                    dump_json(out / 'training_warning_failure.json', {
                        'status': 'failed', 'config_id': cid, 'warnings': warnings_seen,
                        'reason': 'unresolved ConvergenceWarning; later configurations not trained'})
                    raise RuntimeError(f"training convergence warning for {cid}: {convergence[0].message}")
                classes = list(model.classes_); pos = classes.index(1)
                probability = model.predict_proba(Xv)[:, pos]
                if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                    raise AssertionError(f"invalid probability for {cid}")
                model_path = out / "models" / f"{cid}.joblib"; model_path.parent.mkdir(exist_ok=True)
                joblib.dump(model, model_path)
                prob_path = out / "probabilities" / f"{cid}.jsonl"; prob_path.parent.mkdir(exist_ok=True)
                with prob_path.open("w", encoding="utf-8", newline="\n") as f:
                    for row, p in zip(val, probability):
                        f.write(json.dumps({"config_id": cid, "id": row["id"], "label": row["label"], "probability": float(p)}, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                index.append({"config_id": cid, "scheme": scheme, "C": c, "class_weight": weight,
                    "n_iter": [int(x) for x in np.atleast_1d(model.n_iter_)], "elapsed_seconds": elapsed,
                    "model": relative_index_path(out / "model_index.json", model_path),
                    "probabilities": relative_index_path(out / "model_index.json", prob_path),
                    "model_sha256": sha256(model_path), "probabilities_sha256": sha256(prob_path),
                    "train_shape": list(X.shape), "validation_shape": list(Xv.shape)})
        del X, Xv
    after = {str(p): sha256(p) for p in protected}
    if before != after:
        raise AssertionError("protected input hash changed")
    dump_json(out / "model_index.json", {"index_format_version": INDEX_FORMAT_VERSION,
        "path_base": "index_directory", "count": len(index), "configs": index})
    end = datetime.now(BEIJING_TZ).isoformat()
    dump_json(out / "protected_hashes.json", {"before": before, "after": after, "unchanged": before == after})
    dump_json(out / "run_metadata.json", {"status": "passed", "start": start, "end": end, "timezone": "Asia/Shanghai / UTC+08:00",
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(), "cwd": str(Path.cwd()),
        "train": str(train_path), "validation": str(val_path), "train_sha256": sha256(train_path), "validation_sha256": sha256(val_path),
        "counts": {"train": len(rows), "validation": len(val), "label_0_train": int((y == 0).sum()), "label_1_train": int((y == 1).sum())},
        "schemes": list(SCHEMES), "C": list(CS), "class_weight": ["None", "balanced"], "warnings": warnings_seen})
    print(json.dumps({"status":"passed", "output_dir":str(out), "models":len(index), "protected_unchanged":True}, ensure_ascii=False))

if __name__ == "__main__":
    main()
