"""Run TF-IDF cosine stages 1-3 in isolated output subdirectories."""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCHEMES = ("char_1", "char_1_2", "word_1")
PACKAGES = ("scikit-learn", "numpy", "scipy", "jieba", "joblib")

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def dependency_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"缺少依赖 {name}，请先按 README 安装，不会自动安装依赖") from exc
    return versions

def expected_targets(root):
    targets = [root / "logs" / "run.log", root / "logs" / "run_manifest.json",
               root / "stage1" / "TFIDF_STAGE1.md",
               root / "stage1" / "reports" / "tfidf_cosine_stage1.json",
               root / "stage1" / "artifacts" / "tfidf_cosine" / "requirements.txt",
               root / "stage1" / "artifacts" / "tfidf_cosine" / "environment_freeze.txt",
               root / "stage2" / "artifacts" / "tfidf_cosine" / "selected_config.json",
               root / "stage2" / "reports" / "tfidf_cosine_stage2.json",
               root / "stage2" / "results" / "tfidf_cosine" / "threshold_comparison.csv",
               root / "stage2" / "results" / "tfidf_cosine" / "representation_summary.csv",
               root / "stage2" / "results" / "tfidf_cosine" / "validation_selected_predictions.jsonl",
               root / "stage3" / "reports" / "tfidf_cosine_stage3.json",
               root / "stage3" / "results" / "tfidf_cosine" / "evaluation_similarity.jsonl",
               root / "stage3" / "results" / "tfidf_cosine" / "evaluation_predictions.jsonl"]
    for scheme in SCHEMES:
        targets.extend([root / "stage1" / "artifacts" / "tfidf_cosine" / scheme / "vectorizer.joblib",
                        root / "stage1" / "artifacts" / "tfidf_cosine" / scheme / "metadata.json",
                        root / "stage1" / "results" / "tfidf_cosine" / f"validation_similarity_{scheme}.jsonl"])
    return targets

def check_inputs(data_dir):
    paths = {name: (data_dir / f"{name}.jsonl").resolve() for name in ("train", "validation", "evaluation")}
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少统一数据文件:\n" + "\n".join(missing))
    return paths

def append_log(log_path, label, command, cwd, completed, started, ended):
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(json.dumps({"event": "subprocess", "label": label, "command": [str(x) for x in command],
                              "cwd": str(cwd), "started": started, "ended": ended,
                              "exit_code": completed.returncode, "stdout": completed.stdout,
                              "stderr": completed.stderr}, ensure_ascii=False) + "\n")

def run_step(label, command, cwd, log_path):
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    ended = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    append_log(log_path, label, command, cwd, completed, started, ended)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def validate_final(evaluation, predictions):
    eval_rows = [json.loads(line) for line in evaluation.read_text(encoding="utf-8").splitlines() if line]
    pred_rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line]
    if len(eval_rows) != len(pred_rows):
        raise ValueError(f"预测数量不一致: evaluation={len(eval_rows)}, predictions={len(pred_rows)}")
    for line_no, (source, result) in enumerate(zip(eval_rows, pred_rows), 1):
        if set(result) != {"id", "sentence1", "sentence2", "label", "prediction", "method", "latency_ms"}:
            raise ValueError(f"{predictions}:{line_no}: 七字段不完整")
        for key in ("id", "sentence1", "sentence2", "label"):
            if result[key] != source[key]:
                raise ValueError(f"{predictions}:{line_no}: {key} 与 evaluation 不一致")
        if type(result["prediction"]) is not int or result["prediction"] not in (0, 1):
            raise ValueError(f"{predictions}:{line_no}: prediction 不是整数 0/1")
    return len(eval_rows)

def main():
    # Windows consoles may use a legacy code page; keep UTF-8 logs intact while
    # preventing diagnostic output from failing on non-ASCII paths or text.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir.resolve()
    output = args.output_root.resolve()
    inputs = check_inputs(data_dir)
    versions = dependency_versions()
    if output.exists():
        raise SystemExit(f"拒绝覆盖已有输出目录，请指定新的 --output-root: {output}")
    targets = expected_targets(output)
    conflicts = [str(p) for p in targets if p.exists()]
    if conflicts:
        raise SystemExit("拒绝覆盖已有产物，请指定新的 --output-root:\n" + "\n".join(conflicts))
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "run.log"
    log_path.write_text("TF-IDF cosine unified run\n", encoding="utf-8", newline="\n")
    manifest = {"status": "running", "python": sys.executable, "python_version": sys.version,
                "packages": versions, "data_dir": str(data_dir),
                "inputs": {k: {"path": str(v)} for k, v in inputs.items()},
                "output_root": str(output), "stages": [], "preflight_targets": [str(p) for p in targets]}
    (logs / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        stage1, stage2, stage3 = output / "stage1", output / "stage2", output / "stage3"
        py = sys.executable
        run_step("stage1", [py, "-B", str(base / "src" / "run_tfidf_stage1.py"), "--train", str(inputs["train"]),
                             "--validation", str(inputs["validation"]), "--output-root", str(stage1)], base, log_path)
        doc = base / "TFIDF_STAGE1.md"
        if not doc.is_file():
            raise FileNotFoundError(f"阶段 2 所需辅助文件不存在: {doc}")
        shutil.copy2(doc, stage1 / "TFIDF_STAGE1.md")
        run_step("stage2", [py, "-B", str(base / "src" / "run_tfidf_stage2.py"), "--validation", str(inputs["validation"]),
                             "--stage1-root", str(stage1), "--scores-dir", str(stage1 / "results" / "tfidf_cosine"),
                             "--models-dir", str(stage1 / "artifacts" / "tfidf_cosine"),
                             "--stage1-report", str(stage1 / "reports" / "tfidf_cosine_stage1.json"), "--output-root", str(stage2)], base, log_path)
        config = stage2 / "artifacts" / "tfidf_cosine" / "selected_config.json"
        run_step("stage3", [py, "-B", str(base / "src" / "run_tfidf_stage3.py"), "--config", str(config),
                             "--evaluation", str(inputs["evaluation"]), "--output-root", str(stage3)], base, log_path)
        prediction = stage3 / "results" / "tfidf_cosine" / "evaluation_predictions.jsonl"
        run_step("prediction-validation", [py, "-B", str(base / "src" / "validate_tfidf_predictions.py"),
                             "--evaluation", str(inputs["evaluation"]), "--predictions", str(prediction)], base, log_path)
        n = validate_final(inputs["evaluation"], prediction)
        selected = load_json(config)
        report = load_json(stage3 / "reports" / "tfidf_cosine_stage3.json")
        manifest.update({"status": "passed", "stages": ["stage1", "stage2", "stage3", "prediction-validation"],
                         "inputs": {k: {"path": str(v), "sha256": sha256(v)} for k, v in inputs.items()},
                         "selected_config": {k: selected[k] for k in ("method", "representation", "threshold_index", "threshold", "operator")},
                         "evaluation_samples": n, "metrics": report.get("metrics"),
                         "prediction_path": str(prediction), "log_path": str(log_path)})
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        (logs / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"统一流程失败: {exc}", file=sys.stderr)
        raise
    (logs / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
