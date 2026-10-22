"""Run focused nine-case evidence regressions and retain raw subprocess logs."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "provenance" / "closeout_runs"
INPUTS = ROOT / "error_analysis" / "inputs"
OLD_EVIDENCE = ROOT / "error_analysis" / "evidence"
TOL = 1e-12
PROTECTED = [
    "data/processed/train.jsonl",
    "data/processed/validation.jsonl",
    "data/processed/evaluation.jsonl",
    "artifacts/tfidf_cosine_full_run_final/stage1/artifacts/tfidf_cosine/char_1_2/vectorizer.joblib",
    "artifacts/tfidf_cosine_full_run_final/stage1/artifacts/tfidf_cosine/char_1_2/metadata.json",
    "artifacts/tfidf_cosine_full_run_final/stage2/artifacts/tfidf_cosine/selected_config.json",
    "artifacts/tfidf_cosine_full_run_final/stage2/results/tfidf_cosine/threshold_comparison.csv",
    "artifacts/tfidf_cosine_full_run_final/stage2/results/tfidf_cosine/representation_summary.csv",
    "artifacts/tfidf_cosine_full_run_final/stage3/results/tfidf_cosine/evaluation_predictions.jsonl",
    "artifacts/tfidf_cosine_full_run_final/stage3/results/tfidf_cosine/evaluation_similarity.jsonl",
    "artifacts/tfidf_cosine_full_run_final/stage3/reports/tfidf_cosine_stage3.json",
    "artifacts/tfidf_cosine_full_run_final/plots/evaluation_metrics.csv",
    "error_analysis/inputs/evaluation.jsonl",
    "error_analysis/inputs/evaluation_predictions.jsonl",
    "error_analysis/inputs/evaluation_similarity.jsonl",
    "error_analysis/inputs/selected_config.json",
    "error_analysis/inputs/vectorizer.joblib",
    "error_analysis/inputs/metadata.json",
    "error_analysis/evidence/verification.json",
]


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command_for(output: Path, inputs: Path | None = None) -> list[str]:
    if inputs is None:
        inputs = INPUTS
    return [
        sys.executable, "-B", "-X", "utf8",
        str(ROOT / "src" / "extract_tfidf_case_evidence.py"),
        "--evaluation", str(inputs / "evaluation.jsonl"),
        "--predictions", str(inputs / "evaluation_predictions.jsonl"),
        "--similarity", str(inputs / "evaluation_similarity.jsonl"),
        "--config", str(inputs / "selected_config.json"),
        "--vectorizer", str(inputs / "vectorizer.joblib"),
        "--metadata", str(inputs / "metadata.json"),
        "--output-dir", str(output),
    ]


def execute(name: str, command: list[str], cwd: Path, expected_exit: int,
            run_id: str, log_dir: Path) -> dict:
    started = now()
    result = subprocess.run(command, cwd=cwd, capture_output=True)
    ended = now()
    stem = log_dir / name
    stdout_path = stem.with_suffix(".stdout.txt")
    stderr_path = stem.with_suffix(".stderr.txt")
    stdout_path.write_bytes(result.stdout)
    stderr_path.write_bytes(result.stderr)
    return {
        "name": name,
        "run_id": run_id,
        "log_directory": str(log_dir),
        "interpreter": sys.executable,
        "command": command,
        "working_directory": str(cwd),
        "started_at": started,
        "ended_at": ended,
        "exit_code": result.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "expected_exit_code": expected_exit,
        "expectation_met": result.returncode == expected_exit,
        "stdout_text": result.stdout.decode("utf-8", "replace"),
        "stderr_text": result.stderr.decode("utf-8", "replace"),
    }


def write_check_output(name: str, log_dir: Path, stdout: str = "", stderr: str = "") -> tuple[str, str]:
    stdout_path = log_dir / f"{name}.stdout.txt"
    stderr_path = log_dir / f"{name}.stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return str(stdout_path), str(stderr_path)


def equivalent(left, right, path="$", deltas=None) -> bool:
    if deltas is None:
        deltas = []
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(equivalent(left[k], right[k], f"{path}.{k}", deltas) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(equivalent(a, b, f"{path}[{i}]", deltas) for i, (a, b) in enumerate(zip(left, right)))
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        delta = abs(float(left) - float(right))
        deltas.append(delta)
        return math.isclose(float(left), float(right), rel_tol=TOL, abs_tol=TOL)
    return left == right


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused nine-case evidence regressions.")
    parser.add_argument("--run-id", help="Optional safe, unique run directory name.")
    parser.add_argument("--package-root", type=Path, required=True,
                        help="External complete delivery copy with local-only inputs/evidence; never the original")
    parser.add_argument("--output-root", type=Path, required=True, help="New regression output directory")
    return parser.parse_args()


def actual_runner_command() -> list[str]:
    """Return the interpreter options, script path, and arguments used to start this process."""
    original = getattr(sys, "orig_argv", None)
    if original:
        return list(original)
    return [sys.executable, *sys.argv]


def resolve_run_id(requested: str | None) -> str:
    if requested is None:
        timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        return f"{timestamp}_{secrets.token_hex(3)}"
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", requested)
            or requested in {".", ".."} or ".." in requested
            or Path(requested).is_absolute() or "/" in requested or "\\" in requested):
        raise ValueError("--run-id must be one safe directory name; separators, absolute paths, and '..' are not allowed")
    return requested


def main() -> int:
    global ROOT, INPUTS, OLD_EVIDENCE, LOG_ROOT
    args = parse_args()
    ROOT = args.package_root.resolve()
    INPUTS = ROOT / "error_analysis" / "inputs"
    OLD_EVIDENCE = ROOT / "error_analysis" / "evidence"
    missing = [p for p in PROTECTED if not (ROOT / p).is_file()]
    if missing:
        raise SystemExit("Missing external complete-package files: " + ", ".join(missing))
    LOG_ROOT = args.output_root.resolve()
    if LOG_ROOT == ROOT or ROOT in LOG_ROOT.parents:
        raise SystemExit("--output-root must be outside the delivery package")
    LOG_ROOT.mkdir(parents=True, exist_ok=False)
    runner_command = actual_runner_command()
    try:
        run_id = resolve_run_id(args.run_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    log_dir = LOG_ROOT / run_id
    try:
        log_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise SystemExit(f"refusing to overwrite existing regression log directory: {log_dir}") from exc
    temp_root = Path(tempfile.mkdtemp(prefix="tfidf_closeout_regression_", dir=log_dir))
    records = []
    before = {p: sha(ROOT / p) for p in PROTECTED}

    normal_out = temp_root / "normal_evidence"
    normal_cmd = command_for(normal_out)
    normal = execute("01_normal_evidence", normal_cmd, ROOT, 0, run_id, log_dir)
    normal["expected"] = "固定九例均通过现有方向、阈值、稀疏特征及输入哈希核验。"
    normal["actual"] = normal["stdout_text"].strip()
    records.append(normal)

    old_details = OLD_EVIDENCE / "feature_details"
    new_details = normal_out / "feature_details"
    compare_ids = sorted(p.stem for p in old_details.glob("*.json"))
    deltas = []
    same = compare_ids == sorted(p.stem for p in new_details.glob("*.json"))
    changed = []
    for case_id in compare_ids:
        old = json.loads((old_details / f"{case_id}.json").read_text(encoding="utf-8"))
        new = json.loads((new_details / f"{case_id}.json").read_text(encoding="utf-8"))
        if not equivalent(old, new, f"{case_id}", deltas):
            same = False
            changed.append(case_id)
    comparison = {
        "name": "02_old_new_evidence_comparison",
        "operation": "由本脚本在正常提取成功后递归比较旧版与本轮9份feature_details JSON。",
        "started_at": now(),
        "ended_at": now(),
        "exit_code": 0 if same and len(compare_ids) == 9 else 1,
        "expected": "比较全部9例完整JSON字段；数值按rtol=1e-12、atol=1e-12。",
        "actual": {"ids_compared": compare_ids, "all_equal_with_tolerance": same,
                   "changed_ids": changed, "max_absolute_numeric_delta": max(deltas, default=0),
                   "rtol": TOL, "atol": TOL,
                   "fields_scope": "完整递归JSON，包括文本、标签、预测、错误类型、相似度、特征索引、计数、IDF、权重、贡献及摘要。"},
        "interpreter": sys.executable,
        "command": runner_command,
        "working_directory": str(ROOT),
        "stdout": None,
        "stderr": None,
    }
    comparison_stdout, comparison_stderr = write_check_output(
        "02_old_new_evidence_comparison", log_dir,
        json.dumps(comparison["actual"], ensure_ascii=False, indent=2) + "\n",
        "" if same else "One or more case evidence files differ.\n",
    )
    comparison["stdout"] = comparison_stdout
    comparison["stderr"] = comparison_stderr
    records.append(comparison)

    protected_out = OLD_EVIDENCE
    protected_hash = sha(protected_out / "verification.json")
    overwrite = execute("03_existing_output_refusal", command_for(protected_out), ROOT, 1, run_id, log_dir)
    overwrite["expected"] = "拒绝覆盖已存在输出目录。"
    overwrite["actual"] = overwrite["stderr_text"] + overwrite["stdout_text"]
    overwrite["protected_verification_sha256_unchanged"] = sha(protected_out / "verification.json") == protected_hash
    records.append(overwrite)

    counter_dir = temp_root / "counterexample_inputs"
    counter_dir.mkdir()
    for filename in ["evaluation.jsonl", "evaluation_predictions.jsonl", "evaluation_similarity.jsonl", "selected_config.json", "vectorizer.joblib", "metadata.json"]:
        shutil.copy2(INPUTS / filename, counter_dir / filename)
    for filename in ["evaluation.jsonl", "evaluation_predictions.jsonl", "evaluation_similarity.jsonl"]:
        path = counter_dir / filename
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        for row in rows:
            if str(row.get("id")) == "34937":
                predicted = next(json.loads(line)["prediction"] for line in (counter_dir / "evaluation_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line and str(json.loads(line).get("id")) == "34937")
                row["label"] = predicted
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    counter_out = temp_root / "counterexample_output"
    counter = execute("04_synchronized_34937_counterexample", command_for(counter_out, counter_dir), ROOT, 1, run_id, log_dir)
    combined = counter["stdout_text"] + counter["stderr_text"]
    counter["expected"] = "非零退出，且明确指出34937不再满足预期FP方向，不因三文件标签错位失败。"
    counter["actual"] = combined
    counter["direction_failure_confirmed"] = "expected FP" in combined and "actual label=1, prediction=1" in combined
    counter["expectation_met"] = counter["exit_code"] == 1 and counter["direction_failure_confirmed"]
    records.append(counter)

    after = {p: sha(ROOT / p) for p in PROTECTED}
    hash_rows = [{"path": p, "before": before[p], "after": after[p], "same": before[p] == after[p]} for p in PROTECTED]
    hash_record = {
        "name": "05_protected_hash_comparison",
        "operation": "由本脚本在正常提取及反例测试前后，对明确列出的受保护路径计算 SHA-256。",
        "interpreter": sys.executable,
        "command": runner_command,
        "working_directory": str(ROOT),
        "started_at": records[0]["started_at"],
        "ended_at": now(),
        "exit_code": 0 if all(row["same"] for row in hash_rows) else 1,
        "expected": "所有列出的正式数据及历史实验材料哈希保持不变。",
        "actual": {"all_same": all(row["same"] for row in hash_rows), "files": hash_rows},
        "stdout": None,
        "stderr": None,
    }
    hash_stdout, hash_stderr = write_check_output(
        "05_protected_hash_comparison", log_dir,
        json.dumps(hash_record["actual"], ensure_ascii=False, indent=2) + "\n",
        "" if all(row["same"] for row in hash_rows) else "Protected file hash changed.\n",
    )
    hash_record["stdout"] = hash_stdout
    hash_record["stderr"] = hash_stderr
    records.append(hash_record)

    for record in records:
        record.pop("stdout_text", None)
        record.pop("stderr_text", None)
    result = {
        "run_id": run_id,
        "started_at": records[0]["started_at"],
        "ended_at": now(),
        "interpreter": sys.executable,
        "python": sys.version,
        "log_directory": str(log_dir),
        "runner_command": runner_command,
        "working_directory": str(ROOT),
        "temporary_directory": str(temp_root),
        "temporary_directory_disposition": "回归结束后删除；命令与路径保留在本记录。",
        "tests": records,
        "status": "passed" if all(r.get("expectation_met", r["exit_code"] == 0) for r in records) else "failed",
    }
    (log_dir / "regression.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "status": result["status"], "log": str(log_dir / "regression.json"),
                      "temp": str(temp_root), "tests": [{"name": r["name"], "exit_code": r["exit_code"],
                      "expectation_met": r.get("expectation_met", r["exit_code"] == 0)} for r in records]}, ensure_ascii=False))
    shutil.rmtree(temp_root)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
