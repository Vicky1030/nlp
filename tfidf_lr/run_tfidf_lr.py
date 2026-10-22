"""Run the verified TF-IDF + LR stages in one reproducible process tree.

The child stages remain the source of the experiment logic. This wrapper only
chooses fresh output locations, invokes each child with ``sys.executable``,
captures raw logs, and writes a final inventory after all outputs exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BEIJING_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path) -> dict[str, dict[str, int | str]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, int | str]] = {}
    if path.is_file():
        result[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        return result
    for item in sorted(path.rglob("*")):
        if item.is_file():
            result[str(item.relative_to(path)).replace("\\", "/")] = {
                "bytes": item.stat().st_size,
                "sha256": sha256(item),
            }
    return result


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve_user_path(value: Path) -> Path:
    return value.expanduser().resolve()


def run_stage(name: str, args: list[str], cwd: Path, log_dir: Path, inputs: list[Path], output: Path, env: dict[str, str]) -> dict:
    started = datetime.now(BEIJING_TZ).isoformat()
    clock = time.monotonic()
    print(f"[阶段开始] {name}", flush=True)
    stdout_path = log_dir / f"{name}_stdout.txt"
    stderr_path = log_dir / f"{name}_stderr.txt"
    child_env = dict(env, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    child = subprocess.Popen(args, cwd=str(cwd), env=child_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    def forward(pipe, path, terminal):
        # Drain both pipes independently; save original bytes before decoding for display.
        with path.open('wb') as log:
            for line in iter(pipe.readline, b''):
                log.write(line); log.flush()
                terminal.write(line.decode('utf-8', errors='replace')); terminal.flush()
        pipe.close()
    threads = [threading.Thread(target=forward, args=(pipe, path, terminal))
               for pipe, path, terminal in ((child.stdout, stdout_path, sys.stdout), (child.stderr, stderr_path, sys.stderr))]
    for thread in threads: thread.start()
    exit_code = child.wait()
    for thread in threads: thread.join()
    ended = datetime.now(BEIJING_TZ).isoformat()
    elapsed = time.monotonic() - clock
    stdout = stdout_path.read_text(encoding='utf-8')
    stderr = stderr_path.read_text(encoding='utf-8')
    print(f"[阶段结束] {name} exit_code={exit_code} elapsed={elapsed:.3f}s logs={log_dir}", flush=True)
    record = {
        "stage": name,
        "command": {"executable": args[0], "arguments": args[1:]},
        "interpreter": sys.executable,
        "cwd": str(cwd),
        "started_at": started,
        "ended_at": ended,
        "timezone": "Asia/Shanghai / UTC+08:00",
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "expected_exit_code": 0,
        "stdout": stdout,
        "stderr": stderr,
        "input_hashes": {str(p): sha256(p) for p in inputs if p.is_file()},
        "output_directory": str(output),
        "output_hashes_after": inventory(output),
    }
    write_json(log_dir / f"{name}_execution.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TF-IDF + LR stages 1, acceptance, 2, and 3.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing train.jsonl, validation.jsonl, and optionally evaluation.jsonl")
    parser.add_argument("--vectorizer-root", type=Path, required=True, help="Existing frozen TF-IDF vectorizer directory")
    parser.add_argument("--output-root", type=Path, required=True, help="New, empty full-run output directory")
    parser.add_argument("--evaluation", type=Path, default=None, help="Frozen evaluation.jsonl; defaults to data-dir/evaluation.jsonl")
    args = parser.parse_args()

    data_dir = resolve_user_path(args.data_dir)
    vectorizer_root = resolve_user_path(args.vectorizer_root)
    output_root = resolve_user_path(args.output_root)
    evaluation = resolve_user_path(args.evaluation) if args.evaluation else data_dir / "evaluation.jsonl"
    if output_root.exists():
        raise FileExistsError(f"refusing existing output root; choose a new --output-root: {output_root}")
    required = [data_dir / "train.jsonl", data_dir / "validation.jsonl", vectorizer_root]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("missing required input(s):\n" + "\n".join(missing))
    if not evaluation.is_file():
        raise FileNotFoundError(f"evaluation file is missing: {evaluation}")

    stage1 = output_root / "stage1"
    audit = output_root / "acceptance_stage1"
    stage2 = output_root / "stage2"
    stage3 = output_root / "stage3"
    logs = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=False)
    logs.mkdir()
    common_env = os.environ.copy()
    for key, folder in (("TEMP", "temp"), ("TMP", "tmp"), ("PIP_CACHE_DIR", "pip-cache")):
        target = output_root / folder
        target.mkdir()
        common_env[key] = str(target)
    write_json(logs / "inputs.json", {
        "project_root": str(PROJECT_ROOT), "data_dir": str(data_dir),
        "vectorizer_root": str(vectorizer_root), "evaluation": str(evaluation),
        "output_root": str(output_root), "environment": {k: common_env[k] for k in ("TEMP", "TMP", "PIP_CACHE_DIR")},
    })

    python = sys.executable
    stages = [
        ("stage1_train", [python, "-B", str(PROJECT_ROOT / "src/run_tfidf_lr_stage1.py"),
            "--train", str(data_dir / "train.jsonl"), "--validation", str(data_dir / "validation.jsonl"),
            "--vectorizer-root", str(vectorizer_root), "--output-dir", str(stage1)],
            [data_dir / "train.jsonl", data_dir / "validation.jsonl"], stage1),
        ("stage1_acceptance", [python, "-B", str(PROJECT_ROOT / "src/verify_tfidf_lr_stage1.py"),
            "--root", str(PROJECT_ROOT), "--run", str(stage1), "--audit-dir", str(audit),
            "--data-dir", str(data_dir), "--vectorizer-root", str(vectorizer_root)],
            [stage1 / "model_index.json", stage1 / "run_metadata.json"], audit),
        ("stage2_selection", [python, "-B", str(PROJECT_ROOT / "src/run_tfidf_lr_stage2.py"),
            "--root", str(PROJECT_ROOT), "--run", str(stage1), "--acceptance", str(audit / "verification.json"),
            "--data-dir", str(data_dir), "--vectorizer-root", str(vectorizer_root),
            "--output-dir", str(stage2)], [audit / "verification.json"], stage2),
        ("stage3_evaluation", [python, "-B", str(PROJECT_ROOT / "src/run_tfidf_lr_stage3.py"),
            "--config", str(stage2 / "selected_config.json"), "--evaluation", str(evaluation),
            "--stage1-run", str(stage1), "--data-dir", str(data_dir), "--vectorizer-root", str(vectorizer_root),
            "--output-dir", str(stage3)],
            [stage2 / "selected_config.json", evaluation], stage3),
    ]
    records = []
    for name, command, inputs, output in stages:
        record = run_stage(name, command, PROJECT_ROOT, logs, inputs, output, common_env)
        records.append(record)
        if record["exit_code"] != 0:
            summary = {"status": "failed", "failed_stage": name, "stages": records, "output_root": str(output_root)}
            write_json(logs / "run_summary.json", summary)
            print(f"[失败] {name}; 后续阶段未执行；日志：{logs}", file=sys.stderr, flush=True)
            return int(record["exit_code"] or 1)
    final_manifest = inventory(output_root)
    write_json(logs / "run_summary.json", {"status": "passed", "stages": records, "output_root": str(output_root),
        "selected_config": json.loads((stage2 / "selected_config.json").read_text(encoding="utf-8"))["config_id"],
        "stage3_metrics": json.loads((stage3 / "evaluation_metrics.json").read_text(encoding="utf-8")),
        "final_output_hashes": final_manifest})
    write_json(output_root / "full_run_sha256.json", {"excluded": "full_run_sha256.json", "files": inventory(output_root)})
    print(json.dumps({"status": "passed", "output_root": str(output_root), "selected_config_output": records[2]["output_directory"],
                      "stage3_output": str(stage3)}, ensure_ascii=False))
    chosen = json.loads((stage2 / 'selected_config.json').read_text(encoding='utf-8'))
    print(f"获选配置：{chosen['config_id']}；阈值：{chosen['threshold']}\n预测：{stage3 / 'evaluation_predictions.jsonl'}\n指标：{stage3 / 'evaluation_metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
