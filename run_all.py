"""Unique one-command entry point for the AFQMC course project."""
import argparse
import importlib.metadata
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluation.config import OUTPUT_DIR, ROOT
from evaluation.run_all import run_frozen_evaluation
from pipeline.live_runner import METHOD_ORDER, parse_methods, run_live


ROLE_TITLES = {
    "A": "TF-IDF + LR",
    "B": "Qwen Few-shot",
    "C": "Qwen + LoRA",
}


def _label_text(value):
    if value == 1:
        return "1（相似）"
    if value == 0:
        return "0（不相似）"
    return f"{value}（无效）"


def print_hard_case_details(hard_cases):
    """Print complete selected examples so users never need to look IDs up."""
    location = hard_cases.get("path", "hard_case_output.json")
    print(
        f"三类难例：{hard_cases['status']} · {location}", flush=True
    )
    for case in hard_cases.get("cases", []):
        selected = case.get("selected_case")
        if not selected:
            print(f"  {case['title']} · 本次子集未找到", flush=True)
            continue
        predictions = selected["predictions"]
        print(
            f"\n  {case['title']} · 匹配 {case['match_count']} 条 · ID={selected['id']}",
            flush=True,
        )
        print(f"    句子1：{selected['sentence1']}", flush=True)
        print(f"    句子2：{selected['sentence2']}", flush=True)
        print(f"    真实标签：{_label_text(selected['label'])}", flush=True)
        for role in ("A", "B", "C"):
            result = predictions[role]
            verdict = "正确" if result["correct"] else "错误"
            print(
                f"    {role} · {ROLE_TITLES[role]}："
                f"{_label_text(result['prediction'])} · {verdict}",
                flush=True,
            )


def _first_existing(candidates, required_file):
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if (path / required_file).is_file():
            return path.resolve()
    return None


def discover_model_path():
    """Find the base model in an explicit setting or the repository model folder."""
    return _first_existing(
        [
            os.environ.get("AFQMC_QWEN_MODEL_DIR"),
            ROOT / "models" / "qwen3-1.7b",
        ],
        "config.json",
    )


def discover_adapter_path():
    """Find the final LoRA adapter in an explicit setting or repository folder."""
    return _first_existing(
        [
            os.environ.get("AFQMC_LORA_ADAPTER_DIR"),
            ROOT / "models" / "lora_r16_a32_30899",
            ROOT / "models" / "lora_r16_a32_30899" / "checkpoint-7726",
        ],
        "adapter_config.json",
    )


def _runtime_issue(methods):
    """Return a concise reason when the active interpreter cannot run methods."""
    if any(method.startswith("tfidf_") for method in methods):
        try:
            sklearn_version = importlib.metadata.version("scikit-learn")
        except importlib.metadata.PackageNotFoundError:
            sklearn_version = "未安装"
        if sklearn_version != "1.9.1":
            return f"scikit-learn={sklearn_version}，A 的冻结模型需要 1.9.1"
    if any(method.startswith("qwen_") for method in methods):
        if sys.version_info < (3, 11):
            return f"Python={sys.version.split()[0]}，B/C 实时推理需要 Python 3.11+"
        for package in ("torch", "transformers", "peft", "bitsandbytes"):
            try:
                importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                return f"缺少 GPU 依赖 {package}"
        import torch
        if not torch.cuda.is_available():
            return "当前 Python 环境中的 PyTorch 无法使用 CUDA"
    return None


def _python_candidates():
    candidates = []
    configured = os.environ.get("AFQMC_PYTHON")
    if configured:
        candidates.append(Path(configured))
    candidates.append(ROOT / ".venv" / "Scripts" / "python.exe")
    unique = []
    current = Path(sys.executable).resolve()
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path != current and path.is_file() and path not in unique:
            unique.append(path)
    return unique


def _candidate_supports(candidate, methods):
    statements = ["import importlib.metadata as m, sys"]
    if any(method.startswith("tfidf_") for method in methods):
        statements.append("assert m.version('scikit-learn') == '1.9.1'")
    if any(method.startswith("qwen_") for method in methods):
        statements += [
            "assert sys.version_info >= (3, 11)",
            "[m.version(p) for p in ('torch','transformers','peft','bitsandbytes')]",
            "import torch",
            "assert torch.cuda.is_available()",
        ]
    probe = subprocess.run(
        [str(candidate), "-c", ";".join(statements)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=45,
        check=False,
    )
    return probe.returncode == 0


def ensure_runtime(methods, forwarded_args):
    """Re-launch under a compatible project interpreter when possible."""
    issue = _runtime_issue(methods)
    if issue is None:
        return None
    if os.environ.get("AFQMC_AUTO_REEXEC") == "1":
        raise RuntimeError(f"自动切换后的环境仍不可用：{issue}")
    for candidate in _python_candidates():
        if not _candidate_supports(candidate, methods):
            continue
        print(f"[环境] 当前解释器不可用：{issue}", flush=True)
        print(f"[环境] 自动切换并重新启动：{candidate}", flush=True)
        child_env = os.environ.copy()
        child_env["AFQMC_AUTO_REEXEC"] = "1"
        return subprocess.call(
            [str(candidate), str(ROOT / "run_all.py"), *forwarded_args],
            cwd=str(ROOT),
            env=child_env,
        )
    raise RuntimeError(
        f"当前解释器不可用：{issue}。未找到兼容环境；请先安装 requirements-gpu.txt，"
        "或设置 AFQMC_PYTHON 指向准备好的 Python。"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "AFQMC one-command pipeline. With no arguments it runs A/B/C on the "
            "full evaluation set and then invokes D evaluation."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("live", "replay"),
        default="live",
        help="live reruns A/B/C; replay rebuilds D's historical evaluation without GPU",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "evaluation"),
        default="evaluation",
        help="full labeled split used by live mode (default: evaluation)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a deterministic random 80-row evaluation subset for a 2-3 minute recording",
    )
    parser.add_argument("--input", type=Path, help="optional labeled four-field JSONL")
    parser.add_argument(
        "--methods",
        default="members",
        help=(
            "members (A/B/C representatives), A, B, C, all, or comma-separated "
            "concrete method IDs"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="records to run; default 0 means the entire selected input",
    )
    parser.add_argument(
        "--sample-strategy", choices=("head", "random"), default="head"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="local Qwen3-1.7B directory; auto-detected or AFQMC_QWEN_MODEL_DIR",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="final LoRA adapter; auto-detected or AFQMC_LORA_ADAPTER_DIR",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="do not rebuild D's historical frozen-result report after live inference",
    )
    parser.add_argument("--list-methods", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    if args.list_methods:
        print("\n".join(METHOD_ORDER))
        return 0
    if args.seed < 0:
        parser.error("--seed must be non-negative")

    if args.mode == "replay":
        output_dir = (args.output_dir or OUTPUT_DIR).resolve()
        print("[模式] replay：重新校验冻结预测并重建 D 的历史评测。", flush=True)
        run_frozen_evaluation(output_dir, make_plots=not args.no_plots, emit_json=False)
        print(f"历史评测：{output_dir / 'summary_report.md'}", flush=True)
        print(f"历史图像：{output_dir / 'figures'}", flush=True)
        print(f"机器可读记录：{output_dir / 'run_report.json'}", flush=True)
        return 0

    selected_methods = parse_methods(args.methods)
    relaunched = ensure_runtime(selected_methods, forwarded_args)
    if relaunched is not None:
        return relaunched
    if args.quick and args.input:
        parser.error("--quick cannot be combined with --input")
    if args.quick:
        input_path = (ROOT / "data" / "processed" / "evaluation.jsonl").resolve()
        split_name = "quick"
        sample_strategy = "random"
    elif args.input:
        input_path = args.input.resolve()
        split_name = args.split
        sample_strategy = args.sample_strategy
    else:
        input_path = (ROOT / "data" / "processed" / f"{args.split}.jsonl").resolve()
        split_name = args.split
        sample_strategy = args.sample_strategy
    limit = (80 if args.quick else 0) if args.limit is None else args.limit
    model_path = args.model_path or discover_model_path()
    adapter_path = args.adapter_path or discover_adapter_path()
    if any(method.startswith("qwen_") for method in selected_methods) and model_path is None:
        parser.error(
            "Qwen model not found. Download it with the command in README section 5 "
            "to models/qwen3-1.7b, set AFQMC_QWEN_MODEL_DIR, or pass --model-path. "
            "Example: python references\\LoRA\\src\\download_model.py "
            "--model Qwen/Qwen3-1.7B "
            "--revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e "
            "--output models\\qwen3-1.7b"
        )
    if "qwen_lora" in selected_methods and adapter_path is None:
        parser.error(
            "LoRA adapter not found. Put it in models/lora_r16_a32_30899, set "
            "AFQMC_LORA_ADAPTER_DIR, or pass --adapter-path."
        )
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (ROOT / "runs" / f"{split_name}_{stamp}").resolve()

    print("[模式] live：一条命令运行 A/B/C，并自动调用 D 评测。", flush=True)
    if args.quick:
        print(
            f"[QUICK] 从完整 evaluation 固定随机抽取 "
            f"{'全部' if limit == 0 else limit} 条；用于录屏验证流程，"
            "正式指标以全量 evaluation 为准。",
            flush=True,
        )
    print(f"[入口] 数据：{input_path}", flush=True)
    print(f"[入口] 方法：{', '.join(selected_methods)}；seed={args.seed}", flush=True)
    manifest = run_live(
        input_path=input_path,
        split_name=split_name,
        methods=selected_methods,
        limit=limit,
        sample_strategy=sample_strategy,
        seed=args.seed,
        output_dir=output_dir,
        model_path=model_path,
        adapter_path=adapter_path,
        make_plots=not args.no_plots,
    )
    hard_cases = manifest["evaluation"]["hard_cases"]
    print_hard_case_details(hard_cases)
    print(f"本次图像：{output_dir / 'evaluation' / 'figures'}", flush=True)
    if not args.skip_history:
        print("[D 历史评测] 重建仓库内 A/B/C 冻结结果与图像。", flush=True)
        run_frozen_evaluation(
            OUTPUT_DIR, make_plots=not args.no_plots, emit_json=False
        )
        print(f"D 历史评测：{OUTPUT_DIR / 'summary_report.md'}", flush=True)
        print(f"D 历史图像：{OUTPUT_DIR / 'figures'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
