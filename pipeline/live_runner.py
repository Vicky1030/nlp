"""Run A/B/C inference methods on one labeled subset, then call D evaluation."""
import csv
import importlib.util
import importlib.metadata
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluation.api import evaluate_prediction_directory
from evaluation.io_utils import load_jsonl, validate_gold
from evaluation.metrics import calculate_metrics
from evaluation.run_all import sha256


ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = [
    "tfidf_cosine",
    "tfidf_lr",
    "qwen_zero_shot",
    "qwen_few_shot",
    "qwen_lora",
]
METHOD_GROUPS = {
    "a": METHOD_ORDER[:2],
    "b": METHOD_ORDER[2:4],
    "c": METHOD_ORDER[4:],
    "all": METHOD_ORDER,
    # One representative method from each member. B uses the final few-shot prompt.
    "members": ["tfidf_lr", "qwen_few_shot", "qwen_lora"],
}
METHOD_TITLES = {
    "tfidf_cosine": "A · TF-IDF + Cosine",
    "tfidf_lr": "A · TF-IDF + Logistic Regression",
    "qwen_zero_shot": "B · Qwen Zero-shot",
    "qwen_few_shot": "B · Qwen Few-shot",
    "qwen_lora": "C · Qwen + LoRA",
}


def parse_methods(value):
    """Expand A/B/C/all aliases and return a stable, duplicate-free method list."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("--methods must not be empty")
    expanded = []
    for token in (part.strip().lower() for part in value.split(",")):
        if not token:
            continue
        if token in METHOD_GROUPS:
            expanded.extend(METHOD_GROUPS[token])
        elif token in METHOD_ORDER:
            expanded.append(token)
        else:
            choices = ", ".join(["A", "B", "C", "members", "all", *METHOD_ORDER])
            raise ValueError(f"Unknown method {token!r}; choose from: {choices}")
    selected = []
    for method in METHOD_ORDER:
        if method in expanded and method not in selected:
            selected.append(method)
    if not selected:
        raise ValueError("No methods selected")
    return selected


def select_records(records, limit, strategy="head", seed=42):
    """Select a deterministic labeled subset without changing original row order."""
    if limit < 0 or limit > len(records):
        raise ValueError(f"--limit must be between 0 and {len(records)}")
    if limit == 0 or limit == len(records):
        return list(records)
    if strategy == "head":
        return list(records[:limit])
    if strategy != "random":
        raise ValueError("sample strategy must be head or random")
    indexes = sorted(random.Random(seed).sample(range(len(records)), limit))
    return [records[index] for index in indexes]


def _load_module(name, path, import_dir=None):
    path = Path(path).resolve()
    if import_dir is not None:
        sys.path.insert(0, str(Path(import_dir).resolve()))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if import_dir is not None:
            sys.path.pop(0)


def _write_selected_data(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in records:
            clean = {key: row[key] for key in ("id", "sentence1", "sentence2", "label")}
            stream.write(json.dumps(clean, ensure_ascii=False) + "\n")


def _progress(method, completed, total, correct, invalid):
    every = 1 if total <= 20 else max(1, total // 20)
    if completed == 1 or completed == total or completed % every == 0:
        pct = completed * 100 / total
        print(
            f"[{method}] {completed}/{total} ({pct:5.1f}%) · "
            f"当前正确 {correct} · 无效 {invalid}",
            flush=True,
        )


def _run_method(records, output_path, method, infer_one, timing_scope):
    """Execute a method record-by-record and persist the seven-field protocol."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Prediction output already exists: {output_path}")
    produced, correct, invalid = [], 0, 0
    started_at = datetime.now(timezone.utc).isoformat()
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        for completed, row in enumerate(records, 1):
            result = infer_one(row)
            prediction = result.get("prediction")
            record = {
                "id": row["id"],
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "label": row["label"],
                "prediction": prediction,
                "method": method,
                "latency_ms": round(float(result["latency_ms"]), 3),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            produced.append(record)
            correct += int(prediction == row["label"])
            invalid += int(prediction is None)
            _progress(method, completed, len(records), correct, invalid)
    metrics = calculate_metrics(produced)
    summary = {
        "method": method,
        "display_name": METHOD_TITLES[method],
        "status": "complete",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "timing_scope": timing_scope,
        "prediction_file": str(output_path),
        "metrics": metrics,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[{method}] 完成 · Accuracy={metrics['accuracy']:.4f} · "
        f"F1(1)={metrics['f1_positive']:.4f} · Macro-F1={metrics['macro_f1']:.4f}",
        flush=True,
    )
    return summary


def _run_cpu_methods(records, methods, prediction_dir):
    summaries = {}
    selected = {"tfidf_cosine", "tfidf_lr"}.intersection(methods)
    if selected:
        try:
            actual = importlib.metadata.version("scikit-learn")
        except importlib.metadata.PackageNotFoundError:
            actual = "not installed"
        if actual != "1.9.1":
            raise RuntimeError(
                "A's frozen joblib models require scikit-learn==1.9.1; "
                f"the active interpreter has {actual}. Use Python >=3.11 and "
                "install requirements-live.txt in a project virtual environment."
            )
    if "tfidf_cosine" in methods:
        package = ROOT / "references" / "tfidf_cosine_package"
        module = _load_module("afqmc_tfidf_cosine", package / "predict.py", package)
        predictor = module.FrozenPredictor(package)
        predictor.predict_pairs([("中文测试", "中文测试")])

        def infer(row):
            started = time.perf_counter_ns()
            result = predictor.predict_pairs([(row["sentence1"], row["sentence2"])])[0]
            return {
                "prediction": result["prediction"],
                "latency_ms": (time.perf_counter_ns() - started) / 1_000_000,
            }

        summaries["tfidf_cosine"] = _run_method(
            records,
            prediction_dir / "tfidf_cosine.jsonl",
            "tfidf_cosine",
            infer,
            "warmed CPU single-record TF-IDF transforms, cosine and threshold",
        )

    if "tfidf_lr" in methods:
        package = ROOT / "references" / "tfidf_lr_package"
        module = _load_module("afqmc_tfidf_lr", package / "predict.py")
        predictor = module.FrozenPredictor(package)
        for row in records[: min(10, len(records))]:
            predictor._predict_one(row["sentence1"], row["sentence2"])

        def infer(row):
            started = time.perf_counter_ns()
            _, prediction = predictor._predict_one(row["sentence1"], row["sentence2"])
            return {
                "prediction": prediction,
                "latency_ms": (time.perf_counter_ns() - started) / 1_000_000,
            }

        summaries["tfidf_lr"] = _run_method(
            records,
            prediction_dir / "tfidf_lr.jsonl",
            "tfidf_lr",
            infer,
            "warmed CPU single-record TF-IDF transforms, sparse features, predict_proba and threshold",
        )
    return summaries


def _require_directory(path, flag, required_file):
    if path is None:
        raise ValueError(f"{flag} is required for the selected GPU method")
    resolved = Path(path).expanduser().resolve()
    if not (resolved / required_file).is_file():
        raise FileNotFoundError(f"{flag} does not contain {required_file}: {resolved}")
    return resolved


def _run_gpu_methods(records, methods, prediction_dir, calls_dir, model_path, adapter_path, seed):
    requested = [method for method in methods if method.startswith("qwen_")]
    if not requested:
        return {}
    model_path = _require_directory(model_path, "--model-path", "config.json")
    prompt_root = ROOT / "references" / "prompt_engineering_package"
    b_module = _load_module(
        "afqmc_member_b_llm_client",
        prompt_root / "src" / "llm_client.py",
    )
    calls_dir.mkdir(parents=True, exist_ok=True)
    print(f"[GPU] 正在加载一次 NF4 基础模型：{model_path}", flush=True)
    client = b_module.LLMClient(
        model_path=model_path,
        quantization="nf4",
        log_path=calls_dir / "qwen_base.calls.jsonl",
    )
    client.warmup()
    summaries = {}

    prompt_configs = {
        "qwen_zero_shot": prompt_root / "configs" / "near_conditions.config.json",
        "qwen_few_shot": prompt_root / "configs" / "fixed2_last_exact.config.json",
    }
    for method in ("qwen_zero_shot", "qwen_few_shot"):
        if method not in methods:
            continue
        config = json.loads(prompt_configs[method].read_text(encoding="utf-8"))
        if client.model_info["model_fingerprint"] != config["model_fingerprint"]:
            raise ValueError(
                "Local base model fingerprint differs from the frozen B/C experiment. "
                "Use the Qwen3-1.7B files documented in README."
            )

        def infer(row, config=config):
            prompt = config["prompt_template"].format(
                sentence1=row["sentence1"], sentence2=row["sentence2"]
            )
            response = client.generate(
                prompt,
                system_prompt=config["system_prompt"],
                enable_thinking=config["enable_thinking"],
                max_new_tokens=config["max_new_tokens"],
                do_sample=config["do_sample"],
                seed=seed,
                demonstrations=config.get("demonstrations"),
            )
            return {"prediction": response.prediction, "latency_ms": response.latency_ms}

        summaries[method] = _run_method(
            records,
            prediction_dir / f"{method}.jsonl",
            method,
            infer,
            b_module.LATENCY_SCOPE,
        )

    if "qwen_lora" in methods:
        adapter_path = _require_directory(adapter_path, "--adapter-path", "adapter_config.json")
        import torch
        from peft import PeftModel

        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The frozen LoRA protocol requires a CUDA GPU with BF16 support")
        c_module = _load_module(
            "afqmc_member_c_common",
            ROOT / "references" / "LoRA" / "src" / "common.py",
        )
        print(f"[GPU] 在已加载的同一基础模型上挂载 LoRA Adapter：{adapter_path}", flush=True)
        lora_model = PeftModel.from_pretrained(client.model, adapter_path).eval()

        def infer_lora(row):
            messages = c_module.make_messages(row["sentence1"], row["sentence2"])
            _, response = c_module.generate_prediction(
                lora_model, client.tokenizer, messages, client.device, seed=seed
            )
            return {"prediction": response.prediction, "latency_ms": response.latency_ms}

        summaries["qwen_lora"] = _run_method(
            records,
            prediction_dir / "qwen_lora.jsonl",
            "qwen_lora",
            infer_lora,
            c_module.LATENCY_SCOPE,
        )
    return summaries


def _print_final_table(csv_path):
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if rows and rows[0].get("split") == "quick":
        print(
            "\n[提示] QUICK 指标来自固定随机子集，只用于录屏验证流程；"
            "正式成绩请引用 4,316 条全量 evaluation。",
            flush=True,
        )
    print("\n统一评测结果", flush=True)
    print("-" * 92)
    print(f"{'方法':<37} {'Accuracy':>10} {'F1(1)':>10} {'Macro-F1':>10} {'平均延迟ms':>14}")
    print("-" * 92)
    for row in rows:
        latency = row["latency_mean_ms"]
        latency_text = f"{float(latency):.2f}" if latency else "N/A"
        print(
            f"{row['display_name']:<37} {float(row['accuracy']):>10.4f} "
            f"{float(row['f1_positive']):>10.4f} {float(row['macro_f1']):>10.4f} "
            f"{latency_text:>14}"
        )
    print("-" * 92)


def run_live(
    *, input_path, split_name, methods, limit, sample_strategy, seed,
    output_dir, model_path=None, adapter_path=None, make_plots=True,
):
    """Run selected methods, evaluate them and return the run manifest."""
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Live output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_methods = parse_methods(methods) if isinstance(methods, str) else list(methods)
    all_records = validate_gold(load_jsonl(input_path), input_path)
    records = select_records(all_records, limit, sample_strategy, seed)
    selected_path = output_dir / "data" / f"{split_name}_selected.jsonl"
    _write_selected_data(selected_path, records)
    prediction_dir = output_dir / "predictions"
    calls_dir = output_dir / "calls"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[准备] 输入 {len(all_records)} 条；本次选择 {len(records)} 条；"
        f"方法：{', '.join(selected_methods)}；seed={seed}",
        flush=True,
    )
    started = datetime.now(timezone.utc).isoformat()
    summaries = _run_cpu_methods(records, selected_methods, prediction_dir)
    summaries.update(_run_gpu_methods(
        records,
        selected_methods,
        prediction_dir,
        calls_dir,
        model_path,
        adapter_path,
        seed,
    ))

    print("[D 评测] 严格对齐预测与 gold，生成指标、错例、图表和汇总。", flush=True)
    evaluation_dir = output_dir / "evaluation"
    evaluation = evaluate_prediction_directory(
        split_name=split_name,
        gold_path=selected_path,
        prediction_dir=prediction_dir,
        output_dir=evaluation_dir,
        make_plots=make_plots,
    )
    _print_final_table(evaluation["comparison_csv"])
    manifest = {
        "status": "complete",
        "mode": "live",
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "input_sha256": sha256(input_path),
        "split": split_name,
        "input_rows": len(all_records),
        "selected_rows": len(records),
        "selected_ids": [row["id"] for row in records],
        "sample_strategy": sample_strategy,
        "seed": seed,
        "methods": selected_methods,
        "method_summaries": summaries,
        "evaluation": evaluation,
    }
    manifest_path = output_dir / "live_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果总览：{evaluation['summary_html']}", flush=True)
    print(f"运行清单：{manifest_path}", flush=True)
    return manifest
