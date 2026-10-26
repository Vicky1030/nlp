import argparse
import hashlib
import json
from pathlib import Path

from llm_client import LLMClient


ROOT = Path(__file__).resolve().parents[1]


def main():
    default_input = (
        ROOT
        / "data"
        / "AFQMC_全组统一数据包"
        / "data"
        / "processed"
        / "validation.jsonl"
    )

    parser = argparse.ArgumentParser(description="Zero-shot prediction using the local model")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--prompt-file", type=Path, default=ROOT / "results" / "P0.prompt.txt")
    parser.add_argument("--limit", type=int, default=1, help="First N records; 0 means the entire input file")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "zero_shot_predictions.jsonl")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--quantization", choices=("none", "nf4"), default="none")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    content = args.input.read_bytes()
    all_records = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    if not all_records or args.limit < 0 or args.limit > len(all_records):
        parser.error(f"--limit must be between 0 and {len(all_records)}; input must be nonempty")
    records = all_records if args.limit == 0 else all_records[:args.limit]
    template = args.prompt_file.read_text(encoding="utf-8")
    prompts = []
    seen_ids = set()
    for record in records:
        if set(record) != {"id", "sentence1", "sentence2", "label"}:
            raise ValueError("Input must use A's four-field data format")
        if type(record["id"]) is not int or record["id"] in seen_ids:
            raise ValueError("Sample IDs must be unique integers")
        if type(record["label"]) is not int or record["label"] not in (0, 1):
            raise ValueError("Labels must be integers 0 or 1")
        if any(not isinstance(record[key], str) or not record[key].strip() for key in ("sentence1", "sentence2")):
            raise ValueError("Sentences must be nonempty strings")
        seen_ids.add(record["id"])
        prompts.append(template.format(sentence1=record["sentence1"], sentence2=record["sentence2"]))
    print(f"数据检查通过：{args.input.name} 共 {len(all_records)} 条，本次使用前 {len(records)} 条。", flush=True)
    if args.check_data:
        return

    output_path = args.output.resolve()
    summary_path = output_path.with_suffix(".summary.json")
    log_path = output_path.with_suffix(".calls.jsonl")
    for path in (output_path, summary_path, log_path):
        if path.exists():
            raise FileExistsError(f"Output already exists; choose a different --output: {path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = LLMClient(
        log_path=log_path,
        quantization=None if args.quantization == "none" else args.quantization,
    )
    client.warmup()
    summary = {
        "status": "running", "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(content).hexdigest(),
        "selection": "first N records in file order", "selected_ids": [record["id"] for record in records],
        "requested_count": len(records), "completed": 0, "correct": 0, "invalid_count": 0,
        "enable_thinking": args.thinking, "prompt_template": template,
        "prompt_file": str(args.prompt_file.resolve()),
        "prompt_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "model_info": client.model_info, "generation_settings": None,
        "invalid_policy": "Keep prediction=null; count it as incorrect in accuracy",
    }

    def save_summary():
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    save_summary()
    total_latency_ms = 0.0
    try:
        with output_path.open("x", encoding="utf-8") as destination:
            for completed, (record, prompt) in enumerate(zip(records, prompts), start=1):
                response = client.generate(prompt=prompt, enable_thinking=args.thinking)
                prediction = {
                    "id": record["id"], "sentence1": record["sentence1"],
                    "sentence2": record["sentence2"], "label": record["label"],
                    "prediction": response.prediction,
                    "method": "zero_shot_thinking" if args.thinking else "zero_shot",
                    "latency_ms": round(response.latency_ms, 3),
                }
                destination.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                destination.flush()
                total_latency_ms += response.latency_ms
                summary["completed"] = completed
                summary["correct"] += int(response.prediction == record["label"])
                summary["invalid_count"] += int(response.prediction is None)
                summary["generation_settings"] = response.settings
                save_summary()
                if len(records) == 1:
                    print("样本ID：", record["id"])
                    print("句子1：", record["sentence1"])
                    print("句子2：", record["sentence2"])
                    print("模型原始输出：", repr(response.raw_output))
                    print("解析后的预测：", response.prediction)
                    print("真实标签：", record["label"])
                    print("生成耗时：", round(response.latency_ms, 2), "毫秒")
                elif completed % 10 == 0 or completed == len(records):
                    print(f"已完成 {completed}/{len(records)}，正确 {summary['correct']}，无效输出 {summary['invalid_count']}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        save_summary()
        raise
    summary.update(
        status="complete", accuracy=summary["correct"] / len(records),
        invalid_rate=summary["invalid_count"] / len(records),
        mean_latency_ms=total_latency_ms / len(records),
    )
    save_summary()
    print(f"准确率：{summary['accuracy']:.2%}（{summary['correct']}/{len(records)}）")
    print(f"预测文件：{output_path}")
    print(f"统计文件：{summary_path}")


if __name__ == "__main__":
    main()
