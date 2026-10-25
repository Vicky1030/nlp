import argparse
import json
from pathlib import Path

from llm_client import LLMClient, ROOT
from work.compare_nf4_baseline import evaluate, read_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "results" / "near_conditions.config.json")
    parser.add_argument("--input", type=Path, default=ROOT / "data/AFQMC_全组统一数据包/data/processed/validation.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    records = read_records(args.input)
    if args.limit < 0 or args.limit > len(records):
        parser.error("limit must be between zero and the input size")
    records = records[:args.limit] if args.limit else records
    assert records and len({row["id"] for row in records}) == len(records)
    assert all(type(row["label"]) is int and row["label"] in (0, 1) for row in records)
    prompts = [config["prompt_template"].format(sentence1=row["sentence1"], sentence2=row["sentence2"]) for row in records]
    if args.check_data:
        print(f"Validated {len(records)} records and prompt formatting; NF4 candidate={config['candidate_name']}")
        return
    output = args.output.resolve()
    summary_path = output.with_suffix(".summary.json")
    log_path = output.with_suffix(".calls.jsonl")
    for path in (output, summary_path, log_path):
        if path.exists():
            raise FileExistsError(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    client = LLMClient(quantization=config["quantization"], log_path=log_path)
    assert client.model_info["model_fingerprint"] == config["model_fingerprint"]
    client.warmup()
    predictions = []
    with output.open("x", encoding="utf-8") as stream:
        for index, (row, prompt) in enumerate(zip(records, prompts), 1):
            response = client.generate(prompt, system_prompt=config["system_prompt"], enable_thinking=config["enable_thinking"], max_new_tokens=config["max_new_tokens"], do_sample=config["do_sample"], seed=config["seed"], demonstrations=config.get("demonstrations"))
            result = {field: row[field] for field in ("id", "sentence1", "sentence2", "label")}
            result.update(prediction=response.prediction, method=config["method"], latency_ms=response.latency_ms)
            predictions.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 100 == 0 or index == len(records):
                print(f"Completed {index}/{len(records)}", flush=True)
    summary = {"status": "complete", "input": str(args.input.resolve()), "config": config, "model_info": client.model_info, "metrics": evaluate(predictions)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
