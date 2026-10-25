import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from llm_client import LLMClient
from work.compare_nf4_baseline import evaluate, read_records


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def rank_key(metric, baseline):
    differences = [metric["accuracy"] - baseline["accuracy"], metric["macro_f1"] - baseline["macro_f1"], metric["per_class"]["1"]["f1"] - baseline["per_class"]["1"]["f1"]]
    return (min(differences) > 0, min(differences), sum(differences))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=args.resume)
    started = time.monotonic()
    data = ROOT / "data/AFQMC_全组统一数据包/data/processed"
    validation = read_records(data / "validation.jsonl")
    training = {row["id"]: row for row in read_records(data / "train.jsonl")}
    baseline = read_records(ROOT / "outputs/short.validation.predictions.jsonl")
    expected = [(row["id"], row["sentence1"], row["sentence2"], row["label"]) for row in validation]
    assert [(row["id"], row["sentence1"], row["sentence2"], row["label"]) for row in baseline] == expected
    rng = random.Random(2026102402)
    screen = []
    for label, count in ((0, 332), (1, 148)):
        screen.extend(rng.sample([row for row in validation if row["label"] == label], count))
    rng.shuffle(screen)
    selected_ids = {row["id"] for row in screen}
    baseline_screen = evaluate([row for row in baseline if row["id"] in selected_ids])
    examples = [training[example_id] for example_id in (118, 18614, 14112, 29811)]
    positive_examples = [training[example_id] for example_id in (118, 18614, 14112, 3146)]
    short = (ROOT / "results" / "short.prompt.txt").read_text(encoding="utf-8")
    p0 = (ROOT / "results" / "P0.prompt.txt").read_text(encoding="utf-8")
    pair = "句子1：{sentence1}\n句子2：{sentence2}"
    instruction = "判断两个句子的意思是否相同。意思相同输出1，意思不同输出0。"
    candidates = {}

    def add(name, template, system=None, demonstrations=None, example_ids=None):
        candidates[name] = {
            "candidate_name": name, "prompt_template": template, "system_prompt": system,
            "method": "few_shot" if example_ids else "zero_shot",
            "demonstrations": demonstrations, "training_example_ids": example_ids or [],
            "quantization": "nf4", "enable_thinking": False,
            "max_new_tokens": 8, "do_sample": False, "seed": 42,
        }

    add("short_near", short.replace("意思是否相同", "核心意思是否相同或相近").replace("意思相同输出1", "核心意思相同或相近输出1"))
    add("short_similar", "判断两个句子的语义是否相似。相似输出1，不相似输出0。\n" + pair + "\n只输出0或1。")
    add("short_label_anchor", instruction + "\n" + pair + "\n请只给出一个数字标签，不要补写任何句子。\n标签：")
    add("p0_label_anchor", p0.rstrip() + "\n答案：")
    add("intent_balanced", "判断两句话是否在表达同一个问题。同义词、语序变化和求助方式不同可以判1；仅话题相关而具体问题不同则判0。\n" + pair + "\n输出0或1：")
    add("service_intent", "比较两个用户表达的主要诉求。主要诉求相同为1，不同为0。\n" + pair + "\n仅输出数字标签。")
    for name, rows in (("table_2p2n", examples), ("table_3p1n", positive_examples)):
        table = "\n".join(" | ".join((str(index), row["sentence1"], row["sentence2"], str(row["label"]))) for index, row in enumerate(rows, 1))
        add(name, instruction + "\n\n已完成的示例（序号 | 句子1 | 句子2 | 标签）：\n" + table + "\n\n示例到此结束。只判断下面这一对：\n" + pair + "\n标签：", example_ids=[row["id"] for row in rows])
    marked = "\n\n".join("【示例" + str(index) + "】\n句子1：" + row["sentence1"] + "\n句子2：" + row["sentence2"] + "\n正确标签：" + str(row["label"]) for index, row in enumerate(examples, 1))
    add("marked_examples", instruction + "\n\n" + marked + "\n\n【唯一待判断的句对】\n" + pair + "\n只输出该句对的标签，不再编写示例。\n标签：", example_ids=[row["id"] for row in examples])
    add("p0_examples", p0.split("句子1：")[0] + "\n参考已标注示例：\n" + marked + "\n\n现在判断：\n" + pair + "\n只输出0或1，不要解释。", example_ids=[row["id"] for row in examples])
    for name, rows, template, system in (
        ("dialogue_short", examples, short, None),
        ("dialogue_3p1n", positive_examples, short, None),
        ("dialogue_p0", examples, p0, None),
        ("dialogue_system", examples, pair + "\n标签：", instruction + "只输出一个数字0或1。"),
    ):
        demonstrations = [{"user": template.format(sentence1=row["sentence1"], sentence2=row["sentence2"]), "assistant": str(row["label"])} for row in rows]
        add(name, template, system, demonstrations, [row["id"] for row in rows])
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "screen_seed": 2026102402,
        "screen_ids": [row["id"] for row in screen], "baseline_screen": baseline_screen,
        "baseline_full": evaluate(baseline), "candidates": candidates,
        "input_sha256": hashlib.sha256((data / "validation.jsonl").read_bytes()).hexdigest(),
        "train_sha256": hashlib.sha256((data / "train.jsonl").read_bytes()).hexdigest(),
        "ranking_rule": "Prefer simultaneous gains in accuracy, macro-F1 and positive-F1 versus short; otherwise maximize the minimum gain. Compare on the same rows.",
        "note": "Validation is reused development data; evaluation is untouched. Prompts include role-layout variants; quantization and decoding are fixed.",
    }
    if args.resume:
        previous = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for key in ("screen_ids", "input_sha256", "train_sha256"):
            assert previous[key] == manifest[key], key
        manifest = previous
        candidates = manifest["candidates"]
        manifest.setdefault("resumed_utc", []).append(datetime.now(timezone.utc).isoformat())
    else:
        save(folder / "manifest.json", manifest)
    client = LLMClient(quantization="nf4")
    client.warmup()
    if args.resume:
        for key in ("model_fingerprint", "quantization", "torch", "transformers", "dtype"):
            assert manifest["model_info"][key] == client.model_info[key], key
    manifest["model_info"] = client.model_info
    for candidate in candidates.values():
        candidate["model_fingerprint"] = client.model_info["model_fingerprint"]
    save(folder / "manifest.json", manifest)
    results = json.loads((folder / "results.json").read_text(encoding="utf-8")) if args.resume else {"screen": {}, "full": {}}

    def run(name, rows, stage):
        config = candidates[name]
        path = folder / f"{stage}_{name}.jsonl"
        predictions = read_records(path) if args.resume and path.exists() else []
        fields = ("id", "sentence1", "sentence2", "label")
        assert len(predictions) <= len(rows)
        assert [tuple(row[field] for field in fields) for row in predictions] == [tuple(row[field] for field in fields) for row in rows[:len(predictions)]]
        if name in results[stage]:
            assert len(predictions) == len(rows)
            assert evaluate(predictions) == results[stage][name]
            print(f"SKIP verified {stage} {name}", flush=True)
            return
        print(f"START {stage} {name} resume_at={len(predictions)}", flush=True)
        with path.open("a" if predictions else "x", encoding="utf-8") as destination:
            for index, row in enumerate(rows[len(predictions):], len(predictions) + 1):
                response = client.generate(config["prompt_template"].format(sentence1=row["sentence1"], sentence2=row["sentence2"]), system_prompt=config["system_prompt"], demonstrations=config["demonstrations"], enable_thinking=False, do_sample=False, max_new_tokens=8, seed=42, log=False)
                result = {**row, "prediction": response.prediction, "method": config["method"], "latency_ms": response.latency_ms}
                predictions.append(result)
                destination.write(json.dumps({**result, "raw_output": response.raw_output, "finish_reason": response.finish_reason}, ensure_ascii=False) + "\n")
                if index % 100 == 0 or index == len(rows):
                    destination.flush()
                    print(f"{stage} {name} {index}/{len(rows)} elapsed={(time.monotonic()-started)/60:.1f}min", flush=True)
        metric = evaluate(predictions)
        results[stage][name] = metric
        save(folder / "results.json", results)
        print(f"DONE {stage} {name} acc={metric['accuracy']:.4f} macro={metric['macro_f1']:.4f} posF1={metric['per_class']['1']['f1']:.4f} invalid={metric['invalid_count']}", flush=True)

    for name in candidates:
        run(name, screen, "screen")
    ranking = sorted(candidates, key=lambda name: rank_key(results["screen"][name], baseline_screen), reverse=True)
    manifest["screen_ranking"] = ranking
    save(folder / "manifest.json", manifest)
    for name in ranking[:3]:
        run(name, validation, "full")
    manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
    save(folder / "manifest.json", manifest)
    print("REVISION SEARCH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
