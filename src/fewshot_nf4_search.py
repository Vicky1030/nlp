import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from llm_client import LLMClient
from work.compare_nf4_baseline import evaluate, read_records

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/AFQMC_全组统一数据包/data/processed"
PAIR = "句子1：{sentence1}\n句子2：{sentence2}"
INSTRUCTIONS = {
    "exact": "判断两个句子的核心意思是否相同。核心意思相同输出1，意思不同输出0。",
    "near": "判断两个句子的核心意思是否相同或相近。核心意思相同或相近输出1，意思不同输出0。",
    "plain": "判断两个句子的意思是否相同。意思相同输出1，意思不同输出0。",
}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize(sentence):
    return re.sub(r"[^\w]", "", sentence).lower()


def pair_key(row):
    return tuple(sorted((normalize(row["sentence1"]), normalize(row["sentence2"]))))


def features(row):
    counts = Counter()
    for field in ("sentence1", "sentence2"):
        sentence = normalize(row[field])
        for width in (1, 2):
            counts.update(sentence[offset:offset + width] for offset in range(len(sentence) - width + 1))
    return counts


class Retriever:
    def __init__(self, training):
        self.training = training
        vectors = [features(row) for row in training]
        document_frequency = Counter(gram for vector in vectors for gram in vector)
        self.idf = {gram: math.log((1 + len(training)) / (1 + count)) + 1 for gram, count in document_frequency.items()}
        postings = defaultdict(list)
        for index, vector in enumerate(vectors):
            weights = {gram: (1 + math.log(count)) * self.idf[gram] for gram, count in vector.items()}
            norm = math.sqrt(sum(value * value for value in weights.values())) or 1
            for gram, weight in weights.items():
                postings[gram].append((index, weight / norm))
        self.postings = {gram: (np.array([index for index, weight in items], dtype=np.int32),
                               np.array([weight for index, weight in items], dtype=np.float32)) for gram, items in postings.items()}
        self.keys = [pair_key(row) for row in training]
        self.labels = np.array([row["label"] for row in training])

    def retrieve(self, sentence1, sentence2):
        query = {"sentence1": sentence1, "sentence2": sentence2}
        weights = {gram: (1 + math.log(count)) * self.idf[gram] for gram, count in features(query).items() if gram in self.idf}
        norm = math.sqrt(sum(value * value for value in weights.values())) or 1
        scores = np.zeros(len(self.training), dtype=np.float32)
        for gram, weight in weights.items():
            indices, values = self.postings[gram]
            scores[indices] += values * (weight / norm)
        query_key = pair_key(query)
        results = {}
        for name, label in (("all", None), ("negative", 0), ("positive", 1)):
            pool = np.arange(len(self.training)) if label is None else np.flatnonzero(self.labels == label)
            top = pool[np.argpartition(-scores[pool], min(127, len(pool) - 1))[:128]]
            ordered = sorted(top.tolist(), key=lambda index: (-float(scores[index]), index))
            chosen = []
            seen = {query_key}
            for index in ordered:
                if self.keys[index] in seen:
                    continue
                seen.add(self.keys[index])
                chosen.append({"id": self.training[index]["id"], "score": float(scores[index])})
                if len(chosen) == 16:
                    break
            assert len(chosen) == 16
            results[name] = chosen
        return results


def select_examples(config, query, training, retrieval):
    if config["selection"] == "fixed":
        examples = [training[identity] for identity in config["example_ids"]]
    else:
        if config["selection"] in ("balanced", "negative_heavy"):
            positives = config["shots"] // 2 if config["selection"] == "balanced" else 1
            negatives = config["shots"] - positives
            selected = retrieval["negative"][:negatives] + retrieval["positive"][:positives]
            selected.sort(key=lambda row: (row["score"], row["id"]))
        else:
            selected = list(reversed(retrieval["all"][:config["shots"]]))
        if config.get("order") == "near_first":
            selected.reverse()
        examples = [training[row["id"]] for row in selected]
    assert len(examples) == config["shots"]
    assert all(pair_key(example) != pair_key(query) for example in examples)
    return examples


def build_prompt(config, sentence1, sentence2, examples):
    instruction = INSTRUCTIONS[config["instruction"]]
    query = PAIR.format(sentence1=sentence1, sentence2=sentence2)
    ending = "\n只输出0或1，不要解释。"
    if config["layout"] == "dialogue":
        demonstrations = [{"user": instruction + "\n" + PAIR.format(**row) + ending, "assistant": str(row["label"])} for row in examples]
        return instruction + "\n" + query + ending, demonstrations
    if config["layout"] == "compact":
        blocks = [row["sentence1"] + " / " + row["sentence2"] + " → " + str(row["label"]) for row in examples]
        return instruction + "\n示例：\n" + "\n".join(blocks) + "\n\n" + query + ending, None
    blocks = [PAIR.format(**row) + "\n答案：" + str(row["label"]) for row in examples]
    if config["layout"] == "instruction_last":
        return "参考示例：\n" + "\n\n".join(blocks) + "\n\n" + instruction + "\n" + query + ending, None
    prompt = instruction + "\n\n参考示例：\n" + "\n\n".join(blocks) + "\n\n请判断：\n" + query + ending
    return prompt, None


def candidate_configs(refine=False):
    configs = {}

    def add(name, selection, shots, instruction="exact", layout="inline", example_ids=None, order="near_last"):
        configs[name] = {"candidate_name": name, "selection": selection, "shots": shots, "instruction": instruction,
                         "layout": layout, "example_ids": example_ids or [], "order": order,
                         "quantization": "nf4", "enable_thinking": False, "do_sample": False, "max_new_tokens": 8, "seed": 42}

    for layout in ("inline", "dialogue"):
        add(f"fixed2_{layout}", "fixed", 2, layout=layout, example_ids=[118, 18614])
        add(f"fixed2_reverse_{layout}", "fixed", 2, layout=layout, example_ids=[18614, 118])
        add(f"fixed4_{layout}", "fixed", 4, layout=layout, example_ids=[118, 18614, 14112, 29811])
        for selection in ("balanced", "nearest"):
            for shots in (2, 4, 8):
                add(f"{selection}{shots}_{layout}", selection, shots, layout=layout)
    add("balanced4_near", "balanced", 4, instruction="near")
    add("balanced4_plain", "balanced", 4, instruction="plain")
    add("balanced4_near_first", "balanced", 4, order="near_first")
    add("nearest4_near", "nearest", 4, instruction="near")
    if refine:
        configs.clear()
        for instruction in ("near", "plain"):
            add(f"fixed2_{instruction}", "fixed", 2, instruction=instruction, example_ids=[118, 18614])
            add(f"fixed4_{instruction}", "fixed", 4, instruction=instruction, example_ids=[118, 18614, 14112, 29811])
        for instruction in ("exact", "near"):
            add(f"fixed2_last_{instruction}", "fixed", 2, instruction=instruction, layout="instruction_last", example_ids=[118, 18614])
        add("fixed1_positive", "fixed", 1, example_ids=[118])
        add("fixed1_negative", "fixed", 1, example_ids=[18614])
        for instruction in ("exact", "plain"):
            add(f"negative_heavy4_{instruction}", "negative_heavy", 4, instruction=instruction)
        add("fixed2_compact", "fixed", 2, layout="compact", example_ids=[118, 18614])
        add("fixed4_compact", "fixed", 4, layout="compact", example_ids=[118, 18614, 14112, 29811])
        add("balanced4_compact", "balanced", 4, layout="compact")
        add("nearest4_compact", "nearest", 4, layout="compact")
        add("fresh2_exact", "fixed", 2, example_ids=[117, 240])
        add("fresh2_near", "fixed", 2, instruction="near", example_ids=[117, 240])
        add("fresh4_exact", "fixed", 4, example_ids=[117, 240, 143, 76])
        add("fresh4_compact", "fixed", 4, layout="compact", example_ids=[117, 240, 143, 76])
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=args.resume)
    training_rows = read_records(DATA / "train.jsonl")
    validation = read_records(DATA / "validation.jsonl")
    training = {row["id"]: row for row in training_rows}
    assert len(training) == len(training_rows)
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        configs = {config["candidate_name"]: config}
    else:
        configs = candidate_configs(args.refine)
    rng = random.Random(2026102402)
    screen = []
    for label, count in ((0, 332), (1, 148)):
        screen.extend(rng.sample([row for row in validation if row["label"] == label], count))
    rng.shuffle(screen)
    manifest = {"configs": configs, "screen_ids": [row["id"] for row in screen], "instruction_templates": INSTRUCTIONS,
                "train_sha256": hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest(),
                "validation_sha256": hashlib.sha256((DATA / "validation.jsonl").read_bytes()).hexdigest(),
                "retrieval": "Train-only char1/2 sublinear-TF smoothed-IDF cosine; normalized unordered exact query pairs excluded; train pair deduplication. Closest example last by default. Query labels not used.",
                "selection_rule": "Screen accuracy>=0.70 first, then macro-F1 and positive recall; top3 full. Final eligibility requires full accuracy>=0.70. Reused validation, no evaluation."}
    if args.resume:
        previous = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        assert all(previous[key] == manifest[key] for key in manifest)
    save(folder / "manifest.json", manifest)
    retrieval_path = folder / "retrieval.json"
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8")) if args.resume and retrieval_path.exists() else {}
    needed = validation[:args.limit] if args.config and args.limit else validation
    if not retrieval:
        retriever = Retriever(training_rows)
        for index, row in enumerate(needed, 1):
            retrieval[str(row["id"])] = retriever.retrieve(row["sentence1"], row["sentence2"])
            if index % 500 == 0:
                print("RETRIEVAL", index, "/", len(needed), flush=True)
        save(retrieval_path, retrieval)
        del retriever
    client = LLMClient(quantization="nf4")
    expected = json.loads((ROOT / "results" / "near_conditions.config.json").read_text(encoding="utf-8"))
    assert client.model_info["model_fingerprint"] == expected["model_fingerprint"]
    save(folder / "model_info.json", client.model_info)
    client.warmup()
    results = {"screen": {}, "full": {}}

    def run(name, rows, stage):
        config = configs[name]
        destination = folder / f"{stage}_{name}.jsonl"
        predictions = read_records(destination) if args.resume and destination.exists() else []
        fields = ("id", "sentence1", "sentence2", "label")
        assert len(predictions) <= len(rows)
        assert [tuple(row[key] for key in fields) for row in predictions] == [tuple(row[key] for key in fields) for row in rows[:len(predictions)]]
        with destination.open("a" if predictions else "x", encoding="utf-8") as stream:
            for index, row in enumerate(rows[len(predictions):], len(predictions) + 1):
                query = {key: row[key] for key in ("sentence1", "sentence2")}
                examples = select_examples(config, query, training, retrieval[str(row["id"])])
                prompt, demonstrations = build_prompt(config, row["sentence1"], row["sentence2"], examples)
                response = client.generate(prompt, demonstrations=demonstrations, enable_thinking=False,
                                           do_sample=False, max_new_tokens=8, seed=42, log=False)
                result = {**row, "prediction": response.prediction, "method": "few_shot", "latency_ms": response.latency_ms,
                          "raw_output": response.raw_output, "finish_reason": response.finish_reason,
                          "example_ids": [example["id"] for example in examples],
                          "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
                predictions.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                if index % 200 == 0:
                    print(stage, name, index, "/", len(rows), flush=True)
        metrics = evaluate(predictions)
        results[stage][name] = metrics
        save(folder / "results.json", results)
        print("DONE", stage, name, "acc", round(metrics["accuracy"], 4), "macro", round(metrics["macro_f1"], 4),
              "recall", round(metrics["per_class"]["1"]["recall"], 4), "invalid", metrics["invalid_count"], flush=True)

    if args.config:
        run(next(iter(configs)), needed, "full")
    else:
        for name in configs:
            run(name, screen, "screen")
        ranking = sorted(configs, key=lambda name: (results["screen"][name]["accuracy"] >= .70,
                         results["screen"][name]["macro_f1"], results["screen"][name]["per_class"]["1"]["recall"]), reverse=True)
        save(folder / "ranking.json", ranking)
        for name in ranking[:3]:
            run(name, validation, "full")
    print("FEWSHOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
