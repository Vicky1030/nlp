"""B 对齐的 AFQMC 提示、生成、解析、指纹与批量预测公共实现。"""

import copy
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


MODEL_ID = "Qwen/Qwen3-1.7B"
SEED = 42
MAX_NEW_TOKENS = 8
LATENCY_SCOPE = (
    "GPU model.generate only; synchronized; excludes model loading, "
    "tokenization, decoding and log writes"
)
TASK_PROMPT_TEMPLATE = (
    "请判断下面两个中文句子是否表达相同或相近的核心意思。\n\n"
    "判断规则：\n"
    "- 核心意思相同或相近，输出1。\n"
    "- 核心意思不同，输出0。\n"
    "- 不要仅根据共同关键词判断，要关注意图、否定和条件的差别。\n\n"
    "句子1：{sentence1}\n"
    "句子2：{sentence2}\n\n"
    "只输出0或1，不要解释。\n"
)
PROMPT_TEMPLATE_SHA256 = hashlib.sha256(
    TASK_PROMPT_TEMPLATE.encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class GenerationResult:
    text: str
    raw_output: str
    thinking_complete: bool | None
    prediction: int | None
    finish_reason: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    settings: dict

    def to_dict(self):
        return asdict(self)


def make_user_prompt(sentence1: str, sentence2: str) -> str:
    return TASK_PROMPT_TEMPLATE.format(sentence1=sentence1, sentence2=sentence2)


def make_messages(sentence1: str, sentence2: str):
    return [{"role": "user", "content": make_user_prompt(sentence1, sentence2)}]


def split_output(raw_output: str, enable_thinking: bool = False):
    if "</think>" in raw_output:
        return raw_output.rsplit("</think>", 1)[1].strip(), True
    if enable_thinking or "<think>" in raw_output:
        return "", False
    return raw_output.strip(), None


def parse_prediction(
    raw_output: str, *, enable_thinking: bool = False, finish_reason: str = "eos"
):
    answer, thinking_complete = split_output(raw_output, enable_thinking)
    prediction = (
        int(answer)
        if answer in ("0", "1") and finish_reason != "length"
        else None
    )
    return answer, thinking_complete, prediction


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fingerprint_model(model_path: Path):
    """使用与 B 分支 llm_client.py 相同的模型组合指纹算法。"""
    model_path = Path(model_path).resolve()
    filenames = {
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt",
        "chat_template.jinja", "special_tokens_map.json", "added_tokens.json",
    }
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        filenames.add(index_path.name)
        for filename in index["weight_map"].values():
            if Path(filename).name != filename:
                raise ValueError("Weight index must reference files directly inside model_path")
            if not (model_path / filename).is_file():
                raise FileNotFoundError(model_path / filename)
            filenames.add(filename)
    else:
        filenames.update(path.name for path in model_path.glob("*.safetensors"))
    if not any(name.endswith(".safetensors") for name in filenames):
        raise FileNotFoundError("No local safetensors weights found")
    files = {}
    for filename in sorted(filenames):
        path = model_path / filename
        if not path.is_file():
            continue
        files[filename] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return fingerprint, files


def fingerprint_directory(directory: Path):
    directory = Path(directory).resolve()
    files = {}
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if not files:
        raise FileNotFoundError(f"No files found in {directory}")
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return fingerprint, files


def render_model_inputs(tokenizer, messages, device):
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    return prompt_text, inputs


def make_generation_config(model, tokenizer):
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.max_new_tokens = MAX_NEW_TOKENS
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    generation_config.pad_token_id = tokenizer.eos_token_id
    return generation_config


def generate_prediction(model, tokenizer, messages, device, seed: int = SEED):
    prompt_text, inputs = render_model_inputs(tokenizer, messages, device)
    input_tokens = inputs["input_ids"].shape[1]
    if input_tokens + MAX_NEW_TOKENS > model.config.max_position_embeddings:
        raise ValueError("Input plus output budget exceeds model context")
    generation_config = make_generation_config(model, tokenizer)
    settings = {
        "enable_thinking": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "seed": seed,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "use_model_defaults": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device_index]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                generation_config=generation_config,
                use_model_defaults=False,
            )
        torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000
    tokens = generated[0, input_tokens:].tolist()
    eos_ids = generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, (tuple, list)) else [eos_ids])
    finish_reason = (
        "eos"
        if tokens and tokens[-1] in eos_ids
        else "length"
        if len(tokens) >= MAX_NEW_TOKENS
        else "stopped"
    )
    raw_output = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    answer, thinking_complete, prediction = parse_prediction(
        raw_output,
        enable_thinking=False,
        finish_reason=finish_reason,
    )
    result = GenerationResult(
        text=answer,
        raw_output=raw_output,
        thinking_complete=thinking_complete,
        prediction=prediction,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=len(tokens),
        latency_ms=latency_ms,
        settings=settings,
    )
    return prompt_text, result


def read_prediction_records(input_path: Path, max_samples: int):
    input_path = Path(input_path).resolve()
    content = input_path.read_bytes()
    all_records = [
        json.loads(line)
        for line in content.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not all_records:
        raise ValueError("Input data must be nonempty")
    if max_samples < 0 or max_samples > len(all_records):
        raise ValueError(f"max_samples must be between 0 and {len(all_records)}")
    seen_ids = set()
    for record in all_records:
        if set(record) != {"id", "sentence1", "sentence2", "label"}:
            raise ValueError("Input must use the four-field AFQMC data format")
        if type(record["id"]) is not int or record["id"] in seen_ids:
            raise ValueError("Sample IDs must be unique integers")
        if type(record["label"]) is not int or record["label"] not in (0, 1):
            raise ValueError("Labels must be integers 0 or 1")
        if any(
            not isinstance(record[key], str) or not record[key].strip()
            for key in ("sentence1", "sentence2")
        ):
            raise ValueError("Sentences must be nonempty strings")
        seen_ids.add(record["id"])
    records = all_records if max_samples == 0 else all_records[:max_samples]
    return content, all_records, records


def write_json(path: Path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_prediction_file(
    *, model, tokenizer, input_path, output_path, method, model_info,
    max_samples=0, seed=SEED,
):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    summary_path = output_path.with_suffix(".summary.json")
    calls_path = output_path.with_suffix(".calls.jsonl")
    for path in (output_path, summary_path, calls_path):
        if path.exists():
            raise FileExistsError(f"Output already exists; choose another path: {path}")
    content, all_records, records = read_prediction_records(input_path, max_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", torch.cuda.current_device())
    seed_everything(seed)
    warmup_messages = make_messages("花呗怎么提高额度", "如何提升花呗额度")
    generate_prediction(model, tokenizer, warmup_messages, device, seed=seed)
    summary = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "input_sha256": hashlib.sha256(content).hexdigest(),
        "selection": "first N records in file order",
        "input_total": len(all_records),
        "selected_ids": [record["id"] for record in records],
        "requested_count": len(records),
        "completed": 0,
        "correct": 0,
        "invalid_count": 0,
        "method": method,
        "prompt_template": TASK_PROMPT_TEMPLATE,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "model_info": model_info,
        "generation_settings": None,
        "latency_scope": LATENCY_SCOPE,
        "invalid_policy": "Keep prediction=null and count it as incorrect",
    }
    write_json(summary_path, summary)
    total_latency_ms = 0.0
    try:
        with output_path.open("x", encoding="utf-8") as destination, calls_path.open(
            "x", encoding="utf-8"
        ) as calls:
            for completed, record in enumerate(records, start=1):
                messages = make_messages(record["sentence1"], record["sentence2"])
                prompt_text, response = generate_prediction(
                    model, tokenizer, messages, device, seed=seed
                )
                prediction_record = {
                    "id": record["id"],
                    "sentence1": record["sentence1"],
                    "sentence2": record["sentence2"],
                    "label": record["label"],
                    "prediction": response.prediction,
                    "method": method,
                    "latency_ms": round(response.latency_ms, 3),
                }
                call_record = {
                    "event": "generation",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "id": record["id"],
                    "messages": messages,
                    "rendered_prompt": prompt_text,
                    "model_fingerprint": model_info["model_fingerprint"],
                    **response.to_dict(),
                }
                destination.write(json.dumps(prediction_record, ensure_ascii=False) + "\n")
                destination.flush()
                calls.write(json.dumps(call_record, ensure_ascii=False) + "\n")
                calls.flush()
                total_latency_ms += response.latency_ms
                summary["completed"] = completed
                summary["correct"] += int(response.prediction == record["label"])
                summary["invalid_count"] += int(response.prediction is None)
                summary["generation_settings"] = response.settings
                write_json(summary_path, summary)
                if completed % 10 == 0 or completed == len(records):
                    print(
                        f"已完成 {completed}/{len(records)}，正确 {summary['correct']}，"
                        f"无效输出 {summary['invalid_count']}",
                        flush=True,
                    )
    except (Exception, KeyboardInterrupt) as error:
        summary.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(summary_path, summary)
        raise
    summary.update(
        status="complete",
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        accuracy=summary["correct"] / len(records),
        invalid_rate=summary["invalid_count"] / len(records),
        mean_latency_ms=total_latency_ms / len(records),
    )
    write_json(summary_path, summary)
    return output_path, summary_path, calls_path, summary
