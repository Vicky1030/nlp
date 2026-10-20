import argparse
import copy
import errno
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
LATENCY_SCOPE = "GPU model.generate only; synchronized; excludes model loading, tokenization, decoding, retries and log writes"


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
    request_latency_ms: float
    attempts: int
    model_id: str
    model_fingerprint: str
    settings: dict

    def to_dict(self):
        return asdict(self)


def split_output(raw_output, enable_thinking):
    if "</think>" in raw_output:
        return raw_output.rsplit("</think>", 1)[1].strip(), True
    if enable_thinking or "<think>" in raw_output:
        return "", False
    return raw_output.strip(), None


def fingerprint_model(model_path):
    filenames = {
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
        "special_tokens_map.json", "added_tokens.json",
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
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        files[filename] = {"bytes": path.stat().st_size, "sha256": digest}
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return fingerprint, files


class LLMClient:
    def __init__(
        self, model_path=ROOT / "models", *, model_id="Qwen/Qwen3-1.7B",
        log_path=None, max_retries=1, retry_delay=0.5,
    ):
        if type(max_retries) is not int or not 0 <= max_retries <= 3:
            raise ValueError("max_retries must be an integer between 0 and 3")
        if not isinstance(retry_delay, (int, float)) or not math.isfinite(retry_delay) or retry_delay < 0:
            raise ValueError("retry_delay must be finite and nonnegative")
        self.model_path = Path(model_path).resolve()
        if not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(f"Local model config missing: {self.model_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use this project's CUDA-enabled Python environment")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model_id = model_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.log_path = Path(log_path).resolve() if log_path is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint, files = fingerprint_model(self.model_path)
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=self.dtype,
        ).to(self.device).eval()
        torch.cuda.synchronize(self.device)
        self._model_info = {
            "model_id": model_id, "model_path": str(self.model_path),
            "model_fingerprint": fingerprint, "files": files,
            "model_type": self.model.config.model_type,
            "declared_revision": getattr(self.model.config, "_commit_hash", None),
            "version_note": "Local content fingerprint identifies actual files; publisher revision is not inferred",
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(self.device), "dtype": str(self.dtype),
            "quantization": None, "model_load_ms": (time.perf_counter() - started) * 1000,
            "latency_scope": LATENCY_SCOPE,
        }
        self._log("model_loaded", self._model_info)

    @property
    def model_info(self):
        return copy.deepcopy(self._model_info)

    def _log(self, event, payload):
        if self.log_path is not None:
            record = {"event": event, "timestamp_utc": datetime.now(timezone.utc).isoformat(), **payload}
            with self.log_path.open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    def generate(
        self, prompt, *, system_prompt=None, enable_thinking=False,
        max_new_tokens=None, do_sample=None, temperature=0.6, top_p=0.95,
        top_k=20, seed=42, log=True,
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
            raise ValueError("system_prompt must be a nonempty string or None")
        if type(enable_thinking) is not bool:
            raise ValueError("enable_thinking must be a bool")
        if max_new_tokens is None:
            max_new_tokens = 2048 if enable_thinking else 8
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if do_sample is None:
            do_sample = enable_thinking
        if type(do_sample) is not bool:
            raise ValueError("do_sample must be a bool")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if do_sample:
            if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature must be finite and positive")
            if not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0 < top_p <= 1:
                raise ValueError("top_p must be in (0, 1]")
            if type(top_k) is not int or top_k < 0:
                raise ValueError("top_k must be a nonnegative integer")
        settings = {
            "enable_thinking": enable_thinking, "max_new_tokens": max_new_tokens,
            "do_sample": do_sample, "seed": seed,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None, "top_k": top_k if do_sample else None,
        }
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        request_started = time.perf_counter()
        for attempt in range(1, self.max_retries + 2):
            try:
                result = self._generate_once(messages, settings, attempt, request_started)
            except Exception as error:
                transient = isinstance(error, (TimeoutError, ConnectionError)) or (
                    isinstance(error, OSError) and error.errno in {errno.EAGAIN, errno.EINTR, errno.ETIMEDOUT}
                )
                retry = transient and attempt <= self.max_retries
                if log:
                    self._log("generation_error", {
                        "model_fingerprint": self._model_info["model_fingerprint"],
                        "messages": messages, "settings": settings, "attempt": attempt,
                        "error_type": type(error).__name__, "error": str(error), "will_retry": retry,
                    })
                if not retry:
                    raise
                time.sleep(self.retry_delay * attempt)
                continue
            if log:
                self._log("generation", {"messages": messages, **result.to_dict()})
            return result

    def _generate_once(self, messages, settings, attempt, request_started):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=settings["enable_thinking"],
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_tokens = inputs["input_ids"].shape[1]
        if input_tokens + settings["max_new_tokens"] > self.model.config.max_position_embeddings:
            raise ValueError("Input plus output budget exceeds model context; shorten the prompt or output budget")
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.do_sample = settings["do_sample"]
        generation_config.max_new_tokens = settings["max_new_tokens"]
        generation_config.temperature = settings["temperature"]
        generation_config.top_p = settings["top_p"]
        generation_config.top_k = settings["top_k"]
        generation_config.pad_token_id = self.tokenizer.eos_token_id
        with torch.random.fork_rng(devices=[self.device.index]):
            torch.random.default_generator.manual_seed(settings["seed"])
            torch.cuda.default_generators[self.device.index].manual_seed(settings["seed"])
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs, generation_config=generation_config, use_model_defaults=False,
                )
            torch.cuda.synchronize(self.device)
            latency_ms = (time.perf_counter() - started) * 1000
        tokens = generated[0, input_tokens:].tolist()
        eos_ids = generation_config.eos_token_id
        eos_ids = set(eos_ids if isinstance(eos_ids, (tuple, list)) else [eos_ids])
        finish_reason = "eos" if tokens and tokens[-1] in eos_ids else "length" if len(tokens) >= settings["max_new_tokens"] else "stopped"
        raw_output = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        answer, thinking_complete = split_output(raw_output, settings["enable_thinking"])
        prediction = int(answer) if answer in ("0", "1") and finish_reason != "length" else None
        return GenerationResult(
            text=answer, raw_output=raw_output, thinking_complete=thinking_complete,
            prediction=prediction, finish_reason=finish_reason,
            input_tokens=input_tokens, output_tokens=len(tokens),
            latency_ms=latency_ms, request_latency_ms=(time.perf_counter() - request_started) * 1000,
            attempts=attempt, model_id=self.model_id,
            model_fingerprint=self._model_info["model_fingerprint"], settings=dict(settings),
        )

    def warmup(self):
        self.generate("只输出数字0。", max_new_tokens=8, log=False)


def main():
    parser = argparse.ArgumentParser(description="Local Qwen model client; no network downloads")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models")
    parser.add_argument("--prompt", default="句子1：花呗怎么提高额度？\n句子2：如何提升花呗额度？")
    parser.add_argument("--system-prompt", default="判断两个句子是否表达相同或相近的意思。相似输出1，不相似输出0。最终只输出0或1。")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", type=Path, default=ROOT / "outputs" / "llm_client_calls.jsonl")
    args = parser.parse_args()
    client = LLMClient(args.model_path, log_path=args.log)
    client.warmup()
    result = client.generate(
        args.prompt, system_prompt=args.system_prompt, enable_thinking=args.thinking,
        max_new_tokens=args.max_new_tokens, seed=args.seed,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
