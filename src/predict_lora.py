"""用 B 对齐协议在 NF4 基座上加载 LoRA Adapter 并批量预测。"""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import (
    MODEL_ID, fingerprint_directory, fingerprint_model, run_prediction_file,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=MODEL_ID)
    p.add_argument('--adapter', required=True)
    p.add_argument('--input', default='data/demo/validation.jsonl')
    p.add_argument('--output', default='results/lora_demo.jsonl')
    p.add_argument('--method', default='lora_r8')
    p.add_argument('--max-samples', type=int, default=0, help='0=全部样本')
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('NF4 推理需要 NVIDIA CUDA GPU。')
    if not torch.cuda.is_bf16_supported():
        raise SystemExit('本实验固定 NF4 + BF16 compute；当前 GPU 不支持 BF16。')
    model_path = Path(a.model).resolve()
    adapter_path = Path(a.adapter).resolve()
    if not (model_path / 'config.json').is_file():
        raise FileNotFoundError(f'本实验要求本地模型目录：{model_path}')
    if not (adapter_path / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'Adapter 配置不存在：{adapter_path}')
    model_fingerprint, model_files = fingerprint_model(model_path)
    adapter_fingerprint, adapter_files = fingerprint_directory(adapter_path)
    adapter_config = json.loads(
        (adapter_path / 'adapter_config.json').read_text(encoding='utf-8')
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.bfloat16,
        quantization_config=quant_config, device_map={'': 0},
    )
    model = PeftModel.from_pretrained(base, adapter_path).eval()
    torch.cuda.synchronize()
    model_info = {
        'model_id': MODEL_ID,
        'model_path': str(model_path),
        'model_fingerprint': model_fingerprint,
        'files': model_files,
        'model_type': base.config.model_type,
        'declared_revision': getattr(base.config, '_commit_hash', None),
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'peft': importlib.metadata.version('peft'),
        'bitsandbytes': importlib.metadata.version('bitsandbytes'),
        'gpu': torch.cuda.get_device_name(0),
        'dtype': str(torch.bfloat16),
        'quantization': {
            'load_in_4bit': True,
            'bnb_4bit_quant_type': 'nf4',
            'bnb_4bit_use_double_quant': True,
            'bnb_4bit_compute_dtype': str(torch.bfloat16),
        },
        'adapter': {
            'path': str(adapter_path),
            'fingerprint': adapter_fingerprint,
            'files': adapter_files,
            'config': adapter_config,
        },
        'model_load_ms': (time.perf_counter() - started) * 1000,
    }
    output, summary, calls, result = run_prediction_file(
        model=model,
        tokenizer=tokenizer,
        input_path=a.input,
        output_path=a.output,
        method=a.method,
        model_info=model_info,
        max_samples=a.max_samples,
    )
    print(f'完成 {result["completed"]} 条；正确 {result["correct"]} 条；无效 {result["invalid_count"]} 条')
    print('预测文件：', output)
    print('统计文件：', summary)
    print('调用日志：', calls)


if __name__ == '__main__':
    main()
