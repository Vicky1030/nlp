"""以 B 对齐协议运行 Qwen3-1.7B 的 AFQMC NF4 原始模型基准。"""
import argparse
import importlib.metadata
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import MODEL_ID, fingerprint_model, run_prediction_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=MODEL_ID)
    p.add_argument('--input', default='data/demo/validation.jsonl')
    p.add_argument('--output', default='results/base_validation_full.jsonl')
    p.add_argument('--method', default='base_4bit')
    p.add_argument('--max-samples', type=int, default=0, help='0=全部样本')
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('NF4 推理需要 NVIDIA CUDA GPU。')
    if not torch.cuda.is_bf16_supported():
        raise SystemExit('本实验固定 NF4 + BF16 compute；当前 GPU 不支持 BF16。')
    model_path = Path(a.model).resolve()
    if not (model_path / 'config.json').is_file():
        raise FileNotFoundError(f'本实验要求本地模型目录：{model_path}')
    model_fingerprint, model_files = fingerprint_model(model_path)
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
    model = base.eval()
    torch.cuda.synchronize()
    model_info = {
        'model_id': MODEL_ID,
        'model_path': str(model_path),
        'model_fingerprint': model_fingerprint,
        'files': model_files,
        'model_type': model.config.model_type,
        'declared_revision': getattr(model.config, '_commit_hash', None),
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'bitsandbytes': importlib.metadata.version('bitsandbytes'),
        'gpu': torch.cuda.get_device_name(0),
        'dtype': str(torch.bfloat16),
        'quantization': {
            'load_in_4bit': True,
            'bnb_4bit_quant_type': 'nf4',
            'bnb_4bit_use_double_quant': True,
            'bnb_4bit_compute_dtype': str(torch.bfloat16),
        },
        'adapter': None,
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
