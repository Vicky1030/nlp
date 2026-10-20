"""原始 Qwen3-1.7B 的 AFQMC 批量验证，与 predict_lora.py 保持相同推理协议。"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from common import MODEL_ID, make_messages


def parse_prediction(text):
    value = text.strip()
    match = re.fullmatch(r'([01])\s*[。.]?', value)
    return int(match.group(1)) if match else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=MODEL_ID)
    p.add_argument('--input', default='data/demo/validation.jsonl')
    p.add_argument('--output', default='results/base_validation_full.jsonl')
    p.add_argument('--method', default='base_4bit')
    p.add_argument('--max-samples', type=int, default=0, help='0=全部样本')
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('此示例使用 CUDA 4-bit 推理；请在支持 CUDA 的机器运行。')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
    )
    base = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=dtype, quantization_config=quant_config, device_map={'': 0},
    )
    model = base
    model.eval()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    count = 0
    correct = 0
    # 加载模型和读盘不计入 latency_ms；首次推理可能包含预热开销。
    with open(a.input, encoding='utf-8') as fin, open(a.output, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            ids = tokenizer.apply_chat_template(
                make_messages(row['sentence1'], row['sentence2']),
                tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_tensors='pt',
            ).to('cuda')
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=ids, max_new_tokens=8, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            latency = (time.perf_counter() - start) * 1000
            answer = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True)
            prediction = parse_prediction(answer)
            if prediction is None:
                errors += 1
                print(f'警告: id={row["id"]} 输出无法解析: {answer!r}')
            if prediction is not None and prediction == row.get('label'):
                correct += 1
            result = {
                'id': row['id'], 'sentence1': row['sentence1'],
                'sentence2': row['sentence2'], 'label': row.get('label'),
                'prediction': prediction, 'method': a.method,
                'latency_ms': round(latency, 3),
            }
            fout.write(json.dumps(result, ensure_ascii=False) + '\n')
            count += 1
            if a.max_samples > 0 and count >= a.max_samples:
                break
    print(f'完成 {count} 条预测；解析失败 {errors} 条；正确 {correct} 条；结果写入 {a.output}')
    print('注意：首次推理含 CUDA 初始化/预热成本，正式延迟分析应单独预热并统一计时口径。')


if __name__ == '__main__':
    main()
