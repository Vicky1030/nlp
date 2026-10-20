"""测试 RTX 5060 是否能加载 Qwen3-1.7B 并完成不更新权重的 0/1 推理。"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from common import MODEL_ID, make_messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=MODEL_ID)
    parser.add_argument('--precision', choices=('4bit', 'bf16'), default='4bit')
    parser.add_argument('--sentence1', default='花呗怎么提高额度')
    parser.add_argument('--sentence2', default='如何提升花呗额度')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA 不可用。请先运行 python src/check_env.py。')

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if args.precision == 'bf16' and dtype != torch.bfloat16:
        raise SystemExit('本机不支持 BF16；请选择 --precision 4bit。')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {'dtype': dtype, 'device_map': {'': 0}}
    if args.precision == '4bit':
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    print(f'加载 Qwen3-1.7B (mode={args.precision}, compute={dtype}) ...')
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    inputs = tokenizer.apply_chat_template(
        make_messages(args.sentence1, args.sentence2),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors='pt',
    ).to('cuda')
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=8,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    answer = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True
    ).strip()
    print('问题1：', args.sentence1)
    print('问题2：', args.sentence2)
    print('模型原始回答：', repr(answer))
    print('推理耗时（首次含预热）：', round((time.perf_counter()-start)*1000, 2), 'ms')
    print('显存峰值：', round(torch.cuda.max_memory_reserved() / (1024**3), 3), 'GiB')
    print('SUCCESS: 基础模型已实际完成推理。')
    if answer not in ('0', '1'):
        print('提示：模型成功运行，但原始回答不符合 0/1 格式；这属于输出解析问题。')


if __name__ == '__main__':
    main()
