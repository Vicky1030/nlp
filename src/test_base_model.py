"""测试 RTX 5060 是否能加载 Qwen3-1.7B 并完成不更新权重的 0/1 推理。"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from common import MODEL_ID, generate_prediction, make_messages


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
    device = torch.device('cuda', torch.cuda.current_device())
    generate_prediction(
        model, tokenizer,
        make_messages('花呗怎么提高额度', '如何提升花呗额度'),
        device,
    )
    _, result = generate_prediction(
        model, tokenizer, make_messages(args.sentence1, args.sentence2), device,
    )
    print('问题1：', args.sentence1)
    print('问题2：', args.sentence2)
    print('模型原始回答：', repr(result.raw_output))
    print('解析后的预测：', result.prediction)
    print('结束原因：', result.finish_reason)
    print('推理耗时（已预热）：', round(result.latency_ms, 2), 'ms')
    print('显存峰值：', round(torch.cuda.max_memory_reserved() / (1024**3), 3), 'GiB')
    print('SUCCESS: 基础模型已实际完成推理。')
    if result.prediction is None:
        print('提示：模型成功运行，但原始回答不符合 0/1 格式；这属于输出解析问题。')


if __name__ == '__main__':
    main()
