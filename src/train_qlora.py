"""Qwen3-1.7B 4-bit QLoRA，供 8GB NVIDIA GPU 小规模语义匹配试验。"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments,
)
from common import MODEL_ID, make_messages


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get('label') not in (0, 1):
                raise ValueError(f'{path}:{lineno} label 必须为整数 0 或 1')
            for key in ('id', 'sentence1', 'sentence2'):
                if key not in item:
                    raise ValueError(f'{path}:{lineno} 缺少 {key}')
            yield item


class MatchDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.samples = []
        self.dropped = 0
        for row in read_jsonl(path):
            # Qwen3 统一关闭 thinking。这里的 assistant 前缀必须与预测一致。
            prompt_ids = tokenizer.apply_chat_template(
                make_messages(row['sentence1'], row['sentence2']),
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
            )
            # 训练时只对答案和回合结束 token 计算 loss，绝不训练题目文本。
            answer_ids = tokenizer(str(row['label']), add_special_tokens=False)['input_ids']
            answer_ids = answer_ids + [tokenizer.eos_token_id]
            input_ids = prompt_ids + answer_ids
            if len(input_ids) > max_length:
                self.dropped += 1
                continue
            self.samples.append({
                'input_ids': input_ids,
                'attention_mask': [1] * len(input_ids),
                'labels': [-100] * len(prompt_ids) + answer_ids,
            })
        if not self.samples:
            raise ValueError(f'{path} 处理后没有样本。增大 --max-length 或检查数据。')
        print(f'{path}: 有效 {len(self.samples)} 条；超长排除 {self.dropped} 条')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        maxlen = max(len(f['input_ids']) for f in features)
        ids, mask, labels = [], [], []
        for f in features:
            padding = maxlen - len(f['input_ids'])
            ids.append(f['input_ids'] + [self.pad_id] * padding)
            mask.append(f['attention_mask'] + [0] * padding)
            labels.append(f['labels'] + [-100] * padding)
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=MODEL_ID, help='HF 模型 ID 或本地模型目录')
    p.add_argument('--train', default='data/demo/train.jsonl')
    p.add_argument('--validation', default='data/demo/validation.jsonl')
    p.add_argument('--output', default='outputs/smoke_r8')
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=int, default=16)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--epochs', type=float, default=2.0)
    p.add_argument('--max-steps', type=int, default=5, help='初始演示 5 step；正式训练设 -1')
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('QLoRA 训练需要 NVIDIA CUDA GPU。')
    if not (a.rank > 0 and a.alpha > 0 and a.grad_accum > 0):
        raise ValueError('rank/alpha/grad-accum 必须大于零')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    output_dir = Path(a.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data = MatchDataset(a.train, tokenizer, a.max_length)
    val_data = MatchDataset(a.validation, tokenizer, a.max_length)
    print('开始加载 4-bit 基础模型...')
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=dtype, quantization_config=quant_config,
        device_map={'': 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj'], bias='none', task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.learning_rate,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        eval_strategy='no',
        save_strategy='no',
        logging_steps=1,
        report_to='none',
        optim='adamw_torch',
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        max_grad_norm=0.3,
        seed=a.seed,
        data_seed=a.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    result = trainer.train()
    model.save_pretrained(output_dir / 'adapter')
    tokenizer.save_pretrained(output_dir / 'adapter')
    eval_metrics = trainer.evaluate()
    cfg = vars(a).copy()
    cfg.update({
        'dtype': str(dtype),
        'train_count': len(train_data),
        'train_dropped': train_data.dropped,
        'val_count': len(val_data),
        'val_dropped': val_data.dropped,
        'max_vram_reserved_gib': round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        'train_metrics': result.metrics,
        'eval_metrics': eval_metrics,
    })
    (output_dir / 'run_config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / 'trainer_log.json').write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SUCCESS: Adapter 保存到：', output_dir / 'adapter')
    print('验证 loss:', eval_metrics.get('eval_loss'))
    print('显存峰值 GiB:', cfg['max_vram_reserved_gib'])


if __name__ == '__main__':
    main()
