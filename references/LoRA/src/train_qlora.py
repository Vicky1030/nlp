"""Qwen3-1.7B 4-bit QLoRA，供 8GB NVIDIA GPU 小规模语义匹配试验。"""
import argparse
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments,
)
from common import (
    MODEL_ID, PROMPT_TEMPLATE_SHA256, TASK_PROMPT_TEMPLATE,
    file_sha256, fingerprint_directory, fingerprint_model,
    make_messages, seed_everything, write_json,
)


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
            prompt_text = tokenizer.apply_chat_template(
                make_messages(row['sentence1'], row['sentence2']),
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            prompt_ids = tokenizer(
                prompt_text, add_special_tokens=False,
            )['input_ids']
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


def is_complete_checkpoint(path):
    path = Path(path)
    state_files = (
        'trainer_state.json', 'optimizer.pt', 'scheduler.pt',
        'rng_state.pth', 'training_args.bin',
    )
    model_files = (
        'adapter_model.safetensors', 'adapter_model.bin',
        'model.safetensors', 'pytorch_model.bin',
    )
    return (
        path.is_dir()
        and all((path / name).is_file() for name in state_files)
        and any((path / name).is_file() for name in model_files)
    )


def find_latest_complete_checkpoint(output_dir):
    candidates = []
    for path in Path(output_dir).glob('checkpoint-*'):
        try:
            step = int(path.name.removeprefix('checkpoint-'))
        except ValueError:
            continue
        candidates.append((step, path))
    for _, path in sorted(candidates, reverse=True):
        if is_complete_checkpoint(path):
            return path.resolve()
    return None


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
    p.add_argument(
        '--save-steps', type=int, default=250,
        help='每隔多少个 optimizer step 保存完整 Trainer checkpoint',
    )
    p.add_argument(
        '--resume-from-checkpoint', nargs='?', const='auto', default=None,
        metavar='PATH_OR_AUTO',
        help='从 checkpoint 恢复；不带路径或传 auto 时自动选择输出目录中的最新 checkpoint',
    )
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('QLoRA 训练需要 NVIDIA CUDA GPU。')
    if not torch.cuda.is_bf16_supported():
        raise SystemExit('本实验固定 NF4 + BF16 compute；当前 GPU 不支持 BF16。')
    if not (a.rank > 0 and a.alpha > 0 and a.grad_accum > 0 and a.save_steps > 0):
        raise ValueError('rank/alpha/grad-accum/save-steps 必须大于零')
    seed_everything(a.seed)
    output_dir = Path(a.output).resolve()
    model_path = Path(a.model).resolve()
    train_path = Path(a.train).resolve()
    validation_path = Path(a.validation).resolve()
    if not (model_path / 'config.json').is_file():
        raise FileNotFoundError(f'本实验要求本地模型目录：{model_path}')
    model_fingerprint, model_files = fingerprint_model(model_path)
    current_cfg = vars(a).copy()
    current_cfg.update({
        'status': 'running',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'model': str(model_path),
        'model_fingerprint': model_fingerprint,
        'model_files': model_files,
        'train': str(train_path),
        'train_sha256': file_sha256(train_path),
        'validation': str(validation_path),
        'validation_sha256': file_sha256(validation_path),
        'prompt_template': TASK_PROMPT_TEMPLATE,
        'prompt_template_sha256': PROMPT_TEMPLATE_SHA256,
        'message_structure': 'one user message; no system message; no few-shot examples',
        'chat_template': {
            'tokenize': False,
            'add_generation_prompt': True,
            'enable_thinking': False,
            'tokenizer_add_special_tokens': False,
        },
        'quantization': {
            'load_in_4bit': True,
            'bnb_4bit_quant_type': 'nf4',
            'bnb_4bit_use_double_quant': True,
            'bnb_4bit_compute_dtype': str(torch.bfloat16),
        },
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'peft': importlib.metadata.version('peft'),
        'bitsandbytes': importlib.metadata.version('bitsandbytes'),
        'gpu': torch.cuda.get_device_name(0),
        'checkpointing': {
            'strategy': 'steps',
            'save_steps': a.save_steps,
            'save_total_limit': 2,
            'save_only_model': False,
            'selection_rule': 'last epoch; checkpoints are only for recovery',
        },
    })

    resume_checkpoint = None
    if a.resume_from_checkpoint is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f'训练输出目录非空，拒绝覆盖：{output_dir}')
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = current_cfg
    else:
        if not output_dir.is_dir():
            raise FileNotFoundError(f'恢复训练要求输出目录已存在：{output_dir}')
        previous_config_path = output_dir / 'run_config.json'
        if not previous_config_path.is_file():
            raise FileNotFoundError(f'恢复训练缺少 run_config.json：{previous_config_path}')
        if a.resume_from_checkpoint == 'auto':
            latest = find_latest_complete_checkpoint(output_dir)
            if latest is None:
                raise FileNotFoundError(
                    f'输出目录中没有完整、可恢复的 checkpoint：{output_dir}'
                )
            resume_checkpoint = latest
        else:
            resume_checkpoint = Path(a.resume_from_checkpoint).resolve()
            if not resume_checkpoint.is_dir():
                raise FileNotFoundError(f'checkpoint 目录不存在：{resume_checkpoint}')
            if resume_checkpoint.parent != output_dir:
                raise ValueError(
                    f'checkpoint 必须直接位于本次输出目录中：{output_dir}'
                )
            if not is_complete_checkpoint(resume_checkpoint):
                raise ValueError(f'checkpoint 文件不完整，拒绝恢复：{resume_checkpoint}')

        previous_cfg = json.loads(previous_config_path.read_text(encoding='utf-8'))
        if previous_cfg.get('status') == 'complete':
            raise ValueError('该训练已经完成，不应再从 checkpoint 恢复')
        compatibility_keys = (
            'model_fingerprint', 'train_sha256', 'validation_sha256',
            'prompt_template_sha256', 'rank', 'alpha', 'max_length', 'epochs',
            'max_steps', 'learning_rate', 'grad_accum', 'seed', 'save_steps',
        )
        mismatches = {
            key: {'previous': previous_cfg.get(key), 'current': current_cfg.get(key)}
            for key in compatibility_keys
            if previous_cfg.get(key) != current_cfg.get(key)
        }
        if mismatches:
            raise ValueError(
                '恢复参数与原训练不一致，拒绝继续：'
                + json.dumps(mismatches, ensure_ascii=False)
            )
        resume_events = list(previous_cfg.get('resume_events', []))
        resume_events.append({
            'resumed_at_utc': datetime.now(timezone.utc).isoformat(),
            'checkpoint': str(resume_checkpoint),
            'previous_status': previous_cfg.get('status'),
            'previous_error': previous_cfg.get('error'),
        })
        cfg = previous_cfg
        for key in (
            'error', 'finished_at_utc', 'elapsed_seconds', 'trainer_log_entries',
        ):
            cfg.pop(key, None)
        cfg.update({
            'status': 'running',
            'resume_from_checkpoint': str(resume_checkpoint),
            'resume_events': resume_events,
            'checkpointing': current_cfg['checkpointing'],
        })
    write_json(output_dir / 'run_config.json', cfg)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data = MatchDataset(train_path, tokenizer, a.max_length)
    val_data = MatchDataset(validation_path, tokenizer, a.max_length)
    print('开始加载 4-bit 基础模型...')
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=dtype,
        quantization_config=quant_config, device_map={'': 0},
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
        save_strategy='steps',
        save_steps=a.save_steps,
        save_total_limit=2,
        save_only_model=False,
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
    train_started = time.perf_counter()
    try:
        if resume_checkpoint is not None:
            print('从 checkpoint 恢复训练：', resume_checkpoint)
        result = trainer.train(
            resume_from_checkpoint=(
                str(resume_checkpoint) if resume_checkpoint is not None else None
            )
        )
        model.save_pretrained(output_dir / 'adapter')
        tokenizer.save_pretrained(output_dir / 'adapter')
        eval_metrics = trainer.evaluate()
    except (Exception, KeyboardInterrupt) as error:
        cfg.update({
            'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
            'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'error': f'{type(error).__name__}: {error}',
            'elapsed_seconds': time.perf_counter() - train_started,
            'trainer_log_entries': len(trainer.state.log_history),
        })
        write_json(output_dir / 'run_config.json', cfg)
        write_json(output_dir / 'trainer_log.json', trainer.state.log_history)
        raise
    adapter_fingerprint, adapter_files = fingerprint_directory(output_dir / 'adapter')
    cfg.update({
        'status': 'complete',
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': time.perf_counter() - train_started,
        'dtype': str(dtype),
        'train_count': len(train_data),
        'train_dropped': train_data.dropped,
        'val_count': len(val_data),
        'val_dropped': val_data.dropped,
        'lora_scaling_alpha_over_rank': a.alpha / a.rank,
        'max_vram_reserved_gib': round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        'train_metrics': result.metrics,
        'eval_metrics': eval_metrics,
        'adapter_fingerprint': adapter_fingerprint,
        'adapter_files': adapter_files,
    })
    write_json(output_dir / 'run_config.json', cfg)
    write_json(output_dir / 'trainer_log.json', trainer.state.log_history)
    print('SUCCESS: Adapter 保存到：', output_dir / 'adapter')
    print('验证 loss:', eval_metrics.get('eval_loss'))
    print('显存峰值 GiB:', cfg['max_vram_reserved_gib'])


if __name__ == '__main__':
    main()
