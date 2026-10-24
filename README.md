# Qwen3-1.7B AFQMC 语义匹配：统一基准与双秩 QLoRA

本项目记录在统一协议下完成的 Qwen3-1.7B 中文语义匹配实验。本版在上一版 P0 探索性实验的基础上，对齐了正式 Zero-shot 提示词、消息结构、生成参数、解析规则、模型版本和验证集，然后完成：

- Qwen3-1.7B 4-bit NF4 Zero-shot 基准；
- 3,000 条 AFQMC 数据的 QLoRA rank=8 / alpha=16 训练；
- 3,000 条 AFQMC 数据的 QLoRA rank=16 / alpha=32 训练；
- 三个模型在同一份 3,433 条 validation 上的完整逐条预测和评估。

> **本次提交范围：** 只包含协议对齐后的 NF4 Base、3,000 条 r8 和 3,000 条 r16 实验。30,899 条 rank=16 训练及其结果保留在本地 `full_train_v1` 命名空间，将在后续单独整理和提交；本文不使用那部分结果。

## 1. 任务与统一协议

任务是判断两个中文句子是否表达相同或相近的核心意思：`1` 表示相同或相近，`0` 表示不同。

训练和推理都使用一条 `user` 消息，不设置 `system` 消息，也不加入 Few-shot 示例。协议对齐后的完整模板为：

```text
请判断下面两个中文句子是否表达相同或相近的核心意思。

判断规则：
- 核心意思相同或相近，输出1。
- 核心意思不同，输出0。
- 不要仅根据共同关键词判断，要关注意图、否定和条件的差别。

句子1：{sentence1}
句子2：{sentence2}

只输出0或1，不要解释。
```

模板 SHA256：

```text
bd1ab4b528a42f9acd467858088be6059505e536c367d7e65592a6702c0f91e0
```

统一推理协议如下：

- Qwen chat template：`tokenize=False`、`add_generation_prompt=True`、`enable_thinking=False`；
- tokenizer：`add_special_tokens=False`，避免重复添加特殊 token；
- 解码：贪心生成，`do_sample=False`、`max_new_tokens=8`；
- 显式关闭 temperature、top-p 和 top-k，并设置 `use_model_defaults=False`；
- 每条样本使用固定随机种子 42；
- 只接受完整输出 `0` 或 `1`；截断、解释文字或其他内容均记为无效预测；
- 无效预测保留在总样本分母中并按错误计入，不静默丢弃。

Zero-shot 基准只加载基础模型，不加载 Adapter；这里的 Zero-shot 表示提示中不提供带答案示例，也没有针对 AFQMC 进行额外微调。

## 2. 公平对比设计

本版主表中的 Base、r8 和 r16 均在同一台机器上使用以下共同条件：

- 相同 Qwen3-1.7B 本地权重与模型组合指纹；
- 相同 3,433 条 AFQMC validation 及样本顺序；
- 相同正式提示词和 chat template；
- 相同 4-bit NF4 加载方式与 BF16 计算类型；
- 相同生成参数、严格解析规则和评估程序。

因此，Base 与两个 QLoRA 模型之间的主要变化是是否加载 Adapter，以及 Adapter 的 rank/alpha。

未量化 BF16 Zero-shot 结果可作为提示工程参考，但不能直接作为本项目 QLoRA 的增益基线。BF16 与 NF4 会改变输出分布；本项目判断微调收益时只比较 `NF4 Base` 与 `NF4 Base + Adapter`。

## 3. 仓库结构

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── common.py                  # 对齐提示词、生成、解析、指纹与批量预测
│   ├── check_env.py               # CUDA、GPU 与 PyTorch 环境检查
│   ├── download_model.py          # 下载固定版本的基础模型
│   ├── test_base_model.py         # 单对句子的基础模型测试
│   ├── train_qlora.py             # NF4 QLoRA 训练、checkpoint 与恢复
│   ├── predict_base_afqmc.py      # NF4 Base 批量预测
│   ├── predict_lora.py            # NF4 Base + Adapter 批量预测
│   ├── evaluate.py                # 指标、无效率、混淆矩阵与延迟统计
│   └── validate_data.py           # 数据格式、标签、ID 与跨划分检查
├── data/
│   ├── afqmc_shared/              # 组内统一 AFQMC 划分
│   └── afqmc_pilot/               # 3,000 条训练子集与 100 条训练期验证子集
├── outputs/b_aligned_v1/
│   ├── qlora_r8_a16_retry1/       # r8/a16 最终 Adapter、配置与训练日志
│   └── qlora_r16_a32/             # r16/a32 最终 Adapter、配置与训练日志
├── results/b_aligned_v1/validation/
│   ├── base_4bit.*                # NF4 Base 逐条结果、调用记录和运行摘要
│   ├── lora_r8_a16.*              # r8 完整 validation 结果
│   └── lora_r16_a32.*             # r16 完整 validation 结果
└── reports/b_aligned_v1/
    ├── base_4bit_metrics.json
    ├── lora_r8_a16_metrics.json
    └── lora_r16_a32_metrics.json
```

`models/` 不进入仓库，复现时按固定版本下载。`full_train_v1`、失败运行、诊断运行、控制台日志和可快速重跑的 smoke 产物不属于本次提交。

## 4. 环境、模型与数据

### 4.1 实验环境

| 组件 | 实际版本 |
|---|---:|
| Python | 3.11.17 |
| PyTorch | 2.9.1+cu128 |
| CUDA runtime | 12.8 |
| transformers | 4.57.6 |
| peft | 0.19.1 |
| accelerate | 1.15.0 |
| bitsandbytes | 0.50.2 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB |

Windows PowerShell 环境建立方式：

```powershell
conda create -n qwen3_repro python=3.11 -y
conda activate qwen3_repro

python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r .\requirements.txt

python .\src\check_env.py
```

### 4.2 基础模型

基础模型为 `Qwen/Qwen3-1.7B`，固定 Hugging Face revision：

```text
70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
```

实验记录的本地模型组合指纹为：

```text
a9d6d3c0e389377afced62546d63c8c64fb4b26e9495b11de8f1662afb735c5c
```

下载命令：

```powershell
python .\src\download_model.py `
  --model Qwen/Qwen3-1.7B `
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e `
  --output .\models\qwen3-1.7b
```

### 4.3 数据划分

| 文件 | 样本数 | 标签 0 | 标签 1 | 用途 |
|---|---:|---:|---:|---|
| `data/afqmc_shared/train.jsonl` | 30,899 | 21,383 | 9,516 | 统一训练池；本次只取既有 3,000 条子集 |
| `data/afqmc_shared/validation.jsonl` | 3,433 | 2,376 | 1,057 | Base、r8、r16 的共同完整验证集 |
| `data/afqmc_shared/evaluation.jsonl` | 4,316 | 2,978 | 1,338 | 保留的最终评估集；本次没有使用 |
| `data/afqmc_pilot/train_3000.jsonl` | 3,000 | 2,077 | 923 | r8 与 r16 的共同训练集 |
| `data/afqmc_pilot/validation_100.jsonl` | 100 | 69 | 31 | 训练结束后的 loss 检查 |

关键文件哈希：

```text
train_3000.jsonl     39299587180176c0b2550dc173314e6311556f15bafdd0f79f50082764e638bb
validation_100.jsonl 22475d9480f616ad9e58ef1096434f7abaaf9947f5b7367406ad1df3e195232c
validation.jsonl     4fcdbfcc3dd252e7f385e6535510df0fbe8f488e0cf0d7681d380049a6b6b8f5
```

数据检查命令：

```powershell
python .\src\validate_data.py `
  --train .\data\afqmc_shared\train.jsonl `
  --val .\data\afqmc_shared\validation.jsonl `
  --test .\data\afqmc_shared\evaluation.jsonl
```

## 5. 训练配置

两个实验都从同一个原始基础模型独立开始，没有从另一组 Adapter 继续训练。

| 配置 | r8 | r16 |
|---|---:|---:|
| 训练样本 / 训练期验证 | 3,000 / 100 | 3,000 / 100 |
| LoRA rank / alpha | 8 / 16 | 16 / 32 |
| alpha / rank | 2.0 | 2.0 |
| LoRA dropout | 0.05 | 0.05 |
| 目标模块 | `q_proj`, `v_proj` | `q_proj`, `v_proj` |
| 基础模型加载 | 4-bit NF4 + double quant | 4-bit NF4 + double quant |
| 计算类型 | BF16 | BF16 |
| epoch / optimizer step | 2 / 750 | 2 / 750 |
| 单卡 batch / 梯度累积 | 1 / 8 | 1 / 8 |
| max length | 256 | 256 |
| 学习率 / 优化器 | 2e-4 / AdamW | 2e-4 / AdamW |
| seed / data seed | 42 / 42 | 42 / 42 |
| 梯度检查点 | 开启 | 开启 |
| max grad norm | 0.3 | 0.3 |

训练脚本每 250 个 optimizer step 保存可恢复 checkpoint，拒绝覆盖非空输出目录，并在恢复时核对模型、数据、提示词和核心超参数。最终 Adapter 在训练结束后保存，训练期 validation 只用于 loss 检查。

## 6. 复现步骤

以下命令使用新目录，避免覆盖仓库中的现有实验记录。

### 6.1 运行对齐后的 NF4 Zero-shot 基准

```powershell
python .\src\predict_base_afqmc.py `
  --model .\models\qwen3-1.7b `
  --input .\data\afqmc_shared\validation.jsonl `
  --max-samples 0 `
  --output .\results\repro_b_aligned\base_4bit.jsonl `
  --method zero_shot_b_aligned_4bit

python .\src\evaluate.py `
  .\results\repro_b_aligned\base_4bit.jsonl `
  --output .\reports\repro_b_aligned\base_4bit_metrics.json
```

### 6.2 训练 rank=8 / alpha=16

```powershell
python .\src\train_qlora.py `
  --model .\models\qwen3-1.7b `
  --train .\data\afqmc_pilot\train_3000.jsonl `
  --validation .\data\afqmc_pilot\validation_100.jsonl `
  --rank 8 --alpha 16 `
  --max-length 256 --epochs 2 --max-steps -1 `
  --learning-rate 0.0002 --grad-accum 8 --seed 42 `
  --save-steps 250 `
  --output .\outputs\repro_b_aligned\qlora_r8_a16
```

如训练被安全中断，可在参数完全一致时自动从最新完整 checkpoint 恢复：

```powershell
python .\src\train_qlora.py `
  --model .\models\qwen3-1.7b `
  --train .\data\afqmc_pilot\train_3000.jsonl `
  --validation .\data\afqmc_pilot\validation_100.jsonl `
  --rank 8 --alpha 16 `
  --max-length 256 --epochs 2 --max-steps -1 `
  --learning-rate 0.0002 --grad-accum 8 --seed 42 `
  --save-steps 250 `
  --resume-from-checkpoint auto `
  --output .\outputs\repro_b_aligned\qlora_r8_a16
```

### 6.3 训练 rank=16 / alpha=32

```powershell
python .\src\train_qlora.py `
  --model .\models\qwen3-1.7b `
  --train .\data\afqmc_pilot\train_3000.jsonl `
  --validation .\data\afqmc_pilot\validation_100.jsonl `
  --rank 16 --alpha 32 `
  --max-length 256 --epochs 2 --max-steps -1 `
  --learning-rate 0.0002 --grad-accum 8 --seed 42 `
  --save-steps 250 `
  --output .\outputs\repro_b_aligned\qlora_r16_a32
```

### 6.4 完整 validation 预测与评估

```powershell
python .\src\predict_lora.py `
  --model .\models\qwen3-1.7b `
  --adapter .\outputs\repro_b_aligned\qlora_r8_a16\adapter `
  --input .\data\afqmc_shared\validation.jsonl `
  --max-samples 0 `
  --output .\results\repro_b_aligned\lora_r8_a16.jsonl `
  --method lora_r8_a16

python .\src\predict_lora.py `
  --model .\models\qwen3-1.7b `
  --adapter .\outputs\repro_b_aligned\qlora_r16_a32\adapter `
  --input .\data\afqmc_shared\validation.jsonl `
  --max-samples 0 `
  --output .\results\repro_b_aligned\lora_r16_a32.jsonl `
  --method lora_r16_a32

python .\src\evaluate.py `
  .\results\repro_b_aligned\lora_r8_a16.jsonl `
  --output .\reports\repro_b_aligned\lora_r8_a16_metrics.json

python .\src\evaluate.py `
  .\results\repro_b_aligned\lora_r16_a32.jsonl `
  --output .\reports\repro_b_aligned\lora_r16_a32_metrics.json
```

每次批量预测会同时生成三类文件：

- `*.jsonl`：统一七字段逐条预测；
- `*.calls.jsonl`：提示文本、原始输出、token 数、结束原因和生成设置；
- `*.summary.json`：模型/Adapter 指纹、输入哈希、运行状态、总体正确数和延迟统计。

## 7. 完整 validation 结果

所有结果均来自同一份 3,433 条 validation；无效输出均为 0。

| 模型 | Accuracy | Macro-F1 | P(1) | R(1) | F1(1) | 无效输出率 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B NF4 Zero-shot | 62.6566% | 58.9861% | 41.6605% | 53.1693% | 46.7165% | 0.0000% |
| 3,000 条 QLoRA r8/a16 | 73.3178% | 65.3152% | 59.6974% | 41.0596% | 48.6547% | 0.0000% |
| 3,000 条 QLoRA r16/a32 | **74.1625%** | **67.2411%** | **60.6516%** | **45.7900%** | **52.1833%** | 0.0000% |

以 `[[TN, FP], [FN, TP]]` 表示混淆矩阵：

```text
NF4 Base：[[1589, 787],
           [ 495, 562]]

QLoRA r8：[[2083, 293],
           [ 623, 434]]

QLoRA r16：[[2062, 314],
            [ 573, 484]]
```

相对 NF4 Base：

- r8 的 Accuracy 提高 10.6612 个百分点，Macro-F1 提高 6.3291 个百分点；
- r16 的 Accuracy 提高 11.5060 个百分点，Macro-F1 提高 8.2550 个百分点；
- 两个 Adapter 都显著减少了假阳性，并提高了总体准确率和类别均衡指标。

在相同训练数据和主要超参数下，r16 相对 r8：

- Accuracy 提高 0.8447 个百分点；
- Macro-F1 提高 1.9260 个百分点；
- 正类 Recall 提高 4.7304 个百分点；
- 正类 F1 提高 3.5286 个百分点。

因此，在本次 3,000 条训练实验中，r16/a32 是两个 QLoRA 配置中表现更好的方案。需要注意：这只是一次固定 seed 的 validation 对比，不能据此断言更大的 rank 在所有设置下都一定更好。

## 8. 训练过程记录

| 项目 | r8/a16 | r16/a32 |
|---|---:|---:|
| 训练状态 | complete | complete |
| 训练 step | 750 | 750 |
| 总耗时 | 10,884.70 s（约 3.02 h） | 16,066.66 s（约 4.46 h） |
| Trainer train runtime | 10,824.54 s | 15,998.86 s |
| 最终 train loss | 0.271357 | 0.266036 |
| 100 条 eval loss | 0.239183 | 0.242870 |
| 最大保留显存 | 3.428 GiB | 3.434 GiB |
| Adapter 权重大小 | 6,437,272 bytes | 12,859,984 bytes |
| Adapter 指纹 | `5ab26571...9160c9` | `26f07005...93cb9` |

r8 首次正式输出目录只留下了 `status=running` 的不完整配置，未产生最终 Adapter；随后使用全新目录 `qlora_r8_a16_retry1` 从原始基础模型重新完成训练。仓库只保留这次成功运行，不把不完整目录当作实验结果。

记录的推理延迟只覆盖同步后的 GPU `model.generate`，不含模型加载、tokenization、decode 和日志写入；它适合检查同一次运行，不应视为严格硬件性能基准。

## 9. 与旧 P0 结果的关系

上一版 P0 Base 在 validation 上的 Accuracy 为 69.21%，但它把 3,433 条样本全部预测成 0，Macro-F1 只有 40.90%，正类 Recall 和 F1 都为 0。69.21% 恰好等于验证集负类占比，属于类别不均衡下的多数类假高分。

对齐后的 NF4 Zero-shot Base Accuracy 降为 62.66%，但已经能够识别正类：Macro-F1 提升到 58.99%，正类 Recall 提升到 53.17%。因此本项目同时报告 Accuracy、Macro-F1 和正类指标，不用 Accuracy 单独判断模型质量。

本版训练与推理均使用新的正式提示词，不能把旧 P0 Adapter 与本版结果混在同一受控对比中。

## 10. 实验边界与后续工作

- 本文全部模型选择和比较都基于 `validation.jsonl`，没有使用 `evaluation.jsonl` 调参；
- 3,000 条 r8/r16 都只运行一个 seed，尚未给出多随机种子均值和方差；
- Few-shot 提示工程方案尚需放入同一 NF4 推理框架，才能与 QLoRA 做“只改提示 vs. 参数微调”的公平比较；
- 30,899 条 rank=16 属于下一阶段，将单独上传训练产物、结果并再次更新 README；
- 在训练方案完全冻结前，不使用保留的 evaluation 划分反复选择参数。

本次提交公开保留代码、固定配置、Adapter、逐条预测和指标文件，便于任何读者复现并核查“对齐 NF4 基准 → 3,000 条双秩 QLoRA → 同协议 validation 对比”的完整过程。
