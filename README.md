# Qwen3-1.7B AFQMC 中文语义匹配 Zero-shot

本项目记录课程项目成员 B 使用原始 Qwen3-1.7B 完成的本地 Zero-shot 实验，包括统一模型调用接口、单条推理检查、100 条验证数据试跑，以及全部 3,433 条验证数据上的预测与评估。

> **实验定位：** 本文呈现 BF16 未量化、非思考模式下的 Zero-shot 验证集基线，作为 B 后续 Prompt 优化与 Few-shot 实验的对照。当前配置与 C 的 P0、NF4 QLoRA 探索性实验尚未完全对齐，因此暂不作为双方已统一的共同基线，也不是最终 evaluation 测试结论。

本次不训练模型、不加载微调 Adapter、不提供 Few-shot 示例。本文不包含早期随机 100 条思考模式试跑的成绩；不同子集、提示词和解码设置的历史结果不能直接混作本次实验的受控对比。

## 1. 任务与提示词协议

任务是判断两个中文句子的核心语义是否相同或相近：`1` 表示相同或相近，`0` 表示不同。

推理时只有一条 `user` 消息，没有单独的 `system` 消息。完整模板保存在 `zero_shot.txt`：

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

每条样本只将两个句子填入模板，不将真实标签传给模型。程序调用模型自带的 Qwen chat template，设置 `add_generation_prompt=True` 和 `enable_thinking=False`，随后以 `add_special_tokens=False` 分词。

`llm_client.py` 单独运行时的命令行默认 system prompt 不适用于本次实验；本次入口是 `zero_shot.py`。

### 1.1 生成设置

| 参数 | 实际值 |
|---|---|
| 思考模式 | 关闭，`enable_thinking=False` |
| 解码 | 贪心解码，`do_sample=False` |
| 最大生成长度 | `max_new_tokens=8` |
| 随机种子 | `seed=42` |
| temperature / top_p / top_k | 均为 None，不启用采样 |
| 单次样本数 | 1 |
| 模型精度 | BF16，未量化 |
| 模型状态 | eval，使用 inference_mode |

生成配置从模型自带配置复制后覆盖上述参数。调用 `model.generate` 时使用 `use_model_defaults=False`，避免模型默认值覆盖显式设置。

### 1.2 标签解析

- 解码时跳过特殊 token，并去除首尾空白。
- 如含有 `</think>`，取最后一个 `</think>` 后的内容；未闭合的思考内容不作为答案。
- 最终内容必须严格等于 `"0"` 或 `"1"`，才转换为整数预测。
- `"1。"`、`"答案是1"` 等均记为无效输出 `null`。
- 因生成长度上限被截断的输出记为 `null`；在上限处正常生成结束标记不视为截断。
- 无效输出保留，计算准确率时计错，不删除样本。

本次完整验证集没有无效输出。后续若出现无效输出，需与组员进一步统一其在 F1 和混淆矩阵中的处理口径。

## 2. 项目结构

下面列出与本文实验有关的本地文件；不表示所有目录都应上传 GitHub。

```text
.
├── README_Zero-shot.md
├── requirements.txt
├── .gitignore
├── llm_client.py                  # 模型加载、生成、解析与调用日志
├── LLM_CLIENT.md                  # 模型接口说明
├── zero_shot.py                   # 单条与批量 Zero-shot 推理入口
├── zero_shot.txt                  # 本次完整提示词
├── Zero-shot实验配置与结果.md      # 供成员 C 对齐配置的详细说明
├── data/
│   └── AFQMC_全组统一数据包/
│       └── data/processed/
│           ├── train.jsonl
│           ├── validation.jsonl
│           └── evaluation.jsonl
├── models/                       # 本地基础模型与分词器文件
└── outputs/
    ├── zero_shot_100.jsonl
    ├── zero_shot_100.summary.json
    ├── zero_shot_100.calls.jsonl
    ├── zero_shot_validation.jsonl
    ├── zero_shot_validation.summary.json
    └── zero_shot_validation.calls.jsonl
```

基础模型权重直接放在 `models/`，不是 `models/qwen3-1.7b/` 子目录。模型、虚拟环境和缓存不进入仓库；逐条预测含有原始句子与标签，公开分享前应确认统一数据包的再分发许可。

## 3. 环境与依赖

本实验在 Windows 原生 Python 虚拟环境中完成，不需要 WSL。

| 组件 | 已核实信息 |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.8.0+cu126 |
| PyTorch CUDA 构建版本 | 12.6 |
| Transformers | 4.57.1 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 模型加载精度 | torch.bfloat16 |
| 4-bit / 8-bit 量化 | 均未使用 |

`requirements.txt` 固定了 PyTorch 和 Transformers 的基础版本，safetensors 使用版本范围，尚不是完整的依赖锁文件。

新环境可使用以下 PowerShell 命令准备依赖；已配置好的本地环境不需要重复安装：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

本地调用接口要求 CUDA 可用。模型从本地加载，运行预测时不自动下载模型。

## 4. 模型与数据

### 4.1 基础模型

模型为 `Qwen/Qwen3-1.7B`，未微调、未加载 Adapter。

本次记录的模型文件组合指纹：

```text
a9d6d3c0e389377afced62546d63c8c64fb4b26e9495b11de8f1662afb735c5c
```

该指纹由项目代码根据本地权重、配置和分词器等文件计算，详细文件哈希保存在运行 summary 的 `model_info.files` 中。它不是官方模型提交号；当前记录不能据此推断官方 revision，也不能直接套用 C 文档中的 revision。

复现时应按相同方法核对模型文件，不能仅凭相同的模型名称认定权重版本完全一致。

### 4.2 统一数据划分

数据来自 A 同学提供的全组统一 AFQMC 数据包，保留原始划分、ID、句子和标签。

| 文件 | 样本数 | 标签 0 | 标签 1 | B 的用途 |
|---|---:|---:|---:|---|
| train.jsonl | 30,899 | 21,383 | 9,516 | 后续选择 Few-shot 示例；本次未使用 |
| validation.jsonl | 3,433 | 2,376 | 1,057 | 提示词开发与本次基线评估 |
| evaluation.jsonl | 4,316 | 2,978 | 1,338 | 方案固定后的最终评估；本文不报告成绩 |

每条输入使用四字段：`id`、`sentence1`、`sentence2`、`label`。

本次完整验证集文件的 SHA256：

```text
4fcdbfcc3dd252e7f385e6535510df0fbe8f488e0cf0d7681d380049a6b6b8f5
```

已核对完整预测文件：共 3,433 条，每条的 ID、两个句子及真实标签均与原验证集按顺序一致。

## 5. 实验流程与复现

以下命令均在项目根目录执行，使用已配置的虚拟环境和本地模型。

### 5.1 数据检查

```powershell
.\.venv\Scripts\python.exe zero_shot.py --limit 0 --check-data
```

检查字段、标签、句子内容和 ID，不加载模型。

### 5.2 单条推理

```powershell
.\.venv\Scripts\python.exe zero_shot.py --limit 1 --output outputs\repro_zero_shot_single.jsonl
```

默认从 validation 读取数据。程序显示句子、原始输出、解析标签、真实标签和生成耗时，用于检查流程。

### 5.3 前 100 条试跑

```powershell
.\.venv\Scripts\python.exe zero_shot.py --limit 100 --output outputs\repro_zero_shot_100.jsonl
```

此次历史试跑使用文件顺序的前 100 条，正确 67 条，Accuracy 为 67.00%，无效输出为 0。它用于检查批量流程，不代表完整验证集表现。

该子集不是早期 `scripts/run_zero_shot.py` 使用 seed=42 随机抽取的 100 条，不能把两次结果当作同一实验。

### 5.4 完整验证集

```powershell
.\.venv\Scripts\python.exe zero_shot.py --limit 0 --output outputs\repro_zero_shot_validation.jsonl
```

`--limit 0` 表示全部输入数据。未传入 `--thinking`，因此关闭思考模式。

程序只加载一次模型，预热后逐条预测，每 10 条报告进度。每条预测写入后刷新文件，并更新运行统计；异常或中断会记录状态，不将部分结果标记为完整实验。

上述复现命令使用新的文件名，避免覆盖本次历史结果。程序会检查预测、summary 和 calls 文件是否已存在，如存在则要求更换输出名称。

## 6. 完整验证结果

本次预测文件为 `outputs/zero_shot_validation.jsonl`。Accuracy 由运行 summary 记录，其余分类指标根据逐条预测补算。

以标签 1 为正类；F1 在下表使用 0—1 数值，Precision、Recall 和 Accuracy 使用百分数。

| 方法 | Accuracy | Macro-F1 | 正类 Precision | 正类 Recall | 正类 F1 | 无效输出率 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B BF16 Zero-shot | 70.55% | 0.5388 | 57.37% | 16.93% | 0.2615 | 0.00% |

共预测 3,433 条，正确 2,422 条。以 `[[TN, FP], [FN, TP]]` 表示混淆矩阵：

```text
[[2243, 133],
 [ 878, 179]]
```

指标计算方式：

- Accuracy：`(TP + TN) / 总样本数`。
- 正类 Precision：`TP / (TP + FP)`。
- 正类 Recall：`TP / (TP + FN)`。
- 正类 F1：`2TP / (2TP + FP + FN)`。
- Macro-F1：分别计算标签 0、1 的 F1，再取算术平均值。

模型预测 3,121 条为 0，312 条为 1；1,057 条真实正类中只识别出 179 条，漏判 878 条。全部预测为 0 的多数类基线准确率为 69.21%，本次比它高约 1.34 个百分点。

这表明当前设置下模型明显偏向负类，后续改进应关注正类 Recall、F1 及具体错误样本，而不只看 Accuracy。这些结果本身不能确定偏差来自提示词、模型能力还是其他因素，需要受控实验验证。

### 6.1 延迟口径

本次 summary 记录的平均单条生成耗时为 **69.57 ms**，只统计经过 CUDA 同步的 `model.generate` 时间，不包含模型加载、模板构造、分词、解码、重试等待和日志写入。

正式预测前有独立预热。该数值不是端到端请求耗时，不应直接与不同 GPU、量化方式或计时口径下的耗时比较。

## 7. 输出协议与组内协作

逐条预测遵循 D 同学要求的七字段：

```text
id, sentence1, sentence2, label, prediction, method, latency_ms
```

本次 `method` 为 `zero_shot`，`prediction` 为 0 或 1，无效时为 `null`。模型原始输出、输入消息和生成参数另存 calls 日志，不额外加入七字段预测文件。

与 C 当前提供的 P0 文档相比，存在以下差异：

| 项目 | B 本次实验 | C 提供的 P0 文档 |
|---|---|---|
| 提示词 | 相同或相近核心意思，并提示关注意图、否定和条件 | 初版简化的相同/不同判断提示 |
| 基础模型加载 | BF16，未量化 | NF4 4-bit、double quantization |
| 思考模式 | 关闭 | 关闭 |
| 解码 | 贪心，最多 8 token | 贪心，最多 8 token |
| 标签解析 | 严格接受 0/1，不接受末尾标点 | 允许末尾一个常见句号或感叹号 |

C 的配置来自其提供的说明，本文没有独立复跑 C 的实验。

由于提示词、量化和解析协议不同，不能把 B 的成绩与 C 的微调成绩之差直接解释为纯粹的微调收益。C 可在自身固定配置下比较原始模型与 Adapter；双方最终横向比较前，应统一协议，或明确报告差异并补充对照实验。

## 8. 已完成与后续工作

已完成：

- 本地 Qwen3-1.7B 加载与统一模型调用接口。
- 固定 Zero-shot 提示词、非思考贪心推理和严格标签解析。
- 单条检查与前 100 条批量试跑。
- 全部 3,433 条验证集预测、记录核对和分类指标计算。
- 保存逐条预测、运行配置、模型文件指纹及原始调用日志。

后续工作：

- 分析漏判正类及误判负类，形成提示词修改依据。
- 在固定模型和生成设置下开展 Prompt 对比。
- 仅从 train 选择 Few-shot 示例，在 validation 上选择方案。
- 与 C、D 对齐最终推理精度、提示词、解析规则和评估口径。
- 补充完整环境锁文件和可核验的官方模型 revision。
- 完善统一数据来源、版本和再分发许可说明。
- 方案固定后，在共同 evaluation 上进行最终评估。

当前预测默认保存在被 `.gitignore` 忽略的 `outputs/` 中，提交结果时需明确选择文件。公开仓库可优先保留代码与本文；含原始数据的逐条预测是否公开，应按数据许可和小组约定处理。

## 9. 实验范围

本文只呈现 B 已完成的 BF16、非思考 Zero-shot 验证基线，不将计划中的 Few-shot、NF4 或完整思考模式实验写成已完成结果。

本次基线的主要价值是提供可检查的起点：后续提示工程实验保持数据和推理设置可追踪，使用独立输出名称保存结果，并在最终方案确定前保留 evaluation 数据。
