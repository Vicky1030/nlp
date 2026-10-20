# AFQMC：B 同学 Zero-shot 实验配置与结果

本文记录 B 同学已完成的 Qwen3-1.7B Zero-shot 验证集实验，供 C 同学对齐微调前后的评估配置。

本次结果属于验证集成绩，不是最终 evaluation 测试成绩。

## 1. 模型与运行环境

| 项目 | 本次配置 |
|---|---|
| 模型 | Qwen/Qwen3-1.7B |
| 模型状态 | 原始模型，未微调 |
| 加载精度 | BF16（torch.bfloat16） |
| 量化 | 无 |
| 运行环境 | Windows，本地 CUDA GPU |
| PyTorch | 2.8.0+cu126 |
| Transformers | 4.57.1 |
| 每批样本数 | 1 |

本地模型文件组合指纹：

```text
a9d6d3c0e389377afced62546d63c8c64fb4b26e9495b11de8f1662afb735c5c
```

这是项目代码根据本地模型与分词器等文件计算的内容指纹，不是官方 revision 编号。若要比较指纹，双方应使用相同的指纹计算方法；仅模型名称一致不能保证文件版本完全相同。

## 2. 验证集

| 项目 | 本次配置 |
|---|---|
| 来源 | A 同学提供的全组统一数据包 |
| 文件 | validation.jsonl |
| 使用范围 | 全部 3,433 条，按文件顺序处理 |
| 标签 0：语义不同 | 2,376 条 |
| 标签 1：语义相同或相近 | 1,057 条 |
| 数据修改 | 未重新划分，保留原始 ID 和标签 |

验证集 SHA256：

```text
4fcdbfcc3dd252e7f385e6535510df0fbe8f488e0cf0d7681d380049a6b6b8f5
```

数据路径：

```text
data/AFQMC_全组统一数据包/data/processed/validation.jsonl
```

本次未使用 evaluation.jsonl。最终方案确定后，再使用统一的 4,316 条 evaluation 数据进行最终评估。

## 3. 完整提示词

### System prompt

没有设置，消息列表中没有 system 消息。

注意：单独运行 llm_client.py 时的命令行默认 system prompt，不适用于本次通过 zero_shot.py 进行的实验。

### User prompt

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

每条样本用实际句子替换 `{sentence1}` 和 `{sentence2}`，不向模型提供真实标签，也不提供 Few-shot 示例。

使用模型自带的 `tokenizer.apply_chat_template` 组织输入：

```text
tokenize=False
add_generation_prompt=True
enable_thinking=False
```

随后分词时设置 `add_special_tokens=False`，避免重复添加特殊 token。

## 4. 思考开关与生成参数

| 参数 | 本次设置 | 说明 |
|---|---|---|
| enable_thinking | False | 关闭思考模式 |
| do_sample | False | 不随机采样，使用贪心解码 |
| max_new_tokens | 8 | 最多生成 8 个 token，token 不等同于汉字 |
| seed | 42 | 固定随机种子；此次不随机采样 |
| temperature | None | 不启用 |
| top_p | None | 不启用 |
| top_k | None | 不启用 |

生成配置以模型自带配置为基础，再显式覆盖上述参数；`pad_token_id` 使用分词器的 `eos_token_id`。调用 `model.generate` 时设置 `use_model_defaults=False`，避免模型默认配置覆盖显式设置。

模型只加载一次，正式预测前进行一次预热。

## 5. 标签解析规则

1. 解码时跳过特殊 token，并去除首尾空白。
2. 如果输出含有 `</think>`，只取最后一个 `</think>` 后面的内容。
3. 如果存在未闭合的思考内容，不从其中提取预测标签。
4. 最终答案必须严格等于字符串 `"0"` 或 `"1"`，才转换为整数标签。
5. `"1。"`、`"答案是1"`、`"相同"` 等输出均记为无效，保存为 `null`。
6. 如果生成因达到输出长度上限而截断，也记为 `null`；在上限位置正常生成结束标记的情况不视为截断。
7. 无效输出保留，计算准确率时计错，不删除对应样本。

本次无效输出为 0 条，因此下文的二分类 Precision、Recall 和 F1 均按正常二分类公式计算。若后续实验出现无效输出，双方还需统一其在 F1 和混淆矩阵中的处理方式，不能只删除后计算成绩。

## 6. 指标定义与实验结果

以标签 1 为正类：

- TP：真实为 1，预测为 1。
- TN：真实为 0，预测为 0。
- FP：真实为 0，预测为 1。
- FN：真实为 1，预测为 0。

### 混淆矩阵

行是真实标签，列是预测标签。

| 真实标签 \ 预测标签 | 预测 0 | 预测 1 |
|---|---:|---:|
| 真实 0 | TN = 2,243 | FP = 133 |
| 真实 1 | FN = 878 | TP = 179 |

### 指标

| 指标 | 计算方式 | 结果 |
|---|---|---:|
| 样本数 | 全部验证样本 | 3,433 |
| 正确数 | TP + TN | 2,422 |
| Accuracy | (TP + TN) / 总样本数 | 70.55% |
| 正类 Precision | TP / (TP + FP) | 57.37% |
| 正类 Recall | TP / (TP + FN) | 16.93% |
| 正类 F1 | 2TP / (2TP + FP + FN) | 0.2615 |
| Macro-F1 | 标签 0 和标签 1 的 F1 的算术平均值 | 0.5388 |
| 无效输出 | prediction 为 null 的条数 | 0 |

正类 F1 和 Macro-F1 是不同指标，报告和组内汇总时必须明确名称。

模型共预测 3,121 条为 0、312 条为 1，主要问题是漏判正类。全部预测为 0 的准确率为 69.21%，本次准确率比该多数类基线高约 1.34 个百分点。因此不能只看准确率，还要比较 Recall 和 F1。

## 7. 预测输出与耗时口径

逐条预测采用 D 同学要求的七字段格式：

| 字段 | 含义 |
|---|---|
| id | 原始样本 ID |
| sentence1 | 第一个句子 |
| sentence2 | 第二个句子 |
| label | 真实标签 |
| prediction | 预测标签 0、1，或无效时的 null |
| method | 本次为 zero_shot |
| latency_ms | 单条样本生成耗时，单位毫秒 |

`latency_ms` 仅统计 GPU 上 `model.generate` 的时间，计时前后进行 CUDA 同步。不包含模型加载、提示词构造、分词、解码、重试等待和日志写入时间。

不同硬件上的耗时不能直接归因于提示工程或微调本身；比较时应同时注明硬件和计时范围。

## 8. 运行命令与结果文件

在 Windows PowerShell 中运行：

```powershell
cd D:\nlp
.\.venv\Scripts\python.exe zero_shot.py --limit 0 --output outputs\zero_shot_validation.jsonl
```

`--limit 0` 表示使用全部输入数据。未传入 `--thinking`，所以关闭思考模式。

以上是本次运行命令。程序拒绝覆盖已有结果；如需重新运行，请更换 `--output` 文件名。

| 文件 | 内容 |
|---|---|
| zero_shot.txt | 提示词模板 |
| zero_shot.py | 批量预测入口 |
| llm_client.py | 模型加载、生成、解析和调用日志 |
| outputs/zero_shot_validation.jsonl | 逐条预测结果 |
| outputs/zero_shot_validation.summary.json | 本次配置、模型文件指纹、准确率及完成状态 |
| outputs/zero_shot_validation.calls.jsonl | 逐次调用的输入、原始输出和生成设置 |

summary 文件由程序记录准确率等信息；本文中的 Precision、Recall、F1 和混淆矩阵根据逐条预测结果补算。

## 9. 与 C 同学对齐的方式

C 同学可以立即进行微调，与 B 的提示工程实验并行。

若 C 采用相同的基础模型文件、验证集、完整推理提示词、思考开关、生成参数、标签解析和指标定义，可以把本次结果作为共同的原始模型 Zero-shot 验证基线。

如果 C 的推理提示词、量化方式或评估流程不同，应使用自己的固定流程补跑原始模型，再与微调后的模型对照，并记录配置差异。微调前后保持相同的推理设置，更便于判断微调带来的变化。

C 用 train 训练、validation 选择配置；B 的 Few-shot 示例也应从 train 中选择。双方方案固定后，再在统一的 evaluation 上评估共同基线及最终方案。
