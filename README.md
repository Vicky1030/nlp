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
