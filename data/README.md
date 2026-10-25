# 数据说明

本仓库不重复提交全组数据包。请从 A 同学提供的统一数据包取得数据，并在本目录解压为：

```text
data/AFQMC_全组统一数据包/data/processed/train.jsonl
data/AFQMC_全组统一数据包/data/processed/validation.jsonl
data/AFQMC_全组统一数据包/data/processed/evaluation.jsonl
```

三个文件分别用于训练集示例选择、提示词开发验证和冻结后的最终评估：

| 文件 | 条数 | 用途 |
|---|---:|---|
| `train.jsonl` | 30,899 | 只用于选择 Few-shot 示例 |
| `validation.jsonl` | 3,433 | 提示词与示例开发、方案选择 |
| `evaluation.jsonl` | 4,316 | 方案冻结后的一次最终评估 |

最终评估使用的数据哈希记录在 `results/manifest.json`。不要把 validation 或 evaluation 的标签提供给模型，也不要用 evaluation 结果继续调提示词。
