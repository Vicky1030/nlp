# AFQMC Prompt Engineering（成员 B）

本仓库保存成员 B 对 AFQMC 中文语义匹配任务的提示工程、统一本地推理接口和可复核结果。基础模型为 `Qwen/Qwen3-1.7B`，最终比较统一使用 NF4 4-bit、关闭思考模式、greedy 解码、`max_new_tokens=8`、`seed=42`。本仓库不包含模型权重、虚拟环境和全组数据包。

## 目录

```text
data/       数据路径说明；数据包不随仓库提交
outputs/    冻结 evaluation 的逐条预测，以及开发期 short 基线
reports/    最终报告、提示词实验报告和完整实验目录
results/    配置、提示词、汇总指标、哈希和实验总表
src/        llm_client、Zero-shot/Few-shot 脚本及评估辅助模块
```

## 环境准备

1. 按 `data/README.md` 放置 A 同学的统一数据包。
2. 将本地 `Qwen/Qwen3-1.7B` 权重放在仓库根目录的 `models/`，权重不会上传。
3. 使用带 CUDA 的 Python 环境安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

NF4 需要 NVIDIA CUDA、`bitsandbytes` 和可用的 GPU；代码只从本地 `models/` 读取，不自动下载模型。

## 运行示例

先检查数据，不加载模型：

```powershell
.\.venv\Scripts\python.exe src\zero_shot.py --quantization nf4 --prompt-file results\near_conditions.prompt.txt --input data\AFQMC_全组统一数据包\data\processed\validation.jsonl --limit 0 --check-data
```

运行 Zero-shot（请使用新的输出文件名，避免覆盖已有证据）：

```powershell
.\.venv\Scripts\python.exe src\zero_shot.py --quantization nf4 --prompt-file results\near_conditions.prompt.txt --input data\AFQMC_全组统一数据包\data\processed\validation.jsonl --limit 0 --output outputs\near_conditions.validation.new.jsonl
```

运行固定 Few-shot 配置：

```powershell
.\.venv\Scripts\python.exe src\predict_nf4_selected.py --config results\fixed2_last_exact.config.json --input data\AFQMC_全组统一数据包\data\processed\validation.jsonl --output outputs\fixed2_last_exact.validation.new.jsonl
```

`src/llm_client.py` 严格解析最终输出为 `0` 或 `1`；无效输出保留为 `null` 并计为错误，不从指标分母中删除。完整接口说明见 `LLM_CLIENT.md`。

## 冻结方案与最终结果

用户在查看 evaluation 前冻结了 `near_conditions`（Zero-shot）和 `fixed2_last_exact`（Few-shot），`P0` 为未修改基线。evaluation 共 4,316 条，结果如下：

| 方案 | Accuracy | 正类 Precision | 正类 Recall | 正类 F1 | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| P0 baseline | 62.14% | 41.75% | 55.98% | 0.4783 | 0.5906 |
| Zero-shot `near_conditions` | 67.08% | 46.50% | 41.26% | 0.4372 | 0.6023 |
| Few-shot `fixed2_last_exact` | 62.30% | 40.11% | 43.80% | 0.4187 | 0.5699 |

逐条预测在 `outputs/`，原始配置、提示词、模型指纹和数据哈希在 `results/`，解释和过程记录在 `reports/最终冻结方案与Evaluation结果.md`。冻结预测文件除七个基础字段外还保留 `raw_output`、`finish_reason`、`prompt_sha256` 和 `example_ids` 等审计字段；这些字段不参与指标计算。evaluation 只用于冻结后的最终报告，不用于继续挑选提示词。

## 结果边界

- `results/prompt_results.csv` 是开发期 Zero-shot/Few-shot 实验目录，不是把所有实验都视为最终方案。
- `short.validation.predictions.jsonl` 是提示词探索时使用的历史开发基线；最终选定结果以冻结报告为准。
- 预测文件不包含模型权重和调用日志；调用日志可能包含完整提示词，默认由 `.gitignore` 排除。
- 数据和模型的大文件需由组内成员按说明准备，不要将个人路径、API key 或本地缓存提交到 GitHub。
- `outputs/` 的逐条结果含原始句子和标签；若仓库设为公开，请先确认课程、数据集许可和组内共享规则允许再上传。
