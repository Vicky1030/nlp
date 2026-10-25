# 逐条预测

- `P0.evaluation.predictions.jsonl`：冻结的 Zero-shot 基线评估预测。
- `near_conditions.evaluation.predictions.jsonl`：冻结的 Zero-shot 候选评估预测。
- `fixed2_last_exact.evaluation.predictions.jsonl`：冻结的 Few-shot 候选评估预测。
- `short.validation.predictions.jsonl`：提示词探索期的 NF4 validation 基线，不是最终选定方案。

最终指标与配置分别见 `../results/`。前三份文件包含 evaluation 原始句对和标签；公开仓库前需确认数据许可和课程要求。若不允许再发布原始数据，可不上传逐条预测，只提交指标、配置和哈希。
