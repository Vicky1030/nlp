# 整合包交付边界

- `results/` 中的 JSONL 是统一评价使用的实际预测记录。
- `references/member_a/` 包含 A 的冻结推理包代码与模型文件，便于追溯和需要时重跑传统模型。
- `references/member_b/` 保留 B 的提示词、配置、推理代码和说明；B 的最终 evaluation 预测不在原压缩包中。
- `references/member_c/` 保留 C 的源代码、训练报告、配置摘要与指标；为控制整合包体积，未重复复制完整 LoRA Adapter 二进制权重。原始 `C_QLoRA_handoff.zip` 应作为权重和完整训练交付的权威来源。
- 一键命令只运行评估、错例分析、结果汇总和可视化，不会自动训练模型或下载 Qwen 基座。
- 当前 A 的预测结果是在本整合工作区使用冻结推理包生成的，类别结果与 A 的验收摘要一致；本机 latency 是新测量值，不应替代 A 历史日志中的 latency。
