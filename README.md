# AFQMC 中文语义匹配：统一评估与一键运行项目（成员 D）

本项目整合 A、B、C 已提交的材料，并实现统一评价、自动汇总、错误分析、可视化和一键运行。**本包不会重新训练大模型**；一键入口读取已保存的预测 JSONL，完成可复现的评估流程。

## 1. 当前实际纳入的结果

### 最终 evaluation（4,316 条）

- A：TF-IDF + Cosine（冻结推理包生成）
- A：TF-IDF + Logistic Regression（冻结推理包生成）
- C：LoRA rank=16 / alpha=32，使用 30,899 条训练样本的最终模型

### validation（3,433 条，单独报告，不与 evaluation 混算）

- A：TF-IDF + Cosine
- A：TF-IDF + Logistic Regression
- C 交付的 NF4 Base Zero-shot 对齐基线
- C：3,000 条训练数据的 LoRA rank=8 / alpha=16
- C：3,000 条训练数据的 LoRA rank=16 / alpha=32
- C：30,899 条训练数据的 LoRA rank=16 / alpha=32

**结果边界：**
1. C 的 r8 和 3,000 条训练的 r16 当前只有完整 validation 预测，没有最终 evaluation 预测，所以只放入 validation 对比，不冒充最终测试成绩。
2. B 的压缩包包含推理代码、提示词、配置和依赖说明，但没有 Zero-shot / Few-shot 的实际预测 JSONL，也没有随包提供基础模型权重。故 B 的两种方法不能出现在真实 evaluation 成绩表中；`model_availability.csv` 会明确标记缺失。C 包中的 `zero_shot_nf4_aligned` 是 C 实验中的对齐基础模型基线，不直接冒称为 B 的最终结果。
3. A 的模型预测由新交付的冻结推理包在本项目统一数据上生成。A 包要求的 scikit-learn 版本与当前运行环境可能不同；生成结果的类别与 A 的冻结验收摘要一致，但本机生成的延迟只能视为本次运行测量值。
4. 延迟来自不同硬件和计时范围（TF-IDF CPU 与 QLoRA GPU），不应直接据此宣称哪种模型速度更快。报告会保留各文件的 `latency_ms` 并注明不可直接横比。
5. 错误分析中的否定、数字、条件、字符重叠标签是**供人工检查的启发式候选标签**，不是自动判定的真实错误原因。

## 2. 目录结构

```text
AFQMC_D_final/
├── data/processed/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── evaluation.jsonl
├── results/
│   ├── evaluation/                 # 最终 evaluation 预测
│   │   ├── tfidf_cosine.jsonl
│   │   ├── tfidf_lr.jsonl
│   │   └── lora_r16_a32_30899_final.jsonl
│   └── validation/                 # 仅用于 validation 对比
│       ├── tfidf_cosine.jsonl
│       ├── tfidf_lr.jsonl
│       ├── zero_shot_nf4_aligned.jsonl
│       ├── lora_r8_a16_3000.jsonl
│       ├── lora_r16_a32_3000.jsonl
│       └── lora_r16_a32_30899.jsonl
├── evaluation/
│   ├── config.py
│   ├── io_utils.py
│   ├── metrics.py
│   ├── error_analysis.py
│   ├── visualize.py
│   └── run_all.py
├── tests/test_metrics.py
├── references/                     # B/C 交付的文档、配置和报告摘录
├── outputs/                        # 一键运行自动生成
├── run_all.bat
├── requirements.txt
└── README.md
```

## 3. 环境与运行

Python 3.10+。核心评估、数据校验、错例导出只使用标准库；图表需要 `matplotlib`。

首次安装：

```bash
python -m pip install -r requirements.txt
```

### 一条命令完成全部评估

在项目根目录执行：

```bash
python -m evaluation.run_all
```

Windows 用户也可以双击 `run_all.bat`。如果没有 matplotlib，指标与错例仍会生成，绘图步骤会提示跳过。

运行单元测试：

```bash
python -m unittest discover -s tests -v
```

## 4. 指标与评测协议

- Accuracy：预测正确数 / 全部样本数。
- Precision、Recall、F1：以 label=1 为正类，同时在 JSON 中保留两个类别的详细指标。
- Macro-F1：分别计算 label 0 和 label 1 的 F1，再做算术平均。
- 混淆矩阵：行是真实标签，列是预测标签，顺序为 `[0, 1]`，即 `[[TN, FP], [FN, TP]]`。
- 无效预测 `null`：保留在分母中并计为错误；同时报告无效数和无效率。
- 延迟：对预测文件中的有效 `latency_ms` 计算均值、中位数和 P95。P95 使用线性分位数插值。
- 所有模型必须与对应的统一 gold 数据按 ID 完全对齐。程序会拒绝重复 ID、缺失/额外 ID、标签不一致以及句子不一致。
- validation 和 evaluation 分开输出；不得把 validation 指标当成最终 evaluation 成绩。

## 5. 输出文件说明

运行后查看 `outputs/`：

| 输出 | 内容 |
|---|---|
| `comparison_results_evaluation.csv` | 最终 evaluation 模型指标主表 |
| `comparison_results_validation.csv` | validation 模型对比表（含 r8/r16 对比） |
| `metrics_evaluation.json` / `metrics_validation.json` | 完整指标、混淆矩阵、延迟统计和 gold 哈希 |
| `confusion_matrices_*.csv` | 各模型 TN、FP、FN、TP |
| `errors/evaluation/`、`errors/validation/` | 各模型逐条错例，含 FP/FN/INVALID 和候选错误标签 |
| `cross_model_errors_*.csv` | 至少有一个模型出错的跨模型对齐样本，可分析某模型错、另一个模型对 |
| `error_summary_*.csv` | 每个模型的错误数量与错误率 |
| `figures/` | Accuracy、Macro-F1、正类 F1 和记录延迟图 |
| `model_availability.csv` | 作业计划方法的结果文件可用性与缺失说明 |
| `run_report.json` | 本次运行时间、生成内容、失败信息和注意事项 |

## 6. 加入 B 的 Zero-shot / Few-shot 结果

请 B 将最终冻结方案在 `data/processed/evaluation.jsonl` 上的预测保存为七字段 JSONL：

```json
{"id":34335,"sentence1":"句子一","sentence2":"句子二","label":0,"prediction":1,"method":"qwen_zero_shot","latency_ms":125.5}
```

把两个文件分别放入：

```text
results/evaluation/qwen_zero_shot.jsonl
results/evaluation/qwen_few_shot.jsonl
```

如果 B 也提供了同一 validation 的预测，可以分别放入 `results/validation/`。然后重新运行同一条命令，无需修改评价代码。`id`、原句和 `label` 必须与统一数据一致；不可重新切分或修改标签。

## 7. 加入新的模型预测

每个模型一个 `.jsonl` 文件，放进对应 split 的目录。最低要求是每行包含 `id`、`label`、`prediction`；推荐保留 `sentence1`、`sentence2`、`method`、`latency_ms`。预测值必须是 `0`、`1` 或 `null`。所有文件必须覆盖对应 gold 的全部 ID。

## 8. 复现与安全注意

- 这套一键入口只评估预测文件，不会自动下载基础大模型或启动 LoRA 训练；训练与推理需要 GPU、模型权重和各自独立的依赖环境。
- 参考目录保留 B 的提示词/配置/代码以及 C 的报告，便于追溯，不代表 B 的最终预测结果已提供。
- 不要把训练集用于最终评价；不要用最终 evaluation 反复选择提示词、阈值或 LoRA 参数。
- 预测文件里记录的 latency 可能来自不同机器和计时口径，正式报告中应注明环境，不应无条件横向比较。
