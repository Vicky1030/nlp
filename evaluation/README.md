# AFQMC 统一评价与错误分析模块（D 负责）

本目录是可直接运行的评价脚手架。它不负责训练模型，而是统一读取各成员生成的预测结果，保证所有模型在同一份 `evaluation.jsonl` 上计算指标。

## 1. 目录结构

```text
AFQMC_D_evaluation_starter/
├── data/processed/evaluation.jsonl
├── results/
│   └── tfidf_cosine.jsonl
├── evaluation/
│   ├── config.py
│   ├── metrics.py
│   ├── evaluate.py
│   ├── error_analysis.py
│   ├── visualize.py
│   ├── run_all.py
│   └── README.md
└── outputs/                  # 程序自动生成
```

## 2. 环境准备

Python 3.9+。核心评价与错误分析只使用 Python 标准库；完整复现的 PNG 绘图需要安装 requirements 中的 matplotlib：

```bash
python -m pip install matplotlib
```

如果暂时不需要图，可以在一键命令中使用 `--no-plots`。

## 3. 预测文件格式

每个模型一个 JSONL 文件，建议命名为模型名，例如 `results/tfidf_cosine.jsonl`。每行一条 JSON，至少包含：

```json
{"id": 34335, "label": 0, "prediction": 0, "method": "tfidf_cosine", "latency_ms": 0.62}
```

建议保留原始字段 `sentence1`、`sentence2`。`id` 必须对应统一数据包里的原始 ID；`label` 必须是标准答案，`prediction` 是模型预测的 0/1。可选 `latency_ms` 是单条推理延迟，单位毫秒。

重要：不要把训练集或验证集预测文件放进 `results/`；程序会把此目录下的 `*.jsonl` 当作待评价的模型预测文件。

## 4. 常用命令

在项目根目录执行：

评价一个模型：

```bash
python -m evaluation.evaluate --predictions results/tfidf_cosine.jsonl --gold data/processed/evaluation.jsonl --output outputs/tfidf_cosine_metrics.json
```

一键评价所有模型：

```bash
python -m evaluation.run_all
```

不绘图的一键评价：

```bash
python -m evaluation.run_all --no-plots
```

单独运行错误分析：

```bash
python -m evaluation.error_analysis
```

单独绘图：

```bash
python -m evaluation.visualize --metrics outputs/comparison_results.csv --output-dir outputs/figures
```

## 5. 自动生成的结果

- `outputs/comparison_results.csv`：多模型指标对比表，可放入报告。
- `outputs/all_metrics.json` 和 `outputs/<模型文件名>_metrics.json`：完整指标与混淆矩阵。
- `outputs/errors/<模型文件名>_errors.csv`：单模型全部错例，含 FP/FN 类型。
- `outputs/cross_model_predictions_and_errors.csv`：按同一 ID 对齐的各模型预测和正确性，便于找“模型 A 错、模型 B 对”等样本。
- `outputs/error_summary.csv`：各模型错误数量与错误率。
- `outputs/figures/`：Accuracy、Macro-F1、正类 F1、平均延迟图（有 matplotlib 时）。

## 6. 指标定义

- Accuracy：全部样本预测正确比例。
- Precision / Recall / F1：以标签 1 为正类。
- Macro-F1：对出现的每个类别分别算 F1，再做算术平均。
- 混淆矩阵：行是真实标签，列是预测标签；标签顺序记录在 JSON 的 `confusion_matrix.labels` 中。
- latency：对预测文件中有效的 `latency_ms` 汇总。不同模型必须统一计时口径后才适合横向比较。

## 7. 注意事项

1. 程序会检查重复 ID、缺失/额外 ID、标签不一致，以及可用时的句子不一致。
2. 各成员提交的结果应使用同一份 `evaluation.jsonl`，不能自行重新切分。
3. `results/` 每个文件代表一个模型；如果同一模型有多份实验结果，请只把最终确定的版本放进此目录，或先将临时结果放到其他目录。
4. 目前示例 `tfidf_cosine.jsonl` 来自组内已提供的 TF-IDF + Cosine 结果。是否还有 TF-IDF + Logistic Regression 结果，需要向负责该实验的成员确认。
