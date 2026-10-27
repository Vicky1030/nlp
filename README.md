# AFQMC 中文语义匹配一键复现与实时推理

本仓库将四位成员的工作合并为一个闭环：A 的 TF-IDF 基线、B 的 Prompt 方法、C 的 QLoRA 模型调用完成预测，D 对本次预测自动进行统一评测、绘图和难例筛选。根目录唯一推荐入口是 `run_all.py`。

## 0. 运行模式先说明

- `--quick`：固定随机抽取 evaluation 的 80 条，真实运行 A/B/C，约 2–3 分钟，适合录屏。
- 无参数：真实运行完整 4,316 条 evaluation，用于正式成绩；参考硬件上的运行时间约为 1 小时 46 分钟。
- `--mode replay`：不加载模型，重新评测仓库内冻结预测，适合无 GPU 的快速核查。

三个代表方法分别是 A 的 `tfidf_lr`、B 的 `qwen_few_shot` 和 C 的 `qwen_lora`。一张 GPU 上按 A → B → C 顺序运行，以避免 B/C 抢占显存。

## 1. 从零准备电脑

以下步骤以 Windows PowerShell 为例。需要：

- [Git for Windows](https://git-scm.com/download/win)；
- [64 位 Python 3.11](https://www.python.org/downloads/windows/)（安装时勾选加入 PATH）；
- NVIDIA GPU、[较新的显卡驱动](https://www.nvidia.com/Download/index.aspx)和 BF16 支持；
- 可下载 Qwen3-1.7B 的网络连接；
- 建议至少预留 10 GB 磁盘空间。

项目运行只依赖仓库内声明的环境和资源路径，不会查找外部 Conda 环境或其他仓库目录。

## 2. 下载仓库

```powershell
git clone https://github.com/Vicky1030/nlp.git
cd nlp
```

后续命令都在仓库根目录执行。

## 3. 创建独立虚拟环境

先确认 Python 3.11 已安装：

```powershell
py -3.11 --version
```

创建项目自己的 `.venv`。不需要激活环境，后续始终显式使用其中的 Python：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

## 4. 安装依赖

先安装项目测试通过的 CUDA 12.8 PyTorch，再安装锁定依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

其中 scikit-learn 固定为 1.9.1，因为 A 的冻结 `joblib` 模型不能由旧版 1.3.2 正确恢复。不要升级系统 Python 的 scikit-learn；只操作仓库内 `.venv`。

## 5. 下载基础模型

最终 LoRA Adapter 已随仓库放在 `models/lora_r16_a32_30899/`。基础模型体积较大，不进入 Git，需要下载固定 revision：

```powershell
.\.venv\Scripts\python.exe references\LoRA\src\download_model.py `
  --model Qwen/Qwen3-1.7B `
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e `
  --output models\qwen3-1.7b
```

下载脚本支持断点续传。正常情况下只需执行一次；后续运行会直接使用仓库内的
`models\qwen3-1.7b`，不会由 `run_all.py` 自动联网下载。

准备完成后的关键文件应为：

```text
models/
├── qwen3-1.7b/
│   ├── config.json
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   └── ...
└── lora_r16_a32_30899/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## 6. 运行前检查

```powershell
.\.venv\Scripts\python.exe check_environment.py
```

看到 `Environment check PASSED` 后再继续。检查内容包括 Python、全部依赖、scikit-learn 版本、CUDA、BF16、基础模型和 Adapter。

## 7. 录屏：2–3 分钟 QUICK

```powershell
.\.venv\Scripts\python.exe run_all.py --quick
```

QUICK 从完整 evaluation 按 `seed=42` 均匀随机抽取 80 条。它不是人工难例池，A/B/C 会真实推理，D 会生成指标、PNG、错例和三类动态难例。参考运行耗时约 120 秒；A/B/C Accuracy 为 0.6250、0.5750、0.6875，Case1/2/3 找到 24、12、6 条。抽样指标只用于验证流程，正式成绩以全量 evaluation 为准。

## 8. 正式全量复现

```powershell
.\.venv\Scripts\python.exe run_all.py
```

该命令实际推理完整 4,316 条 evaluation，并自动执行 D 评测和历史结果重建。不同 GPU 用时不同；RTX 5060 Laptop GPU 的参考运行时间约为 1 小时 46 分钟。

如无 GPU，只核查已经提交的完整预测和评测：

```powershell
.\.venv\Scripts\python.exe run_all.py --mode replay
```

## 9. 结果在哪里

每次实时运行都会创建新目录，不覆盖以前的证据：

```text
runs/<quick或evaluation>_<UTC时间>/
├── data/<split>_selected.jsonl
├── predictions/                    # A/B/C 七字段预测
├── calls/                          # B 的生成调用记录
├── evaluation/
│   ├── comparison_results_<split>.csv
│   ├── metrics_<split>.json
│   ├── confusion_matrices_<split>.csv
│   ├── hard_case_output.json
│   ├── errors/
│   ├── figures/*.png
│   ├── summary_report.md
│   └── summary_report.html
└── live_run_manifest.json
```

终端和 HTML/Markdown 会直接展示难例的两条原句、真实标签、A/B/C 预测及正确/错误，无需按 ID 查找。三种模式始终由本次预测动态判定：

1. Case1：A、B、C 全部正确；
2. Case2：A 错误，B Few-shot、C QLoRA 正确；
3. Case3：A、B Few-shot 错误，仅 C QLoRA 正确。

由于 NF4 数值噪声，复现时具体 ID 可能变化。2026-10-27 的全量运行分别找到 1,750、583、361 条三类模式；这些数量和示例不是写死规则。

## 10. 其他常用命令

```powershell
# 完整 validation
.\.venv\Scripts\python.exe run_all.py --split validation --limit 0

# validation 固定随机 100 条
.\.venv\Scripts\python.exe run_all.py --split validation --limit 100 --sample-strategy random --seed 42

# evaluation 上运行全部五种方法
.\.venv\Scripts\python.exe run_all.py --split evaluation --limit 0 --methods all
```

方法别名：`members` 是默认三方法；`A` 是两种 TF-IDF；`B` 是 Zero-shot 和 Few-shot；`C` 是 LoRA；`all` 是全部五种。

## 11. 常见错误

- 报 `scikit-learn=1.3.2`：说明用了系统 Python。请使用 `.\.venv\Scripts\python.exe`。
- 报找不到 `config.json`：基础模型没有下载到 `models/qwen3-1.7b/`。
- 报 CUDA 不可用：PyTorch CUDA 版本或显卡驱动不匹配；重新安装适合该机器的官方 CUDA PyTorch。
- 只想快速检查代码和历史结果：使用 `--mode replay`，不要启动全量实时推理。
- 模型放在其他磁盘：可设置 `AFQMC_QWEN_MODEL_DIR` 和 `AFQMC_LORA_ADAPTER_DIR`，路径应指向对应文件夹。

## 冻结结果与 D 评测说明

以下内容说明仓库内已纳入的冻结结果、统一协议和评测产物。

## 1. 当前实际纳入的结果

### 最终 evaluation（4,316 条）

- A：TF-IDF + Cosine（冻结推理包生成）
- A：TF-IDF + Logistic Regression（冻结推理包生成）
- B：Qwen Zero-shot P0 基线
- B：Qwen Zero-shot 最终方案 `near_conditions`
- B：Qwen Few-shot 最终方案 `fixed2_last_exact`
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
2. B 的 P0、最终 Zero-shot `near_conditions` 和最终 Few-shot `fixed2_last_exact` prediction 已纳入 `results/evaluation/`，会出现在真实 evaluation 成绩表中。C 包中的 `zero_shot_nf4_aligned` 仍只作为 C 实验里的对齐基础模型基线，不冒称为 B 的最终结果。
3. A 的模型预测由新交付的冻结推理包在本项目统一数据上生成。A 包要求的 scikit-learn 版本与当前运行环境可能不同；生成结果的类别与 A 的冻结验收摘要一致，延迟仅代表对应运行过程的测量值。
4. 延迟来自不同硬件和计时范围（TF-IDF CPU 与 QLoRA GPU），不应直接据此宣称哪种模型速度更快。报告会保留各文件的 `latency_ms` 并注明不可直接横比。
5. 错误分析中的否定、数字、条件、字符重叠标签是**供人工检查的启发式候选标签**，不是自动判定的真实错误原因。

## 2. 目录结构

```text
AFQMC_D_final/
├── data/
│   └── processed/
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── evaluation.jsonl
├── results/
│   ├── evaluation/                 # 最终 evaluation 预测
│   │   ├── tfidf_cosine.jsonl
│   │   ├── tfidf_lr.jsonl
│   │   ├── P0.predictions.jsonl
│   │   ├── near_conditions.predictions.jsonl
│   │   ├── fixed2_last_exact.predictions.jsonl
│   │   └── lora_r16_a32_30899_final.jsonl
│   └── validation/                 # 仅用于 validation 对比
│       ├── tfidf_cosine.jsonl
│       ├── tfidf_lr.jsonl
│       ├── zero_shot_nf4_aligned.jsonl
│       ├── lora_r8_a16_3000.jsonl
│       ├── lora_r16_a32_3000.jsonl
│       └── lora_r16_a32_30899.jsonl
├── evaluation/
│   ├── api.py
│   ├── config.py
│   ├── io_utils.py
│   ├── metrics.py
│   ├── error_analysis.py
│   ├── visualize.py
│   └── run_all.py
├── pipeline/live_runner.py         # A/B/C 实时推理编排与进度显示
├── tests/
│   ├── test_metrics.py
│   └── test_pipeline.py
├── references/                     # A/B/C 冻结推理包、配置和报告摘录
├── models/
│   ├── qwen3-1.7b/                 # 下载后生成，Git 忽略
│   └── lora_r16_a32_30899/         # 仓库自带的最终 Adapter
├── outputs/                        # D 的历史全量评测输出
├── runs/                           # live 模式的独立运行目录
├── check_environment.py            # 环境、CUDA 和模型预检
├── run_all.py                      # 唯一推荐入口
├── run_project.py                  # 旧命令兼容别名
├── run_all.bat
├── requirements.txt
├── requirements-live.txt
├── requirements-gpu.txt
└── README.md
```

## 3. 开发者检查

环境建立后可运行单元测试；不要改用系统 Python：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
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

## 6. B 的 Zero-shot / Few-shot 结果格式

B 的最终冻结结果已按下列七字段 JSONL 纳入仓库：

```json
{"id":34335,"sentence1":"句子一","sentence2":"句子二","label":0,"prediction":1,"method":"qwen_zero_shot","latency_ms":125.5}
```

对应文件为：

```text
results/evaluation/near_conditions.predictions.jsonl
results/evaluation/fixed2_last_exact.predictions.jsonl
```

新增冻结预测可放入对应 split 的 `results/` 目录后运行 `.\.venv\Scripts\python.exe run_all.py --mode replay`；实时模式则会在单次运行目录内自动生成并评测。`id`、原句和 `label` 必须与统一数据一致；不可重新切分或修改标签。

## 7. 加入新的模型预测

每个模型一个 `.jsonl` 文件，放进对应 split 的目录。最低要求是每行包含 `id`、`label`、`prediction`；推荐保留 `sentence1`、`sentence2`、`method`、`latency_ms`。预测值必须是 `0`、`1` 或 `null`。所有文件必须覆盖对应 gold 的全部 ID。

## 8. 复现与安全注意

- 默认 live 入口会实际调用三个代表方法；`--mode replay` 只评估冻结预测。两种模式都不会自动下载基础模型或重新训练 LoRA。
- 参考目录保留 A/B/C 的冻结推理代码、提示词、配置和报告，便于追溯；B 的最终 evaluation 预测也已纳入。
- 不要把训练集用于最终评价；不要用最终 evaluation 反复选择提示词、阈值或 LoRA 参数。
- 预测文件里记录的 latency 可能来自不同机器和计时口径，正式报告中应注明环境，不应无条件横向比较。
