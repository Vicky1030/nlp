# Qwen3-1.7B AFQMC 语义匹配 QLoRA

本项目记录课程项目成员 C 在原提示词 P0 下完成的 Qwen3-1.7B 语义匹配探索性实验，包括 4-bit 基础模型推理、Demo 闭环、500 条真实数据预实验、3000 条 AFQMC 数据的 rank=8 QLoRA 训练，以及基础模型与微调模型在同一完整验证集上的评估。

> **实验定位：** P0 是尚未与成员 B 的基准提示词对齐的初版简化提示。本次工作的目标是完整验证 rank=8 QLoRA 从数据检查、训练、Adapter 保存加载到全量验证的技术流程，并观察微调是否带来改善。因此，本次结果属于探索性/测试性结果，不作为全组正式基线或最终实验结论。

本项目不包含后续 P1 实验。与成员 B 对齐提示词、输入格式和推理精度后的实验将另行开展。

## 1. 任务与 P0 协议

任务是判断两个中文句子的语义是否相同：`1` 表示相同，`0` 表示不同。

训练和推理均由 `src/common.py` 生成单条 `user` 消息，没有单独的 `system` 消息。完整模板为：

```text
请判断下面两个中文句子表达的意思是否相同。相同输出1，不同输出0。只能输出一个数字，不要解释。
句子1：{sentence1}
句子2：{sentence2}
```

训练时，assistant 目标是标签 `0` 或 `1`，提示部分不计入损失。训练与推理都调用 Qwen chat template，并设置 `enable_thinking=False`。推理使用贪心解码：`do_sample=False`、`max_new_tokens=8`；解析器只接受单个 `0`/`1`（允许首尾空白及末尾一个常见句号或感叹号），其余输出记为无效预测。

## 2. 仓库结构

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── src/
│   ├── common.py                  # P0 指令与 user 消息模板
│   ├── check_env.py               # CUDA、GPU 与 PyTorch 环境检查
│   ├── download_model.py          # 下载并记录基础模型来源
│   ├── test_base_model.py         # 单对句子的基础模型推理
│   ├── train_qlora.py             # 4-bit QLoRA 训练
│   ├── predict_base_afqmc.py      # 基础模型批量预测
│   ├── predict_lora.py            # 加载 Adapter 后批量预测
│   ├── evaluate.py                # 指标、无效率、混淆矩阵与延迟统计
│   └── validate_data.py           # 数据格式、标签、ID 和跨划分重复检查
├── data/
│   ├── demo/                      # 16 条训练、4 条验证的最小闭环数据
│   ├── afqmc_shared/              # 组内统一 AFQMC 数据包
│   └── afqmc_pilot/               # 500/3000 训练子集与 100 条验证子集
├── models/qwen3-1.7b/             # 复现时按固定版本下载
├── outputs/                       # Adapter、run_config 与 trainer_log
│   ├── windows_smoke_r8/
│   ├── afqmc_pilot_r8/
│   └── afqmc_3000_r8/
└── results/
    ├── base_test10.jsonl
    ├── windows_lora_demo.jsonl
    ├── afqmc_pilot_r8_100.jsonl
    ├── base_validation_full.jsonl
    └── lora_r8_validation_full.jsonl
```

上面的结构只保留 Windows 下完成 P0 实验所需的代码、数据、训练产物和评估结果，不包含早期 WSL Demo 产物。基础模型目录 `models/` 不进入仓库，由复现者按固定版本下载。

## 3. 环境与依赖

本实验在 Windows Conda 环境中完成。基础模型权重以 4-bit NF4 加载，量化计算类型为 BF16；实验环境如下：

| 组件 | 已核实版本 |
|---|---:|
| Python | 3.11.17 |
| PyTorch | 2.9.1+cu128 |
| CUDA runtime | 12.8 |
| transformers | 4.57.6 |
| peft | 0.19.1 |
| accelerate | 1.15.0 |
| bitsandbytes | 0.50.2 |
| huggingface_hub | 0.36.2 |
| safetensors | 0.8.0 |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.1 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB |

`requirements.txt` 保存兼容版本范围，并按设计将 PyTorch 独立安装。下面是在 Windows PowerShell 中建立独立环境的复现入口：

```powershell
conda create -n qwen3_repro python=3.11 -y
conda activate qwen3_repro

python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r .\requirements.txt

python .\src\check_env.py
```

表中的版本是本次实验使用的实际版本；后续可增加环境锁文件以进一步固定依赖。

## 4. 模型与数据

### 4.1 基础模型

基础模型为 `Qwen/Qwen3-1.7B`。`download_source.json` 记录当时请求的是 `main`，本地 Hugging Face 下载元数据一致指向提交：

```text
70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
```

复现时使用该固定提交：

```powershell
python .\src\download_model.py `
  --model Qwen/Qwen3-1.7B `
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e `
  --output .\models\qwen3-1.7b
```

下载后的基础模型保存在 `models/qwen3-1.7b/`。仓库不包含基础模型权重，复现时通过上述命令获取。

### 4.2 数据划分

真实数据来自组内提供的统一 AFQMC 数据包，数据来源和版本信息由课程项目组统一维护。

| 文件 | 样本数 | 标签 0 | 标签 1 | 本工程用途 |
|---|---:|---:|---:|---|
| `data/afqmc_shared/train.jsonl` | 30,899 | 21,383 | 9,516 | 统一训练池；本次未全量训练 |
| `data/afqmc_shared/validation.jsonl` | 3,433 | 2,376 | 1,057 | 基础模型与探索性微调模型的共同完整验证集 |
| `data/afqmc_shared/evaluation.jsonl` | 4,316 | 2,978 | 1,338 | 保留的独立评估划分；本项目尚未报告其成绩 |
| `data/afqmc_pilot/train_500.jsonl` | 500 | 346 | 154 | 真实数据预实验训练集 |
| `data/afqmc_pilot/train_3000.jsonl` | 3,000 | 2,077 | 923 | rank=8 探索性训练集 |
| `data/afqmc_pilot/validation_100.jsonl` | 100 | 69 | 31 | 预实验及 3000 条训练的内部验证 |
| `data/demo/train.jsonl` | 16 | 8 | 8 | Demo 训练 |
| `data/demo/validation.jsonl` | 4 | 2 | 2 | Demo 验证与加载测试 |

`validate_data.py` 对三个统一划分的检查结果为：字段与标签合法、各划分无重复 ID，且 train/validation/evaluation 之间没有重复句子对。500 条和 3000 条训练文件都能逐条匹配统一训练池，100 条验证文件能逐条匹配统一验证集。

本版本未保存 `train_500.jsonl`、`train_3000.jsonl` 和 `validation_100.jsonl` 的抽样脚本及抽样种子。现有文件表明，500 条中有 361 条同时出现在 3000 条子集中，两份训练文件不是嵌套子集。

放置好组内数据包后执行：

```powershell
python .\src\validate_data.py `
  --train .\data\afqmc_shared\train.jsonl `
  --val .\data\afqmc_shared\validation.jsonl `
  --test .\data\afqmc_shared\evaluation.jsonl
```

## 5. 三阶段实验记录

### 5.1 Demo：功能闭环

Windows Demo 使用 16 条训练数据、4 条验证数据和 5 个优化步，完成了训练、Adapter 保存、重新加载和预测。对应产物是 `outputs/windows_smoke_r8/` 与 `results/windows_lora_demo.jsonl`；4 条 Demo 预测全部正确，但样本过小，不能作为模型效果结论。

用独立输出目录重新运行：

```powershell
python .\src\test_base_model.py `
  --model .\models\qwen3-1.7b `
  --precision 4bit

python .\src\train_qlora.py `
  --model .\models\qwen3-1.7b `
  --train .\data\demo\train.jsonl `
  --validation .\data\demo\validation.jsonl `
  --rank 8 --alpha 16 --max-length 256 `
  --epochs 2 --max-steps 5 `
  --learning-rate 0.0002 --grad-accum 8 --seed 42 `
  --output .\outputs\repro_demo_r8

python .\src\predict_lora.py `
  --model .\models\qwen3-1.7b `
  --adapter .\outputs\repro_demo_r8\adapter `
  --input .\data\demo\validation.jsonl `
  --output .\results\repro_demo_r8.jsonl `
  --method lora_r8

python .\src\evaluate.py .\results\repro_demo_r8.jsonl
```

### 5.2 500 条真实数据预实验

工程中存在 `outputs/afqmc_pilot_r8/adapter/` 和 100 条验证预测 `results/afqmc_pilot_r8_100.jsonl`。Adapter 配置能够确认 rank=8、alpha=16、LoRA dropout=0.05、目标模块为 `q_proj`/`v_proj`。

500 条预实验未保存 `run_config.json` 和 `trainer_log.json`，因此没有独立记录该次运行的学习率、epoch、梯度累积、耗时和显存峰值。下面的命令使用现有训练脚本与 3000 条探索性实验参数复现实验流程：

```powershell
python .\src\train_qlora.py `
  --model .\models\qwen3-1.7b `
  --train .\data\afqmc_pilot\train_500.jsonl `
  --validation .\data\afqmc_pilot\validation_100.jsonl `
  --rank 8 --alpha 16 --max-length 256 `
  --epochs 2 --max-steps -1 `
  --learning-rate 0.0002 --grad-accum 8 --seed 42 `
  --output .\outputs\repro_afqmc_pilot_r8

python .\src\predict_lora.py `
  --model .\models\qwen3-1.7b `
  --adapter .\outputs\repro_afqmc_pilot_r8\adapter `
  --input .\data\afqmc_pilot\validation_100.jsonl `
  --output .\results\repro_afqmc_pilot_r8_100.jsonl `
  --method lora_r8

python .\src\evaluate.py .\results\repro_afqmc_pilot_r8_100.jsonl
```

已有 100 条预测的复算结果为 Accuracy 65.00%、Macro-F1 63.6892%、正类 Recall 74.1935%、无效预测 0，混淆矩阵为 `[[42, 27], [8, 23]]`。该结果只用于小规模流程检查，不与 3433 条完整验证成绩直接横向比较。

### 5.3 3000 条 rank=8 探索性全流程实验

`outputs/afqmc_3000_r8/run_config.json` 和训练日志记录的关键配置如下：

| 项目 | 实际值 |
|---|---|
| 基础模型加载 | 4-bit NF4，double quantization |
| 计算类型 | BF16 |
| LoRA | rank=8，alpha=16，dropout=0.05 |
| LoRA 目标模块 | `q_proj`, `v_proj` |
| 训练数据 / 内部验证 | 3000 / 100 |
| epoch / optimizer steps | 2 / 750 |
| max length | 256 |
| 单卡 batch / 梯度累积 | 1 / 8 |
| 学习率 / 优化器 | 2e-4 / `adamw_torch` |
| 随机种子 / data seed | 42 / 42 |
| 梯度检查点 | 开启 |
| 训练中周期评估与保存 | 均关闭；训练结束后评估并保存 Adapter |

该次运行没有丢弃超长样本。记录的训练耗时为 5719.28 秒（约 95.32 分钟），最大保留显存为 3.428 GiB，最终训练 loss 为 0.269061，100 条内部验证的 eval loss 为 0.239659。

使用新目录复现，避免覆盖原产物：

```powershell
python .\src\train_qlora.py `
  --model .\models\qwen3-1.7b `
  --train .\data\afqmc_pilot\train_3000.jsonl `
  --validation .\data\afqmc_pilot\validation_100.jsonl `
  --rank 8 --alpha 16 --max-length 256 `
  --epochs 2 --max-steps -1 `
  --learning-rate 0.0002 --grad-accum 8 --seed 42 `
  --output .\outputs\repro_afqmc_3000_r8
```

基础模型与新 Adapter 在同一 3433 条完整验证集上预测并评估：

```powershell
python .\src\predict_base_afqmc.py `
  --model .\models\qwen3-1.7b `
  --input .\data\afqmc_shared\validation.jsonl `
  --output .\results\repro_base_validation_full.jsonl `
  --method base_4bit

python .\src\predict_lora.py `
  --model .\models\qwen3-1.7b `
  --adapter .\outputs\repro_afqmc_3000_r8\adapter `
  --input .\data\afqmc_shared\validation.jsonl `
  --output .\results\repro_lora_r8_validation_full.jsonl `
  --method lora_r8

python .\src\evaluate.py .\results\repro_base_validation_full.jsonl
python .\src\evaluate.py .\results\repro_lora_r8_validation_full.jsonl
```

## 6. 探索性完整验证结果

下表由现有两份逐条预测文件使用 `src/evaluate.py` 复算。两个文件都包含 3433 条记录，并与 `data/afqmc_shared/validation.jsonl` 按位置逐条一致。

| 模型 | Accuracy | Macro-F1 | 正类 Precision | 正类 Recall | 正类 F1 | 无效输出率 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B 4-bit 基础模型 | 69.2106% | 40.9020% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| P0 探索性 QLoRA rank=8（3000 条） | **73.4343%** | **66.3064%** | 59.0966% | **44.5601%** | 50.8091% | 0.0000% |

以 `[[TN, FP], [FN, TP]]` 表示混淆矩阵：

```text
基础模型：[[2376,   0],
          [1057,   0]]

QLoRA r8：[[2050, 326],
          [ 586, 471]]
```

基础模型将全部样本预测为 0；QLoRA 后 Accuracy 提高 4.2237 个百分点，Macro-F1 提高 25.4043 个百分点，并开始识别正类，但正类 Recall 仍只有 44.5601%，是后续改进重点。

这一结果说明当前训练流程能够使模型摆脱基础模型“全部预测为 0”的退化行为，但正类识别能力仍不理想。由于 P0 提示较为简化，且尚未与成员 B 的基准协议对齐，该结果用于验证方法与发现问题，不用于代表最终模型性能。

预测文件记录的单样本顺序推理延迟仅供本机内部参考：基础模型均值 611.914 ms、中位数 747.563 ms；QLoRA 均值 636.884 ms、中位数 336.239 ms。该计时包含首次调用影响，没有独立 warm-up，也不是严格性能基准。

以上全部是验证集成绩，不是 `evaluation.jsonl` 上的最终独立测试成绩。

## 7. 已完成与待完成

已完成：

- P0、非思考模式下的基础模型 4-bit 推理。
- Demo 训练、Adapter 保存、重新加载、预测和评估闭环。
- 500 条真实数据预实验及 100 条验证预测。
- 3000 条数据、rank=8、2 epochs 的探索性 QLoRA 全流程训练。
- 基础模型与探索性 Adapter 在同一 3433 条完整验证集上的逐条评估。

后续工作：

- 统一数据包的来源 URL、版本和再分发许可说明。
- 500/3000/100 子集的生成脚本、抽样种子，以及 500 条历史训练的完整配置和日志。
- 可锁定完整 Python 环境的 lock 文件。
- 在 `evaluation.jsonl` 上开展最终独立评估；本文现有结果均为验证集成绩。
- 与成员 B 对齐基准提示词、输入格式和推理精度后，重新开展正式 rank=8/rank=16 实验。
- 在统一协议下开展思考/非思考模式受控对比及后续 P1 实验。

## 8. 实验范围

本仓库呈现成员 C 在 Windows 环境下完成的 P0、非思考模式探索性实验。它证明了现有代码和硬件能够完成 Qwen3-1.7B 的 4-bit QLoRA 训练与全量验证，也保留了初版提示下的可复查结果。由于当前协议与成员 B 尚未对齐，本结果不作为全组共同基线；后续正式实验将使用新的输出目录，不覆盖这里保存的探索性记录。
