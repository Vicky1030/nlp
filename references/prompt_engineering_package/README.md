# B 同学：提示词推理与测评代码

本压缩包只包含提示词实验和预测结果指标计算所需的代码，不包含训练/微调代码、模型权重、数据集或已有预测结果。

## 目录

- `src/llm_client.py`：加载本地 Qwen 模型并执行生成；支持 NF4 量化。
- `src/zero_shot.py`：按 JSONL 输入运行零样本提示词推理。
- `src/predict_nf4_selected.py`：按配置运行保存的 NF4 Zero-shot/Few-shot 候选。
- `src/evaluate_predictions.py`：根据预测 JSONL 计算 Accuracy、各类 Precision/Recall/F1、Macro-F1 和混淆矩阵。
- `prompts/`、`configs/`：P0、near_conditions、fixed2_last_exact 三个候选提示词及实验配置。
- `src/work/compare_nf4_baseline.py`：B 自己的 NF4/BF16 配对实验核验脚本，依赖原实验的汇总、输出及提示词文件，不是通用评测入口。

## 环境要求

- Windows、Python 虚拟环境、可用 NVIDIA CUDA GPU。
- 已下载本地模型并放在包目录下的 `models/`；此压缩包不含模型文件。
- 输入数据为 UTF-8 JSONL，每行字段为 `id`（整数）、`sentence1`、`sentence2`（非空字符串）、`label`（0 或 1）。数据集需由组内同学另行提供。
- 在包目录执行 `pip install -r requirements.txt`。PyTorch 安装需匹配机器 CUDA 环境；若 requirements 中固定版本与本机 CUDA wheel 不兼容，应先安装对应 CUDA 版本的 PyTorch。

## 运行示例

先检查数据和提示词格式，不加载模型：

```powershell
python src/zero_shot.py --input data/validation.jsonl --prompt-file prompts/P0.prompt.txt --limit 10 --check-data
```

运行 NF4 零样本预测（先把验证集放到 `data/validation.jsonl`）：

```powershell
python src/zero_shot.py --input data/validation.jsonl --prompt-file prompts/P0.prompt.txt --limit 0 --quantization nf4 --output outputs/P0_nf4.jsonl
```

计算预测文件指标：

```powershell
python src/evaluate_predictions.py outputs/P0_nf4.jsonl
```

运行已保存的 NF4 候选配置：

```powershell
python src/predict_nf4_selected.py --config configs/near_conditions.config.json --input data/validation.jsonl --output outputs/near_conditions.jsonl
```

该预测脚本的配置会校验模型文件指纹；不同模型文件必须先由实验负责人确认并更新配置，不能直接删掉校验绕过。

## 与 C 同学微调模型的边界

本包的 `llm_client.py` 加载基础模型，不加载 LoRA Adapter，因此适用于 B 的提示词推理结果复现和通用指标计算。若 D 要评测 C 微调后的 Adapter，应同时向 C 索取其 Adapter 推理脚本、Adapter 权重位置/获取方式、模型基座版本及运行参数；不能用本包的基础模型结果替代 Adapter 评测。

## 结果与数据注意事项

预测结果通常含原句和标签；公开分享前先确认数据许可和团队仓库约定。不要把模型权重、完整数据集、虚拟环境或缓存目录放进代码压缩包。
