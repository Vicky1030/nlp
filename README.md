# AFQMC：B 同学本地试跑

工作目录：`D:\nlp`。模型为 `Qwen/Qwen3-1.7B`，使用 BF16、关闭思考模式，以单条 Zero-shot 推理先测 100 条 validation。此实验不训练模型，不读取 evaluation。

## 目录

- `.venv/`：独立 Python 3.11 环境。
- `data/AFQMC_全组统一数据包.zip`：A 同学原数据包的副本，不重新划分。
- `models/`：手动下载的 Qwen3-1.7B 模型文件。
- `scripts/run_zero_shot.py`：预测、计时、指标与元数据记录。
- `llm_client.py`：后续 Zero-shot/Few-shot 共用的本地模型调用接口。
- `LLM_CLIENT.md`：公共接口的调用方式、日志和计时定义。
- `scripts/download_model.py`：下载官方模型。
- `work/`：依赖和模型下载缓存。
- `outputs/`：实验结果。

## 已完成安装后的运行方法

在 PowerShell 执行，不必激活虚拟环境：

```powershell
cd D:\nlp
.\run_smoke.ps1
```

若系统策略阻止执行 ps1，可直接运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_zero_shot.py --data-zip data\AFQMC_全组统一数据包.zip --model models --limit 100 --output outputs\smoke_manual_01
```

每次使用新的输出目录，避免覆盖。预测文件使用全组约定的七字段。无效生成不会被强行改成标签，程序会停止并在 metadata.json 记录原输出；部分结果不当作完整结果提交。

100 条抽样固定 seed=42，仅用于检查流程、初步查看效果与估算速度。延迟包含单条模板构建、分词、GPU 推理及解码，排除加载、预热及文件写入；正式比较前与 D 统一口径。小样本指标不能代表完整验证集或最终评估集。

下一步为完整 validation 的提示词实验，再加入仅从 train 选择示例的 Few-shot。正式实验需固定模型权重版本和依赖版本，并与 C 使用相同基础模型。
