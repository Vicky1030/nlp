# TF-IDF＋余弦相似度：成员D冻结推理包

本包只做冻结推理，不包含训练、TF-IDF拟合、参数搜索、绘图、报告材料或数据文件。

固定方法为 `tfidf_cosine`，表示为 `char_1_2`，阈值为 `0.28`，判断规则为 `similarity >= 0.28`。相似度是余弦相似度，不是概率。模型只加载一次；输入标签只写入结果记录，不进入向量转换或相似度计算。

## 文件

- `predict.py`：CLI和Python函数接口。
- `model/vectorizer.joblib`：冻结char_1_2向量器。
- `model/metadata.json`：模型身份、训练数据哈希和参数metadata。
- `frozen_config.json`：包内相对路径的冻结配置。
- `data_loader.py`：严格UTF-8无BOM四字段JSONL读取。
- `requirements.txt`：实际推理依赖。
- `acceptance_summary.md`：本轮包外验证摘要。

## 安装

虚拟环境、临时目录、pip缓存和推理结果均放在包外D盘工作目录。以下示例支持中文和空格路径；请按实际位置修改 `$Package` 和 `$Work`，不要在包内创建 `.venv`：

```powershell
$Package = 'D:\TFIDF 推理包 02'
$Work = 'D:\TFIDF 推理包工作区\run_01'
if (Test-Path -LiteralPath $Work) { throw '工作目录已存在，请换用新的目录' }
New-Item -ItemType Directory -Path "$Work\temp","$Work\tmp","$Work\pip-cache","$Work\results" | Out-Null
$env:TEMP = "$Work\temp"
$env:TMP = "$Work\tmp"
$env:PIP_CACHE_DIR = "$Work\pip-cache"
python -B -m venv "$Work\venv"
$Py = "$Work\venv\Scripts\python.exe"
& $Py -B -m pip install -r "$Package\requirements.txt"
```

也可以使用已安装且版本匹配的环境。模型反序列化只应使用可信来源的模型文件。

## 输入与命令

输入是UTF-8、无BOM、LF换行的JSONL，每行严格包含 `id`、`sentence1`、`sentence2`、`label`；推理只读取前两句，label仅用于原样记录。通过 `--output-dir` 指定新的结果目录，不要求先切换到该目录；已有目标文件会被拒绝：

```powershell
& $Py -B "$Package\predict.py" --input "D:\path\evaluation.jsonl" --output-dir "$Work\results\inference_01"
```

输入和输出路径均由参数指定；从其他工作目录启动时，将 `predict.py` 改为包的绝对路径即可。包外测试使用正式evaluation时，不能把该数据复制进本包。

## Python接口

```python
from predict import FrozenPredictor

predictor = FrozenPredictor()
results = predictor.predict_pairs([
    ("花呗怎么还款", "花呗如何还款")
])
# 每项为 {"prediction": 0 or 1, "similarity": float}
```

函数返回顺序与输入一致。顶层 `predict_pairs` 每次调用都会创建预测器并加载模型；多次调用建议复用 `FrozenPredictor` 实例。CLI本身复用同一个实例处理全部输入。输入可为任意可迭代对象，但每项必须是恰好两个字符串的列表或元组；空输入返回空列表。

## 输出

`evaluation_predictions.jsonl`严格七字段：`id`、`sentence1`、`sentence2`、`label`、`prediction`、`method`、`latency_ms`。`method`固定为 `tfidf_cosine`。`evaluation_similarity.jsonl`单独保存 `id`、`label`、`similarity`，通过ID关联。CLI的 `latency_ms` 是预热后单条CPU推理计时，包含文本转换、两次transform、稀疏余弦和阈值判断，不包含模型加载、预热、读写、评分或结果组装；计时会随环境变化。Python函数调用方可自行计时。

输出目录已有任一目标文件时，程序拒绝运行，不删除、不清空、不覆盖历史输出。
