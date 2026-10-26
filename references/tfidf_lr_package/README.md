# TF-IDF + Logistic Regression 冻结测评推理包

本包只执行冻结推理，不训练、不拟合 TF-IDF、不搜索参数。配置为
`char_1_2__C1p0__cw_balanced`，阈值为 `0.53`，规则为 `probability >= 0.53`。

## 安装与运行

在包外的 D 盘工作目录操作，示例路径可按需修改；包内不要创建虚拟环境、缓存或输出：

```powershell
$Package = 'D:\TFIDF_LR_测评推理包_02'
$Work = 'D:\TFIDF_LR_推理工作 中文 空格'
$env:TEMP = Join-Path $Work 'temp'
$env:TMP = Join-Path $Work 'tmp'
$env:PIP_CACHE_DIR = Join-Path $Work 'pip-cache'
New-Item -ItemType Directory -Force $Work,$env:TEMP,$env:TMP,$env:PIP_CACHE_DIR | Out-Null
py -3.14 -m venv (Join-Path $Work 'venv')
$Python = Join-Path $Work 'venv\Scripts\python.exe'
& $Python -B -m pip install -r (Join-Path $Package 'requirements.txt')
$EvaluationPath = 'D:\inputs\evaluation.jsonl'
$Output = Join-Path $Work 'result_01'
& $Python -B (Join-Path $Package 'predict.py') --input $EvaluationPath --output-dir $Output
```

命令可从任意工作目录启动；`--output-dir` 必须指定新的目录，已有目录会被拒绝，不会删除或覆盖结果。输入 JSONL 每行严格包含 `id、sentence1、sentence2、label` 四字段。标签只写入结果，不进入模型推理。

## Python 接口

```python
from predict import FrozenPredictor

predictor = FrozenPredictor()
results = predictor.predict_pairs([
    ("花呗怎么还款", "花呗如何还款")
])
```

`FrozenPredictor` 初始化时加载向量器和 LR 模型各一次，多次调用建议复用实例。顶层 `predict_pairs` 便利函数每次调用都会新建预测器并加载模型。输入可为列表、元组或生成器；每项必须是恰好两个非空白字符串。空输入返回 `[]`。输入原句不会被 strip 或改写。

模型对两句话分别 transform，构造 CSR 特征 `[abs(A-B); A.multiply(B)]`，再由 `predict_proba` 取得标签 1 的概率，并按阈值判断。probability 是标签 1 的概率估计，不是相似度，也不表示预测正确率。

CLI 的 `latency_ms` 包含文本处理、两次 TF-IDF transform、稀疏特征构造、`predict_proba` 和阈值判断；排除模型加载、预热、文件读写、评分和结果组装。CLI 先预热前 10 条记录（不足 10 条则全部预热）。

输出文件为 `predictions.jsonl`（七字段 `id、sentence1、sentence2、label、prediction、method、latency_ms`）、`probabilities.jsonl`（`id、probability、threshold`）和 `run_summary.json`。仅从可信来源加载包内模型文件。

Python 3.14 和 `requirements.txt` 中的固定版本已验证。

许可和模型/数据使用边界见 `THIRD_PARTY_NOTICES.md`。本包不包含任何训练数据。
