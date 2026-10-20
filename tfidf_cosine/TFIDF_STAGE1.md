# TF-IDF＋余弦相似度：第一阶段运行说明

> 日常复现请以交付包根 README.md 为准。本文件是统一入口复制和第二阶段检查所需的程序辅助文件，请保留；以下阶段说明原样保留。

本阶段仅使用正式 train、validation。三种表示按 char_1、char_1_2、word_1 顺序分别拟合，仅保存向量器及内部验证集相似度。没有阈值搜索、方案选择、分类预测、分类指标、LR 或实验分析；不读取 evaluation 或官方 Test。

## 环境与运行

在项目根目录打开 PowerShell。项目独立环境为 .venv-tfidf，不修改系统 Python，也不应将虚拟环境加入交付包。

已有环境重新运行：

```powershell
.\.venv-tfidf\Scripts\python.exe -B -X utf8 .\outputs\src\run_tfidf_stage1.py `
  --train .\outputs\data\processed\train.jsonl `
  --validation .\outputs\data\processed\validation.jsonl `
  --output-root .\outputs
```

在其他机器创建环境（建议采用报告中相同 Python 与依赖版本）：

```powershell
python -m venv .venv-tfidf
.\.venv-tfidf\Scripts\python.exe -m pip install -r .\outputs\artifacts\tfidf_cosine\requirements.txt
```

requirements.txt 固定五个直接依赖；environment_freeze.txt 记录完整实际环境。--output-root 可指定其他结果根目录，其下直接创建 artifacts/、results/、reports/。不使用桌面绝对路径，不使用 deliverables 副本。

## 口径与接口

拟合文档按 train 文件顺序，逐条 sentence1、sentence2 交错展开；重复文本保留，不把标签或 ID 当作特征。train_only_fit 路径完成后才读取 validation，验证路径仅 transform。相似度函数仅接收向量/句子字符串，标签只在写结果时关联。

char_1 使用 sklearn char 1-gram，char_1_2 使用 sklearn char 1/2-gram。这里不是未经内部处理的 Unicode 拆分：sklearn char 分析器会将连续两个及以上空白合并成一个 ASCII 空格，再生成字符 n-gram；本程序不额外规范化，lowercase=False、strip_accents=None。实际实现源码和行为探针记录于报告。

word_1 使用 tfidf_common.jieba_tokens 顶层可导入函数；jieba 精确模式、HMM=False，独立 Tokenizer 使用随包默认词典，不加载用户词典、不增删词；只过滤纯空白 token，数字、标点、否定词和脱敏符号均保留，token_pattern=None。默认词典实际路径与 SHA-256 见报告。

共同参数：use_idf=True、smooth_idf=True、norm=l2、min_df=1、max_df=1.0、max_features=None、sublinear_tf=False、binary=False、lowercase=False、stop_words=None、dtype=numpy.float64。各向量器全部实际参数记录于对应 metadata.json。

使用稀疏逐行点积，无 N×N 矩阵，不把完整特征矩阵稠密化。任一侧零向量时得分为 0，记录四种零向量计数。非有限数或超过 1e-12 容差的越界会报错；容差内才裁剪到 [0,1]。JSON 浮点数保留 Python 可往返精度，不先四舍五入。

results/tfidf_cosine/ 下每条仅为 id、label、similarity，保持 validation 顺序。这是中间分数，不是七字段预测文件，不填 prediction、method 或 latency_ms。现有 data_loader.py 仍仅用于严格四字段输入。

## 保存与验证

artifacts/tfidf_cosine/ 各方案包含 vectorizer.joblib、metadata.json。加载 word_1 时，outputs/src 必须可导入（例如 sys.path.insert(0, 'outputs/src')），以找到 tfidf_common；使用报告中的同版本依赖。只加载可信 joblib。

入口自动执行：ID/标签逐行对账、数值及零向量检查、固定样本逐对标准 cosine_similarity 对照、独立合成验证独有特征泄漏测试、词表/IDF 不变检查，以及每方案新 Python 子进程重新加载并重算全部 validation（rtol=atol=1e-12）。报告位于 reports/tfidf_cosine_stage1.json，记录运行事实、环境、词典和代码哈希。现有数据准备材料不覆盖。
