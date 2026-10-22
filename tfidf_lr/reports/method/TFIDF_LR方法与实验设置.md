# TF-IDF＋Logistic Regression方法与实验设置

## 1. 任务与数据划分

任务输入为中文句子对，输出二分类标签：0表示不相似，1表示语义相似。当前统一数据包记录了train 30,899对、validation 3,433对、evaluation 4,316对。三者均为带标签的句子对，但用途不同：train句子的两列文本用于拟合TF-IDF词表和IDF，train句子对特征及标签用于训练LR；validation用于比较候选模型和阈值；evaluation只用于配置冻结后的最终评估。

本实验没有重新拟合TF-IDF，而是复用包内已经拟合的三套向量器。其metadata记录每套向量器由30,899个训练句对的两列文本组成61,798条拟合文档，训练数据哈希与本次统一train一致。validation与evaluation在对应阶段只调用transform。evaluation对应数据说明中所述的统一AFQMC Dev评估集；它不是官方无标签Test，官方Test没有纳入本实验。

数据说明要求统一划分、保留ID和标签，不重新划分。该说明还记录处理副本排除过两条同标签逆序重复（7812保留7487，33870保留21154），以及移除一处Train原始第12078行句子开头的U+FEFF；未做其他文本规范化。补查原数据准备manifest及处理报告后，确认其记载按标签分层划出10%作为validation：按类别0、1顺序，将各类记录先按ID排序，再使用同一个`random.Random(42)`依次打乱；每类验证数量按十进制ROUND_HALF_UP取整，取打乱后的前若干条。处理后的官方Train形成train/validation，官方Dev形成evaluation，不由上述随机划分产生。三份输出大小和SHA-256与当前数据一致。见[数据划分依据](数据划分依据.md)及必要字段摘录；这些是历史记录的只读对账，本轮没有重新划分。划分种子与第4节LR训练种子分别有独立依据。

### 方案选择依据

选择TF-IDF＋LR作为传统监督基线，是因为它能利用训练标签学习句子对特征权重，而余弦阈值方法直接对句向量相似度作决策。当前实现使用高维CSR稀疏特征，适合本地CPU实验；线性模型系数与特征贡献还能帮助分析部分判断依据，但不能代表完整语义理解或唯一因果证明。该基线便于与余弦、Prompt和LoRA方法按统一数据与指标比较。以上是方案和实现的选择理由，不是新增对比实验，不意味着LR必然更优，也不包含其他组员的成绩。

## 2. TF-IDF文本表示

三种表示分别是字符1-gram（`char_1`）、字符1与2-gram（`char_1_2`）和jieba词级1-gram（`word_1`）。所有向量器均使用UTF-8内容输入、`float64`、`use_idf=True`、`smooth_idf=True`、`norm='l2'`、`min_df=1`、`max_df=1.0`、`max_features=None`、`sublinear_tf=False`、`binary=False`、`lowercase=False`、不使用stop words，且没有自定义preprocessor或vocabulary。

字符表示使用scikit-learn的`analyzer='char'`。`char_1`的`ngram_range=(1,1)`；`char_1_2`为`(1,2)`。metadata记录其预处理采用`lowercase=False`、不去重音，并按scikit-learn字符分析流程处理连续空白；没有额外文本规范化。词级表示使用`analyzer='word'`、`ngram_range=(1,1)`及项目内`jieba_tokens`，以jieba自带词典、`cut_all=False`、`HMM=False`切词，再滤除纯空白token。该设置下`token_pattern=None`，不使用正则默认token pattern。字符N-gram直接描述相邻字符片段，词级表示先依赖分词边界；这是两类稀疏文本表示的比较，不是神经语言模型或PPL实验。

各项实际参数在三个metadata中一致，除analyzer、ngram范围及word_1的tokenizer/token_pattern差异外。代码设置原始词频，不启用次线性TF。用常见记号表示，未归一化TF为词项在文档中的计数`tf(t,d)`；平滑IDF为：

`idf(t) = log((1 + N) / (1 + df(t))) + 1`

其中`N`为拟合文档数，`df(t)`为包含该特征的文档数。TF-IDF权重经L2归一化：每个单句向量除以其L2范数（零向量保持为零）。拟合语料由每个train句子对的sentence1和sentence2分别展开，因此共61,798条文本；LR阶段复用已有对象，validation/evaluation只转换，不更新词表或IDF。

## 3. 句子对特征

设同一向量器转换两个句子后得到稀疏向量`A`和`B`，特征按以下顺序构造：

`x = [|A-B| ; A⊙B]`

`|A-B|`是逐位置绝对差，`A⊙B`是逐位置乘积；拼接时绝对差在前、乘积在后。词表大小为`V`时，每一部分有`V`维，总维数为`2V`。代码使用SciPy `hstack(..., format='csr')`，保持CSR稀疏格式和float64；标签仅作为训练目标，不拼入特征。单句TF-IDF在transform时按L2规范化；拼接后的句子对特征没有额外归一化。

**示意（不是实际样本的TF-IDF值）：**`A=[1,1,0]`，`B=[1,0,1]`；绝对差为`[0,1,1]`，乘积为`[1,0,0]`，拼接结果为`[0,1,1,1,0,0]`。

该构造交换两句话时保持不变，因为绝对差和逐位置乘积均对交换对称。它表示的是句子对的对称关系，不能区分sentence1与sentence2的角色方向；但单句字符双字仍编码各自句内相邻字符的局部顺序，不能据此说句内字符顺序完全丢失。

## 4. Logistic Regression与训练设置

LR对句子对特征计算线性决策值`z = wᵀx + b`，再通过sigmoid得到标签1概率估计：`p(y=1|x) = 1 / (1 + exp(-z))`。权重`w`对各特征作线性加权，`b`为截距。概率是模型输出的概率估计，不是预测正确率，也不是余弦相似度。最终类别由独立阈值规则决定。

所有候选使用L2正则化和`liblinear`求解器，训练参数为`fit_intercept=True`、`intercept_scaling=1`、`dual=False`、`max_iter=5000`、`tol=0.0001`、`random_state=42`、`l1_ratio=0.0`、`n_jobs=None`、`verbose=0`、`warm_start=False`。候选`C`为0.1、1、10；`class_weight`为None或balanced。C越小，正则化越强。balanced按训练集标签频数设置类别权重；类别权重改变训练损失中的类别权重，阈值则改变概率转类别的决策边界，两者作用环节不同。实际获选模型使用`predict_proba`推理；求解器名称描述训练求解过程，不表示推理也调用该优化器。推理在CPU运行，未使用GPU。

阶段一代码捕获并记录训练警告，未解决的`ConvergenceWarning`会阻断后续配置；阶段一验收记录18组模型均完成参数与迭代记录核验，`convergence_warning_count=0`。最终训练运行记录含18条普通`FutureWarning`，内容是scikit-learn对显式`penalty`参数的弃用提示；验收没有要求普通警告必须出现特定数量，也没有把它当作收敛错误。更早的一次尝试因把普通警告当作阻断而停止于第一组，后续正式运行保留`penalty='l2'`并仅让收敛警告阻断。上述为历史运行记录，本阶段没有重新训练。

## 5. 参数搜索与选择

搜索组合为3种表示×3个C×2种类别权重，共18组LR模型。每组使用同一份已保存的validation概率，评估102个阈值：`0.00`至`1.00`（步长`0.01`），以及`1.01`，总计`18×102=1,836`个“配置＋阈值”结果。阈值1.01用于使所有概率均判为0；0.5只是普通候选阈值，不是额外训练实验。阈值变化只重新计算类别和评分，不重新拟合模型。

选择按以下六级顺序，前项相同时才比较后项：①Macro-F1降序；②Accuracy降序；③阈值降序；④C升序；⑤None优先于balanced；⑥表示顺序`char_1`、`char_1_2`、`word_1`。评分实现以混淆计数构造精确分数（`Fraction`）参与排序，报告中的小数仅用于展示。先在每个模型的102个阈值中选择，再比较18组模型，与将1,836行一次性按同一六级规则排序的结果一致。

## 6. 评价指标

以标签1为正类，令TP、TN、FP、FN表示混淆计数：

- `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
- `Precision(1) = TP / (TP + FP)`
- `Recall(1) = TP / (TP + FN)`
- `F1(1) = 2 × Precision × Recall / (Precision + Recall)`
- `Macro-F1 = (F1(0) + F1(1)) / 2`，固定包含标签`[0,1]`

零分母按0处理。混淆矩阵的行为真实标签、列为预测标签，标签顺序为`[0,1]`，即`[[TN,FP],[FN,TP]]`。选择主要依据Macro-F1，是为了同时计入两个类别的F1；只看Accuracy可能在类别不均衡时掩盖较弱类别的表现。这是指标口径的考虑，不代表Macro-F1在所有任务中都必然更合适。最终真实成绩及固定全0参照见第一阶段材料；本报告不重复完整结果表，也不作显著性或普遍优势结论。

## 7. 冻结配置与最终预测

冻结配置为`char_1_2__C1p0__cw_balanced`，阈值`0.53`。流程是：读取evaluation句子对→加载保存的TF-IDF向量器→对两句话transform→按`[|A-B|; A⊙B]`构造CSR特征→LR计算标签1概率→若`p≥0.53`预测1，否则预测0→保存记录。冻结后未把validation合并进train重训。

预测记录字段为`id`、`sentence1`、`sentence2`、`label`、`prediction`、`method`、`latency_ms`，其中`method`为`tfidf_lr`。标签1概率另存于同阶段`evaluation_probabilities.jsonl`，每行以相同顺序的`id`对应，并包含`probability`和`threshold`。evaluation标签在该阶段用于评分与结果记录，不进入特征，不用于训练，也不用于阈值选择。可复现冻结推理需同时保留TF-IDF对象、LR模型、冻结配置及特征构造/推理代码；配置文件本身不能替代这些对象和实现。

## 8. 与TF-IDF＋余弦阈值方法的区别

| 方法 | 句子对打分 | 学习过程 | 决策 |
|---|---|---|---|
| TF-IDF＋余弦阈值 | 两句TF-IDF向量计算余弦相似度 | 不训练句子对分类器 | 在validation上根据带标签结果选择相似度阈值，再比较相似度与阈值 |
| TF-IDF＋LR | 构造绝对差与逐位置乘积特征 | 用train句子对标签学习LR权重 | 在validation上选择标签1概率阈值，再据此预测 |

余弦方法虽不学习LR分类权重，其validation阈值选择仍使用validation标签，不能说成完全不使用标签。本合并包没有可用于本报告核实的余弦最终成绩对照，因此这里只比较方法机制，不给出性能比较。
