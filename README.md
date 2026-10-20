# AFQMC D 统一评价模块

先看 `evaluation/README.md`。当前包已放入统一的 evaluation 标准集与 TF-IDF + Cosine 基线预测，可先运行基线评价，再把其他组员的最终预测 JSONL 放入 `results/`。

快速开始（项目根目录）：

```bash
python -m evaluation.run_all
```

如绘图依赖未安装：
```bash
python -m evaluation.run_all --no-plots
```
