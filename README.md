# NLP experiments

This repository keeps the core experiment code for two AFQMC sentence-pair methods:

- [`tfidf_cosine/`](./tfidf_cosine/): fit three TF-IDF representations, compare 306 validation representation/threshold candidates, freeze `char_1_2` at `0.28`, and predict with `similarity >= threshold`.
- [`tfidf_lr/`](./tfidf_lr/): start from three already-fitted TF-IDF vectorizers, train 18 Logistic Regression candidates, compare 1,836 validation model/threshold candidates, freeze `char_1_2__C1p0__cw_balanced` at `0.53`, and predict with `probability >= threshold`.

Each module README documents its external data/model requirements, complete pipeline and frozen-inference command. Authorized data and local model artifacts are not added at the repository root.

Plotting, ECharts/browser export, report generation and historical closeout tooling were removed from the current source tree. Complete figures, reports and historical acceptance materials remain in separate local delivery packages; Git history was not rewritten.
