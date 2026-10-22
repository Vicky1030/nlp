# TF-IDF + Logistic Regression

This directory contains the reproducible source release for the TF-IDF + Logistic Regression method. It does not publish AFQMC sentence data, saved models/vectorizers, prediction or probability rows, error lists, or execution logs. Those resources remain local because the available delivery records do not establish public redistribution permission.

## Method and frozen protocol

Pair features are sparse CSR matrices `[abs(A-B); A.multiply(B)]`: the absolute difference is first and the elementwise product second, giving `2V` features for a vocabulary of size `V`. The released pipeline starts from three already-fitted TF-IDF vectorizers (`char_1`, `char_1_2`, `word_1`); it does not refit TF-IDF. It trains 18 LR candidates (3 representations x 3 C values x 2 class-weight settings), evaluates 102 thresholds per candidate (1,836 model/threshold comparisons), applies the recorded six-level selection rule, and evaluates only after freezing the selected configuration. Validation is not merged back into training.

The LR training seed is `random_state=42`. This is separate from the seed used by the original data-preparation process. The released code consumes frozen train/validation/evaluation files and does not repartition them.

The historical selected configuration is `char_1_2__C1p0__cw_balanced` with threshold `0.53`; label 1 is predicted when `p >= 0.53`. The recorded evaluation reference is 4,316 rows, confusion matrix `[[2108,870],[667,671]]`, Accuracy about `0.643883`, and Macro-F1 about `0.599485`. These are reference values, not values recomputed by this source-only release.

## External resources

Prepare these resources outside the repository and pass their paths explicitly:

- `train.jsonl`, `validation.jsonl`, and `evaluation.jsonl`, each with `id`, `sentence1`, `sentence2`, `label` fields;
- three fitted vectorizers under a directory shaped like `char_1/vectorizer.joblib`, `char_1_2/vectorizer.joblib`, and `word_1/vectorizer.joblib`, plus their metadata;
- for quick inference, a frozen selected configuration, selected LR model, selected vectorizer, and the stage-1 model index/probability files required by the protection checks.

No verified public download URL is available for the unified processed data or the derived model/vectorizer artifacts. The code can process authorized local resources; cloning this repository alone is therefore not sufficient for a complete run.

## Environment and commands

The tested dependency pins are in `requirements.txt` (Python 3.14.0 was used for the historical Windows verification). From the repository root:

```powershell
$Work = Join-Path 'D:\' ('TFIDF_LR_runs_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $Work -ErrorAction Stop | Out-Null
$env:TEMP = Join-Path $Work 'temp'
$env:TMP = Join-Path $Work 'tmp'
$env:PIP_CACHE_DIR = Join-Path $Work 'pip-cache'
$env:PYTHONDONTWRITEBYTECODE = '1'
New-Item -ItemType Directory -Force $env:TEMP,$env:TMP,$env:PIP_CACHE_DIR | Out-Null
py -3.14 -m venv (Join-Path $Work 'venv')
$Python = Join-Path $Work 'venv\Scripts\python.exe'
& $Python -B -m pip install -r .\tfidf_lr\requirements.txt
```

Complete LR reproduction starts from the supplied vectorizers and writes only to a new external output root:

```powershell
& $Python -B .\tfidf_lr\run_tfidf_lr.py `
  --data-dir 'D:\authorized\afqmc\processed' `
  --vectorizer-root 'D:\authorized\tfidf_vectorizers' `
  --evaluation 'D:\authorized\afqmc\processed\evaluation.jsonl' `
  --output-root (Join-Path $Work 'full_reproduction_01')
```

Quick frozen inference likewise requires external frozen artifacts and a new output directory:

```powershell
& $Python -B .\tfidf_lr\src\run_tfidf_lr_stage3.py `
  --config 'D:\authorized\lr_stage2\selected_config.json' `
  --evaluation 'D:\authorized\afqmc\processed\evaluation.jsonl' `
  --stage1-run 'D:\authorized\lr_stage1' `
  --output-dir (Join-Path $Work 'quick_inference_01')
```

Missing resources fail explicitly. Existing output directories are rejected; use a new external directory for every run. The source tree is not an output location.

## Included materials

`src/` contains the loader, sparse feature construction, three pipeline stages, frozen inference and verification helpers. `reports/tables/` contains aggregate parameter/metric tables without sentence text. `reports/method/` contains the method summary, implementation notes and the supplied PNG flowchart. Runtime models, vectorizers, data, sentence-level outputs and logs are intentionally excluded.

