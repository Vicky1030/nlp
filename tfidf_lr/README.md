# TF-IDF + Logistic Regression

This module preserves the core LR experiment and does not publish sentence data, trained LR models, probabilities or prediction rows. It starts from three already-fitted TF-IDF vectorizers (`char_1`, `char_1_2`, `word_1`); this repository does not add a TF-IDF fitting stage to the LR pipeline.

Stage 1 trains 18 LR candidates (3 representations x 3 C values x 2 class-weight settings) using sparse pair features `[abs(A-B); A.multiply(B)]`. Stage 2 compares 102 thresholds for every candidate (1,836 rows) with the original six-level selection rule. Stage 3 evaluates only after freezing the configuration. The recorded choice is `char_1_2__C1p0__cw_balanced`, threshold `0.53`, with `probability >= threshold`.

## External resources and environment

Provide authorized external `train.jsonl`, `validation.jsonl`, `evaluation.jsonl` files and the three fitted vectorizer directories, each containing `vectorizer.joblib` and metadata. Cloning the repository alone is not sufficient for training or frozen inference because derived LR artifacts are intentionally external.

```powershell
$Module = (Resolve-Path .\tfidf_lr).Path
$Data = 'D:\authorized\afqmc\processed'
$Vectorizers = 'D:\authorized\tfidf_vectorizers'
$Work = Join-Path 'D:\' ('TFIDF_LR_run_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $Work -ErrorAction Stop | Out-Null
$env:TEMP = Join-Path $Work 'temp'
$env:TMP = Join-Path $Work 'tmp'
$env:PIP_CACHE_DIR = Join-Path $Work 'pip-cache'
$env:PYTHONDONTWRITEBYTECODE = '1'
New-Item -ItemType Directory -Force $env:TEMP,$env:TMP,$env:PIP_CACHE_DIR | Out-Null
py -3.14 -m venv (Join-Path $Work 'venv')
$Python = Join-Path $Work 'venv\Scripts\python.exe'
& $Python -B -m pip install -r (Join-Path $Module 'requirements.txt')
```

## Complete LR experiment

```powershell
& $Python -B (Join-Path $Module 'run_tfidf_lr.py') `
  --data-dir $Data `
  --vectorizer-root $Vectorizers `
  --evaluation (Join-Path $Data 'evaluation.jsonl') `
  --output-root (Join-Path $Work 'full_run')
```

The output root must be new. The wrapper runs LR training, stage-1 reload acceptance, validation selection, frozen evaluation and output integrity checks in order. It does not refit TF-IDF, merge validation into training or search parameters using evaluation.

Historical evaluation references are 4,316 rows, confusion matrix `[[2108,870],[667,671]]`, Accuracy about `0.643883`, and Macro-F1 about `0.599485`.

## Frozen inference

Frozen inference requires the complete external stage-2 selected-artifact directory and the stage-1 run used by its protection checks:

```powershell
& $Python -B (Join-Path $Module 'src\run_tfidf_lr_stage3.py') `
  --config 'D:\authorized\lr_stage2\selected_config.json' `
  --evaluation (Join-Path $Data 'evaluation.jsonl') `
  --stage1-run 'D:\authorized\lr_stage1' `
  --data-dir $Data `
  --vectorizer-root $Vectorizers `
  --output-dir (Join-Path $Work 'frozen_inference')
```

`src/` retains the strict loader, sparse feature construction, model-index resolution, metric calculation, all three stages and the reload/fresh-process verification helpers required by the core pipeline. `reports/tables/` retains small aggregate parameter and metric summaries.

Method prose, flowchart images, plotting/report generation and historical presentation materials were removed from this streamlined repository version. Complete materials remain in the separate local delivery package.
