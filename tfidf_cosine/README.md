# TF-IDF + Cosine Similarity

This module preserves the core three-stage experiment: fit `char_1`, `char_1_2` and `word_1` TF-IDF vectorizers on internal training text; compare 102 thresholds per representation on validation (306 candidates); freeze the selected configuration; then predict and score evaluation. The protocol remains `char_1_2`, threshold `0.28`, and `similarity >= threshold`.

## External data and environment

AFQMC sentence data is not redistributed here. Supply an authorized external directory containing `train.jsonl`, `validation.jsonl` and `evaluation.jsonl`, each with exactly `id`, `sentence1`, `sentence2`, `label`. `tools/check_processed.py` checks the recorded hashes, counts and non-overlapping IDs. A mismatch is a stop condition; do not regenerate or replace the formal split merely to pass the check.

From the repository root, keep environments, caches and outputs outside the clone:

```powershell
$Module = (Resolve-Path .\tfidf_cosine).Path
$Data = 'D:\authorized\afqmc\processed'
$Work = Join-Path 'D:\' ('TFIDF_Cosine_run_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $Work -ErrorAction Stop | Out-Null
$env:TEMP = Join-Path $Work 'temp'
$env:TMP = Join-Path $Work 'tmp'
$env:PIP_CACHE_DIR = Join-Path $Work 'pip-cache'
$env:PYTHONDONTWRITEBYTECODE = '1'
New-Item -ItemType Directory -Force $env:TEMP,$env:TMP,$env:PIP_CACHE_DIR | Out-Null
py -3.14 -m venv (Join-Path $Work 'venv')
$Python = Join-Path $Work 'venv\Scripts\python.exe'
& $Python -B -m pip install -r (Join-Path $Module 'requirements.txt')
& $Python -B -X utf8 (Join-Path $Module 'tools\check_processed.py') --data-dir $Data
```

## Complete experiment

```powershell
& $Python -B -X utf8 (Join-Path $Module 'run_tfidf_cosine.py') `
  --data-dir $Data `
  --output-root (Join-Path $Work 'full_run')
```

The output root must be new. Stage 1 fits the three vectorizers using only train text; stage 2 performs the fixed 306-candidate validation comparison; stage 3 uses the frozen choice for evaluation and validates the seven-field prediction output. `TFIDF_STAGE1.md` is retained because the unified entry copies it into stage 1 and stage 2 checks it.

Historical expected results are `char_1_2`, threshold `0.28`, 4,316 evaluation predictions, Accuracy about `0.603105`, Macro-F1 about `0.557542`, and confusion matrix `[[1994,984],[729,609]]`. These are comparison references, not hard-coded evaluation inputs.

## Frozen inference

The repository includes the historical selected configuration and vectorizer. Supply the authorized external evaluation file and a new output root:

```powershell
& $Python -B -X utf8 (Join-Path $Module 'src\run_tfidf_stage3.py') `
  --config (Join-Path $Module 'artifacts\tfidf_cosine_full_run_final\stage2\artifacts\tfidf_cosine\selected_config.json') `
  --evaluation (Join-Path $Data 'evaluation.jsonl') `
  --output-root (Join-Path $Work 'frozen_inference')
```

The committed vectorizers and metadata are trusted-input-only `joblib` artifacts. Core aggregate materials retained under `artifacts/` include the three vectorizers, validation similarities, selected configuration, representation summary and full 306-row threshold table. `reports/processed_hashes.json` records formal split integrity; it is not a data license.

Plotting, ECharts/Playwright resources, error-case analysis, report generation, release packaging and historical closeout scripts are not part of this streamlined repository version. They remain in the separate local delivery package.
