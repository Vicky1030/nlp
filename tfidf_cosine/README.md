# TF-IDF + Cosine Similarity

This directory contains only the TF-IDF + cosine similarity method. It preserves the three-stage scripts, shared loader and tokenizer/scoring modules, plotting and error-analysis utilities, case-feature extraction, regression checks, stage-one helper document, frozen models/configuration, validation comparison tables, and method figures. No LR code is included.

## Data and preparation

The AFQMC source and processed rows are not redistributed here. Their source, version authenticity, and public-sharing authorization are unconfirmed in the accompanying preparation record. Do not infer a data license from this code or publish sentence-level predictions, error lists, case evidence, or reports containing source sentences. Obtain the data through the course's authorized channel and follow its terms. Keep the existing unified processed files in an external directory; do not split or regenerate them for these commands.

Expected files are `train.jsonl`, `validation.jsonl`, and `evaluation.jsonl`, with exactly `id`, `sentence1`, `sentence2`, and integer `label` fields. `tools/check_processed.py` validates schema, record counts, non-overlapping IDs, and the known SHA-256 hashes before either run. A mismatch is a stop condition, not an invitation to preprocess or split the data again.

## Install and verify data

PowerShell examples below start at the repository root and run from this method directory:

```powershell
Set-Location .\tfidf_cosine
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -B -X utf8 .\tools\check_processed.py --data-dir "D:\authorized\afqmc\processed"
```

No Node.js, Playwright, browser, or model service is required for the Python experiments. These dependencies are optional for re-exporting the supplied ECharts files.

## One-command complete reproduction

After the environment and all three verified external data files are ready, this command fits the three TF-IDF representations on train only, compares the 306 predeclared representation/threshold configurations on validation, freezes the selected model, and then predicts and scores evaluation:

```powershell
Set-Location .\tfidf_cosine
.\.venv\Scripts\python.exe -B -X utf8 .\run_tfidf_cosine.py --data-dir "D:\authorized\afqmc\processed" --output-root ".\reproduced\full_01"
```

It prints the selected method, representation and threshold, prediction count, metrics, and output paths. Failure stops the pipeline with a nonzero exit. Output directories must be new and are never overwritten. The expected historical check is `tfidf_cosine`, `char_1_2`, threshold `0.28`, 4,316 evaluation rows, Accuracy about `0.603105`, Macro-F1 about `0.557542`, and `[[1994,984],[729,609]]`; these are comparison references, not hard-coded outputs.

## Frozen-model quick verification

After the environment and the existing unified evaluation file are ready, the following loads the committed historical vectorizer and frozen threshold and runs final inference only. It does not refit TF-IDF or search/select thresholds:

```powershell
Set-Location .\tfidf_cosine
.\.venv\Scripts\python.exe -B -X utf8 .\src\run_tfidf_stage3.py --config ".\artifacts\tfidf_cosine_full_run_final\stage2\artifacts\tfidf_cosine\selected_config.json" --evaluation "D:\authorized\afqmc\processed\evaluation.jsonl" --output-root ".\reproduced\quick_01"
```

The final prediction and similarity files include source sentences and labels. Keep them local unless data-sharing permission is confirmed.

## Included and intentionally local-only materials

The committed historical models are ordinary trusted-input-only `joblib` files and should be loaded only from this repository. Included validation comparison CSVs, aggregate score tables, metrics figures, method-flow figures, model metadata, and ECharts resource notices/licenses are selected materials without AFQMC source sentences. The `data/processed/`, `error_analysis/inputs/`, `error_analysis/evidence/`, full stage-three predictions/similarities, error samples and full-run logs from the source delivery remain in the separate complete local backup and are excluded from Git. `TFIDF_STAGE1.md` remains in place because the unified entry and stage checks require it.

Historical model metadata retains the versions, parameters, and hashes needed to understand and load the frozen models. Personal absolute paths to the original Python executable and jieba dictionary were removed from the release copy; neither path is a runtime dependency. Regenerated chart summaries use relative paths.

The original source archive and complete local delivery contain `reports/data_manifest.json`. It is intentionally not published: it records personal absolute input paths and the source package leaves download provenance, official-version authenticity, and license authorization unverified. This is not a missing-file or copy error. The data-preparation source is retained separately from that report. `reports/processed_hashes.json` contains integrity checks only; it does not establish a license or grant access. `reports/release_sha256.json` inventories the files in this repository release; the source archive's `package_sha256.json` is retained only in the full local backup and is not reused as a Git manifest.

The auxiliary `src/inspect_data.py`, `src/prepare_data.py`, and `src/validate_data.py` code is retained for transparency and reuse, but is not part of the default TF-IDF reproduction. These utilities require authorized external raw inputs and/or historical local audit artifacts that are deliberately not included. Run `inspect_data.py` with explicit `--train`, `--dev`, `--test`, and `--output-dir` paths. `prepare_data.py` requires `--audit`, explicit `--train`, `--dev`, `--test`, and a new `--output-root`; it reproduces a documented preparation procedure and must not be used to replace the existing unified processed files for the model commands above. `validate_data.py` is a historical acceptance harness for a complete prepared-data package containing its audit and manifest reports, so it cannot run against this source-only Git release. Its inputs and generated reports may contain source sentences and must remain local unless sharing is authorized. No data-preparation command is implied by the model reproduction instructions.

The ECharts 5.6.0 JavaScript bundle and Noto CJK font are accompanied by their source/identity notes and license texts under `report_materials/echarts_v6/`. No project code license is added because the source package has none.

## Optional analysis and figures

All commands below use new, unused output directories. They need the external evaluation rows or generated evaluation outputs, which may contain source sentences.

```powershell
Set-Location .\tfidf_cosine
.\.venv\Scripts\python.exe -B -X utf8 .\src\plot_tfidf_results.py --train "D:\authorized\afqmc\processed\train.jsonl" --threshold-table .\artifacts\tfidf_cosine_full_run_final\stage2\results\tfidf_cosine\threshold_comparison.csv --summary .\artifacts\tfidf_cosine_full_run_final\stage2\results\tfidf_cosine\representation_summary.csv --stage3-report .\verification_runs\full_01\stage3\reports\tfidf_cosine_stage3.json --predictions .\verification_runs\full_01\stage3\results\tfidf_cosine\evaluation_predictions.jsonl --error-report .\verification_runs\full_01\stage3\reports\tfidf_cosine_stage3.json --output-dir .\reproduced\plots_01
.\.venv\Scripts\python.exe -B -X utf8 .\src\analyze_tfidf_errors.py --predictions .\verification_runs\full_01\stage3\results\tfidf_cosine\evaluation_predictions.jsonl --similarity .\verification_runs\full_01\stage3\results\tfidf_cosine\evaluation_similarity.jsonl --config .\verification_runs\full_01\stage2\artifacts\tfidf_cosine\selected_config.json --output-dir .\reproduced\errors_01
.\.venv\Scripts\python.exe -B -X utf8 .\src\extract_tfidf_case_evidence.py --evaluation "D:\authorized\afqmc\processed\evaluation.jsonl" --predictions .\verification_runs\full_01\stage3\results\tfidf_cosine\evaluation_predictions.jsonl --similarity .\verification_runs\full_01\stage3\results\tfidf_cosine\evaluation_similarity.jsonl --config .\verification_runs\full_01\stage2\artifacts\tfidf_cosine\selected_config.json --vectorizer .\verification_runs\full_01\stage1\artifacts\tfidf_cosine\char_1_2\vectorizer.joblib --metadata .\verification_runs\full_01\stage1\artifacts\tfidf_cosine\char_1_2\metadata.json --output-dir .\reproduced\case_evidence_01
```

To render the supplied flowchart afresh, pass the chosen output directory and (optionally) a font file: `python .\provenance\render_flowchart_closeout.py --output-dir .\reproduced\flowchart_01 --font .\report_materials\echarts_v6\ReportCJK.font`. PNG export of the supplied offline HTML charts is optional and requires Node, npm dependencies and Chromium; use `npm ci --cache .\npm-cache` and `node .\report_materials\echarts_v6\export_echarts_png.mjs --html-dir .\report_materials\echarts_v6 --output-dir .\reproduced\echarts_png_01`. The HTML is based on aggregate comparison data; the generator source is retained as an analysis reference, but its original full-result prediction input is local-only.
