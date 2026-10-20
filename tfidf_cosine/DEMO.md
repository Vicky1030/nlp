# Demo: TF-IDF + cosine similarity

**Suggested cut:** reserve about 25-35 seconds of the combined group video for this method; keep the complete group video under two minutes. Prepare the environment and authorized unified data before recording. Show the name and student ID only as required by the course; this document does not guess either value.

## Complete reproduction

From the repository root:

```powershell
Set-Location .\tfidf_cosine
.\.venv\Scripts\python.exe -B -X utf8 .\run_tfidf_cosine.py --data-dir "D:\authorized\afqmc\processed" --output-root ".\reproduced\demo_full_01"
```

**Say:** “We compare paired sentences using TF-IDF vectors and cosine similarity. The full run fits on train, checks 306 preset configurations on validation, freezes the choice, then predicts evaluation. This takes longer than the clip, so I am showing the real startup and completed output with the middle omitted.” Only accelerate or cut the recording if this edit is disclosed. Point to the actual selected configuration, prediction count, and metrics printed by the run.

## Quick alternative

If recording only the quick inference, use:

```powershell
Set-Location .\tfidf_cosine
.\.venv\Scripts\python.exe -B -X utf8 .\src\run_tfidf_stage3.py --config ".\artifacts\tfidf_cosine_full_run_final\stage2\artifacts\tfidf_cosine\selected_config.json" --evaluation "D:\authorized\afqmc\processed\evaluation.jsonl" --output-root ".\reproduced\demo_quick_01"
```

**Say:** “This quick check loads the already frozen model and threshold. It does not redo fitting or configuration selection.” Do not present it as a full reproduction.

## Existing failure example

One real failure example is evaluation ID `34937` in the local `error_analysis/九例失败案例分析.md` and `error_analysis/evidence/feature_details/34937.json`. The example includes source sentences and is intentionally not in the public repository because sharing authorization is unconfirmed. If showing its text, open it from the authorized local delivery copy; do not expose it on the public GitHub page. Describe only the recorded true label, prediction, similarity and feature evidence; do not embellish it with a new failure or claim.
