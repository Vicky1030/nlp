"""Generate per-model errors and cross-model error/disagreement tables."""
import csv
import re
from pathlib import Path

NEGATION = ("不", "没", "无", "未", "不能", "无法", "不可以", "不是", "没有", "取消", "禁止")
CONDITION = ("如果", "除非", "必须", "才", "是否", "能否", "可以吗", "吗")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?|[一二三四五六七八九十百千万亿]+")


def candidate_tags(sentence1, sentence2):
    tags = []
    a, b = sentence1 or "", sentence2 or ""
    if any(k in a for k in NEGATION) != any(k in b for k in NEGATION):
        tags.append("negation_difference_candidate")
    if NUMBER_RE.findall(a) != NUMBER_RE.findall(b):
        tags.append("number_difference_candidate")
    if any(k in a for k in CONDITION) != any(k in b for k in CONDITION):
        tags.append("condition_or_question_candidate")
    # High character overlap is only a weak heuristic, not a semantic diagnosis.
    aset, bset = set(a), set(b)
    overlap = len(aset & bset) / max(1, len(aset | bset))
    if overlap >= 0.55 and a != b:
        tags.append("high_character_overlap_candidate")
    if not tags:
        tags.append("manual_review")
    return ";".join(tags)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze_group(aligned_models, output_dir, split_name):
    """
    aligned_models: {model_name: aligned rows in gold order}.
    Produces error examples and cross-model agreement patterns.
    """
    out = Path(output_dir)
    errors_dir = out / "errors" / split_name
    errors_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(aligned_models)
    gold_rows = next(iter(aligned_models.values())) if aligned_models else []
    summary_rows = []
    by_model = {}

    for model, rows in aligned_models.items():
        wrong = []
        by_id = {r["id"]: r for r in rows}
        by_model[model] = by_id
        for r in rows:
            pred = r.get("prediction")
            if pred is None or int(pred) != int(r["label"]):
                wrong.append({
                    "id": r["id"], "sentence1": r["sentence1"], "sentence2": r["sentence2"],
                    "label": r["label"], "prediction": "" if pred is None else pred,
                    "method": r.get("method", model),
                    "error_type": "INVALID" if pred is None else ("FP" if r["label"] == 0 else "FN"),
                    "candidate_error_tags": candidate_tags(r["sentence1"], r["sentence2"]),
                    "latency_ms": r.get("latency_ms", ""),
                })
        write_csv(errors_dir / f"{model}_errors.csv", wrong)
        summary_rows.append({"model": model, "n_samples": len(rows),
                             "errors": len(wrong), "error_rate": len(wrong)/len(rows) if rows else 0})
    write_csv(out / f"error_summary_{split_name}.csv", summary_rows)

    cross_rows = []
    for g in gold_rows:
        row = {"id": g["id"], "sentence1": g["sentence1"], "sentence2": g["sentence2"],
               "label": g["label"]}
        flags = []
        preds = []
        for model in model_names:
            r = by_model[model][g["id"]]
            pred = r.get("prediction")
            correct = pred is not None and int(pred) == int(g["label"])
            row[f"{model}_prediction"] = "" if pred is None else pred
            row[f"{model}_correct"] = int(correct)
            flags.append(correct)
            preds.append(pred)
        if all(flags):
            pattern = "all_correct"
        elif not any(flags):
            pattern = "all_wrong"
        elif len(set(preds)) > 1:
            pattern = "models_disagree"
        else:
            pattern = "partial_correct_same_prediction_or_invalid"
        row["error_pattern"] = pattern
        if not all(flags):
            row["candidate_error_tags"] = candidate_tags(g["sentence1"], g["sentence2"])
            cross_rows.append(row)
    fields = ["id", "sentence1", "sentence2", "label"]
    for model in model_names:
        fields.extend([f"{model}_prediction", f"{model}_correct"])
    fields += ["error_pattern", "candidate_error_tags"]
    write_csv(out / f"cross_model_errors_{split_name}.csv", cross_rows, fields)
    return {"models": model_names, "error_rows": len(cross_rows)}
