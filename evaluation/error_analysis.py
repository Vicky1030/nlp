"""Generate per-model errors and cross-model error/disagreement tables."""
import csv
import json
import re
from datetime import datetime, timezone
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


THREE_METHOD_CASES = (
    (
        "case1_all_correct",
        "Case1：A、B、C 全部正确",
        "简单样例：传统基线、提示方法和 LoRA 微调模型均能正确判断。",
        lambda a, b, c: a and b and c,
    ),
    (
        "case2_a_wrong_bc_correct",
        "Case2：A 错误，B、C 正确",
        "传统 TF-IDF 基线失效，而 Few-shot 提示与 LoRA 微调均正确。",
        lambda a, b, c: not a and b and c,
    ),
    (
        "case3_ab_wrong_c_correct",
        "Case3：A、B 错误，仅 C 正确",
        "传统基线与 Few-shot 提示均失败，仅 LoRA 微调模型正确，用于展示微调增益。",
        lambda a, b, c: not a and not b and c,
    ),
)


def analyze_three_method_cases(aligned_models, output_dir):
    """Select the three course-required patterns from this run's predictions."""
    roles = {
        "A": "tfidf_lr",
        "B": "qwen_few_shot",
        "C": "qwen_lora",
    }
    missing = [model for model in roles.values() if model not in aligned_models]
    output_path = Path(output_dir) / "hard_case_output.json"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "unavailable" if missing else "complete",
        "method_roles": roles,
        "reproducibility_note": (
            "由于 NF4 量化推理存在微小数值噪声，复现运行时满足各模式的具体样本 ID "
            "可能变化；本文件始终由 error_analysis.py 根据本次预测自动筛选。"
        ),
        "pattern_definitions": [
            {"case_type": key, "title": title, "explanation": explanation}
            for key, title, explanation, _ in THREE_METHOD_CASES
        ],
        "cases": [],
        "missing_models": missing,
    }
    if not missing:
        indexed = {
            role: {row["id"]: row for row in aligned_models[model]}
            for role, model in roles.items()
        }
        gold_rows = aligned_models[roles["A"]]
        matches = {case_type: [] for case_type, _, _, _ in THREE_METHOD_CASES}
        for gold in gold_rows:
            sample_id = gold["id"]
            role_rows = {role: indexed[role][sample_id] for role in roles}
            correct = {
                role: row.get("prediction") is not None
                and int(row["prediction"]) == int(gold["label"])
                for role, row in role_rows.items()
            }
            for case_type, _, _, predicate in THREE_METHOD_CASES:
                if predicate(correct["A"], correct["B"], correct["C"]):
                    matches[case_type].append((gold, role_rows, correct))

        for case_type, title, explanation, _ in THREE_METHOD_CASES:
            candidates = matches[case_type]
            selected = None
            if candidates:
                gold, role_rows, correct = candidates[0]
                selected = {
                    "id": gold["id"],
                    "sentence1": gold["sentence1"],
                    "sentence2": gold["sentence2"],
                    "label": int(gold["label"]),
                    "predictions": {
                        role: {
                            "model": roles[role],
                            "prediction": role_rows[role].get("prediction"),
                            "correct": correct[role],
                            "latency_ms": role_rows[role].get("latency_ms"),
                        }
                        for role in roles
                    },
                    "candidate_error_tags": candidate_tags(
                        gold["sentence1"], gold["sentence2"]
                    ),
                }
            payload["cases"].append({
                "case_type": case_type,
                "title": title,
                "explanation": explanation,
                "match_count": len(candidates),
                "pattern_satisfied": selected is not None,
                "selected_case": selected,
            })
        payload["status"] = (
            "complete" if all(case["pattern_satisfied"] for case in payload["cases"])
            else "partial"
        )
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"path": str(output_path), **payload}
