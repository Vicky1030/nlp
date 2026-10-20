"""Prepare human-review candidates from frozen TF-IDF evaluation errors."""
import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from tfidf_common import jieba_tokens

FIELDS = ("id", "sentence1", "sentence2", "label", "prediction", "similarity", "threshold", "error_type")
OPERATIONS = ("查询", "查看", "找不到", "找回", "申请", "开通", "关闭", "还款", "退款", "充值", "转账", "支付", "绑定", "解绑", "修改", "更改", "取消", "提额", "提现", "收款", "借款", "认证", "验证", "领取", "使用", "冻结", "解冻", "账单", "明细", "额度", "返现")
NEGATIONS = ("没有", "无法", "未曾", "未", "不是", "并非", "不能", "没", "不", "无")
GENERIC_SHARED = {"为什么", "怎么", "如何", "可以", "是否", "有没有", "能不能", "花呗", "借呗", "蚂蚁", "支付宝", "现在", "还有", "使用", "多少", "要求", "我的", "这个", "用", "过", "了", "有"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open("rb") as f:
        for line_no, raw in enumerate(f, 1):
            if line_no == 1 and raw.startswith(b"\xef\xbb\xbf"):
                raise ValueError(f"{path}:{line_no}: UTF-8 BOM is not allowed")
            try:
                rows.append(json.loads(raw.decode("utf-8")))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token_jaccard(left, right):
    a, b = set(jieba_tokens(left)), set(jieba_tokens(right))
    return len(a & b) / len(a | b) if a | b else 1.0


def snippets(text, terms, width=8):
    found = []
    for term in sorted(set(terms), key=len, reverse=True):
        for match in re.finditer(re.escape(term), text):
            start = max(0, match.start() - width)
            end = min(len(text), match.end() + width)
            found.append({"term": term, "text": text[start:end], "start": match.start(), "end": match.end()})
    return sorted(found, key=lambda x: (x["start"], -len(x["term"])))


def operation_terms(text):
    return [term for term in OPERATIONS if term in text]


def negation_terms(text):
    result = []
    for term in NEGATIONS:
        for match in re.finditer(re.escape(term), text):
            left = text[max(0, match.start() - 2):match.start()]
            right = text[match.end():match.end() + 2]
            context = text[max(0, match.start() - 2):min(len(text), match.end() + 2)]
            # Do not turn the interrogative forms “能不能” and “有没有” into
            # a negation candidate merely because they contain 不/没/有.
            if ("能不能" in context or "有没有" in context or "是不是" in context or
                    "是否" in context or (term == "没" and "有没" in context)):
                continue
            result.append(term)
    return sorted(set(result), key=len, reverse=True)


def base_record(row, score, threshold):
    error_type = "FP" if row["label"] == 0 and row["prediction"] == 1 else "FN"
    return {"id": row["id"], "sentence1": row["sentence1"], "sentence2": row["sentence2"],
            "label": row["label"], "prediction": row["prediction"], "similarity": float(score),
            "threshold": threshold, "error_type": error_type}


def make_candidates(error):
    left, right = error["sentence1"], error["sentence2"]
    overlap = token_jaccard(left, right)
    ops_left, ops_right = operation_terms(left), operation_terms(right)
    neg_left, neg_right = negation_terms(left), negation_terms(right)
    candidates = []
    if error["error_type"] == "FP" and overlap >= 0.5 and ops_left and ops_right and set(ops_left) != set(ops_right):
        candidates.append({"category": "A. 操作意图不同", "quality": overlap,
                           "rule_trigger": f"FP；词级集合 Jaccard={overlap:.3f}>=0.5；两句操作词集合不同",
                           "evidence_spans": {"sentence1": snippets(left, ops_left), "sentence2": snippets(right, ops_right)},
                           "ambiguity_note": "操作词差异仅为客观筛选依据，是否属于不同操作意图待人工判断。"})
    shared = sorted((set(jieba_tokens(left)) & set(jieba_tokens(right))), key=lambda x: (-len(x), x))
    shared = [x for x in shared if len(x.strip()) >= 2 and not x.isspace() and x not in GENERIC_SHARED]
    if error["error_type"] == "FN" and overlap <= 0.25 and shared:
        candidates.append({"category": "B. 同义表达差异", "quality": -overlap,
                           "rule_trigger": f"FN；词级集合 Jaccard={overlap:.3f}<=0.25；仍有共享词面 {shared[:5]}",
                           "evidence_spans": {"sentence1": snippets(left, shared[:5]), "sentence2": snippets(right, shared[:5])},
                           "ambiguity_note": "低词面重合和共享词语不能证明同义，需人工核对两句需求是否一致。"})
    if error["error_type"] == "FP" and bool(neg_left) != bool(neg_right):
        terms = sorted(set(neg_left + neg_right), key=len, reverse=True)
        candidates.append({"category": "C. 否定语义差异", "quality": abs(error["similarity"] - error["threshold"]),
                           "rule_trigger": "FP；两句强否定词线索不对称；已排除能不能/有没有问句形式",
                           "evidence_spans": {"sentence1": snippets(left, neg_left), "sentence2": snippets(right, neg_right)},
                           "ambiguity_note": "否定词作用范围和整句语义待人工判断，未因出现单个否定字词直接定类。"})
    return candidates


def metric_counts(labels, predictions):
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    def div(a, b): return a / b if b else 0.0
    p1, r1 = div(tp, tp + fp), div(tp, tp + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    p0, r0 = div(tn, tn + fn), div(tn, tn + fp)
    f0 = div(2 * tn, 2 * tn + fn + fp)
    return {"n_samples": len(labels), "accuracy": div(tn + tp, len(labels)), "precision_1": p1,
            "recall_1": r1, "f1_1": f1, "macro_f1": (f0 + f1) / 2,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "predicted_0": predictions.count(0), "predicted_1": predictions.count(1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--similarity", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    targets = [out / name for name in ("all_errors.jsonl", "candidate_cases.jsonl", "case_review.md", "error_summary.json", "metrics_comparison.json")]
    conflicts = [str(p) for p in targets if p.exists()]
    if conflicts:
        raise SystemExit("Refusing to overwrite existing analysis files; choose a new --output-dir:\n" + "\n".join(conflicts))
    evaluation, predictions, similarities = map(read_jsonl, (args.evaluation, args.predictions, args.similarity))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    require(config.get("representation") == "char_1_2" and float(config.get("threshold")) == 0.28 and config.get("operator") == ">=", "冻结配置不是 char_1_2、0.28、>=")
    require(len(evaluation) == len(predictions) == len(similarities), "输入文件数量不一致")
    seen = set(); errors = []; candidates = defaultdict(list)
    threshold = float(config["threshold"])
    for line_no, (source, pred, sim) in enumerate(zip(evaluation, predictions, similarities), 1):
        require(source["id"] == pred["id"] == sim["id"], f"line {line_no}: ID 不一致")
        for key in ("sentence1", "sentence2", "label"):
            require(source[key] == pred[key], f"line {line_no}: {key} 与 prediction 不一致")
        require(pred["id"] not in seen, f"line {line_no}: duplicate ID")
        seen.add(pred["id"])
        score = float(sim["similarity"])
        require(math.isfinite(score) and -1e-12 <= score <= 1 + 1e-12, f"line {line_no}: invalid similarity")
        expected = int(score >= threshold)
        require(pred["prediction"] == expected, f"line {line_no}: prediction violates similarity >= threshold")
        if pred["prediction"] != source["label"]:
            error = base_record(pred, score, threshold)
            errors.append(error)
            for candidate in make_candidates(error):
                candidates[candidate["category"]].append({**error, **candidate, "review_status": "pending_human_review"})
    labels = [r["label"] for r in evaluation]; predictions_only = [r["prediction"] for r in predictions]
    metrics = metric_counts(labels, predictions_only)
    majority = 0 if Counter(labels)[0] >= Counter(labels)[1] else 1
    baseline = metric_counts(labels, [majority] * len(labels))
    selected = {}
    for category, rows in candidates.items():
        rows.sort(key=lambda x: x["quality"], reverse=True)
        selected[category] = rows[:5]
    raw_overlap_ids = defaultdict(list)
    for category, rows in candidates.items():
        for row in rows:
            raw_overlap_ids[row["id"]].append(category)
    flat_candidates = [row for rows in selected.values() for row in rows]
    overlap_ids = defaultdict(list)
    for row in flat_candidates:
        overlap_ids[row["id"]].append(row["category"])
    for row in flat_candidates:
        row["overlapping_categories"] = sorted(set(overlap_ids[row["id"]]))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "all_errors.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in errors: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (out / "candidate_cases.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in flat_candidates: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    review = ["# TF-IDF 最终评估失败案例人工复核表", "", "所有条目均来自实际 `prediction != label` 的 evaluation 错误。建议类别、触发规则和片段只是候选证据，`review_status` 统一为 `pending_human_review`，不得视为已确认归因。", "", f"冻结配置：`{config['representation']}`，`similarity >= {threshold}`。", "", f"候选 ID 如重复命中多个类别，类别重叠不计为多个独立样本。"]
    for category in ("A. 操作意图不同", "B. 同义表达差异", "C. 否定语义差异"):
        review += ["", f"## {category}", ""]
        rows = selected.get(category, [])
        if not rows:
            review.append("未筛出足够候选；需要人工从全部错误中补充或说明不足。")
        for idx, row in enumerate(rows, 1):
            review += [f"### 候选 {idx}：ID {row['id']}", f"- sentence1：{row['sentence1']}", f"- sentence2：{row['sentence2']}", f"- label={row['label']}，prediction={row['prediction']}，{row['error_type']}，similarity={row['similarity']:.15g}，threshold={threshold}", f"- rule_trigger：{row['rule_trigger']}", f"- evidence_spans：{json.dumps(row['evidence_spans'], ensure_ascii=False)}", f"- 重叠类别：{', '.join(row['overlapping_categories'])}", "- 语义关系核查：待人工填写", "- 是否接受该类别：待人工填写", "- 原因分析：待人工填写", "- 方法局限：待人工填写", "- 是否选入最终 6 例：待人工填写", ""]
    review += ["## 使用边界", "", "A/B/C 候选数量不是已确认类别数量，也不是错误比例。数字差异未单列为 C；`能不能`、`有没有` 不作为自动否定依据。疑似标签问题、背景不足或语义不确定的样本应另行标记。"]
    (out / "case_review.md").write_text("\n".join(review) + "\n", encoding="utf-8", newline="\n")
    span_checks = []
    for row in flat_candidates:
        span_checks.extend(snippet["text"].replace(" ", "") in row["sentence1"].replace(" ", "") for snippet in row["evidence_spans"]["sentence1"])
        span_checks.extend(snippet["text"].replace(" ", "") in row["sentence2"].replace(" ", "") for snippet in row["evidence_spans"]["sentence2"])
    summary = {"status": "passed", "command": subprocess.list2cmdline([str(__import__('sys').executable), *map(str, __import__('sys').argv)]), "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in (("evaluation", args.evaluation), ("predictions", args.predictions), ("similarity", args.similarity), ("config", args.config))}, "configuration": {"representation": config["representation"], "threshold": threshold, "operator": config["operator"]}, "counts": {"total": len(evaluation), "correct": len(evaluation) - len(errors), "errors": len(errors), "FP": metrics["FP"], "FN": metrics["FN"]}, "metrics": metrics, "all_zero_baseline": baseline, "candidate_counts": {key: len(value) for key, value in selected.items()}, "candidate_unique_ids": {key: len({row["id"] for row in value}) for key, value in selected.items()}, "candidate_overlap": {str(key): value for key, value in overlap_ids.items() if len(set(value)) > 1}, "raw_rule_candidate_counts": {key: len(value) for key, value in candidates.items()}, "raw_rule_candidate_unique_ids": {key: len({row["id"] for row in value}) for key, value in candidates.items()}, "raw_rule_candidate_overlap": {str(key): value for key, value in raw_overlap_ids.items() if len(set(value)) > 1}, "candidate_limit": 5, "review_status": "pending_human_review", "checks": {"prediction_alignment": True, "similarity_alignment": True, "prediction_rule_similarity_ge_threshold": True, "all_errors_count_equals_FP_plus_FN": len(errors) == metrics["FP"] + metrics["FN"], "confusion_total_equals_n": sum(metrics[key] for key in ("TN", "FP", "FN", "TP")) == len(evaluation), "candidate_ids_are_errors": all(row["id"] in {item["id"] for item in errors} for row in flat_candidates), "evidence_spans_are_substrings": all(span_checks)}, "files": {"all_errors": str(out / "all_errors.jsonl"), "candidate_cases": str(out / "candidate_cases.jsonl"), "case_review": str(out / "case_review.md")}}
    (out / "error_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out / "metrics_comparison.json").write_text(json.dumps({"model": metrics, "all_zero_baseline": baseline, "confusion_matrix_order": "[[TN,FP],[FN,TP]]", "labels": [0, 1], "zero_division": 0}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "passed", "total": len(evaluation), "errors": len(errors), "FP": metrics["FP"], "FN": metrics["FN"], "candidate_counts": summary["candidate_counts"], "candidate_unique_ids": summary["candidate_unique_ids"], "overlap_ids": summary["candidate_overlap"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
