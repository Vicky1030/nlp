"""筛选 TF-IDF 最终评估错误中的 A/C 类人工复核候选。"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation", type=Path, required=True)
    ap.add_argument("--span-map", type=Path, required=True,
                    help="External authorized JSON: ID -> [sentence1 span, sentence2 span]")
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--similarity", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {out}")
    evaluation, predictions, similarities = map(read_jsonl, (args.evaluation, args.predictions, args.similarity))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (config.get("representation"), float(config.get("threshold")), config.get("operator")) != ("char_1_2", 0.28, ">="):
        raise SystemExit("Frozen configuration mismatch")
    if not (len(evaluation) == len(predictions) == len(similarities)):
        raise SystemExit("Input row counts differ")
    by_id = {}
    all_errors = []
    labels = []
    for i, (src, pred, sim) in enumerate(zip(evaluation, predictions, similarities), 1):
        if src["id"] in by_id:
            raise SystemExit(f"duplicate ID at line {i}: {src['id']}")
        if any(src[k] != pred[k] for k in ("id", "sentence1", "sentence2", "label")) or src["id"] != sim["id"]:
            raise SystemExit(f"row alignment failure at line {i}")
        score = float(sim["similarity"])
        expected = int(score >= 0.28)
        if pred["prediction"] != expected:
            raise SystemExit(f"threshold mismatch at ID {src['id']}")
        by_id[src["id"]] = (src, pred, score)
        labels.append(src["label"])
        if src["label"] != pred["prediction"]:
            all_errors.append({"id": src["id"], "sentence1": src["sentence1"], "sentence2": src["sentence2"],
                               "label": src["label"], "prediction": pred["prediction"], "similarity": score,
                               "threshold": 0.28, "error_type": "FP" if pred["prediction"] == 1 else "FN"})
    # Curated after inspecting the complete FP pool. These are retrieval candidates, not confirmed causes.
    groups = {
        "A": [36785, 37117, 38524, 34685, 37087, 34436],
        "C1": [35570, 38010, 38638, 37181, 37220],
        "C2": [36720, 35714, 37275],
        "C3": [37914, 34877, 37771, 38141, 38041],
    }
    old = {34937, 38237, 36449, 36735, 37568, 37619, 35533, 36632, 38141, 38041}
    span_map = {int(k): v for k, v in json.loads(args.span_map.read_text(encoding="utf-8")).items()}
    if not span_map or any(len(v) != 2 or any(not isinstance(s, str) for s in v) for v in span_map.values()):
        raise SystemExit("span-map JSON must map IDs to two string fields")
    records = []
    for group, ids in groups.items():
        for ident in ids:
            if ident in old or ident not in by_id:
                continue
            src, pred, score = by_id[ident]
            if src["label"] != 0 or pred["prediction"] != 1:
                continue
            s1, s2 = src["sentence1"], src["sentence2"]
            if group == "A":
                category, subtype = "A. 业务意图混淆", "动作/对象/业务目标不同"
                trigger = "两句共享同一业务词面，但请求动作、对象或目标存在可直接引用的差异；仅作检索依据。"
                questions = "两句的动作—对象—目标分别是什么？差异是否足以改变用户实际需求，而非仅为详略或语气差异？"
                spans = {"sentence1": span_map.get(ident, (s1, s2))[0], "sentence2": span_map.get(ident, (s1, s2))[1]}
            elif group == "C1":
                category, subtype = "C. 否定语义识别不足", "C1 明确状态/事实否定"
                trigger = "同一事项出现对应的肯定/否定状态或明确失败状态；否定结构及其作用范围需人工核对。"
                questions = "两句是否谈同一对象和事项？否定词/结构具体作用于哪个动作或状态？除否定外是否还有重要差异？"
                spans = {"sentence1": span_map.get(ident, (s1, s2))[0], "sentence2": span_map.get(ident, (s1, s2))[1]}
            elif group == "C2":
                category, subtype = "C. 否定语义识别不足", "C2 请求/条件否定"
                trigger = "同一事项下，肯定请求与否定请求/条件并列出现；不是仅凭否定字计类。"
                questions = "否定是否改变了请求目标或前提条件？两句是否仍围绕同一对象和操作？"
                spans = {"sentence1": span_map.get(ident, (s1, s2))[0], "sentence2": span_map.get(ident, (s1, s2))[1]}
            else:
                category, subtype = "C. 否定语义识别不足", "C3 问句意图差异备选"
                trigger = "同一事项的可行性/恢复或不可用状态问句形成对照；列为问句意图备选，不等同于明确事实否定。"
                questions = "这是支持性询问还是实际失败后的原因/恢复询问？是否存在除问句形式之外的状态差异？"
                spans = {"sentence1": span_map.get(ident, (s1, s2))[0], "sentence2": span_map.get(ident, (s1, s2))[1]}
            records.append({"id": ident, "sentence1": s1, "sentence2": s2, "label": 0, "prediction": 1,
                            "error_type": "FP", "similarity": score, "threshold": 0.28,
                            "margin": score - 0.28, "suggested_category": category, "subtype": subtype,
                            "key_difference_spans": spans, "retrieval_trigger": trigger,
                            "human_questions": questions,
                            "ambiguity_note": "自动筛选候选；可能存在其他业务或语义差异，需人工阅读原句确认。",
                            "overlap_note": "未与本轮排除的旧候选重复；如与其他新候选语义重复，最终案例只计一个ID。",
                            "review_status": "pending_human_review"})
    # Limit A to 8 and C total to 12, preserving explicit subgroup order.
    selected = [r for r in records if r["suggested_category"].startswith("A.")][:8]
    selected += [r for r in records if r["suggested_category"].startswith("C.")][:12]
    selected_ids = [r["id"] for r in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise SystemExit("duplicate candidate ID across new groups")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "additional_candidates.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    lines = ["# TF-IDF 最终评估新增候选人工复核表", "", "所有条目均为真实 FP，且只使用冻结配置的最终评估结果。建议类别、片段和问题是筛选材料，不是人工确认。review_status 统一为 `pending_human_review`。", "", "## A：业务意图混淆（最多8条）", ""]
    for title, rows in (("A", [r for r in selected if r["suggested_category"].startswith("A.")]),
                        ("C1", [r for r in selected if r["subtype"].startswith("C1")]),
                        ("C2", [r for r in selected if r["subtype"].startswith("C2")]),
                        ("C3", [r for r in selected if r["subtype"].startswith("C3")])):
        if title != "A": lines += [f"## {title}：否定语义识别不足", ""]
        for idx, r in enumerate(rows, 1):
            lines += [f"### {title} 候选 {idx}：ID {r['id']}", f"- sentence1：{r['sentence1']}", f"- sentence2：{r['sentence2']}",
                      f"- label={r['label']}，prediction={r['prediction']}，error_type={r['error_type']}，similarity={r['similarity']:.15g}，threshold=0.28，margin={r['margin']:.15g}",
                      f"- 建议类别：{r['suggested_category']}；子组：{r['subtype']}", f"- 原文关键片段：sentence1={r['key_difference_spans']['sentence1']}；sentence2={r['key_difference_spans']['sentence2']}",
                      f"- 检索命中依据：{r['retrieval_trigger']}", f"- 人工需回答：{r['human_questions']}", f"- 歧义提示：{r['ambiguity_note']}",
                      "- 语义关系核查：待人工填写", "- 是否接受该类别：待人工填写", "- 原因分析/方法局限：待人工填写", "- 是否选入最终案例：待人工填写", ""]
    (out / "additional_case_review.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    fp = sum(r["error_type"] == "FP" for r in all_errors); fn = sum(r["error_type"] == "FN" for r in all_errors)
    cats = Counter(r["subtype"] for r in selected)
    summary = {"status": "candidate_screening_only", "inputs": {str(p): sha256(p) for p in (args.evaluation, args.predictions, args.similarity, args.config)},
               "frozen_config": {"representation": "char_1_2", "threshold": 0.28, "operator": ">="},
               "n_samples": len(evaluation), "errors": len(all_errors), "FP": fp, "FN": fn,
               "candidate_counts": {"A": sum(r["suggested_category"].startswith("A.") for r in selected), "C1": cats.get("C1 明确状态/事实否定", 0), "C2": cats.get("C2 请求/条件否定", 0), "C3": cats.get("C3 问句意图差异备选", 0), "C_total": sum(r["suggested_category"].startswith("C.") for r in selected)},
               "candidate_ids": selected_ids, "old_excluded_ids": sorted(old),
               "screening_notes": ["旧规则将通用否定词和问句线索混在一起；本次按A、C1、C2、C3分组并保留原句。", "候选数量不等于人工确认数量，重叠语义由人工决定。"],
               "command": "python outputs/src/prepare_tfidf_additional_candidates.py --evaluation ... --predictions ... --similarity ... --config ... --output-dir ..."}
    (out / "screening_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "n_errors": len(all_errors), "FP": fp, "FN": fn, "candidate_counts": summary["candidate_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
