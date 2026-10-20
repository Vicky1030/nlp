"""单模型评价命令行程序。"""
import argparse
import json
import sys
from pathlib import Path

try:
    from .config import GOLD_FILE, OUTPUT_DIR, POSITIVE_LABEL, REQUIRED_PREDICTION_FIELDS
    from .metrics import calculate_metrics, latency_summary
except ImportError:  # 允许从 evaluation 目录直接运行
    from config import GOLD_FILE, OUTPUT_DIR, POSITIVE_LABEL, REQUIRED_PREDICTION_FIELDS
    from metrics import calculate_metrics, latency_summary


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} 第 {line_no} 行必须是 JSON 对象")
            rows.append(row)
    return rows


def validate_predictions(predictions, gold=None):
    seen = set()
    for i, row in enumerate(predictions, 1):
        missing = [k for k in REQUIRED_PREDICTION_FIELDS if k not in row]
        if missing:
            raise ValueError(f"预测文件第 {i} 条缺少字段：{missing}")
        rid = str(row["id"])
        if rid in seen:
            raise ValueError(f"预测文件存在重复 id：{rid}")
        seen.add(rid)
        if row["label"] not in (0, 1) or row["prediction"] not in (0, 1):
            raise ValueError(f"id={rid} 的 label/prediction 必须为整数 0 或 1")
    if gold is not None:
        gold_by_id = {str(r["id"]): r for r in gold}
        pred_by_id = {str(r["id"]): r for r in predictions}
        missing = set(gold_by_id) - set(pred_by_id)
        extra = set(pred_by_id) - set(gold_by_id)
        if missing or extra:
            raise ValueError(
                f"预测与标准集 ID 不一致：缺少 {len(missing)} 条，额外 {len(extra)} 条。"
                f"缺少示例={list(sorted(missing))[:5]}，额外示例={list(sorted(extra))[:5]}"
            )
        for rid, g in gold_by_id.items():
            p = pred_by_id[rid]
            if int(p["label"]) != int(g["label"]):
                raise ValueError(f"id={rid} 的预测文件 label 与标准 evaluation.jsonl 不一致")
            for field in ("sentence1", "sentence2"):
                if field in p and field in g and p[field] != g[field]:
                    raise ValueError(f"id={rid} 的 {field} 与标准数据不一致")
        # 以 gold 顺序统一计算，避免各模型行顺序不同导致横向比较错位
        predictions = [pred_by_id[str(g["id"])] for g in gold]
    return predictions


def evaluate_file(prediction_path, gold_path=None, output_path=None):
    predictions = load_jsonl(prediction_path)
    gold = load_jsonl(gold_path) if gold_path else None
    predictions = validate_predictions(predictions, gold)
    y_true = [int(r["label"]) for r in predictions]
    y_pred = [int(r["prediction"]) for r in predictions]
    result = calculate_metrics(y_true, y_pred, POSITIVE_LABEL)
    result.update(latency_summary(predictions))
    result["model"] = predictions[0].get("method", Path(prediction_path).stem) if predictions else Path(prediction_path).stem
    result["prediction_file"] = str(prediction_path)
    result["gold_file"] = str(gold_path) if gold_path else None
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="评价一个 AFQMC 模型的 JSONL 预测结果")
    parser.add_argument("--predictions", required=True, help="模型预测 JSONL 路径")
    parser.add_argument("--gold", default=str(GOLD_FILE), help="标准 evaluation.jsonl 路径")
    parser.add_argument("--output", default=None, help="评价结果 JSON 输出路径")
    args = parser.parse_args()
    try:
        result = evaluate_file(args.predictions, args.gold, args.output)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
