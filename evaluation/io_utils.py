"""Strict JSONL IO and prediction-vs-gold alignment checks."""
import json
from pathlib import Path


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: each JSONL row must be an object")
            rows.append(row)
    return rows


def validate_gold(gold, path="<gold>"):
    if not gold:
        raise ValueError(f"{path}: empty gold dataset")
    seen = set()
    for i, row in enumerate(gold, 1):
        for field in ("id", "sentence1", "sentence2", "label"):
            if field not in row:
                raise ValueError(f"{path}: row {i} missing {field}")
        rid = row["id"]
        if isinstance(rid, bool) or not isinstance(rid, int):
            raise ValueError(f"{path}: row {i} id must be an integer")
        if rid in seen:
            raise ValueError(f"{path}: duplicate id={rid}")
        seen.add(rid)
        if row["label"] not in (0, 1) or isinstance(row["label"], bool):
            raise ValueError(f"{path}: id={rid} label must be integer 0/1")
        for field in ("sentence1", "sentence2"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"{path}: id={rid} {field} must be a non-empty string")
    return gold


def align_predictions(predictions, gold, path="<predictions>"):
    if not predictions:
        raise ValueError(f"{path}: empty predictions")
    gold = validate_gold(gold)
    gold_by_id = {r["id"]: r for r in gold}
    pred_by_id = {}
    for i, row in enumerate(predictions, 1):
        for field in ("id", "label", "prediction"):
            if field not in row:
                raise ValueError(f"{path}: row {i} missing {field}")
        rid = row["id"]
        if isinstance(rid, bool) or not isinstance(rid, int):
            raise ValueError(f"{path}: row {i} id must be integer")
        if rid in pred_by_id:
            raise ValueError(f"{path}: duplicate id={rid}")
        if row["label"] not in (0, 1) or isinstance(row["label"], bool):
            raise ValueError(f"{path}: id={rid} label must be integer 0/1")
        pred = row["prediction"]
        if pred is not None and (pred not in (0, 1) or isinstance(pred, bool)):
            raise ValueError(f"{path}: id={rid} prediction must be 0, 1, or null")
        pred_by_id[rid] = row

    missing = sorted(set(gold_by_id) - set(pred_by_id))
    extra = sorted(set(pred_by_id) - set(gold_by_id))
    if missing or extra:
        raise ValueError(
            f"{path}: IDs mismatch: missing={len(missing)} {missing[:5]}, "
            f"extra={len(extra)} {extra[:5]}"
        )

    aligned = []
    for g in gold:
        p = pred_by_id[g["id"]]
        if int(p["label"]) != int(g["label"]):
            raise ValueError(f"{path}: id={g['id']} label differs from gold")
        for field in ("sentence1", "sentence2"):
            if field in p and p[field] != g[field]:
                raise ValueError(f"{path}: id={g['id']} {field} differs from gold")
        # preserve gold ordering and canonical gold text/label
        aligned.append({**g, **p, "id": g["id"], "label": g["label"],
                        "sentence1": g["sentence1"], "sentence2": g["sentence2"]})
    return aligned
