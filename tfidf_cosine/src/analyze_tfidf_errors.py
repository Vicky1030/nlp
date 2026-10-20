"""Analyze fixed-configuration TF-IDF cosine errors without changing predictions."""
import argparse
import json
import math
import re
from pathlib import Path

from tfidf_common import jieba_tokens


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def token_jaccard(a, b):
    left, right = set(jieba_tokens(a)), set(jieba_tokens(b))
    return len(left & right) / len(left | right) if left | right else 1.0


def candidate_categories(row):
    categories = []
    overlap = token_jaccard(row['sentence1'], row['sentence2'])
    if row['label'] == 0 and row['prediction'] == 1 and overlap >= 0.5:
        categories.append(('A', f'词级集合 Jaccard={overlap:.3f}，候选：用词重合较多但标签为非相似'))
    if row['label'] == 1 and row['prediction'] == 0 and overlap <= 0.25:
        categories.append(('B', f'词级集合 Jaccard={overlap:.3f}，候选：词面差异较大但标签为相似'))
    neg = ('不', '没', '无', '未', '不是', '无法', '没有', '并非', '别', '勿')
    neg_left = any(x in row['sentence1'] for x in neg); neg_right = any(x in row['sentence2'] for x in neg)
    numbers_left = set(re.findall(r'\d+(?:\.\d+)?%?', row['sentence1']))
    numbers_right = set(re.findall(r'\d+(?:\.\d+)?%?', row['sentence2']))
    if row['label'] != row['prediction'] and (neg_left != neg_right or numbers_left != numbers_right):
        details = []
        if neg_left != neg_right: details.append('否定词线索不同')
        if numbers_left != numbers_right: details.append('数字/比例线索不同')
        categories.append(('C', '候选：' + '、'.join(details) + '，待人工复核'))
    return categories


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--similarity', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(); out = args.output_dir.resolve()
    targets = [out/'error_samples.jsonl', out/'error_analysis.json', out/'error_analysis.md']
    clash = [str(x) for x in targets if x.exists()]
    if clash: raise SystemExit('Refusing to overwrite existing output files; choose a new --output-dir:\n'+'\n'.join(clash))
    predictions = read_jsonl(args.predictions); scores = read_jsonl(args.similarity); config = json.loads(args.config.read_text(encoding='utf-8'))
    if len(predictions) != len(scores): raise ValueError('prediction/similarity count mismatch')
    errors=[]; categories={'A':[],'B':[],'C':[]}
    threshold=float(config['threshold'])
    for row, score in zip(predictions, scores):
        if row['id'] != score['id'] or row['label'] != score['label']: raise ValueError('prediction/similarity alignment mismatch')
        if row['prediction'] not in (0,1) or not math.isfinite(float(score['similarity'])): raise ValueError('invalid prediction or similarity')
        if row['prediction'] != row['label']:
            item={k:row[k] for k in ('id','sentence1','sentence2','label','prediction')}
            item.update(similarity=float(score['similarity']), threshold=threshold, candidates=[])
            for code, reason in candidate_categories(row):
                item['candidates'].append({'category':code,'reason':reason}); categories[code].append(item)
            errors.append(item)
    fp=sum(x['label']==0 and x['prediction']==1 for x in errors); fn=sum(x['label']==1 and x['prediction']==0 for x in errors)
    negatives=sum(x['label']==0 for x in predictions); positives=sum(x['label']==1 for x in predictions)
    for row in errors:
        row['candidates'] = list(row['candidates'])
    out.mkdir(parents=True, exist_ok=True)
    with (out/'error_samples.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for row in errors: stream.write(json.dumps(row, ensure_ascii=False, separators=(',',':'))+'\n')
    summary={'status':'passed','n_samples':len(predictions),'errors':len(errors),'false_positive':fp,'false_negative':fn,
             'false_positive_rate_denominator':'all actual label=0 samples','false_positive_rate':fp/negatives if negatives else 0,
             'false_negative_rate_denominator':'all actual label=1 samples','false_negative_rate':fn/positives if positives else 0,
             'error_rate':len(errors)/len(predictions) if predictions else 0,'threshold':threshold,
             'candidate_counts':{k:len(v) for k,v in categories.items()},'candidate_overlap_note':'categories are heuristic candidates for manual review and may overlap',
             'examples':{k:v[:3] for k,v in categories.items()}}
    (out/'error_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    md=['# TF-IDF 错误分析','',f'固定配置：`{config["representation"]}`，阈值 `{threshold}`，规则 `similarity >= threshold`。本分析只解释已生成预测，不重新调参。','',f'- 总样本：{len(predictions)}；错误：{len(errors)}；错误率（分母为全部评估样本）：{summary["error_rate"]:.6f}',f'- 假阳性 FP：{fp}；比例分母为实际 label=0 样本（{negatives}）：{summary["false_positive_rate"]:.6f}',f'- 假阴性 FN：{fn}；比例分母为实际 label=1 样本（{positives}）：{summary["false_negative_rate"]:.6f}','']
    labels={'A':'A. 用词重合较多但表达可能不同','B':'B. 表达方式不同但意思可能相同','C':'C. 否定、数字或比例等线索可能不同'}
    for code,title in labels.items():
        md += [f'## {title}', '', '以下为规则筛选的候选，均需人工复核；类别可重叠。']
        ex=categories[code][:3]
        if not ex: md.append('实际错误中没有筛出足够候选样本。')
        for i,row in enumerate(ex,1):
            md += [f'### 候选 {i}（id={row["id"]}）',f'- sentence1：{row["sentence1"]}',f'- sentence2：{row["sentence2"]}',f'- label={row["label"]}，prediction={row["prediction"]}，similarity={row["similarity"]:.15g}，threshold={row["threshold"]}',f'- 依据：{next(x["reason"] for x in row["candidates"] if x["category"]==code)}','']
    md += ['## 边界','', 'A/B/C 是候选分析方向，不是自动确认的错误原因。标签疑点或语义原因不确定的样本仍需人工复核；本程序不修改标签、句子或预测。']
    (out/'error_analysis.md').write_text('\n'.join(md)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
