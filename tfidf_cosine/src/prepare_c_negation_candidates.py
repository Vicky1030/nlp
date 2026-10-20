"""从冻结 TF-IDF 最终评估 FP 中导出 C 类否定语义人工复核候选。"""
import argparse, hashlib, json
from pathlib import Path

EXCLUDED = {35570,38638,38010,37181,37220,36720,35714,37275,37914,34877,37771,38141,38041,37619,35533,36632}

def read(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--evaluation',type=Path,required=True); ap.add_argument('--predictions',type=Path,required=True); ap.add_argument('--similarity',type=Path,required=True); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--curated',type=Path,required=True,help='External authorized JSON: ID -> [subgroup, sentence1 span, sentence2 span, review note]')
    a=ap.parse_args()
    CURATED = {int(k): v for k, v in json.loads(a.curated.read_text(encoding='utf-8')).items()}
    if not CURATED or any(len(v) != 4 or any(not isinstance(s, str) for s in v) for v in CURATED.values()):
        raise SystemExit('curated JSON must map IDs to four non-empty string fields')
    out=a.output_dir.resolve()
    if out.exists() and any(out.iterdir()): raise SystemExit(f'non-empty output directory: {out}')
    e,p,s=map(read,(a.evaluation,a.predictions,a.similarity)); cfg=json.loads(a.config.read_text(encoding='utf-8'))
    if cfg.get('representation')!='char_1_2' or float(cfg.get('threshold'))!=.28 or cfg.get('operator')!='>=': raise SystemExit('frozen config mismatch')
    rows=[]; seen=set(); fp=0
    for i,(x,y,z) in enumerate(zip(e,p,s),1):
        if x['id'] in seen: raise SystemExit(f'duplicate ID {x["id"]}')
        seen.add(x['id']); score=float(z['similarity'])
        if any(x[k]!=y[k] for k in ('id','sentence1','sentence2','label')) or x['id']!=z['id']: raise SystemExit(f'alignment failure line {i}')
        if y['prediction'] != int(score>=.28): raise SystemExit(f'threshold mismatch ID {x["id"]}')
        if x['label']==0 and y['prediction']==1:
            fp+=1
            if x['id'] in CURATED and x['id'] not in EXCLUDED:
                sub,sp1,sp2,why=CURATED[x['id']]
                if sp1 not in x['sentence1'] or sp2 not in x['sentence2']: raise SystemExit(f'span mismatch {x["id"]}')
                rows.append({'id':x['id'],'sentence1':x['sentence1'],'sentence2':x['sentence2'],'label':x['label'],'prediction':y['prediction'],'similarity':score,'threshold':.28,'margin':score-.28,'subgroup':sub,'retrieval_rule':'FP；否定/失败/未完成结构与对应肯定或条件表达形成不对称线索；已排除问句模板的机械匹配。','negation_spans':{'sentence1':sp1,'sentence2':sp2},'negation_scope_pending':why,'other_semantic_differences':'需人工核对是否存在对象、场景或业务阶段差异。','ambiguity_note':'候选筛选依据，不是人工确认；如两句只是不同问法，应降级或排除。','review_status':'pending_human_review'})
    if len(rows)!=len({r['id'] for r in rows}): raise SystemExit('duplicate candidate IDs')
    out.mkdir(parents=True,exist_ok=True)
    (out/'c_negation_candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows),encoding='utf-8',newline='\n')
    lines=['# C类否定语义人工复核候选','','全部条目来自冻结配置 `char_1_2`、阈值 `0.28` 下的真实 FP。候选规则、否定片段和说明仅供筛选，`review_status` 统一为 `pending_human_review`。','']
    for subgroup in ('C1','C2'):
        lines += [f'## {subgroup}', '']
        for i,r in enumerate([x for x in rows if x['subgroup']==subgroup],1):
            lines += [f'### 候选 {i}：ID {r["id"]}',f'- sentence1：{r["sentence1"]}',f'- sentence2：{r["sentence2"]}',f'- label={r["label"]}，prediction={r["prediction"]}，similarity={r["similarity"]:.15g}，threshold=0.28，margin={r["margin"]:.15g}',f'- 原文否定/对照片段：sentence1={r["negation_spans"]["sentence1"]}；sentence2={r["negation_spans"]["sentence2"]}',f'- 检索规则：{r["retrieval_rule"]}',f'- 否定作用待核查：{r["negation_scope_pending"]}',f'- 其他语义差异：{r["other_semantic_differences"]}',f'- 歧义提示：{r["ambiguity_note"]}','- 是否确属C类：待人工填写','- 是否选入最终案例：待人工填写','']
    (out/'c_negation_review.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    summary={'status':'candidate_screening_only','inputs':{str(p):sha(p) for p in (a.evaluation,a.predictions,a.similarity,a.config)},'frozen_config':{'representation':'char_1_2','threshold':.28,'operator':'>='},'n_samples':len(e),'FP':fp,'new_candidate_count':len(rows),'subgroup_counts':{k:sum(r['subgroup']==k for r in rows) for k in ('C1','C2')},'candidate_ids':[r['id'] for r in rows],'excluded_ids':sorted(EXCLUDED),'coverage':['all 984 FP traversed','asymmetric negation and failure/status terms','interrogative templates excluded from primary retrieval; condition questions kept only as lower-priority candidates'],'notes':['未声称数据中绝对不存在其他候选。','候选不足以自动构成最终归因，需人工核对。'],'command':'python outputs/src/prepare_c_negation_candidates.py --evaluation ... --predictions ... --similarity ... --config ... --output-dir ...'}
    (out/'c_negation_screening_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'FP':fp,'new_candidates':len(rows),'subgroups':summary['subgroup_counts'],'output':str(out)},ensure_ascii=False))
if __name__=='__main__': main()
