"""Extract sparse char 1-2 gram evidence for fixed frozen evaluation cases."""
import argparse, csv, hashlib, json, math, platform, sys
from collections import Counter
from pathlib import Path

CASE_GROUPS = {
    34937: "A. 业务意图混淆", 37087: "A. 业务意图混淆", 38524: "A. 业务意图混淆",
    37557: "B. 同义表达识别不足", 36035: "B. 同义表达识别不足", 38599: "B. 同义表达识别不足",
    35570: "C. 否定语义识别不足", 38638: "C. 否定语义识别不足", 37891: "C. 否定语义识别不足",
}
EXPECTED_TYPES = {34937: "FP", 37087: "FP", 38524: "FP", 37557: "FN", 36035: "FN", 38599: "FN", 35570: "FP", 38638: "FP", 37891: "FP"}
PHRASES = {
    34937: ["还款", "支付"], 37087: ["使用", "申请"], 38524: ["网商贷", "借呗", "关闭", "使用"],
    37557: ["人工客服", "人工服务"], 36035: ["逾期", "我还了", "不可以用了", "还款后", "不能使用"],
    38599: ["提额度", "提高花呗的额度"], 35570: ["不支持", "可以", "支付", "付款"],
    38638: ["还能用", "付不了款", "关闭"], 37891: ["还款了", "没有还", "再借"],
}

def read_jsonl(p): return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x): return math.isfinite(float(x))
def is_binary_int(x): return isinstance(x, int) and not isinstance(x, bool) and x in (0, 1)
def close(a, b, rtol=1e-12, atol=1e-12): return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b)) <= atol + rtol*abs(float(b))

def metric(labels, preds):
    tn=sum(y==0 and p==0 for y,p in zip(labels,preds)); fp=sum(y==0 and p==1 for y,p in zip(labels,preds)); fn=sum(y==1 and p==0 for y,p in zip(labels,preds)); tp=sum(y==1 and p==1 for y,p in zip(labels,preds))
    d=lambda a,b:a/b if b else 0.0
    p1=d(tp,tp+fp); r1=d(tp,tp+fn); f1=d(2*tp,2*tp+fp+fn); f0=d(2*tn,2*tn+fp+fn)
    return {"n_samples":len(labels),"accuracy":d(tn+tp,len(labels)),"precision_1":p1,"recall_1":r1,"f1_1":f1,"macro_f1":(f0+f1)/2,"confusion_matrix_0_1":[[tn,fp],[fn,tp]],"TN":tn,"FP":fp,"FN":fn,"TP":tp,"predicted_0":preds.count(0),"predicted_1":preds.count(1)}

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation', 'predictions', 'similarity', 'config', 'vectorizer', 'metadata'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def ensure_output_targets_absent(output_dir):
    targets = [
        output_dir / name for name in (
            'selected_cases.jsonl', 'case_summary.csv', 'feature_evidence.md',
            'case_review_with_evidence.md', 'metrics_comparison.json',
            'metrics_comparison.csv', 'verification.json', 'README.md',
        )
    ] + [output_dir / 'feature_details']
    conflicts = [str(path) for path in targets if path.exists()]
    if conflicts:
        raise SystemExit('Refusing overwrite; choose a new output directory:\n' + '\n'.join(conflicts))


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    ensure_output_targets_absent(out)
    protected_paths=(args.evaluation,args.predictions,args.similarity,args.config,args.vectorizer,args.metadata)
    protected_before={str(x.resolve()):sha(x) for x in protected_paths}
    e,p,s=map(read_jsonl,(args.evaluation,args.predictions,args.similarity)); cfg=json.loads(args.config.read_text(encoding='utf-8')); meta=json.loads(args.metadata.read_text(encoding='utf-8'))
    if cfg.get('representation')!='char_1_2' or float(cfg.get('threshold'))!=.28 or cfg.get('operator')!='>=': raise SystemExit('Frozen config mismatch')
    if len(e)!=len(p) or len(e)!=len(s): raise SystemExit('Input length mismatch')
    by={}; labels=[]; preds=[]
    for i,(x,y,z) in enumerate(zip(e,p,s),1):
        for obj, name in ((x,'evaluation'),(y,'prediction'),(z,'similarity')):
            if not isinstance(obj, dict): raise SystemExit(f'{name} line {i}: JSON object required')
        if not is_binary_int(x.get('label')) or not is_binary_int(y.get('label')) or not is_binary_int(y.get('prediction')): raise SystemExit(f'line {i}: label/prediction must be integer 0/1, not bool or other type')
        if not isinstance(x.get('id'), int) or isinstance(x.get('id'), bool): raise SystemExit(f'line {i}: ID must be integer')
        if x['id'] in by: raise SystemExit(f'duplicate ID {x["id"]}')
        if any(x[k]!=y[k] for k in ('id','sentence1','sentence2','label')) or x['id']!=z['id']: raise SystemExit(f'alignment failure line {i}')
        if 'label' in z:
            if not is_binary_int(z['label']): raise SystemExit(f'similarity label must be integer 0/1 at ID {x["id"]}')
            if z['label'] != x['label']: raise SystemExit(f'similarity label mismatch ID {x["id"]}')
        if not isinstance(z.get('id'), int) or isinstance(z.get('id'), bool): raise SystemExit(f'similarity ID must be integer at line {i}')
        try: score=float(z['similarity'])
        except (KeyError,TypeError,ValueError): raise SystemExit(f'invalid similarity at ID {x["id"]}')
        if not finite(score) or score < -1e-12 or score > 1+1e-12: raise SystemExit(f'similarity out of range at ID {x["id"]}')
        expected=int(score>=.28)
        if y['prediction']!=expected: raise SystemExit(f'prediction threshold mismatch {x["id"]}')
        by[x['id']]=(x,y,score); labels.append(x['label']); preds.append(y['prediction'])
    missing=set(CASE_GROUPS)-set(by)
    if missing: raise SystemExit(f'fixed case IDs missing: {sorted(missing)}')
    for ident, expected_type in EXPECTED_TYPES.items():
        src,pr,_=by[ident]; actual_type='FP' if src['label']==0 and pr['prediction']==1 else ('FN' if src['label']==1 and pr['prediction']==0 else 'CORRECT_OR_INVALID')
        if actual_type != expected_type: raise SystemExit(f'fixed case ID {ident}: expected {expected_type}, actual label={src["label"]}, prediction={pr["prediction"]}')
    import joblib
    vectorizer_sha=sha(args.vectorizer)
    expected_model_sha=cfg.get('vectorizer',{}).get('sha256')
    if expected_model_sha and expected_model_sha != vectorizer_sha: raise SystemExit('selected_config vectorizer SHA-256 mismatch')
    if meta.get('model_sha256') and meta['model_sha256'] != vectorizer_sha: raise SystemExit('metadata model_sha256 mismatch')
    vec=joblib.load(args.vectorizer); analyzer=vec.build_analyzer(); vocab=vec.vocabulary_; idf=vec.idf_; names=vec.get_feature_names_out(); threshold=.28
    if vec.analyzer!='char' or vec.ngram_range!=(1,2) or vec.norm!='l2' or str(vec.dtype) not in ('<class \'numpy.float64\'>', 'float64'): raise SystemExit('vectorizer parameters mismatch')
    mp=meta.get('parameters',{})
    if meta.get('scheme') and meta['scheme'] != 'char_1_2': raise SystemExit(f'metadata scheme mismatch: {meta["scheme"]}')
    for key, expected in (('analyzer','char'),('ngram_range',[1,2]),('norm','l2'),('use_idf',True),('smooth_idf',True),('min_df',1),('max_df',1.0),('lowercase',False)):
        if key in mp and mp[key] != expected: raise SystemExit(f'metadata parameter mismatch: {key}={mp[key]!r}, expected {expected!r}')
    if 'dtype' in mp and mp['dtype'] not in ('numpy.float64','float64'): raise SystemExit(f'metadata dtype mismatch: {mp["dtype"]!r}')
    vocab_before=dict(vec.vocabulary_); idf_before=vec.idf_.copy()
    out.mkdir(parents=True,exist_ok=True); (out/'feature_details').mkdir()
    selected=[]; rows=[]; max_score_err=0.0; max_contrib_err=0.0
    for ident in CASE_GROUPS:
        src,pr,stored=by[ident]; x1,x2=src['sentence1'],src['sentence2']; mats=vec.transform([x1,x2]); recs=[]; analyzed=[]
        for text,row in zip((x1,x2),mats):
            toks=analyzer(text); counts=Counter(toks); analyzed.append((toks,counts))
            weights={int(j):float(v) for j,v in zip(row.indices,row.data)}
            analyzed[-1]=(toks,counts,weights)
        for side,(_,counts,weights) in enumerate(analyzed):
            raw={vocab[t]:float(count)*float(idf[vocab[t]]) for t,count in counts.items() if t in vocab}
            raw_norm=math.sqrt(sum(v*v for v in raw.values()))
            expected={j:(v/raw_norm if raw_norm else 0.0) for j,v in raw.items()}
            if set(expected)!=set(weights) or any(not close(weights[j],expected[j]) for j in expected):
                raise SystemExit(f'TF-IDF count/IDF/weight mismatch ID {ident}, sentence {side+1}')
            if any(not finite(v) or v < 0 for v in weights.values()): raise SystemExit(f'invalid TF-IDF weight ID {ident}, sentence {side+1}')
        w1=analyzed[0][2]; w2=analyzed[1][2]; common=sorted(set(w1)&set(w2),key=lambda j:(-(w1[j]*w2[j]),names[j]))
        unique1=sorted(set(w1)-set(w2),key=lambda j:names[j]); unique2=sorted(set(w2)-set(w1),key=lambda j:names[j])
        common_rows=[]
        for j in common: common_rows.append({'index':j,'feature':str(names[j]),'ngram_length':len(names[j]),'count_sentence1':analyzed[0][1][names[j]],'count_sentence2':analyzed[1][1][names[j]],'idf':float(idf[j]),'weight_sentence1':w1[j],'weight_sentence2':w2[j],'contribution':w1[j]*w2[j]})
        def uniq(js, side): return [{'index':j,'feature':str(names[j]),'ngram_length':len(names[j]),'count':analyzed[side][1][names[j]],'idf':float(idf[j]),'weight':(w1 if side==0 else w2)[j]} for j in js]
        oov=[]
        for side,(toks,counts,weights) in enumerate(analyzed):
            vals=[]
            for t,c in sorted(counts.items()):
                if t not in vocab: vals.append({'feature':t,'ngram_length':len(t),'count':c})
            oov.append({'sentence':side+1,'features':vals})
        row1_norm=float(math.sqrt(sum(v*v for v in w1.values()))); row2_norm=float(math.sqrt(sum(v*v for v in w2.values())))
        dot=float(mats[0].multiply(mats[1]).sum()); contrib=sum(r['contribution'] for r in common_rows); score_err=abs(dot-stored); contrib_err=abs(contrib-dot)
        if not finite(dot) or not finite(contrib) or dot < -1e-15 or contrib < -1e-15: raise SystemExit(f'non-finite/negative similarity evidence ID {ident}')
        if not close(dot,stored): raise SystemExit(f'similarity mismatch ID {ident}: stored={stored:.17g}, recomputed={dot:.17g}, error={score_err:.17g}')
        if not close(contrib,dot): raise SystemExit(f'contribution sum mismatch ID {ident}: dot={dot:.17g}, sum={contrib:.17g}, error={contrib_err:.17g}')
        expected_error=EXPECTED_TYPES[ident]; actual_error='FP' if src['label']==0 and pr['prediction']==1 else 'FN' if src['label']==1 and pr['prediction']==0 else 'INVALID'
        if actual_error != expected_error: raise SystemExit(f'error type mismatch ID {ident}: expected {expected_error}, actual {actual_error}')
        if set(common)&(set(unique1)|set(unique2)) or set(unique1)&set(unique2) or (set(common)|set(unique1)|set(unique2)) != (set(w1)|set(w2)): raise SystemExit(f'feature partition mismatch ID {ident}')
        if any(not finite(z['contribution']) or z['contribution'] < -1e-15 for z in common_rows): raise SystemExit(f'invalid feature contribution ID {ident}')
        expected_order=sorted(common_rows,key=lambda z:(-z['contribution'],z['feature']))
        if common_rows != expected_order: raise SystemExit(f'common feature sort mismatch ID {ident}')
        top_sum=sum(r['contribution'] for r in common_rows[:10]); max_contrib_err=max(max_contrib_err,contrib_err); max_score_err=max(max_score_err,score_err)
        detail={'id':ident,'category':CASE_GROUPS[ident],'sentence1':x1,'sentence2':x2,'label':src['label'],'prediction':pr['prediction'],'error_type':actual_error,'stored_similarity':stored,'recomputed_similarity':dot,'threshold':threshold,'margin':stored-threshold,'sentence1_valid_feature_count':len(w1),'sentence2_valid_feature_count':len(w2),'sentence1_l2_norm':row1_norm,'sentence2_l2_norm':row2_norm,'zero_vector':{'sentence1':not w1,'sentence2':not w2},'common_feature_count':len(common_rows),'common_features':common_rows,'unique_sentence1_features':uniq(unique1,0),'unique_sentence2_features':uniq(unique2,1),'out_of_vocabulary_features':oov,'summary':{'contribution_sum':contrib,'top10_contribution_sum':top_sum,'top10_fraction':(top_sum/contrib if contrib else None),'max_similarity_error':score_err,'contribution_sum_error':contrib_err},'key_expression_checks':[]}
        for phrase in PHRASES[ident]:
            found=[]; start=0
            while True:
                pos=x1.find(phrase,start)
                if pos<0: break
                found.append({'start':pos,'end':pos+len(phrase),'text':phrase,'analyzed_features':[t for t in analyzer(phrase)],'note':'片段分析仅作覆盖定位；完整句子的双字符跨边界特征仍以完整句明细为准。'}); start=pos+1
            start=0; found2=[]
            while True:
                pos=x2.find(phrase,start)
                if pos<0: break
                found2.append({'start':pos,'end':pos+len(phrase),'text':phrase,'analyzed_features':[t for t in analyzer(phrase)],'note':'片段分析仅作覆盖定位；完整句子的双字符跨边界特征仍以完整句明细为准。'}); start=pos+1
            detail['key_expression_checks'].append({'requested':phrase,'sentence1_matches':found,'sentence2_matches':found2,'status':'found' if found or found2 else 'not_found_in_original_text'})
        (out/'feature_details'/f'{ident}.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); selected.append({'id':ident,'category':CASE_GROUPS[ident],'sentence1':x1,'sentence2':x2,'label':src['label'],'prediction':pr['prediction'],'error_type':detail['error_type'],'stored_similarity':stored,'recomputed_similarity':dot,'threshold':threshold,'margin':stored-threshold,'detail_path':f'feature_details/{ident}.json'})
        rows.append({'id':ident,'category':CASE_GROUPS[ident],'sentence1':x1,'sentence2':x2,'label':src['label'],'prediction':pr['prediction'],'error_type':detail['error_type'],'similarity':stored,'recomputed_similarity':dot,'margin':stored-threshold,'common_features':len(common_rows),'sentence1_features':len(w1),'sentence2_features':len(w2),'top10_sum':detail['summary']['top10_contribution_sum'],'top10_fraction':detail['summary']['top10_fraction']})
    (out/'selected_cases.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in selected),encoding='utf-8')
    with (out/'case_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    allm=metric(labels,preds); bas=metric(labels,[0]*len(labels)); (out/'metrics_comparison.json').write_text(json.dumps({'evaluation_metrics':allm,'all_zero_baseline':bas,'baseline_definition':'固定全部预测0，不根据最终评估标签选择预测类别','note':'事实核对，不用于重新选择配置。'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'metrics_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(['method','accuracy','precision_1','recall_1','f1_1','macro_f1','TN','FP','FN','TP']); w.writerow(['tfidf_cosine']+[allm[k] for k in ('accuracy','precision_1','recall_1','f1_1','macro_f1','TN','FP','FN','TP')]); w.writerow(['all_zero_baseline']+[bas[k] for k in ('accuracy','precision_1','recall_1','f1_1','macro_f1','TN','FP','FN','TP')])
    md=['# 九个固定失败案例的 TF-IDF 特征证据','', '本材料只提取冻结模型的计算证据，不把特征贡献直接当作唯一错误原因；人工分类、原因和最终选用状态均待填写。','',f'冻结配置：`char_1_2`，`similarity >= {threshold}`。共同特征贡献和为对应稀疏 L2 点积。','']
    for r in rows:
        d=json.loads((out/'feature_details'/f'{r["id"]}.json').read_text(encoding='utf-8')); md += [f'## ID {r["id"]}｜{r["category"]}',f'- 原句1：{r["sentence1"]}',f'- 原句2：{r["sentence2"]}',f'- label={r["label"]}，prediction={r["prediction"]}，{r["error_type"]}，similarity={r["similarity"]:.15g}，复算={r["recomputed_similarity"]:.15g}，margin={r["margin"]:.15g}',f'- 有效特征数：{r["sentence1_features"]} / {r["sentence2_features"]}；共同特征数：{r["common_features"]}；Top10贡献和：{r["top10_sum"]:.15g}；Top10占比：{r["top10_fraction"] if r["top10_fraction"] is not None else "null"}', '- Top10共同特征：']
        for z in d['common_features'][:10]: md.append(f'  - `{z["feature"]}` (idx {z["index"]}, idf={z["idf"]:.8g}, w1={z["weight_sentence1"]:.8g}, w2={z["weight_sentence2"]:.8g}, contribution={z["contribution"]:.8g})')
        md += [f'- 完整明细：`feature_details/{r["id"]}.json`','- 片段覆盖检查：见明细中的 `key_expression_checks`。','- 特征证据支持的解释：待人工填写','- 仍不能确定的内容：待人工填写','- 可能的方法局限：待人工填写','- 最终选用状态：待人工填写','']
    md += ['## 三类局限总结（待人工填写）','', 'A 类：待人工填写','', 'B 类：待人工填写','', 'C 类：待人工填写','']
    (out/'feature_evidence.md').write_text('\n'.join(md),encoding='utf-8',newline='\n')
    review=['# 九个固定案例人工复核表','','以下九条来自真实最终评估错误。分类是预设分析对象，不表示已证明具体原因。请结合 feature_evidence.md 和完整 JSON 明细填写。','']
    for r in selected: review += [f'## ID {r["id"]}｜{r["category"]}',f'- sentence1：{r["sentence1"]}',f'- sentence2：{r["sentence2"]}',f'- label={r["label"]}，prediction={r["prediction"]}，{r["error_type"]}，similarity={r["stored_similarity"]:.15g}，threshold=0.28，margin={r["margin"]:.15g}',f'- 完整证据：`feature_details/{r["id"]}.json`','- 我对两句语义的理解：待人工填写','- 是否接受类别及理由：待人工填写','- 特征证据支持的解释：待人工填写','- 仍不能确定的内容：待人工填写','- 可能的方法局限：待人工填写','- 最终选用状态：待人工填写','']
    (out/'case_review_with_evidence.md').write_text('\n'.join(review),encoding='utf-8',newline='\n')
    vocab_after=dict(vec.vocabulary_); idf_after=vec.idf_.copy(); protected_after={str(x.resolve()):sha(x) for x in protected_paths}
    if protected_after != protected_before: raise SystemExit('protected input hash changed during run')
    if vocab_after != vocab_before or not (idf_after == idf_before).all(): raise SystemExit('model vocabulary or IDF changed during transform')
    actual_types={ident:('FP' if by[ident][0]['label']==0 and by[ident][1]['prediction']==1 else 'FN') for ident in CASE_GROUPS}
    import numpy, scipy, sklearn
    ver={'status':'passed','inputs':{'before':protected_before,'after':protected_after,'unchanged':protected_after==protected_before},'n_samples':len(e),'selected_ids':list(CASE_GROUPS),'checks':{'nine_rows_aligned':True,'all_selected_errors':all(v in ('FP','FN') for v in actual_types.values()),'expected_error_types':EXPECTED_TYPES,'actual_error_types':actual_types,'group_error_types':{'A':all(actual_types[i]=='FP' for i in (34937,37087,38524)),'B':all(actual_types[i]=='FN' for i in (37557,36035,38599)),'C':all(actual_types[i]=='FP' for i in (35570,38638,37891))},'max_recomputed_similarity_error':max_score_err,'max_contribution_sum_error':max_contrib_err,'rtol_atol':{'rtol':1e-12,'atol':1e-12},'similarity_and_contribution_checks_passed':True,'feature_partition_checks_passed':True,'model_vocabulary_unchanged':vocab_after==vocab_before,'model_idf_unchanged':bool((idf_after==idf_before).all()),'protected_hashes_after_same_as_before':protected_after==protected_before},'environment':{'executable':sys.executable,'python':sys.version,'platform':platform.platform(),'packages':{'numpy':numpy.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'joblib':joblib.__version__}},'command':' '.join(sys.argv)}
    (out/'verification.json').write_text(json.dumps(ver,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'README.md').write_text('# TF-IDF 九个失败案例证据包\n\n本目录只复用冻结 `char_1_2` 向量器对九个预先指定的 evaluation 错误提取稀疏特征证据。未 fit、未调参、未改写正式预测。`feature_details/` 保存全部共同、独有和词表外特征；Markdown 只展示 Top10。人工归因和最终案例选择均待填写。\n\n运行脚本需要显式提供 evaluation、prediction、similarity、selected_config、vectorizer、metadata，并指定新的 `--output-dir`。已有目标会拒绝覆盖。\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'n_cases':len(selected),'max_similarity_error':max_score_err,'max_contribution_error':max_contrib_err},ensure_ascii=False))
if __name__=='__main__': main()
