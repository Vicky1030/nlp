"""Read-only acceptance checks for an existing TF-IDF + LR stage-1 run."""
from __future__ import annotations
import argparse, hashlib, json, sys, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
import joblib, numpy as np
from scipy.sparse import hstack
from sklearn.exceptions import ConvergenceWarning
from data_loader import load_data
from run_tfidf_lr_stage1 import SCHEMES, CS, WEIGHTS, config_id, pair_features
from index_paths import load_model_index, resolve_index_path
BEIJING_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")

def audit_training_warnings(records):
    """Ordinary warnings are evidence, never a required experimental result."""
    from collections import Counter
    counts = dict(Counter(w.get('category', 'unknown') for w in records))
    convergence = [w for w in records if 'ConvergenceWarning' in w.get('category', '')
                   or 'converg' in w.get('message', '').lower()]
    if convergence:
        raise ValueError(f'unresolved training convergence warning(s): {convergence!r}')
    return {'records': records, 'category_counts': counts,
            'futurewarning_count': counts.get('FutureWarning', 0),
            'convergence_warning_count': len(convergence), 'unresolved_convergence_issue': False}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read_probs(path):
    rows=[]
    for n,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        obj=json.loads(line)
        assert set(obj)=={'config_id','id','label','probability'}, (path,n,set(obj))
        rows.append(obj)
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True); p.add_argument('--run',type=Path,required=True); p.add_argument('--audit-dir',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=None); p.add_argument('--vectorizer-root',type=Path,default=None)
    a=p.parse_args(); root=a.root.resolve(); run=a.run.resolve(); audit=a.audit_dir.resolve()
    data_dir=(a.data_dir.resolve() if a.data_dir is not None else root/'data'/'processed')
    vectorizer_root=(a.vectorizer_root.resolve() if a.vectorizer_root is not None else root/'artifacts'/'tfidf_cosine')
    train_path=data_dir/'train.jsonl'; validation_path=data_dir/'validation.jsonl'
    missing=[str(p) for p in (train_path,validation_path,vectorizer_root) if not p.exists()]
    if missing: raise FileNotFoundError('missing explicit stage-1 acceptance input(s):\n'+'\n'.join(missing))
    if audit.exists() and any(audit.iterdir()): raise FileExistsError(f'audit directory must be new and empty: {audit}')
    audit.mkdir(parents=True,exist_ok=True)
    start=datetime.now(BEIJING_TZ).isoformat()
    train=load_data(train_path); val=load_data(validation_path)
    index_path=run/'model_index.json'; idx=load_model_index(index_path)
    run_meta=json.loads((run/'run_metadata.json').read_text(encoding='utf-8'))
    expected={config_id(s,c,w) for s in SCHEMES for c in CS for w in WEIGHTS}
    ids=[x['config_id'] for x in idx['configs']]
    assert idx['count']==18 and len(ids)==18 and len(set(ids))==18 and set(ids)==expected
    files=sorted([*run.glob('models/*.joblib'),*run.glob('probabilities/*.jsonl')])
    before={str(f):sha(f) for f in files}
    vectors={s:joblib.load(vectorizer_root/s/'vectorizer.joblib') for s in SCHEMES}
    vector_states={s:(dict(v.vocabulary_),v.idf_.copy()) for s,v in vectors.items()}
    train_features={}; val_features={}; alignment={}
    for s,v in vectors.items():
        X=pair_features(v,train); V=pair_features(v,val)
        assert X.shape==(len(train),2*len(v.vocabulary_)) and V.shape==(len(val),2*len(v.vocabulary_))
        assert X.format=='csr' and V.format=='csr'
        train_features[s]=X; val_features[s]=V
        # Verify sparse rows are generated from the matching record at several file positions.
        picks=sorted({0,1,len(train)//2,len(train)-1})
        for i in picks:
            assert (pair_features(v,[train[i]])-X[i]).nnz==0
            assert (pair_features(v,[val[i]])-V[i]).nnz==0 if i<len(val) else True
        alignment[s]={'train_rows':X.shape[0],'validation_rows':V.shape[0],
                      'train_id_label_pairs_in_file_order':[[r['id'],r['label']] for r in train],
                      'validation_id_label_pairs_in_file_order':[[r['id'],r['label']] for r in val],
                      'sampled_train_feature_rows_match_same_record':True,
                      'sampled_validation_feature_rows_match_same_record':True,
                      'feature_width':X.shape[1], 'format':X.format}
    # Explicit synthetic arithmetic establishes [abs(A-B); A*B] order.
    from scipy.sparse import csr_matrix
    sa=csr_matrix([[1.,0.,2.]]); sb=csr_matrix([[0.,3.,2.]])
    syn=hstack((abs(sa-sb),sa.multiply(sb)),format='csr')
    assert syn.format=='csr' and syn.shape==(1,6) and syn.toarray().tolist()==[[1.,3.,0.,0.,0.,4.]]
    reloads=[]; warnings_seen=[]; models={}; probs={}
    for s in SCHEMES:
        v=vectors[s]
        for c in CS:
            for cw in WEIGHTS:
                cid=config_id(s,c,cw)
                entry=next(x for x in idx['configs'] if x['config_id']==cid)
                mp=resolve_index_path(index_path,idx,entry['model'],'model')
                pp=resolve_index_path(index_path,idx,entry['probabilities'],'probabilities')
                assert mp.is_file() and pp.is_file()
                if sha(mp)!=entry['model_sha256']:
                    raise ValueError(f'{cid}: model SHA-256 mismatch before load: {mp}')
                if sha(pp)!=entry['probabilities_sha256']:
                    raise ValueError(f'{cid}: probability SHA-256 mismatch: {pp}')
                m=joblib.load(mp); models[cid]=m
                pars=m.get_params(deep=False)
                required={'solver':'liblinear','penalty':'l2','C':c,'class_weight':cw,'fit_intercept':True,
                          'intercept_scaling':1,'dual':False,'max_iter':5000,'tol':1e-4,'random_state':42}
                actual={k:pars[k] for k in required}
                assert actual==required,(cid,actual,required)
                assert m.n_iter_.size==1 and int(m.n_iter_[0])<=5000
                assert [int(x) for x in m.n_iter_]==entry['n_iter']
                saved=read_probs(pp); assert len(saved)==3433
                assert all(r['config_id']==cid for r in saved)
                assert [(r['id'],r['label']) for r in saved]==[(r['id'],r['label']) for r in val]
                arr=np.asarray([r['probability'] for r in saved],dtype=np.float64)
                assert np.isfinite(arr).all() and ((arr>=0)&(arr<=1)).all()
                pos=list(m.classes_).index(1)
                recalculated=m.predict_proba(val_features[s])[:,pos]
                diff=np.abs(recalculated-arr)
                np.testing.assert_allclose(recalculated,arr,rtol=1e-12,atol=1e-12)
                # Full validation swap-invariance for features and predictions.
                swapped=[{'id':r['id'],'label':r['label'],'sentence1':r['sentence2'],'sentence2':r['sentence1']} for r in val]
                Xswap=pair_features(v,swapped)
                delta=val_features[s]-Xswap
                assert delta.nnz==0, (cid,'swapped features')
                pswap=m.predict_proba(Xswap)[:,pos]
                np.testing.assert_allclose(pswap,recalculated,rtol=1e-12,atol=1e-12)
                # Alter in-memory labels only; inference inputs and scores must not change.
                changed=[dict(r,label=1-r['label']) for r in val]
                Xchanged=pair_features(v,changed)
                assert (Xchanged-val_features[s]).nnz==0
                np.testing.assert_array_equal(m.predict_proba(Xchanged)[:,pos],recalculated)
                reloads.append({'config_id':cid,'samples':len(saved),'max_absolute_error':float(diff.max()),'passed':True,
                    'classes':m.classes_.tolist(),'positive_probability_class':1,'parameters':actual,'n_iter':[int(x) for x in m.n_iter_]})
                probs[cid]=arr
    for s,v in vectors.items():
        assert v.vocabulary_==vector_states[s][0] and np.array_equal(v.idf_,vector_states[s][1])
    # Real collision test: the runner must reject an occupied target and leave bytes intact.
    probe=audit/'overwrite_probe'; (probe/'models').mkdir(parents=True); (probe/'probabilities').mkdir()
    cid=sorted(expected)[0]; sentinels=[probe/'models'/f'{cid}.joblib',probe/'probabilities'/f'{cid}.jsonl']
    for i,f in enumerate(sentinels): f.write_bytes(f'preserve-{i}'.encode())
    sentinel_before={str(f):sha(f) for f in sentinels}
    command=[sys.executable,'-B',str(root/'src/run_tfidf_lr_stage1.py'),'--train',str(train_path),
        '--validation',str(validation_path),'--vectorizer-root',str(vectorizer_root),'--output-dir',str(probe)]
    import subprocess
    collision=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',cwd=str(root))
    sentinel_after={str(f):sha(f) for f in sentinels}
    assert collision.returncode!=0 and 'Refusing to overwrite' in collision.stderr and sentinel_before==sentinel_after
    after={str(f):sha(f) for f in files}
    assert before==after
    warning_records=run_meta.get('warnings',[])
    warning_audit=audit_training_warnings(warning_records)
    end=datetime.now(BEIJING_TZ).isoformat()
    result={'status':'passed','started_at':start,'ended_at':end,'timezone':'Asia/Shanghai / UTC+08:00',
      'interpreter':sys.executable,'python':sys.version,'cwd':str(Path.cwd()),'run_dir':str(run),
      'data_dir':str(data_dir),'vectorizer_root':str(vectorizer_root),'train_path':str(train_path),'validation_path':str(validation_path),
      'scope':'reload/integrity acceptance only; no training, predictions, classification metrics, thresholds, evaluation or Test access',
      'config_count':18,'configuration_ids':sorted(ids),'feature_definition':'CSR hstack([abs(A-B), A.multiply(B)])',
      'synthetic_feature_check':{'expected':[1.,3.,0.,0.,0.,4.],'shape':[1,6],'format':'csr','passed':True},
      'alignment_checks':alignment,'probability_reloads':reloads,
      'swap_invariance':{'all_validation_rows':3433,'feature_matrix_equal':True,'all_18_probabilities_equal_within_1e-12':True},
      'label_independence':{'all_18_probabilities_unchanged_after_in_memory_label_flip':True},
      'vectorizer_transform_state_unchanged':True,
      'probability_files':{'count':len(list((run/'probabilities').glob('*.jsonl'))),'rows_each':3433,'all_fields_exact':True},
      'models':{'count':len(list((run/'models').glob('*.joblib'))),'all_loadable':True,'parameters_and_n_iter_recorded':True},
      'training_warning_audit':warning_audit,
      'l2_history':{'first_attempt':'停止于第一组配置，因为脚本把所有警告均作为阻断；penalty="l2"触发scikit-learn 1.9.1 FutureWarning。',
        'second_attempt':'只修改警告处理：记录所有警告，仅ConvergenceWarning阻断；LogisticRegression仍传入penalty="l2"，本次实载模型get_params确认penalty=l2。实验参数未变。'},
      'overwrite_test':{'exit_code':collision.returncode,'stdout':collision.stdout,'stderr':collision.stderr,
        'sentinel_hashes_before':sentinel_before,'sentinel_hashes_after':sentinel_after,'unchanged':sentinel_before==sentinel_after},
      'original_model_probability_hashes_before':before,'original_model_probability_hashes_after':after,
      'original_files_unchanged':before==after}
    (audit/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','configs':18,'max_absolute_error':max(x['max_absolute_error'] for x in reloads),'overwrite_refused':True},ensure_ascii=False))

if __name__=='__main__': main()
