"""Cached validation threshold comparison, exact selection and timed inference.

No fitting and no evaluation/Test access. Paths in config are relative to config.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from data_loader import load_data, strict_json
from tfidf_common import SCHEMES, dictionary_info, jieba_tokens, transform_pairs

BASE = Path(__file__).resolve().parents[1]
RULES = {
    'within_scheme': ['Macro-F1 descending', 'Accuracy descending', 'threshold_index descending'],
    'across_schemes': ['best Macro-F1 descending', 'best Accuracy descending', 'scheme order ascending'],
    'scheme_order': list(SCHEMES),
    'ranking': 'Exact Fraction from integer confusion counts; no rounded metrics or tie tolerance',
}
CLAIM = '在预设三种表示和 102 个候选阈值中，验证集按既定规则选出的最佳配置。'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def ensure_targets_absent(paths):
    conflicts = [str(Path(p).resolve()) for p in paths if Path(p).exists()]
    if conflicts:
        raise FileExistsError('Refusing to overwrite existing output files; choose a new --output-root:\n' + '\n'.join(conflicts))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8', newline='\n')


def relative(path, directory):
    return Path(os.path.relpath(path, directory)).as_posix()


def load_scores(path, validation):
    raw = path.read_bytes()
    need(not raw.startswith(b'\xef\xbb\xbf'), f'{path}:1: BOM not allowed')
    rows = []
    seen = set()
    for line, text in enumerate(raw.decode('utf8').splitlines(), 1):
        try:
            r = strict_json(text)
            need(type(r) is dict and set(r)=={'id','label','similarity'}, 'expected exactly id,label,similarity')
            need(type(r['id']) is int and r['id'] not in seen, 'invalid or duplicate integer ID')
            need(type(r['label']) is int and r['label'] in (0,1), 'invalid integer label')
            s = r['similarity']
            need(type(s) in (float,int) and math.isfinite(s) and 0 <= s <= 1, 'invalid finite similarity in [0,1]')
            need(line <= len(validation), 'extra score row')
            v = validation[line-1]
            need((r['id'],r['label']) == (v['id'],v['label']), 'ID/label/order differs from validation')
            seen.add(r['id']); rows.append(r)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f'{path}:{line}: {exc}') from exc
    need(len(rows)==len(validation), f'{path}: EOF: {len(rows)} != {len(validation)}')
    return np.array([r['similarity'] for r in rows], dtype=np.float64)


def predict(scores, index):
    return (np.asarray(scores) >= index/100).astype(np.int64)


def safe_fraction(a, b):
    return Fraction(a,b) if b else Fraction(0)


def exact_metrics(row):
    tn,fp,fn,tp = (int(row[k]) for k in ('TN','FP','FN','TP'))
    return {
        'accuracy': safe_fraction(tn+tp,tn+fp+fn+tp),
        'precision_1': safe_fraction(tp,tp+fp),
        'recall_1': safe_fraction(tp,tp+fn),
        'f1_1': safe_fraction(2*tp,2*tp+fp+fn),
        'macro_f1': (safe_fraction(2*tp,2*tp+fp+fn)+safe_fraction(2*tn,2*tn+fp+fn))/2,
    }


def score(y, prediction, scheme, index):
    tn,fp,fn,tp = map(int,confusion_matrix(y,prediction,labels=[0,1]).ravel())
    row = {'scheme':scheme,'threshold_index':index,'threshold':index/100,'n_samples':len(y),
           'accuracy':float(accuracy_score(y,prediction)),
           'precision_1':float(precision_score(y,prediction,pos_label=1,zero_division=0)),
           'recall_1':float(recall_score(y,prediction,pos_label=1,zero_division=0)),
           'f1_1':float(f1_score(y,prediction,pos_label=1,zero_division=0)),
           'macro_f1':float(f1_score(y,prediction,labels=[0,1],average='macro',zero_division=0)),
           'TN':tn,'FP':fp,'FN':fn,'TP':tp,'predicted_0':tn+fn,'predicted_1':fp+tp}
    need(tn+fp+fn+tp==len(y),'confusion total differs')
    # Independent formulas use integer counts; sklearn is the scoring implementation.
    for name,value in exact_metrics(row).items():
        need(math.isclose(row[name],float(value),rel_tol=1e-14,abs_tol=1e-14), 'metric formula mismatch: '+name)
    return row


def within_key(row):
    m = exact_metrics(row)
    return m['macro_f1'],m['accuracy'],row['threshold_index']


def across_key(row):
    m = exact_metrics(row)
    return m['macro_f1'],m['accuracy'],-SCHEMES.index(row['scheme'])


def synthetic_checks():
    need(predict([0.,.5,1.],50).tolist()==[0,1,1],'>= equality failed')
    need(predict([0.,.5,1.],0).tolist()==[1,1,1],'threshold 0 failed')
    need(predict([0.,.5,1.],101).tolist()==[0,0,0],'threshold 1.01 failed')
    a = score(np.array([0,1]), np.array([0,1]), 'char_1', 20)
    b = dict(a,threshold_index=90,threshold=.9)
    need(max([a,b],key=within_key)['threshold_index']==90,'within tie rule failed')
    tied = [dict(a,scheme=s) for s in reversed(SCHEMES)]
    need(max(tied,key=across_key)['scheme']=='char_1','across tie rule failed')
    for y,p in [([0,1],[0,0]),([0,1],[1,1]),([0,0],[0,0]),([1,1],[1,1])]:
        score(np.array(y),np.array(p),'char_1',101)
    # Different exact fractions must not be treated as ties.
    need(Fraction(1,3)!=Fraction(333333333333,1000000000000),'Fraction distinction')
    return {'threshold_equality':True,'endpoints':True,'within_tie_larger_threshold':True,
            'across_tie_fixed_order':True,'single_prediction_class_zero_division':True}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)  # Python float repr: round-trip precision


def model_predict(vectorizer, sentence1, sentence2, threshold_index):
    scores,_ = transform_pairs(vectorizer,[sentence1],[sentence2])
    value = float(scores[0])
    return value,int(value >= threshold_index/100)


def warmup(vectorizer):
    jieba_tokens('固定预热：你好，世界123。')
    for _ in range(3):
        model_predict(vectorizer,'固定预热句子甲。','固定预热句子乙。',50)


def reload_check(config_path):
    config = strict_json(config_path.read_text(encoding='utf8'))
    root = config_path.parent
    model = (root/config['vectorizer']['path']).resolve()
    validation_path = (root/config['validation']['path']).resolve()
    score_path = (root/config['similarity_inputs'][config['representation']]['path']).resolve()
    need(sha(model)==config['vectorizer']['sha256'],'reload model hash differs')
    need(sha(validation_path)==config['validation']['sha256'],'reload validation hash differs')
    need(sha(score_path)==config['similarity_inputs'][config['representation']]['sha256'],'reload scores hash differs')
    validation = load_data(validation_path); cached = load_scores(score_path,validation)
    v = joblib.load(model)
    scores,_ = transform_pairs(v,[r['sentence1'] for r in validation],[r['sentence2'] for r in validation])
    np.testing.assert_allclose(scores,cached,rtol=1e-12,atol=1e-12)
    need(np.array_equal(predict(scores,config['threshold_index']),predict(cached,config['threshold_index'])),'reload label mismatch')
    saved_path = (root/config['validation_predictions']['path']).resolve()
    need(sha(saved_path)==config['validation_predictions']['sha256'],'reload predictions hash differs')
    saved = [strict_json(s) for s in saved_path.read_text(encoding='utf8').splitlines()]
    need(len(saved)==len(validation),'reload row count')
    need([r['prediction'] for r in saved]==predict(scores,config['threshold_index']).tolist(),'reload != saved predictions')
    print(json.dumps({'passed':True,'rows':len(validation),'max_absolute_similarity_difference':float(np.max(np.abs(scores-cached))),
                      'prediction_labels_identical':True,'rtol':1e-12,'atol':1e-12}))


def run(args):
    validation_path = args.validation.resolve(); stage1_root = args.stage1_root.resolve(); output = args.output_root.resolve()
    score_dir = args.scores_dir.resolve() if args.scores_dir else stage1_root/'results/tfidf_cosine'
    model_dir = args.models_dir.resolve() if args.models_dir else stage1_root/'artifacts/tfidf_cosine'
    stage1_report = args.stage1_report.resolve() if args.stage1_report else stage1_root/'reports/tfidf_cosine_stage1.json'
    result_dir = output/'results/tfidf_cosine'; config_dir=output/'artifacts/tfidf_cosine'
    config_path=config_dir/'selected_config.json'; pred_path=result_dir/'validation_selected_predictions.jsonl'
    targets = [result_dir/'threshold_comparison.csv',result_dir/'representation_summary.csv',pred_path,config_path,output/'reports/tfidf_cosine_stage2.json']
    ensure_targets_absent(targets)
    first = strict_json(stage1_report.read_text(encoding='utf8'))
    for name,expected_hash in first['code_sha256'].items():
        need(sha(Path(__file__).with_name(name))==expected_hash,'stage1 code changed: '+name)
    validation = load_data(validation_path); need(bool(validation),'empty validation')
    need(sha(validation_path)==first['inputs']['validation']['sha256'],'validation SHA-256 differs from stage1')
    paths = {s:{'score':score_dir/f'validation_similarity_{s}.jsonl','model':model_dir/s/'vectorizer.joblib','metadata':model_dir/s/'metadata.json'} for s in SCHEMES}
    protected = [validation_path,stage1_report,stage1_root/'TFIDF_STAGE1.md']
    protected += [p for mapping in paths.values() for p in mapping.values()]
    protected += [Path(__file__).with_name(n) for n in ('data_loader.py','tfidf_common.py','run_tfidf_stage1.py')]
    before = {str(p):sha(p) for p in protected if p.is_file()}
    metadata = {}; cached = {}
    for s in SCHEMES:
        expected = {Path(p).name:h for p,h in first['schemes'][s]['files'].items()}
        for kind,p in paths[s].items():
            need(p.name in expected and sha(p)==expected[p.name],f'{p}: stage1 SHA-256 mismatch ({kind})')
        metadata[s] = strict_json(paths[s]['metadata'].read_text(encoding='utf8'))
        need(metadata[s]['scheme']==s and metadata[s]['model_sha256']==sha(paths[s]['model']),s+' metadata mismatch')
        cached[s] = load_scores(paths[s]['score'],validation)
    y = np.array([r['label'] for r in validation],dtype=np.int64)
    tests = synthetic_checks(); comparison=[]; best=[]
    reference = score(y,np.zeros(len(y),dtype=np.int64),'all_zero_reference',101)
    for s in SCHEMES:
        rows=[]
        for index in range(102):
            prediction = predict(cached[s],index)
            row = score(y,prediction,s,index); rows.append(row)
        need(rows[0]['predicted_1']==len(y) and rows[-1]['predicted_0']==len(y),'endpoint coverage')
        need(exact_metrics(rows[-1])==exact_metrics(reference),'all-zero reference mismatch')
        need(all(rows[-1][k]==reference[k] for k in ('TN','FP','FN','TP')),'all-zero counts mismatch')
        comparison.extend(rows); best.append(max(rows,key=within_key))
    need(len(comparison)==306 and len({(r['scheme'],r['threshold_index']) for r in comparison})==306,'306-grid incomplete')
    winner = max(best,key=across_key); scheme=winner['scheme']; index=winner['threshold_index']
    need(not set(p.resolve() for p in targets).intersection(p.resolve() for p in protected),'output aliases protected input')
    write_csv(targets[0],comparison)
    summary=[dict(r,vocabulary_size=metadata[r['scheme']]['vocabulary_size'],selected=(r['scheme']==scheme)) for r in best]
    write_csv(targets[1],summary)
    with targets[0].open(encoding='utf8',newline='') as f:
        csv_rows=list(csv.DictReader(f))
    need(len(csv_rows)==306,'CSV row count')
    for a,b in zip(comparison,csv_rows):
        for k,v in a.items():
            need(b[k]==v if isinstance(v,str) else float(b[k])==v,'CSV roundtrip mismatch '+k)
    # Selection finished. Load only the winner; no transform inside threshold search.
    v = joblib.load(paths[scheme]['model'])
    vocabulary_before=dict(v.vocabulary_); idf_before=v.idf_.copy()
    warmup(v)
    inferred=[]; scores=[]; latencies=[]
    for row in validation:
        start=perf_counter_ns()
        value,prediction=model_predict(v,row['sentence1'],row['sentence2'],index)
        elapsed=perf_counter_ns()-start
        latency=elapsed/1_000_000
        need(math.isfinite(latency) and latency>=0,'invalid latency')
        # Record assembly is outside timed section. Neither labels nor IDs enter inference.
        inferred.append(dict(row,prediction=prediction,method='tfidf_cosine',latency_ms=latency))
        scores.append(value); latencies.append(latency)
    np.testing.assert_allclose(scores,cached[scheme],rtol=1e-12,atol=1e-12)
    labels=np.array([r['prediction'] for r in inferred]); cached_labels=predict(cached[scheme],index)
    mismatches=np.flatnonzero(labels!=cached_labels)
    if len(mismatches):
        details=[{'id':validation[i]['id'],'cached':float(cached[scheme][i]),'recomputed':scores[i],'threshold':index/100} for i in mismatches]
        raise ValueError('Threshold-boundary mismatch (no inputs changed): '+json.dumps(details))
    need(v.vocabulary_==vocabulary_before and np.array_equal(v.idf_,idf_before),'inference modified vectorizer state')
    pred_path.parent.mkdir(parents=True,exist_ok=True)
    with pred_path.open('w',encoding='utf8',newline='\n') as f:
        for row in inferred:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    reread=[strict_json(s) for s in pred_path.read_text(encoding='utf8').splitlines()]
    need(len(reread)==len(validation),'prediction length')
    for a,b in zip(validation,reread):
        need(set(b)=={'id','sentence1','sentence2','label','prediction','method','latency_ms'},'seven fields required')
        need(all(b[k]==a[k] and type(b[k]) is type(a[k]) for k in a),'prediction not aligned')
        need(type(b['prediction']) is int and b['prediction'] in (0,1) and b['method']=='tfidf_cosine','invalid prediction')
        need(type(b['latency_ms']) in (float,int) and math.isfinite(b['latency_ms']) and b['latency_ms']>=0,'invalid latency output')
    environment={'python':platform.python_version(),'os':platform.platform(),
                 'packages':{n:importlib.metadata.version(n) for n in ('scikit-learn','numpy','scipy','jieba','joblib')}}
    need(environment['packages']==first['environment']['packages'],'dependencies changed since stage1')
    code_hashes={n:sha(Path(__file__).with_name(n)) for n in ('run_tfidf_stage2.py','tfidf_common.py','data_loader.py')}
    tokenization={k:value for k,value in metadata[scheme]['tokenization'].items() if k!='path'}
    need(dictionary_info()['sha256']==first['dictionary']['sha256'],'default dictionary changed')
    config={'method':'tfidf_cosine','representation':scheme,'threshold_index':index,'threshold':index/100,'operator':'>=',
            'schemes':list(SCHEMES),'candidate_thresholds':[{'threshold_index':i,'threshold':i/100} for i in range(102)],
            'selection_rules':RULES,'selection_claim':CLAIM,'validation_metrics':winner,
            'exact_ranking_scores':{k:str(f) for k,f in exact_metrics(winner).items()},
            'parameters':metadata[scheme]['parameters'],'tokenization':tokenization,
            'vectorizer':{'path':relative(paths[scheme]['model'],config_dir),'sha256':sha(paths[scheme]['model'])},
            'validation':{'path':relative(validation_path,config_dir),'sha256':sha(validation_path)},
            'similarity_inputs':{s:{'path':relative(paths[s]['score'],config_dir),'sha256':sha(paths[s]['score'])} for s in SCHEMES},
            'validation_predictions':{'path':relative(pred_path,config_dir),'sha256':sha(pred_path)},
            'training_data_sha256_from_metadata':metadata[scheme]['train_sha256'],
            'environment':environment,'code_sha256':code_hashes,'code_directory':relative(Path(__file__).parent,config_dir),
            'path_resolution':'All paths relative to this selected_config.json directory',
            'latency_scope':'perf_counter_ns: individual sentence pair, text processing/tokenization, two transforms, sparse row cosine, >= decision; excludes load, warmup, I/O, scoring, record assembly',
            'final_evaluation_executed':False}
    write_json(config_path,config)
    command=[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--reload-config',str(config_path),
             '--validation',str(args.validation),'--stage1-root',str(args.stage1_root),'--output-root',str(output)]
    child=subprocess.run(command,capture_output=True,text=True,encoding='utf8')
    need(child.returncode==0,'new-process load failed: '+child.stderr)
    reload_result=strict_json(child.stdout)
    after={str(p):sha(p) for p in protected if p.is_file()}; need(before==after,'protected file modified')
    report={'status':'passed','scope':'Validation selection only; no fit, evaluation/Test read, LR or final evaluation',
            'command':subprocess.list2cmdline([sys.executable,*sys.argv]),
            'comparison_groups':len(comparison),'validation_samples':len(validation),'selection_claim':CLAIM,
            'selection_rules':RULES,'representation_best':summary,'selected':winner,'all_zero_reference':reference,
            'metric_policy':{'labels':[0,1],'zero_division':0,'confusion_matrix':'[[TN,FP],[FN,TP]]',
                             'sklearn_vs_independent_count_formulas':'all 306 rows passed; check tolerance 1e-14 only for implementation verification, never ranking'},
            'checks':{'input_schema_and_alignment':True,'stage1_file_hashes_match':True,'306_unique_groups':True,'all_confusion_totals':True,
                      'CSV_roundtrip':True,'synthetic':tests,'all_zero_matches_threshold_1_01':True,'prediction_seven_fields_aligned':True,
                      'timed_prediction_matches_cached':True,'max_similarity_difference':float(np.max(np.abs(np.array(scores)-cached[scheme]))),
                      'vocabulary_and_idf_unchanged':True,'reload_process':{'exit_code':child.returncode,'result':reload_result},
                      'protected_files_unchanged':True},
            'latency':{'scope':config['latency_scope'],'clock':vars(__import__('time').get_clock_info('perf_counter')),
                       'warmup':'jieba initialized; three synthetic non-evaluation pairs before timing','rows':len(latencies),
                       'min_ms':min(latencies),'max_ms':max(latencies)},
            'environment':environment,'code_sha256':code_hashes,
            'protected_hashes':{relative(Path(p),output):h for p,h in before.items()},
            'outputs':{relative(p,output):sha(p) for p in targets[:-1]},'final_evaluation_executed':False}
    write_json(targets[-1],report)
    print(json.dumps({'status':'passed','groups':len(comparison),'best':summary,'selected':winner,'all_zero':reference,
                      'reload':reload_result},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--stage1-root',type=Path,required=True)
    p.add_argument('--scores-dir',type=Path)
    p.add_argument('--models-dir',type=Path)
    p.add_argument('--stage1-report',type=Path)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--reload-config',type=Path)
    args=p.parse_args()
    if args.reload_config:
        reload_check(args.reload_config.resolve())
    else:
        run(args)


if __name__=='__main__':
    main()
