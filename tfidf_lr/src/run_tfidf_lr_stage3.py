"""Run frozen selected TF-IDF + LR inference on the unified evaluation set."""
import argparse,csv,hashlib,json,math,os,platform,statistics,subprocess,sys,time,importlib.metadata
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np,joblib
from data_loader import load_data
from lr_stage3_runtime import load_frozen,predict_pair,predict_batch
from lr_stage3_metrics import compare_sklearn
from index_paths import load_model_index, resolve_index_path

EXPECTED_EVAL='adb30d8dc949b1605501c577b3d5c9d8a8aa199c3004fd72052c210b83240838'
BEIJING_TZ=timezone(timedelta(hours=8), name='Asia/Shanghai')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def verify_stage2(config_path,root):
    config=json.loads(config_path.read_text(encoding='utf-8'))
    expected={'config_id':'char_1_2__C1p0__cw_balanced','representation':'char_1_2','C':1.0,'class_weight':'balanced','threshold_index':53,'threshold':0.53}
    for k,v in expected.items():
        if config.get(k)!=v: raise ValueError(f'frozen config mismatch for {k}: actual={config.get(k)!r}; expected={v!r}')
    report=json.loads((config_path.parent/'selection_report.json').read_text(encoding='utf-8'))
    if report.get('status')!='frozen' or report.get('candidate_scoring',{}).get('selected_config_id')!=config['config_id']:
        raise ValueError('stage-2 selection report does not support frozen config')
    manifest=json.loads((config_path.parent/'artifact_sha256.json').read_text(encoding='utf-8'))
    for rel in ('selected_config.json','artifacts/selected/models/lr_model.joblib','artifacts/selected/vectorizers/tfidf_vectorizer.joblib',
      'threshold_comparison.csv','model_summary.csv','selection_report.json'):
        path=config_path.parent/rel
        if path.relative_to(config_path.parent).as_posix() not in manifest or sha(path)!=manifest[path.relative_to(config_path.parent).as_posix()]:
            raise ValueError(f'stage-2 frozen artifact hash mismatch: {rel}')
    return config,report

def protected_paths(root,run_dir,stage2_dir,eval_path,data_dir=None,vectorizer_root=None):
    index_path=run_dir/'model_index.json'; index=load_model_index(index_path)
    data_root=Path(data_dir).resolve() if data_dir is not None else root/'data/processed'
    vec_root=Path(vectorizer_root).resolve() if vectorizer_root is not None else root/'artifacts/tfidf_cosine'
    paths=[data_root/'train.jsonl',data_root/'validation.jsonl',eval_path]
    for scheme in ('char_1','char_1_2','word_1'):
        paths.extend([vec_root/scheme/'vectorizer.joblib',vec_root/scheme/'metadata.json'])
    for entry in index['configs']:
        model_path=resolve_index_path(index_path,index,entry['model'],'model')
        probability_path=resolve_index_path(index_path,index,entry['probabilities'],'probabilities')
        if sha(model_path)!=entry.get('model_sha256'):
            raise ValueError(f"model SHA-256 mismatch for {entry.get('config_id')}: {model_path}")
        if sha(probability_path)!=entry.get('probabilities_sha256'):
            raise ValueError(f"probability SHA-256 mismatch for {entry.get('config_id')}: {probability_path}")
        paths.extend([model_path, probability_path])
    paths.extend([stage2_dir/'selected_config.json',stage2_dir/'artifacts/selected/models/lr_model.joblib',
      stage2_dir/'artifacts/selected/vectorizers/tfidf_vectorizer.joblib',stage2_dir/'threshold_comparison.csv',
      stage2_dir/'model_summary.csv',stage2_dir/'selection_report.json'])
    unique=list(dict.fromkeys(p.resolve() for p in paths))
    missing=[str(p) for p in unique if not p.is_file()]
    if missing: raise FileNotFoundError('protected stage input missing:\n'+'\n'.join(missing))
    return unique

def hardware():
    ps="""
    $cpu=(Get-CimInstance Win32_Processor -ErrorAction Stop | Select-Object -First 1 -ExpandProperty Name)
    $physical=(Get-CimInstance Win32_PhysicalMemory -ErrorAction Stop | Measure-Object Capacity -Sum).Sum
    $os=Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
    [pscustomobject]@{cpu=$cpu;installed_memory_bytes=$physical;system_visible_memory_bytes=([int64]$os.TotalVisibleMemorySize*1024);os_caption=$os.Caption;os_version=$os.Version} | ConvertTo-Json -Compress
    """
    try:
        result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',ps],capture_output=True,text=True,encoding='utf-8',check=True)
        return json.loads(result.stdout),None
    except Exception as exc:
        return {'cpu':None,'installed_memory_bytes':None,'system_visible_memory_bytes':None,'os_caption':platform.platform(),'os_version':None},str(exc)

def write_json(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--evaluation',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--stage1-run',type=Path,default=None); ap.add_argument('--data-dir',type=Path,default=None); ap.add_argument('--vectorizer-root',type=Path,default=None)
    a=ap.parse_args(); config_path=a.config.resolve(); eval_path=a.evaluation.resolve(); out=a.output_dir.resolve()
    if out.exists(): raise FileExistsError(f'refusing to overwrite existing output directory; choose new --output-dir: {out}')
    targets=[out/n for n in ('evaluation_predictions.jsonl','evaluation_probabilities.jsonl','evaluation_metrics.json','latency_summary.json','verification.json','execution.json','stdout.txt','stderr.txt','artifact_sha256.json')]
    conflicts=[str(p) for p in targets if p.exists()]
    if conflicts: raise FileExistsError('refusing existing output targets:\n'+'\n'.join(conflicts))
    # Locate the project root from the config tree instead of relying on a
    # historical stage directory name. This supports nested fresh runs.
    root=None
    for candidate in config_path.parents:
        if (candidate/'src').is_dir() and (candidate/'data').is_dir():
            root=candidate; break
    if root is None:
        root=config_path.parent
    frozen,selection=verify_stage2(config_path,root)
    project_root=root
    stage1_run=(a.stage1_run.resolve() if a.stage1_run is not None else project_root/'outputs'/'lr_run_20261020_01')
    if not stage1_run.is_dir():
        raise FileNotFoundError(f"stage-1 run directory is missing: {stage1_run}")
    stage1_index=stage1_run/'model_index.json'
    if not stage1_index.is_file():
        raise FileNotFoundError(f"stage-1 model index is missing: {stage1_index}")
    protect=protected_paths(project_root,stage1_run,config_path.parent,eval_path,a.data_dir,a.vectorizer_root)
    if not protect:
        raise RuntimeError('protected input scope is empty; refusing to report protection as passed')
    protected_before={str(p):sha(p) for p in protect}
    started=datetime.now(BEIJING_TZ).isoformat()
    out.mkdir(parents=True,exist_ok=False)
    if not eval_path.is_file(): raise FileNotFoundError(f'evaluation file missing: {eval_path}')
    eval_hash=sha(eval_path)
    if eval_hash!=EXPECTED_EVAL: raise ValueError(f'evaluation SHA-256 mismatch: {eval_hash}')
    rows=load_data(eval_path); labels=[r['label'] for r in rows]
    counts={str(y):sum(r['label']==y for r in rows) for y in (0,1)}
    if len(rows)!=4316 or counts!={'0':2978,'1':1338}: raise ValueError(f'evaluation counts mismatch: {len(rows)} {counts}')
    frozen,vectorizer,model,pos=load_frozen(config_path); threshold=frozen['threshold']
    # Warm exactly the first ten text pairs before starting the official per-record timing pass.
    for row in rows[:10]: predict_pair(vectorizer,model,pos,row['sentence1'],row['sentence2'],threshold)
    warmup={'records':10,'indices_0based':list(range(10)),'completed_before_timed_pass':True}
    predictions=[]; probabilities=[]; latencies=[]; zero_l=zero_r=0
    for row in rows:
        t0=time.perf_counter_ns()
        probability,prediction,zeros=predict_pair(vectorizer,model,pos,row['sentence1'],row['sentence2'],threshold)
        elapsed=(time.perf_counter_ns()-t0)/1_000_000.0
        if not math.isfinite(elapsed) or elapsed<0: raise ValueError('invalid latency')
        zero_l+=zeros[0]; zero_r+=zeros[1]; latencies.append(elapsed)
        predictions.append({'id':row['id'],'sentence1':row['sentence1'],'sentence2':row['sentence2'],'label':row['label'],
          'prediction':int(prediction),'method':'tfidf_lr','latency_ms':elapsed})
        probabilities.append({'id':row['id'],'probability':probability,'threshold':threshold})
    if len(predictions)!=4316 or len(probabilities)!=4316: raise AssertionError('prediction row count mismatch')
    with (out/'evaluation_predictions.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for r in predictions: f.write(json.dumps(r,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    with (out/'evaluation_probabilities.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for r in probabilities: f.write(json.dumps(r,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    pred=[r['prediction'] for r in predictions]
    lr_metrics,sklearn_metrics=compare_sklearn(labels,pred)
    allzero=[0]*len(labels); zero_metrics,zero_sk=compare_sklearn(labels,allzero)
    metrics={'config_id':frozen['config_id'],'threshold_index':frozen['threshold_index'],'threshold':threshold,'metric_labels':[0,1],
      'matrix_orientation':'rows=true label, columns=predicted label; order [0,1]','LR':lr_metrics,
      'all_zero_reference':zero_metrics,'sklearn_crosscheck':{'LR':sklearn_metrics,'all_zero_reference':zero_sk},
      'confusion_count_sum':sum(lr_metrics[k] for k in ('TN','FP','FN','TP')),'FP_plus_FN':lr_metrics['FP']+lr_metrics['FN']}
    if metrics['confusion_count_sum']!=4316 or metrics['FP_plus_FN']!=lr_metrics['error_count']: raise AssertionError('metric counts mismatch')
    write_json(out/'evaluation_metrics.json',metrics)
    hw,hw_error=hardware(); sampled=datetime.now(BEIJING_TZ).isoformat()
    latency={'count':len(latencies),'unit':'milliseconds','method':'perf_counter_ns, one independent pair per timed call',
      'mean_ms':statistics.fmean(latencies),'median_ms':statistics.median(latencies),'p95_ms':float(np.percentile(latencies,95,method='linear')),
      'p95_definition':'NumPy percentile with method=linear','min_ms':min(latencies),'max_ms':max(latencies),
      'warmup':warmup,'timing_includes':['text preprocessing performed by vectorizer','two TF-IDF transforms','sparse feature construction','LR predict_proba','threshold decision'],
      'timing_excludes':['model loading','file read/write','metric scoring','prediction record assembly','acceptance verification'],
      'hardware_sampled_at':sampled,'hardware':hw,'hardware_query_error':hw_error,'gpu_used':False,
      'gpu_note':'scikit-learn liblinear CPU inference; no GPU path invoked'}
    write_json(out/'latency_summary.json',latency)
    # Independent batch/swap/label checks are outside formal latency measurements.
    batch,batch_pred,features,zeros=predict_batch(vectorizer,model,pos,rows,threshold)
    saved=np.asarray([x['probability'] for x in probabilities],dtype=np.float64); singlepred=np.asarray(pred,dtype=np.int8)
    np.testing.assert_allclose(batch,saved,rtol=1e-12,atol=1e-12)
    if not np.array_equal(batch_pred,singlepred): raise AssertionError('timed single predictions differ from batch predictions')
    swapped=[{'sentence1':r['sentence2'],'sentence2':r['sentence1']} for r in rows]
    swap_p,swap_pred,_,swap_zero=predict_batch(vectorizer,model,pos,swapped,threshold)
    np.testing.assert_allclose(batch,swap_p,rtol=1e-12,atol=1e-12)
    if not np.array_equal(batch_pred,swap_pred): raise AssertionError('sentence swap changes predictions')
    label_flipped=[dict(r,label=1-r['label']) for r in rows]
    flip_p,flip_pred,_,_=predict_batch(vectorizer,model,pos,label_flipped,threshold)
    np.testing.assert_array_equal(batch,flip_p)
    if not np.array_equal(batch_pred,flip_pred): raise AssertionError('label mutation changes predictions')
    # Fresh independent interpreter reload and full-set re-inference.
    cmd=[sys.executable,'-B',str(Path(__file__).with_name('verify_lr_stage3_fresh.py')),'--config',str(config_path),
      '--evaluation',str(eval_path),'--predictions',str(out/'evaluation_predictions.jsonl'),'--probabilities',str(out/'evaluation_probabilities.jsonl')]
    child_env=dict(os.environ,PYTHONIOENCODING='utf-8')
    fresh=subprocess.run(cmd,capture_output=True,cwd=str(Path(__file__).resolve().parents[1]),env=child_env)
    fresh_stdout=fresh.stdout.decode('utf-8',errors='replace'); fresh_stderr=fresh.stderr.decode('utf-8',errors='replace')
    (out/'fresh_process_stdout.txt').write_text(fresh_stdout,encoding='utf-8'); (out/'fresh_process_stderr.txt').write_text(fresh_stderr,encoding='utf-8')
    if fresh.returncode: raise RuntimeError(f'fresh verification failed ({fresh.returncode}): {fresh_stderr}')
    fresh_result=json.loads(fresh_stdout.strip().splitlines()[-1])
    # A second invocation targeting this occupied directory must fail before writing.
    sentinel=out/'overwrite_sentinel.bin'; sentinel.write_bytes(b'preserve-lr-stage3-output')
    sentinel_hash=sha(sentinel)
    collision_cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--config',str(config_path),'--evaluation',str(eval_path),'--output-dir',str(out)]
    collision=subprocess.run(collision_cmd,capture_output=True,cwd=str(Path(__file__).resolve().parents[1]),env=child_env)
    collision_stdout=collision.stdout.decode('utf-8',errors='replace'); collision_stderr=collision.stderr.decode('utf-8',errors='replace')
    if collision.returncode==0 or 'refusing to overwrite existing output directory' not in collision_stderr or sha(sentinel)!=sentinel_hash:
        raise AssertionError(f'existing output directory was not safely refused; exit={collision.returncode}; stderr={collision_stderr}')
    (out/'overwrite_test.json').write_text(json.dumps({'exit_code':collision.returncode,'stdout':collision_stdout,'stderr':collision_stderr,
      'sentinel_sha256_before':sentinel_hash,'sentinel_sha256_after':sha(sentinel),'sentinel_unchanged':sha(sentinel)==sentinel_hash},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Strict output and record alignment checks.
    for original,pr,prob in zip(rows,predictions,probabilities):
        if set(pr)!={'id','sentence1','sentence2','label','prediction','method','latency_ms'}: raise AssertionError('prediction fields mismatch')
        if type(pr['prediction']) is not int or pr['method']!='tfidf_lr': raise AssertionError('prediction type/method mismatch')
        if (pr['id'],pr['sentence1'],pr['sentence2'],pr['label'])!=(original['id'],original['sentence1'],original['sentence2'],original['label']): raise AssertionError('record alignment mismatch')
        if pr['prediction']!=int(prob['probability']>=threshold): raise AssertionError('prediction threshold rule mismatch')
        if type(prob['probability']) not in (int,float) or not math.isfinite(prob['probability']) or not 0<=prob['probability']<=1: raise AssertionError('invalid output probability')
        if type(pr['latency_ms']) not in (int,float) or not math.isfinite(pr['latency_ms']) or pr['latency_ms']<0: raise AssertionError('invalid latency')
    verification={'status':'passed','evaluation':{'rows':len(rows),'label_counts':counts,'sha256':eval_hash,'strict_four_fields':True,'ids_unique':True,'utf8_validated_by_loader':True},
      'frozen_config_matches_protocol':True,'config_id':frozen['config_id'],'threshold':threshold,
      'single_vs_batch':{'probability_max_abs_error':float(np.max(np.abs(saved-batch))),'prediction_mismatches':int(np.count_nonzero(singlepred!=batch_pred))},
      'sentence_swap':{'probability_max_abs_error':float(np.max(np.abs(batch-swap_p))),'prediction_mismatches':int(np.count_nonzero(batch_pred!=swap_pred))},
      'label_flip':{'probability_max_abs_error':float(np.max(np.abs(batch-flip_p))),'prediction_mismatches':int(np.count_nonzero(batch_pred!=flip_pred))},
      'fresh_process':fresh_result,'probability_tolerance':{'rtol':1e-12,'atol':1e-12},
      'zero_vectors':zeros,'prediction_rows':len(predictions),'probability_rows':len(probabilities),
      'prediction_records_aligned':True,'probability_finite_range_checked':True,'latencies_finite_nonnegative':True,
      'metric_counts_sum':metrics['confusion_count_sum'],'overwrite_test':{'exit_code':collision.returncode,'refused':True,'sentinel_unchanged':True}}
    protected_after={str(p):sha(p) for p in protect}
    if protected_before!=protected_after: raise AssertionError('protected train/validation/evaluation/stage1/stage2 inputs changed')
    verification['protection_scope']={'mode':'complete','checked_file_count':len(protect),'paths':[str(p) for p in protect],
      'hashes_before_after_recorded':True,'all_hashes_unchanged':protected_before==protected_after}
    verification['protected_file_hashes']={'before':protected_before,'after':protected_after,'unchanged':True}
    write_json(out/'verification.json',verification)
    write_json(out/'protected_hashes.json',{'before':protected_before,'after':protected_after,'unchanged':True})
    print(json.dumps({'status':'passed','config_id':frozen['config_id'],'threshold':threshold,'rows':len(rows),
      'macro_f1':lr_metrics['macro_f1'],'accuracy':lr_metrics['accuracy'],'output_dir':str(out)},ensure_ascii=False))

if __name__=='__main__': main()
