"""Score the frozen 18 LR candidates over 102 thresholds and freeze one choice."""
import argparse,csv,hashlib,json,math,shutil,subprocess,sys,time,importlib.metadata
from datetime import datetime,timedelta,timezone
from pathlib import Path
from fractions import Fraction
import numpy as np
import joblib
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from lr_stage2_inputs import validate_inputs,sha256
from lr_stage2_scoring import threshold_records,select_best,displayed
from lr_stage2_tests import run_contract_tests
from lr_stage2_inference import load_selected,infer_batch,infer_one
BEIJING_TZ=timezone(timedelta(hours=8), name='Asia/Shanghai')

THRESHOLD_FIELDS=['config_id','representation','C','class_weight','threshold_index','threshold','TN','FP','FN','TP','Accuracy','Precision','Recall','F1','Macro-F1']
MODEL_FIELDS=['config_id','representation','C','class_weight','threshold_index','threshold','TN','FP','FN','TP','Accuracy','Precision','Recall','F1','Macro-F1','vocabulary_size','feature_dimension_2V','training_elapsed_seconds','n_iter','max_iter','convergence_status','training_warning_categories']

def dump_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def validate_stage1_acceptance(path):
    report=json.loads(Path(path).read_text(encoding='utf-8'))
    if report.get('status')!='passed' or len(report.get('probability_reloads',[]))!=18:
        raise ValueError('stage-1 acceptance report is not a passed 18-model reload report')
    if any(not x.get('passed') or x.get('max_absolute_error',math.inf)>0 for x in report['probability_reloads']):
        raise ValueError('stage-1 reload acceptance contains a failure')
    return report

def write_csv(path,fields,rows,header=True):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
        if header: writer.writeheader()
        for row in rows:
            clean={k:row.get(k) for k in fields}
            if clean.get('class_weight') is None: clean['class_weight']='None'
            for k,v in list(clean.items()):
                if isinstance(v,Fraction): clean[k]=str(v)
            writer.writerow(clean)

def crosscheck_sklearn(rows,labels,probabilities):
    by_config={}
    for row in rows: by_config.setdefault(row['config_id'],[]).append(row)
    for cid,group in by_config.items():
        probs=probabilities[cid]
        for row in group:
            pred=(probs>=row['threshold']).astype(np.int8)
            expected={
              'TN':int(((np.asarray(labels)==0)&(pred==0)).sum()),'FP':int(((np.asarray(labels)==0)&(pred==1)).sum()),
              'FN':int(((np.asarray(labels)==1)&(pred==0)).sum()),'TP':int(((np.asarray(labels)==1)&(pred==1)).sum()),
              'Accuracy':accuracy_score(labels,pred), 'Precision':precision_score(labels,pred,zero_division=0),
              'Recall':recall_score(labels,pred,zero_division=0), 'F1':f1_score(labels,pred,zero_division=0),
              'Macro-F1':f1_score(labels,pred,labels=[0,1],average='macro',zero_division=0),
              'confusion_matrix':confusion_matrix(labels,pred,labels=[0,1]).tolist()}
            for key in ('TN','FP','FN','TP'):
                if row[key]!=expected[key]: raise AssertionError(f'{cid} threshold {row["threshold"]}: {key} mismatch')
            for key in ('Accuracy','Precision','Recall','F1','Macro-F1'):
                if not math.isclose(row[key],expected[key],rel_tol=1e-14,abs_tol=1e-14): raise AssertionError(f'{cid} threshold {row["threshold"]}: {key} mismatch')

def choose_rows(validated):
    labels=[r['label'] for r in validated['validation']]
    probs={}; all_rows=[]; model_rows=[]; best_each={}
    for cid,entry in validated['configs'].items():
        probs[cid]=np.asarray(entry['probabilities'],dtype=np.float64)
        rows=threshold_records(entry['config'],labels,probs[cid]); all_rows.extend(rows)
        best=select_best(rows); best_each[cid]=best
        model_rows.append(best)
    if len(all_rows)!=1836: raise AssertionError(f'expected 1836 threshold records, got {len(all_rows)}')
    if any(len([r for r in all_rows if r['config_id']==cid])!=102 for cid in validated['configs']): raise AssertionError('each config must have 102 thresholds')
    direct=select_best(all_rows); staged=select_best(list(best_each.values()))
    if direct['config_id']!=staged['config_id'] or direct['threshold_index']!=staged['threshold_index']:
        raise AssertionError('two-step model-then-threshold choice differs from direct six-level selection')
    crosscheck_sklearn(all_rows,labels,probs)
    return all_rows,model_rows,direct,probs

def selected_inference(root,out,selected,validation,saved_probability):
    cfg_path=out/'selected_config.json'; config,vectorizer,model=load_selected(cfg_path)
    threshold=config['threshold']; batch_prob,batch_pred,X=infer_batch(vectorizer,model,validation,threshold)
    warm_count=min(10,len(validation)); warmup=[]
    for row in validation[:warm_count]:
        p,pred,ms=infer_one(vectorizer,model,row,threshold); warmup.append({'probability':p,'prediction':pred})
    singles=[]; prediction_rows=[]; probability_rows=[]
    for row in validation:
        p,pred,latency=infer_one(vectorizer,model,row,threshold)
        singles.append((p,pred,latency))
        prediction_rows.append({'id':row['id'],'sentence1':row['sentence1'],'sentence2':row['sentence2'],'label':row['label'],
          'prediction':int(float(saved_probability[len(prediction_rows)])>=threshold),'method':'tfidf_lr','latency_ms':latency})
        probability_rows.append({'id':row['id'],'label':row['label'],'probability':float(saved_probability[len(probability_rows)])})
    single_prob=np.asarray([x[0] for x in singles]); single_pred=np.asarray([x[1] for x in singles])
    max_batch=float(np.max(np.abs(batch_prob-single_prob)))
    if max_batch>0 or not np.array_equal(batch_pred,single_pred): raise AssertionError('single/batch inference mismatch')
    saved=np.asarray(saved_probability,dtype=np.float64)
    max_saved=float(np.max(np.abs(batch_prob-saved)))
    np.testing.assert_allclose(batch_prob,saved,rtol=1e-12,atol=1e-12)
    if not np.array_equal(single_pred,(saved>=threshold).astype(np.int8)):
        raise AssertionError('saved probability decisions differ from single-model decisions')
    swapped=[dict(r,sentence1=r['sentence2'],sentence2=r['sentence1']) for r in validation]
    swap_prob,swap_pred,_=infer_batch(vectorizer,model,swapped,threshold)
    max_swap=float(np.max(np.abs(batch_prob-swap_prob)))
    if max_swap>0 or not np.array_equal(batch_pred,swap_pred): raise AssertionError('sentence swap changed inference')
    relabelled=[dict(r,label=1-r['label']) for r in validation]
    label_prob,label_pred,_=infer_batch(vectorizer,model,relabelled,threshold)
    max_label=float(np.max(np.abs(batch_prob-label_prob)))
    if max_label>0 or not np.array_equal(batch_pred,label_pred): raise AssertionError('memory label changed inference')
    with (out/'validation_probabilities.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for row in probability_rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    with (out/'validation_predictions.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for row in prediction_rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    return {'batch_single_max_abs_error':max_batch,'saved_probability_max_abs_error':max_saved,
      'swapped_order_max_abs_error':max_swap,'relabelled_max_abs_error':max_label,
      'batch_single_prediction_mismatches':int(np.count_nonzero(batch_pred!=single_pred)),
      'swapped_prediction_mismatches':int(np.count_nonzero(batch_pred!=swap_pred)),
      'relabelled_prediction_mismatches':int(np.count_nonzero(batch_pred!=label_pred)),
      'probability_tolerance':{'rtol':1e-12,'atol':1e-12},'rows':len(validation),
      'latency_ms_scope':'single-record text processing, two transforms, sparse features, predict_proba, threshold decision; excludes model loading, file I/O, scoring and record assembly',
      'warmup':{'fixed_record_count':warm_count,'performed_before_timing':True,'completed':True},
      'single_latency_ms':{'count':len(singles),'min':float(min(x[2] for x in singles)),'median':float(np.median([x[2] for x in singles])),'max':float(max(x[2] for x in singles))},
      'prediction_fields':['id','sentence1','sentence2','label','prediction','method','latency_ms'],
      'method':'tfidf_lr','selected_artifact_load':'relative selected_config paths'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--acceptance',type=Path,required=True)
    ap.add_argument('--data-dir',type=Path,default=None)
    ap.add_argument('--vectorizer-root',type=Path,default=None)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args(); root=args.root.resolve(); run=args.run.resolve(); acceptance=args.acceptance.resolve(); out=args.output_dir.resolve()
    if out.exists(): raise FileExistsError(f'refusing to write existing output directory; choose a new --output-dir: {out}')
    started=datetime.now(BEIJING_TZ).isoformat()
    acceptance_report=validate_stage1_acceptance(acceptance)
    validated=validate_inputs(root,run,args.data_dir,args.vectorizer_root)
    before=dict(validated['input_hashes'])
    before[str(validated['train_path'])]=sha256(validated['train_path'])
    before[str(validated['validation_path'])]=validated['validation_sha256']
    tests=run_contract_tests()
    all_rows,model_rows,chosen,probabilities=choose_rows(validated)
    labels=[r['label'] for r in validated['validation']]
    if sum(chosen[k] for k in ('TN','FP','FN','TP'))!=len(labels): raise AssertionError('selected confusion counts do not sum to N')
    out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'threshold_comparison.csv',THRESHOLD_FIELDS,[displayed(r) for r in all_rows])
    summary=[]
    stage1_warnings=validated['run_metadata'].get('warnings',[])
    cats=sorted({w.get('category','unknown') for w in stage1_warnings})
    has_conv=any('ConvergenceWarning' in w.get('category','') for w in stage1_warnings)
    for row in model_rows:
        entry=validated['configs'][row['config_id']]; scheme=row['representation']; meta=validated['vectorizers'][scheme]['metadata']
        actual_model=joblib.load(entry['model']);
        expected_params={'solver':'liblinear','penalty':'l2','C':entry['config']['C'],
          'class_weight':entry['config']['class_weight'],'fit_intercept':True,'intercept_scaling':1,
          'dual':False,'max_iter':5000,'tol':1e-4,'random_state':42}
        actual_params=actual_model.get_params(deep=False)
        if any(actual_params.get(k)!=v for k,v in expected_params.items()):
            raise ValueError(f'{row["config_id"]}: fitted model parameters differ from frozen LR protocol')
        sr=dict(row); sr.update({'vocabulary_size':meta['vocabulary_size'],'feature_dimension_2V':2*meta['vocabulary_size'],
          'training_elapsed_seconds':entry['config']['elapsed_seconds'],'n_iter':json.dumps(entry['config']['n_iter']),
          'max_iter':actual_model.max_iter,'convergence_status':'passed' if not has_conv and max(entry['config']['n_iter'])<actual_model.max_iter else 'unresolved',
          'training_warning_categories':';'.join(cats)})
        if sr['convergence_status']!='passed': raise ValueError(f'{row["config_id"]}: unresolved convergence status')
        summary.append(sr)
    write_csv(out/'model_summary.csv',MODEL_FIELDS,[displayed(r) for r in summary],header=True)
    # Stage two artifacts are independent byte-for-byte copies; source remains untouched.
    scheme=chosen['representation']; entry=validated['configs'][chosen['config_id']]; vm=validated['vectorizers'][scheme]
    selected_dir=out/'artifacts'/'selected'; (selected_dir/'models').mkdir(parents=True); (selected_dir/'vectorizers').mkdir(); (selected_dir/'python').mkdir()
    source_model=entry['model']; source_vector=vm['path']
    shutil.copy2(source_model,selected_dir/'models'/'lr_model.joblib')
    shutil.copy2(source_vector,selected_dir/'vectorizers'/'tfidf_vectorizer.joblib')
    shutil.copy2(root/'src'/'tfidf_common.py',selected_dir/'python'/'tfidf_common.py')
    shutil.copy2(root/'src'/'data_loader.py',selected_dir/'python'/'data_loader.py')
    for source,target in ((source_model,selected_dir/'models'/'lr_model.joblib'),(source_vector,selected_dir/'vectorizers'/'tfidf_vectorizer.joblib')):
        if sha256(source)!=sha256(target): raise AssertionError(f'copied artifact hash mismatch: {source}')
    loaded=joblib.load(selected_dir/'models'/'lr_model.joblib'); vector=joblib.load(selected_dir/'vectorizers'/'tfidf_vectorizer.joblib')
    params=loaded.get_params(deep=False); classes=loaded.classes_.tolist(); positive=list(loaded.classes_).index(1)
    config={
      'status':'frozen','config_id':chosen['config_id'],'representation':scheme,
      'selected_artifacts_dir':'artifacts/selected','model_relative_path':'models/lr_model.joblib',
      'vectorizer_relative_path':'vectorizers/tfidf_vectorizer.joblib',
      'class':{'model':f'{loaded.__class__.__module__}.{loaded.__class__.__qualname__}','vectorizer':f'{vector.__class__.__module__}.{vector.__class__.__qualname__}',
        'tokenizer':'tfidf_common.jieba_tokens' if scheme=='word_1' else None,'helper_module':'tfidf_common.py'},
      'dependencies':{'python':validated['run_metadata']['python'],'packages':{n:importlib.metadata.version(n) for n in ('scikit-learn','numpy','scipy','jieba','joblib')}},
      'tfidf_parameters':vm['metadata']['parameters'],'feature_formula':'[abs(A-B); A.multiply(B)]',
      'feature_concatenation_order':['absolute_difference','elementwise_product'],'feature_dimension':2*len(vector.vocabulary_),
      'C':chosen['C'],'class_weight':chosen['class_weight'],'lr_parameters':params,'classes_':classes,
      'positive_label':1,'positive_probability_column_index':positive,'threshold_index':chosen['threshold_index'],'threshold':chosen['threshold'],
      'selection_rule':['Macro-F1 descending','Accuracy descending','threshold descending','C ascending','class_weight None before balanced','representation char_1, char_1_2, word_1'],
      'validation_metrics':{k:chosen[k] for k in ['TN','FP','FN','TP','Accuracy','Precision','Recall','F1','Macro-F1','confusion_matrix']},
      'exact_scores':{k:str(chosen[k]) for k in ['accuracy_exact','precision_exact','recall_exact','f1_exact','f1_0_exact','macro_f1_exact']},
      'data_sha256':{'validation':validated['validation_sha256'],'train':validated['run_metadata']['train_sha256']},
      'source_sha256':{'model':entry['config']['model_sha256'],'probabilities':entry['config']['probabilities_sha256'],'vectorizer':vm['sha256'],
        'copied_model':sha256(selected_dir/'models'/'lr_model.joblib'),'copied_vectorizer':sha256(selected_dir/'vectorizers'/'tfidf_vectorizer.joblib')},
      'source_model_relative_to_project':str(source_model.relative_to(root)).replace('\\','/') if source_model.is_relative_to(root) else str(source_model),
      'source_vectorizer_relative_to_project':str(source_vector.relative_to(root)).replace('\\','/') if source_vector.is_relative_to(root) else str(source_vector)}
    dump_json(out/'selected_config.json',config)
    inference=selected_inference(root,out,chosen,validated['validation'],probabilities[chosen['config_id']])
    # Selected metric reconstruction from frozen probabilities and predictions.
    chosen_prob=probabilities[chosen['config_id']]
    chosen_pred=(chosen_prob>=chosen['threshold']).astype(np.int8)
    target_pred=np.asarray([r['prediction'] for r in map(json.loads,(out/'validation_predictions.jsonl').read_text(encoding='utf-8').splitlines())])
    if not np.array_equal(chosen_pred,target_pred): raise AssertionError('saved selected predictions differ from saved probabilities at selected threshold')
    selected_conf={k:chosen[k] for k in ['TN','FP','FN','TP','Accuracy','Precision','Recall','F1','Macro-F1','confusion_matrix']}
    if sum(chosen[k] for k in ('TN','FP','FN','TP'))!=3433: raise AssertionError('selected confusion count mismatch')
    # Verify all required artifacts and helpers can be resolved in a new process.
    verifier=[sys.executable,'-B',str(root/'src'/'verify_lr_stage2_selected.py'),'--config',str(out/'selected_config.json'),
      '--validation',str(validated['validation_path']),'--probabilities',str(out/'validation_probabilities.jsonl'),
      '--predictions',str(out/'validation_predictions.jsonl')]
    child=subprocess.run(verifier,capture_output=True,text=True,encoding='utf-8',cwd=str(root))
    (out/'fresh_process_stdout.txt').write_text(child.stdout,encoding='utf-8')
    (out/'fresh_process_stderr.txt').write_text(child.stderr,encoding='utf-8')
    if child.returncode: raise RuntimeError(f'fresh process selected artifact check failed ({child.returncode}): {child.stderr}')
    fresh=json.loads(child.stdout.strip().splitlines()[-1])
    # Preserve only explicitly in-scope train/validation, TF-IDF, candidate model and probability inputs.
    after={str(validated['validation_path']):sha256(validated['validation_path']),
      str(validated['train_path']):sha256(validated['train_path'])}
    for scheme,data in validated['vectorizers'].items(): after[str(data['path'])]=sha256(data['path'])
    for cid,data in validated['configs'].items():
        after[str(data['model'])]=sha256(data['model']); after[str(data['probability_path'])]=sha256(data['probability_path'])
    if before!=after: raise AssertionError('stage-1 inputs or outputs changed during stage 2')
    # Actual no-overwrite execution test against an existing directory.
    probe=out/'overwrite_probe'; probe.mkdir()
    sentinel=probe/'sentinel.bin'; sentinel.write_bytes(b'preserve-stage2-output')
    probe_hash=sha256(sentinel)
    collision_cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(root),'--run',str(run),'--acceptance',str(acceptance),'--output-dir',str(probe)]
    collision=subprocess.run(collision_cmd,capture_output=True,text=True,encoding='utf-8',cwd=str(root))
    if collision.returncode==0 or 'refusing to write existing output directory' not in collision.stderr or sha256(sentinel)!=probe_hash:
        raise AssertionError('existing output directory was not rejected safely')
    (out/'overwrite_probe_result.json').write_text(json.dumps({'exit_code':collision.returncode,'stderr':collision.stderr,
      'sentinel_sha256_before':probe_hash,'sentinel_sha256_after':sha256(sentinel),'unchanged':sha256(sentinel)==probe_hash},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    finished=datetime.now(BEIJING_TZ).isoformat()
    report={'status':'frozen','started_at':started,'finished_at':finished,'timezone':'Asia/Shanghai / UTC+08:00',
      'scope':'stage 2 only; no training/refit, evaluation/Test, fixed-threshold experiment, or stage 3',
      'input_validation':{'validation_rows':len(validated['validation']),'label_counts':validated['validation_label_counts'],
        'validation_sha256':validated['validation_sha256'],'configs':18,'probability_rows_each':3433,
        'all_index_hashes_match':True,'stage1_acceptance_passed':True},
      'candidate_scoring':{'threshold_indices':list(range(102)),'threshold_values':['index / 100'],'rows':len(all_rows),
        'each_configuration_threshold_count':102,'all_confusion_counts_sum_to_n':True,'independent_sklearn_metric_crosscheck':'passed for all 1836 rows',
        'score_order_uses_fraction_exact_values':True,'contract_tests':tests,
        'two_step_equals_direct':True,'selected_config_id':chosen['config_id'],'selected_threshold_index':chosen['threshold_index'],
        'selected_threshold':chosen['threshold'],'selected_exact_scores':config['exact_scores'],'selection_rule':config['selection_rule']},
      'selected_inference':inference,'fresh_process_verification':fresh,
      'overwrite_test':{'exit_code':collision.returncode,'refused':True,'sentinel_sha256_unchanged':True},
      'stage1_before_hashes':before,'stage1_after_hashes':after,'stage1_inputs_outputs_unchanged':before==after,
      'artifacts':{'threshold_comparison_rows':len(all_rows),'model_summary_rows':len(summary),
        'all_18_fitted_models_match_frozen_parameters':True,
        'validation_predictions_rows':len(validated['validation']),'selected_model_copy_sha256_matches':True,
        'selected_vectorizer_copy_sha256_matches':True},
      'evaluation_status':'not used','stage3_status':'not started'}
    dump_json(out/'selection_report.json',report)
    readme='''# TF-IDF + LR stage 2 (frozen)\n\n本目录记录阶段二对已有18组阶段一模型及验证概率的评分与冻结结果；不重新训练LR、不重新拟合TF-IDF、不读取evaluation或官方Test。选中模型只在train上训练，validation未合并。\n\n完整复现请阅读项目根目录 `README.md` 的“Complete One-Command Reproduction”，并向统一入口传入本次实际存在的 `--data-dir`、`--vectorizer-root`、`--evaluation` 和全新 `--output-root`。不要使用历史运行目录名称。\n\n本阶段独立脚本支持 `--root`、`--run`、`--acceptance`、`--data-dir`、`--vectorizer-root` 和 `--output-dir`；所有输入路径都必须由调用者显式传入或使用脚本明确的项目内默认值。最终清单 `artifact_sha256.json` 在全部产物（含本说明）写入后生成，且不计入自身哈希。\n'''
    (out/'README_STAGE2.md').write_text(readme,encoding='utf-8')
    # Write the README before the final inventory so the manifest covers every
    # stage-2 artifact. The manifest excludes only itself.
    manifest={str(p.relative_to(out)).replace('\\','/'):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_sha256.json'}
    dump_json(out/'artifact_sha256.json',manifest)
    print(json.dumps({'status':'frozen','output_dir':str(out),'selected_config_id':chosen['config_id'],
      'threshold':chosen['threshold'],'macro_f1':chosen['Macro-F1'],'accuracy':chosen['Accuracy'],
      'scored_candidates':len(all_rows)},ensure_ascii=False))

if __name__=='__main__': main()
