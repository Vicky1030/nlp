"""Fresh-process end-to-end checks of the frozen selected configuration."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from data_loader import load_data
from lr_stage2_inference import load_selected,infer_batch,infer_one

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--probabilities',type=Path,required=True); p.add_argument('--predictions',type=Path,required=True); a=p.parse_args()
    config,vectorizer,model=load_selected(a.config)
    rows=load_data(a.validation); savedp=[json.loads(x) for x in a.probabilities.read_text(encoding='utf-8').splitlines()]
    savedy=[json.loads(x) for x in a.predictions.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==len(savedp)==len(savedy)==3433
    assert [(x['id'],x['label']) for x in savedp]==[(r['id'],r['label']) for r in rows]
    assert [(x['id'],x['label']) for x in savedy]==[(r['id'],r['label']) for r in rows]
    threshold=config['threshold']; batch,batchpred,_=infer_batch(vectorizer,model,rows,threshold)
    saved=np.asarray([x['probability'] for x in savedp],dtype=np.float64)
    prediction=np.asarray([x['prediction'] for x in savedy],dtype=np.int8)
    np.testing.assert_allclose(batch,saved,rtol=1e-12,atol=1e-12)
    assert np.array_equal(batchpred,prediction)
    single=[]; singlepred=[]
    for row in rows:
        prob,pred,_=infer_one(vectorizer,model,row,threshold); single.append(prob); singlepred.append(pred)
    single=np.asarray(single); singlepred=np.asarray(singlepred)
    np.testing.assert_allclose(single,batch,rtol=1e-12,atol=1e-12)
    assert np.array_equal(singlepred,batchpred)
    swapped=[dict(r,sentence1=r['sentence2'],sentence2=r['sentence1']) for r in rows]
    swap_prob,swap_pred,_=infer_batch(vectorizer,model,swapped,threshold)
    np.testing.assert_allclose(swap_prob,batch,rtol=1e-12,atol=1e-12)
    assert np.array_equal(swap_pred,batchpred)
    changed=[dict(r,label=1-r['label']) for r in rows]
    label_prob,label_pred,_=infer_batch(vectorizer,model,changed,threshold)
    np.testing.assert_array_equal(label_prob,batch); assert np.array_equal(label_pred,batchpred)
    print(json.dumps({'status':'passed','config_id':config['config_id'],'rows':len(rows),
      'saved_probability_max_abs_error':float(np.max(np.abs(batch-saved))),
      'single_batch_max_abs_error':float(np.max(np.abs(single-batch))),
      'swap_max_abs_error':float(np.max(np.abs(swap_prob-batch))),
      'relabel_max_abs_error':float(np.max(np.abs(label_prob-batch))),
      'single_batch_prediction_mismatches':int(np.count_nonzero(singlepred!=batchpred)),
      'swap_prediction_mismatches':int(np.count_nonzero(swap_pred!=batchpred)),
      'relabel_prediction_mismatches':int(np.count_nonzero(label_pred!=batchpred)),
      'classes':model.classes_.tolist(),'positive_class':1,
      'model_path':config['model_relative_path'],'vectorizer_path':config['vectorizer_relative_path']},ensure_ascii=False))

if __name__=='__main__': main()
