"""Fresh-process verification of frozen selected artifacts and output files."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from data_loader import load_data
from lr_stage3_runtime import load_frozen,predict_batch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True,type=Path); p.add_argument('--evaluation',required=True,type=Path)
    p.add_argument('--predictions',required=True,type=Path); p.add_argument('--probabilities',required=True,type=Path); a=p.parse_args()
    config,v,m,pos=load_frozen(a.config); rows=load_data(a.evaluation)
    pred=[json.loads(x) for x in a.predictions.read_text(encoding='utf-8').splitlines()]
    probs=[json.loads(x) for x in a.probabilities.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==len(pred)==len(probs)==4316
    assert [(r['id'],r['sentence1'],r['sentence2'],r['label']) for r in rows]==[(r['id'],r['sentence1'],r['sentence2'],r['label']) for r in pred]
    assert [(r['id']) for r in rows]==[r['id'] for r in probs]
    batch,batchpred,_,zeros=predict_batch(v,m,pos,rows,config['threshold'])
    saved=np.asarray([r['probability'] for r in probs]); outputpred=np.asarray([r['prediction'] for r in pred])
    np.testing.assert_allclose(batch,saved,rtol=1e-12,atol=1e-12)
    assert np.array_equal(batchpred,outputpred)
    swapped=[{'sentence1':r['sentence2'],'sentence2':r['sentence1']} for r in rows]
    swap,batchswap,_,_=predict_batch(v,m,pos,swapped,config['threshold'])
    np.testing.assert_allclose(batch,swap,rtol=1e-12,atol=1e-12); assert np.array_equal(batchpred,batchswap)
    flipped=[dict(r,label=1-r['label']) for r in rows]
    labelp,labely,_,_=predict_batch(v,m,pos,flipped,config['threshold'])
    np.testing.assert_array_equal(batch,labelp); assert np.array_equal(batchpred,labely)
    print(json.dumps({'status':'passed','config_id':config['config_id'],'rows':len(rows),
      'saved_probability_max_abs_error':float(np.max(np.abs(batch-saved))),
      'swap_probability_max_abs_error':float(np.max(np.abs(batch-swap))),
      'label_flip_probability_max_abs_error':float(np.max(np.abs(batch-labelp))),
      'saved_prediction_mismatches':int(np.count_nonzero(batchpred!=outputpred)),
      'swap_prediction_mismatches':int(np.count_nonzero(batchpred!=batchswap)),
      'label_flip_prediction_mismatches':int(np.count_nonzero(batchpred!=labely)),
      'zero_vectors':zeros},ensure_ascii=False))

if __name__=='__main__': main()
