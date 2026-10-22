"""Portable frozen TF-IDF + LR prediction; prediction functions accept text only."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from scipy.sparse import hstack

def load_frozen(config_path):
    config_path=Path(config_path).resolve()
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('status')!='frozen': raise ValueError('selected configuration is not frozen')
    artifact_root=(config_path.parent/config['selected_artifacts_dir']).resolve()
    helper_dir=artifact_root/'python'
    if helper_dir.is_dir(): sys.path.insert(0,str(helper_dir))
    model_path=(artifact_root/config['model_relative_path']).resolve()
    vectorizer_path=(artifact_root/config['vectorizer_relative_path']).resolve()
    import hashlib
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
        return h.hexdigest()
    if sha(model_path)!=config['source_sha256']['copied_model']:
        raise ValueError('selected LR model hash mismatch')
    if sha(vectorizer_path)!=config['source_sha256']['copied_vectorizer']:
        raise ValueError('selected TF-IDF vectorizer hash mismatch')
    model=joblib.load(model_path); vectorizer=joblib.load(vectorizer_path)
    params=model.get_params(deep=False)
    for key,value in config['lr_parameters'].items():
        if key in params and params[key]!=value: raise ValueError(f'LR parameter mismatch: {key}')
    if f'{model.__class__.__module__}.{model.__class__.__qualname__}'!=config['class']['model']:
        raise ValueError('LR class mismatch')
    if f'{vectorizer.__class__.__module__}.{vectorizer.__class__.__qualname__}'!=config['class']['vectorizer']:
        raise ValueError('TF-IDF class mismatch')
    actual=vectorizer.get_params(deep=False)
    for key,value in config['tfidf_parameters'].items():
        if key=='dtype': value=np.float64
        if key=='tokenizer' and value is None: continue
        actual_value=actual.get(key)
        if isinstance(actual_value,tuple): actual_value=list(actual_value)
        if actual_value!=value: raise ValueError(f'TF-IDF parameter mismatch: {key}; actual={actual_value!r}, configured={value!r}')
    if len(vectorizer.vocabulary_)*2!=config['feature_dimension']:
        raise ValueError('feature dimension 2V mismatch')
    classes=list(model.classes_)
    if classes!=config['classes_'] or config['positive_label']!=1:
        raise ValueError('model classes differ from frozen configuration')
    positive_column=classes.index(1)
    if positive_column!=config['positive_probability_column_index']:
        raise ValueError('positive-class probability column mismatch')
    return config,vectorizer,model,positive_column

def transform_pair(vectorizer,sentence1,sentence2):
    """Transform two texts with one frozen vectorizer and return one CSR row."""
    left=vectorizer.transform([sentence1])
    right=vectorizer.transform([sentence2])
    zero=(int(left.getnnz(axis=1)[0]==0),int(right.getnnz(axis=1)[0]==0))
    features=hstack((abs(left-right),left.multiply(right)),format='csr',dtype=np.float64)
    if features.shape[1]!=2*len(vectorizer.vocabulary_) or features.dtype!=np.float64:
        raise ValueError('invalid sparse pair feature shape or dtype')
    return features,zero

def predict_pair(vectorizer,model,positive_column,sentence1,sentence2,threshold):
    features,zero=transform_pair(vectorizer,sentence1,sentence2)
    probability=float(model.predict_proba(features)[0,positive_column])
    if not np.isfinite(probability) or not 0.0<=probability<=1.0:
        raise ValueError('invalid probability')
    prediction=int(probability>=threshold)
    return probability,prediction,zero

def predict_batch(vectorizer,model,positive_column,rows,threshold):
    left=vectorizer.transform([r['sentence1'] for r in rows])
    right=vectorizer.transform([r['sentence2'] for r in rows])
    zero_left=np.asarray(left.getnnz(axis=1)==0)
    zero_right=np.asarray(right.getnnz(axis=1)==0)
    features=hstack((abs(left-right),left.multiply(right)),format='csr',dtype=np.float64)
    probabilities=model.predict_proba(features)[:,positive_column]
    if not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any():
        raise ValueError('invalid batch probability')
    predictions=(probabilities>=threshold).astype(np.int8)
    zeros={'sentence1_zero':int(zero_left.sum()),'sentence2_zero':int(zero_right.sum()),
      'either_zero':int((zero_left|zero_right).sum()),'both_zero':int((zero_left&zero_right).sum())}
    return probabilities,predictions,features,zeros
