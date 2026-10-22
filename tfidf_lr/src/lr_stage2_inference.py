"""Selected-model inference helpers shared by the stage-2 runner and fresh-process audit."""
import json, sys, time
from pathlib import Path
import joblib, numpy as np
from scipy.sparse import hstack

def load_selected(config_path):
    config_path=Path(config_path).resolve()
    config=json.loads(config_path.read_text(encoding='utf-8'))
    artifact_root=(config_path.parent/config['selected_artifacts_dir']).resolve()
    helper=artifact_root/'python'
    sys.path.insert(0,str(helper))
    vectorizer=joblib.load(artifact_root/config['vectorizer_relative_path'])
    model=joblib.load(artifact_root/config['model_relative_path'])
    return config,vectorizer,model

def features(vectorizer, rows):
    a=vectorizer.transform([r['sentence1'] for r in rows])
    b=vectorizer.transform([r['sentence2'] for r in rows])
    return hstack((abs(a-b),a.multiply(b)),format='csr')

def positive_column(model):
    return list(model.classes_).index(1)

def infer_one(vectorizer,model,row,threshold):
    start=time.perf_counter()
    X=features(vectorizer,[row])
    probability=float(model.predict_proba(X)[0,positive_column(model)])
    prediction=int(probability>=threshold)
    elapsed=(time.perf_counter()-start)*1000
    return probability,prediction,elapsed

def infer_batch(vectorizer,model,rows,threshold):
    X=features(vectorizer,rows)
    probabilities=model.predict_proba(X)[:,positive_column(model)]
    predictions=(probabilities>=threshold).astype(np.int8)
    return probabilities,predictions,X

