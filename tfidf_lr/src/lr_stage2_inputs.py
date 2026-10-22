"""Read-only validation of the frozen stage-1 inputs and artifacts."""
import hashlib, json
from pathlib import Path
from data_loader import load_data
from run_tfidf_lr_stage1 import SCHEMES,CS,WEIGHTS,config_id
from index_paths import load_model_index, resolve_index_path

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def validate_inputs(root, run, data_dir=None, vectorizer_root=None):
    root=Path(root).resolve(); run=Path(run).resolve()
    data_dir=Path(data_dir).resolve() if data_dir is not None else root/'data/processed'
    vectorizer_root=Path(vectorizer_root).resolve() if vectorizer_root is not None else root/'artifacts/tfidf_cosine'
    train_path=data_dir/'train.jsonl'
    validation_path=data_dir/'validation.jsonl'
    missing=[str(p) for p in (train_path, validation_path, vectorizer_root) if not p.exists()]
    if missing: raise FileNotFoundError('missing explicit stage-2 input(s):\n'+'\n'.join(missing))
    validation=load_data(validation_path)
    counts={y:sum(r['label']==y for r in validation) for y in (0,1)}
    if len(validation)!=3433 or counts!={0:2376,1:1057}:
        raise ValueError(f'validation mismatch: rows={len(validation)}, labels={counts}')
    run_meta=json.loads((run/'run_metadata.json').read_text(encoding='utf-8'))
    val_hash=sha256(validation_path)
    if val_hash!=run_meta['validation_sha256']:
        raise ValueError('validation SHA-256 differs from stage-1 run_metadata')
    index_path=run/'model_index.json'; idx=load_model_index(index_path)
    expected={config_id(s,c,w) for s in SCHEMES for c in CS for w in WEIGHTS}
    actual=[c['config_id'] for c in idx['configs']]
    if idx['count']!=18 or len(actual)!=18 or len(set(actual))!=18 or set(actual)!=expected:
        raise ValueError('stage-1 model index configuration set is incomplete or duplicated')
    by_id={c['config_id']:c for c in idx['configs']}
    records={}; file_hashes={}; vector_metadata={}
    ids_labels=[(r['id'],r['label']) for r in validation]
    for cid in sorted(expected):
        config=by_id[cid]
        if type(config['class_weight']) not in (str,type(None)):
            raise ValueError(f'{cid}: invalid class_weight type')
        if config['class_weight'] not in WEIGHTS:
            raise ValueError(f'{cid}: unsupported class_weight')
        model=resolve_index_path(index_path,idx,config['model'],'model')
        probs=resolve_index_path(index_path,idx,config['probabilities'],'probabilities')
        if not model.is_file() or not probs.is_file(): raise FileNotFoundError(f'{cid}: model/probability missing')
        mh=sha256(model); ph=sha256(probs)
        if mh!=config['model_sha256'] or ph!=config['probabilities_sha256']:
            raise ValueError(f'{cid}: model/probability hash differs from stage-1 index')
        lines=probs.read_text(encoding='utf-8').splitlines()
        if len(lines)!=3433: raise ValueError(f'{cid}: probability row count {len(lines)}')
        parsed=[]; values=[]
        for n,line in enumerate(lines,1):
            row=json.loads(line)
            if set(row)!={'config_id','id','label','probability'}: raise ValueError(f'{cid}:{n}: fields mismatch')
            if row['config_id']!=cid: raise ValueError(f'{cid}:{n}: config_id mismatch')
            if (row['id'],row['label'])!=ids_labels[n-1]: raise ValueError(f'{cid}:{n}: validation order/ID/label mismatch')
            value=float(row['probability'])
            if not (value==value and abs(value)!=float('inf') and 0<=value<=1): raise ValueError(f'{cid}:{n}: invalid probability')
            parsed.append(row); values.append(value)
        records[cid]={'config':config,'model':model,'probability_path':probs,'rows':parsed,'probabilities':values}
        file_hashes[str(model)]=mh; file_hashes[str(probs)]=ph
    for scheme in SCHEMES:
        metadata_path=vectorizer_root/scheme/'metadata.json'
        vector_path=vectorizer_root/scheme/'vectorizer.joblib'
        meta=json.loads(metadata_path.read_text(encoding='utf-8'))
        vh=sha256(vector_path)
        if meta['model_sha256']!=vh: raise ValueError(f'{scheme}: vectorizer hash differs from TF-IDF metadata')
        if meta['train_sha256']!=run_meta['train_sha256']: raise ValueError(f'{scheme}: TF-IDF train hash mismatch')
        vector_metadata[scheme]={'metadata_path':metadata_path,'metadata':meta,'path':vector_path,'sha256':vh}
        file_hashes[str(vector_path)]=vh
    return {'root':root,'run':run,'train_path':train_path,'validation_path':validation_path,'validation':validation,
      'validation_sha256':val_hash,'validation_label_counts':counts,'run_metadata':run_meta,
      'index':idx,'configs':records,'vectorizers':vector_metadata,'input_hashes':file_hashes}
