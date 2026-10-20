"""Stage 2: deterministic AFQMC preparation; no models, downloads or predictions."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import itertools
import json
from pathlib import Path
import platform
import random
import sys
import unicodedata

import inspect_data as audit_tool
from data_loader import load_data, strict_json, validate_record

BASE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def jsonl(rows):
    return ''.join(encode(r)+'\n' for r in rows).encode('utf-8')


def pair(row):
    return tuple(sorted((row['sentence1'], row['sentence2'])))


def distribution(rows):
    counts = Counter(r['label'] for r in rows)
    return {'records': len(rows), 'labels': {str(k): {'count': counts[k], 'ratio': counts[k]/len(rows) if rows else None} for k in (0, 1)},
            'identical_sentence_records': sum(r['sentence1'] == r['sentence2'] for r in rows)}


def check_sets(sets, sources):
    result = {'within': {}, 'between': {}}
    indexes = {}
    for name, rows in sets.items():
        idx = defaultdict(list)
        for r in rows:
            idx[pair(r)].append(r)
        indexes[name] = idx
        groups = []
        conflicts = []
        for key, members in idx.items():
            detail = {'pair': key, 'sources': [sources[r['id']] for r in members]}
            if len(members) > 1:
                groups.append(detail)
            if len({r['label'] for r in members}) > 1:
                conflicts.append(detail)
        result['within'][name] = {'duplicate_groups': len(groups), 'excess_records': sum(len(idx[tuple(g['pair'])])-1 for g in groups), 'groups': groups,
                                 'conflict_groups': len(conflicts), 'conflicts': conflicts}
    for left, right in itertools.combinations(sets, 2):
        keys = sorted(indexes[left].keys() & indexes[right].keys())
        result['between'][left+'--'+right] = {'overlap_groups': len(keys), 'groups': [
            {'pair': k, 'left_sources': [sources[r['id']] for r in indexes[left][k]],
             'right_sources': [sources[r['id']] for r in indexes[right][k]],
             'label_conflict': len({r['label'] for r in indexes[left][k]+indexes[right][k]}) > 1} for k in keys]}
    return result


def prepare(args):
    audit_path = args.audit.resolve()
    old = json.loads(audit_path.read_text(encoding='utf-8'))
    paths = {n: (getattr(args, n) or Path(old['files'][n]['path'])).resolve() for n in ('train', 'dev', 'test')}
    output = args.output_root.resolve()
    report_dir = output/'reports'
    protected = set(paths.values()) | {audit_path, audit_path.with_suffix('.md'), BASE/'src/inspect_data.py'}
    names = ['data/processed/'+n+'.jsonl' for n in ('train','validation','evaluation')]
    names += ['reports/'+n for n in ('processing_changes.jsonl','data_provenance.jsonl','data_preparation.json','data_preparation.md','data_manifest.json','preparation_failure.json')]
    if any((output/n).resolve() in protected for n in names):
        raise ValueError('Output path aliases protected input/audit')
    original = {n: sha(p) for n,p in paths.items()}
    phase1 = {n: {'path': str(paths[n]), 'phase1_path': old['files'][n]['path'], 'expected_sha256': old['files'][n]['sha256'],
                   'actual_sha256': original[n], 'sha256_matches': original[n] == old['files'][n]['sha256'],
                   'path_matches': paths[n] == Path(old['files'][n]['path']).resolve()} for n in paths}
    # Always re-read/re-audit actual input, even if the phase-1 hashes match.
    raw, fresh = {}, {}
    for name, path in paths.items():
        meta, rows = audit_tool.read(path, name)
        fresh[name] = audit_tool.audit(meta, rows, name, 200)
        raw[name] = rows
    changed = any(not v['sha256_matches'] for v in phase1.values())
    report_dir.mkdir(parents=True, exist_ok=True)
    def stop(message, details):
        failure = {'status':'failed','message':message,'details':details,'phase1_comparison':phase1,'fresh_input_audit':fresh}
        target = report_dir/'preparation_failure.json'
        target.write_bytes((json.dumps(failure,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
        raise ValueError(message+'; see '+str(target))
    for name, meta in fresh.items():
        if meta.get('fatal_error') or meta['anomalous_records'] or meta['duplicate_json_keys'] or meta['blank_lines'] or meta['format'] != 'JSONL':
            stop('Unsupported/new input anomaly: '+name, {k:meta.get(k) for k in ('format','fatal_error','anomalies','parse_errors','duplicate_json_keys','blank_lines')})
    # Only known Train leading U+FEFF is removable. Other new Cc/Cf/Cs requires review.
    allowed_warnings = {(w['source']['line'], note['field'], ch['offset_0based'], ch['codepoint'])
                        for w in old['files']['train']['warnings'] for note in w['notes']
                        for ch in note.get('characters', []) if ch['codepoint']=='U+FEFF' and ch['offset_0based']==0}
    changes=[]; sources={}; converted={}; conversions=Counter()
    for name in ('train','dev'):
        rows=[]
        for rawrow in raw[name]:
            o=rawrow['data']; line=rawrow['source']['line']; location=f'{paths[name]}:{line}'
            if set(o) != {'sentence1','sentence2','label'}:
                stop('Unexpected fields; refusing to discard: '+location, o)
            rid=line if name=='train' else len(raw['train'])+line
            source={'id':rid,'original_file':str(paths[name]),'original_line':line}
            sources[rid]=source
            v=o['label']
            if type(v) is int and v in (0,1): lab=v
            elif type(v) is str and v in ('0','1'): lab=int(v); conversions[name]+=1
            else: stop('Invalid label: '+location,o)
            r={'id':rid,'sentence1':o['sentence1'],'sentence2':o['sentence2'],'label':lab}
            for field in ('sentence1','sentence2'):
                text=r[field]
                for pos,ch in enumerate(text):
                    if unicodedata.category(ch) in ('Cc','Cf','Cs'):
                        known = name=='train' and (line,field,pos,'U+%04X'%ord(ch)) in allowed_warnings and pos==0 and ch=='\ufeff'
                        if not known: stop('Unapproved invisible/control character: '+location,{'field':field,'offset':pos,'codepoint':'U+%04X'%ord(ch)})
                if name=='train' and text.startswith('\ufeff'):
                    after=text[1:]
                    changes.append({'action':'remove_leading_U+FEFF','source':source,'field':field,'before':text,'after':after,
                                    'before_escaped':ascii(text),'after_escaped':ascii(after),'reason':'Explicit stage-2 rule; origin/cause not established'})
                    r[field]=after
            validate_record(r,location)
            rows.append(r)
        converted[name]=rows
    for line,field,pos,codepoint in allowed_warnings:
        # Verify the actual known location rather than assuming the old warning still applies.
        matches=[r for r in raw['train'] if r['source']['line']==line]
        if not matches or not matches[0]['data'][field].startswith('\ufeff'):
            stop('Phase-1 expected leading U+FEFF not present',{'line':line,'field':field})
    pre=check_sets(converted,sources)
    if any(v['conflict_groups'] for v in pre['within'].values()): stop('Label conflict after text processing',pre)
    if pre['within']['dev']['duplicate_groups']: stop('Dev duplicate: cannot change official membership',pre)
    if pre['between']['train--dev']['overlap_groups']: stop('Train overlaps Dev after processing',pre)
    kept=[]; seen={}; deleted={}
    for r in converted['train']:
        k=pair(r)
        if k in seen:
            retain=seen[k]
            if retain['label']!=r['label']: stop('Train conflicting labels',{'kept':retain,'removed':r})
            deleted[r['id']]=retain['id']
            changes.append({'action':'deduplicate_train','removed_source':sources[r['id']],'retained_source':sources[retain['id']],
                            'removed_record':r,'retained_record':retain,'reason':'Same unordered processed sentence pair and label; retain earliest original line'})
        else: seen[k]=r; kept.append(r)
    ratio=Decimal(args.validation_ratio)
    if not ratio.is_finite() or not Decimal(0)<ratio<Decimal(1): stop('validation-ratio must be between 0 and 1',args.validation_ratio)
    rng=random.Random(args.seed); training=[]; validation=[]; allocations={}
    for lab in (0,1):
        members=sorted((r for r in kept if r['label']==lab),key=lambda r:r['id'])
        n=int((Decimal(len(members))*ratio).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
        if n==0 or n==len(members): stop('Class too small for requested split',{'label':lab,'n':len(members),'validation':n})
        rng.shuffle(members)
        validation.extend(members[:n]); training.extend(members[n:])
        allocations[str(lab)]={'total':len(members),'validation':n,'train':len(members)-n}
    sets={'train':training,'validation':validation,'evaluation':converted['dev']}
    sets={n:sorted(rows,key=lambda r:r['id']) for n,rows in sets.items()}
    post=check_sets(sets,sources)
    if any(v['duplicate_groups'] or v['conflict_groups'] for v in post['within'].values()) or any(v['overlap_groups'] for v in post['between'].values()): stop('Final pair checks failed',post)
    all_ids=[r['id'] for rows in sets.values() for r in rows]
    if len(set(all_ids))!=len(all_ids): stop('Global ID collision',all_ids)
    dev_ok=all((a['sentence1'],a['sentence2'],a['label'])==(b['data']['sentence1'],b['data']['sentence2'],int(b['data']['label'])) for a,b in zip(sets['evaluation'],raw['dev'])) and len(sets['evaluation'])==len(raw['dev'])
    if not dev_ok: stop('Dev invariant failed',{})
    final={r['id']:n for n,rows in sets.items() for r in rows}
    provenance=[]
    for rid,source in sorted(sources.items()):
        entry=dict(source,final_set=final.get(rid,'excluded_duplicate'))
        if rid in deleted: entry['retained_id']=deleted[rid]
        provenance.append(entry)
    after={n:sha(p) for n,p in paths.items()}
    if original!=after: stop('Raw input changed during run',{'before':original,'after':after})
    cfg={'validation_ratio':str(ratio),'seed':args.seed,'dedup_key':'exact processed unordered pair; no strip',
         'text_rule':'Only remove one known leading U+FEFF in Train sentence field; all other text preserved',
         'id_rule':'Train id=original physical line; Dev id=original Train record count+original physical line',
         'split_algorithm':'Classes 0 then 1; sort by id; shared random.Random(seed).shuffle per class; first n to validation',
         'rounding':'Per-class n=Decimal(class_count)*Decimal(validation_ratio), ROUND_HALF_UP',
         'serialization':'UTF-8 no BOM; LF; compact JSON; keys id,sentence1,sentence2,label; ascending id; terminal newline'}
    versions={'python':platform.python_version(),'implementation':platform.python_implementation(),'dependencies':'Python standard library only; no pip packages'}
    rep={'status':'passed','phase1_comparison':phase1,'input_sha256_after':after,'fresh_input_summary':{n:{k:f[k] for k in ('total_records','valid_records','anomalous_records','label_distribution','fields')} for n,f in fresh.items()},
         'input_changed_since_phase1':changed,'config':cfg,'versions':versions,'input_counts':{n:len(v) for n,v in raw.items()},
         'retained_official_train':distribution(kept),'outputs':{n:distribution(v) for n,v in sets.items()},'allocations':allocations,
         'text_modifications':sum(c['action']=='remove_leading_U+FEFF' for c in changes),'deleted_records':len(deleted),'label_type_conversions':dict(conversions),
         'pre_dedup_checks':pre,'final_checks':post,'invariants':{'raw_hashes_unchanged':original==after,'global_ids_unique':True,'dev_text_membership_order_label_meaning_unchanged':dev_ok,
         'train_count_reconciles':len(raw['train'])-len(deleted)==len(training)+len(validation),'all_four_fields_valid':True},
         'test_policy':'Raw Test preserved; no labeled version, split membership or local metrics',
         'unknowns':{'download_source':'待补充','official_version_authenticity':'待补充','license_authorization':'待补充','U+FEFF_origin_cause':'尚未确认'},
         'scope':'Stage 2 preparation and basic checks only; no models/predictions/results analysis; full reproducibility acceptance deferred to stage 3'}
    if changed:
        rep['fresh_input_audit']=fresh
        rep['phase1_statistics_differences']={n:{k:{'phase1':old['files'][n].get(k),'current':fresh[n].get(k)} for k in ('total_records','label_distribution','fields','warnings') if old['files'][n].get(k)!=fresh[n].get(k)} for n in paths}
    buffers={'data/processed/'+n+'.jsonl':jsonl(v) for n,v in sets.items()}
    buffers['reports/processing_changes.jsonl']=jsonl(changes)
    buffers['reports/data_provenance.jsonl']=jsonl(provenance)
    buffers['reports/data_preparation.json']=(json.dumps(rep,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    lines=['# 第二阶段：统一数据准备与正式划分','','状态：实际处理及基本检查通过；未训练模型、未生成预测。第一阶段报告未覆盖。',
           '', '## 实现与配置','',json.dumps(cfg,ensure_ascii=False,indent=2),'',json.dumps(versions,ensure_ascii=False),
           '', '## 数量与标签分布','','|集合|记录数|label=0|label=1|正类比例|','|---|---:|---:|---:|---:|']
    for n,d in rep['outputs'].items(): lines.append(f'|{n}|{d["records"]}|{d["labels"]["0"]["count"]}|{d["labels"]["1"]["count"]}|{d["labels"]["1"]["ratio"]:.8%}|')
    lines+=['','原始输入数量：'+encode(rep['input_counts'])+'；去重后官方 Train：'+str(len(kept))+'。',
            '','Train 对账：'+str(len(raw['train']))+' − '+str(len(deleted))+' = '+str(len(training))+' + '+str(len(validation))+'。',
            '','## 修改与删除','', '文本修改 '+str(rep['text_modifications'])+' 处；同标签无向重复删除 '+str(len(deleted))+' 条；标签类型转换次数：'+encode(dict(conversions))+'。',
            '', '以下删除仅指处理副本排除，原始数据没有删除。保留原始句子方向；两句相同与共享单句不作为删除理由。',
            '', '~~~json',json.dumps(changes,ensure_ascii=False,indent=2),'~~~',
            '', '## 输入完整性与第一阶段衔接','', '所有输入均重新实际读取并检查。哈希发生变化：'+str(changed)+'；原始文件运行前后未变：'+str(original==after)+'。',
            '', '~~~json',json.dumps(phase1,ensure_ascii=False,indent=2),'~~~',
            '', '## 处理后与输出检查','', '处理后、去重前检查与最终检查（来源可追溯）：', '', '~~~json',json.dumps({'before_dedup':pre,'final':post,'invariants':rep['invariants']},ensure_ascii=False,indent=2),'~~~',
            '', 'Test 只读保留；处理后有标签数据与原始 Test 的无向句子对重叠统计见下一节。',
            '', '## 使用边界与待补充事项','',
            '- TF-IDF 词表和 IDF 只在内部 train 拟合。Prompt、阈值、参数在 validation 选择；evaluation 不用于训练、选择示例或反复调参。',
            '- Test 无真实标签，不生成有标签版本，不参与内部划分或本地指标评估。',
            '- 下载来源、版本真实性和许可：待补充。U+FEFF 来源及成因尚未确认。',
            '- 未执行第三阶段全面复现验收；本次执行实际生成与基本检查。',
            '- 全部来源映射见 data_provenance.jsonl（含被排除记录及 retained_id），修改/删除明细见 processing_changes.jsonl；输入输出 SHA-256 见 data_manifest.json。']
    test_pairs={tuple(sorted((r['data']['sentence1'],r['data']['sentence2']))) for r in raw['test']}
    test_overlap={n:len({pair(r) for r in rows}&test_pairs) for n,rows in {'processed_official_train':kept,'evaluation':sets['evaluation']}.items()}
    rep['processed_vs_raw_test_overlap_groups']=test_overlap
    buffers['reports/data_preparation.json']=(json.dumps(rep,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    lines+=['','## 与原始 Test 重叠复查','',encode(test_overlap)+'。计数为不同原文无向句子对数，未使用 Test 标签，未删除共享单句。']
    buffers['reports/data_preparation.md']=('\n'.join(lines)+'\n').encode('utf-8')
    manifest={'schema_version':1,'config':cfg,'versions':versions,'inputs':phase1,'phase1_audit_sha256':sha(audit_path),
              'code_sha256':{n:sha(Path(__file__).parent/n) for n in ('prepare_data.py','data_loader.py','inspect_data.py')},
              'outputs':{n:{'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)} for n,b in buffers.items()},
              'note':'Manifest excludes its own hash to avoid self-reference. No volatile timestamps in generated files.',
              'unknowns':rep['unknowns']}
    buffers['reports/data_manifest.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    # All checks above complete before publishing any processed dataset.
    for name,b in buffers.items():
        target=output/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(b)
    for n,expected in sets.items():
        actual=load_data(output/'data/processed'/f'{n}.jsonl')
        if actual!=expected: stop('Output readback mismatch',n)
    if {n:sha(p) for n,p in paths.items()}!=original: stop('Raw changed during publish',{})
    print(json.dumps({'outputs':rep['outputs'],'text_modifications':rep['text_modifications'],'deleted_records':len(deleted),'invariants':rep['invariants']},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,default=BASE/'reports/data_audit.json')
    for n in ('train','dev','test'): p.add_argument('--'+n,type=Path)
    p.add_argument('--output-root',type=Path,default=BASE)
    p.add_argument('--validation-ratio',default='0.1')
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    try: prepare(args)
    except (ValueError,OSError,KeyError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr)
        return 1
    return 0


if __name__=='__main__': sys.exit(main())

