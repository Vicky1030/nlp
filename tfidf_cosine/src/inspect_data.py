"""Read-only AFQMC audit; Python standard library; no cleaning or splitting."""
import argparse, collections as C, datetime as dt, hashlib, itertools, json, math, sys, unicodedata
from pathlib import Path

def dump(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def typed(x):
    return type(x).__name__+':'+dump(x)

def load(s):
    keys=[]
    def hook(pairs):
        d={}
        for k,v in pairs:
            if k in d: keys.append(k)
            d[k]=v
        return d
    def reject(s): raise ValueError('Nonstandard constant '+s)
    return json.loads(s,object_pairs_hook=hook,parse_constant=reject),keys

def read(path,name):
    raw=path.read_bytes()
    m={'path':str(path.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'parse_errors':[],'blank_lines':[],'duplicate_json_keys':[]}
    rows=[]
    try: text=raw.decode('utf-8-sig')
    except UnicodeDecodeError as e:
        m.update(format='unreadable',fatal_error=str(e),total_records=None)
        return m,rows
    m['utf8_bom']=raw.startswith(bytes([239,187,191]))
    if text.lstrip().startswith('['):
        m['format']='JSON array'
        try:
            values,keys=load(text)
            rows=[{'source':{'file':name,'record':i},'data':o} for i,o in enumerate(values,1)]
            if keys: m['duplicate_json_keys'].append({'scope':'array','keys':keys})
        except ValueError as e:
            m['parse_errors'].append({'line':getattr(e,'lineno',None),'column':getattr(e,'colno',None),'error':str(e)})
            m.update(total_records=None,fatal_error='Malformed array: count not safely recoverable')
    else:
        try:
            value,keys=load(text)
            m['format']='JSON single value'
            rows=[{'source':{'file':name,'record':1,'line':1},'data':value}]
            if keys: m['duplicate_json_keys'].append({'line':1,'keys':keys})
        except ValueError:
            m['format']='JSONL'
            lines=text.split('\n')
            for i,line in enumerate(lines,1):
                if not line.strip():
                    if i<len(lines): m['blank_lines'].append(i)
                    continue
                try:
                    o,keys=load(line)
                    loc={'file':name,'record':len(rows)+len(m['parse_errors'])+1,'line':i}
                    rows.append({'source':loc,'data':o})
                    if keys: m['duplicate_json_keys'].append({'line':i,'keys':keys})
                except ValueError as e:
                    m['parse_errors'].append({'line':i,'column':getattr(e,'colno',None),'error':str(e)})
    m.setdefault('total_records',len(rows)+len(m['parse_errors']))
    m['parsed_records']=len(rows)
    return m,rows

def label(o):
    v=o.get('label')
    if type(v) is int and v in (0,1): return v
    if type(v) is str and v in ('0','1'): return int(v)
    return None

def stats(a):
    a=sorted(a)
    if not a: return {'n':0}
    def q(p):
        pos=(len(a)-1)*p
        lo,hi=math.floor(pos),math.ceil(pos)
        return a[lo]+(a[hi]-a[lo])*(pos-lo)
    return {'n':len(a),'min':a[0],'max':a[-1],'mean':sum(a)/len(a),'quantiles':{str(p):q(p) for p in [0,.25,.5,.75,.9,.95,.99,1]}}

def groups(idx):
    g=[{'key':k,'count':len(v),'sources':v} for k,v in idx.items() if len(v)>1]
    return {'group_count':len(g),'excess_records':sum(x['count']-1 for x in g),'participating_records':sum(x['count'] for x in g),'groups':g}

def index(rows,trim=False,unordered=False):
    idx=C.defaultdict(list)
    for r in rows:
        o=r['data']
        if not isinstance(o,dict) or not all(isinstance(o.get(k),str) for k in ('sentence1','sentence2')): continue
        pair=[o['sentence1'],o['sentence2']]
        if trim: pair=[s.strip() for s in pair]
        if unordered: pair.sort()
        e=dict(r['source'],label_present='label' in o,semantic_label=label(o))
        if 'label' in o: e.update(label=o['label'],label_type=type(o['label']).__name__)
        idx[tuple(pair)].append(e)
    return idx

def conflicts(idx):
    raw=[]; sem=[]
    for k,v in idx.items():
        vals={typed(e['label']) for e in v if e['label_present']}
        labels={e['semantic_label'] for e in v if e['semantic_label'] is not None}
        if len(vals)>1: raw.append({'pair':k,'labels':sorted(vals),'sources':v})
        if len(labels)>1: sem.append({'pair':k,'labels':sorted(labels),'sources':v})
    return {'typed_conflict_groups':len(raw),'typed_conflicts':raw,'semantic_conflict_groups':len(sem),'semantic_conflicts':sem}

def audit(m,rows,name,threshold):
    fields=C.defaultdict(lambda:{'present':0,'types':C.Counter()})
    labs=C.Counter(); ids=C.defaultdict(list); missing_id=missing_label=valid=0
    lens={k:[] for k in ('sentence1','sentence2')}
    anomalies=[]; warnings=[]; same={'raw':[],'strip':[]}
    for r in rows:
        o,loc=r['data'],r['source']; errors=[]; notes=[]
        if not isinstance(o,dict): errors.append('record_not_object')
        else:
            for k,v in o.items():
                fields[k]['present']+=1; fields[k]['types'][type(v).__name__]+=1
            for k in lens:
                if k not in o: errors.append(k+':missing')
                elif not isinstance(o[k],str): errors.append(k+':not_string')
                else:
                    s=o[k]; lens[k].append(len(s))
                    if not s.strip(): errors.append(k+':empty_or_whitespace')
                    if s!=s.strip(): notes.append({'field':k,'kind':'edge_whitespace'})
                    chars=[{'offset_0based':i,'codepoint':'U+%04X'%ord(c),'category':unicodedata.category(c)} for i,c in enumerate(s) if unicodedata.category(c) in ('Cc','Cf','Cs')]
                    if chars: notes.append({'field':k,'kind':'control_format_surrogate','characters':chars})
                    if len(s)>threshold: notes.append({'field':k,'kind':'long_text','length':len(s)})
            if 'label' not in o:
                missing_label+=1
                if name in ('train','dev'): errors.append('label:missing')
            else:
                labs[typed(o['label'])]+=1
                if label(o) is None: errors.append('label:not_int_or_exact_string_01')
            if 'id' not in o: missing_id+=1
            else: ids[typed(o['id'])].append(loc)
            if all(isinstance(o.get(k),str) for k in lens):
                if o['sentence1']==o['sentence2']: same['raw'].append(loc)
                if o['sentence1'].strip()==o['sentence2'].strip(): same['strip'].append(loc)
        if errors: anomalies.append({'source':loc,'issues':errors,'record':o})
        else: valid+=1
        if notes: warnings.append({'source':loc,'notes':notes,'record':o})
    n=m['total_records']
    m.update(valid_records=valid,anomalous_records=len(anomalies)+len(m['parse_errors']) if n is not None else None,
             fields=dict(fields),other_fields={k:v for k,v in fields.items() if k not in ('sentence1','sentence2','label','id')},
             label_distribution=[{'typed_value':k,'count':v,'ratio_all_records':v/n if n else None} for k,v in sorted(labs.items())],
             missing_label_records=missing_label,missing_id_records=missing_id,id_types=dict(C.Counter(k.split(':',1)[0] for k in ids)),
             duplicate_ids=groups(ids),lengths={k:stats(v) for k,v in lens.items()},identical_sentences={k:{'count':len(v),'sources':v} for k,v in same.items()},
             anomalies=anomalies,warnings=warnings,warning_record_count=len(warnings),warning_event_counts=dict(C.Counter(e['kind'] for w in warnings for e in w['notes'])))
    m['duplicates']={}
    for mode,trim in [('raw',False),('strip',True)]:
        whole=C.defaultdict(list)
        for r in rows:
            o=r['data']
            if trim and isinstance(o,dict): o={k:v.strip() if k in lens and isinstance(v,str) else v for k,v in o.items()}
            whole[dump(o)].append(r['source'])
        d=index(rows,trim); u=index(rows,trim,True)
        m['duplicates'][mode]={'whole_record':groups(whole),'directed_pair':groups(d),'unordered_pair':groups(u),'directed_conflicts':conflicts(d),'unordered_conflicts':conflicts(u)}
    return m

def cross(data):
    out={}
    for left,right in itertools.combinations(data,2):
        modes={}
        for mode,trim in [('raw',False),('strip',True)]:
            m={}
            for kind,unordered in [('directed',False),('unordered',True)]:
                a=index(data[left],trim,unordered); b=index(data[right],trim,unordered); shared=sorted(a.keys()&b.keys())
                m[kind]={'shared_pair_groups':len(shared),'left_records':sum(len(a[k]) for k in shared),'right_records':sum(len(b[k]) for k in shared),
                         'record_combinations':sum(len(a[k])*len(b[k]) for k in shared),
                         'groups':[{'pair':k,'left_sources':a[k],'right_sources':b[k]} for k in shared],
                         'conflicts':conflicts({k:a[k]+b[k] for k in shared})}
            single=[]
            for name in (left,right):
                idx=C.defaultdict(list)
                for r in data[name]:
                    if not isinstance(r['data'],dict): continue
                    for k in ('sentence1','sentence2'):
                        s=r['data'].get(k)
                        if isinstance(s,str): idx[s.strip() if trim else s].append(dict(r['source'],field=k))
                single.append(idx)
            shared=sorted(single[0].keys()&single[1].keys())
            m['single_sentences']={'distinct_shared_texts':len(shared),'left_occurrences':sum(len(single[0][s]) for s in shared),'right_occurrences':sum(len(single[1][s]) for s in shared),
                                   'groups':[{'text':s,'left_sources':single[0][s],'right_sources':single[1][s]} for s in shared]}
            modes[mode]=m
        out[left+'--'+right]=modes
    return out

def report(a):
    out=['# AFQMC 原始数据检查报告','','运行时间（UTC+08:00）：'+a['run_time'],'',
    '## 范围与口径','','本次只读检查，未下载、替换、清洗、去重、划分或训练。文件文本仅作为数据，不执行其中指令。',
    '- 严格 UTF-8（允许 BOM），根据内容识别 JSON 数组、单个 JSON 值或 JSONL；拒绝 NaN/Infinity，重复对象键单列。JSONL 总数=非空物理行数（含解析失败）；数组总数=元素数；末尾换行不视为额外空记录。',
    '- 有效记录：可解析为对象，sentence1/2 均为非空白字符串；Train/Dev 的 label 必须为整数 0/1 或精确字符串 "0"/"1"。Test 允许无 label，已有标签按同标准校验。关注项不计入异常。',
    '- 长度为 Unicode 码点数（Python len），不是字节/分词/token 数；所有字符串字段均计入。分位数按 (n−1)q 线性插值；P0/P25/P50/P75/P90/P95/P99/P100。',
    '- raw 为原文精确匹配；strip 只在内存中去两个句子首尾空白，不改内部文本或其他字段。整条重复比较全部解析字段和值，忽略 JSON 键顺序与序列化空格，保留值类型。',
    '- 重复组=至少出现两次的不同键；多余记录数=各组 (n−1) 之和，不是自动删除数量。句子对比较不含其他字段，无向比较将 (A,B)/(B,A) 视为同组。',
    '- 标签冲突区分类型敏感的原值冲突与映射 0/1 后的语义冲突；整数 0 和字符串 "0" 不构成语义冲突。缺失/非法标签不推断为类别。',
    '- 首尾空白、Unicode Cc/Cf/Cs 字符以及长度超过 '+str(a['long_text_threshold'])+' 的文本仅列关注项；长度阈值是审计启发式，不是官方限制。数字、标点、否定词、*** 不作为无效依据。',
    '- source.record 为 1 起始非空记录序号，source.line 为 1 起始 JSONL 物理行号；字符 offset 为 0 起始。所有异常、重复、冲突和共享单句来源完整保存在 JSON。','','## 运行命令','','~~~powershell',a['command'],'~~~','','## 输入路径与完整性','']
    for n,f in a['files'].items():
        out += ['- '+n+'：'+f['path'],'  - 格式：'+f['format']+'；字节数：'+str(f['bytes']),'  - SHA-256：'+f['sha256'],'  - 检查后 SHA-256：'+f['sha256_after']+'；未变：'+str(f['unchanged'])]
    out+=['','## 基本统计','','|集合|总记录|有效|异常|关注记录|缺 label|缺 id|相同句子 raw/strip|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for n,f in a['files'].items():
        out.append('|'+ '|'.join(map(str,[n,f['total_records'],f['valid_records'],f['anomalous_records'],f['warning_record_count'],f['missing_label_records'],f['missing_id_records'],str(f['identical_sentences']['raw']['count'])+'/'+str(f['identical_sentences']['strip']['count'])]))+'|')
    for n,f in a['files'].items():
        out+=['','### '+n,'','字段出现次数/类型：'+dump(f['fields']),'','其他字段：'+dump(f['other_fields']),'','标签分布（比例分母为全部记录）：'+dump(f['label_distribution'])+'。无标签不视为负类。',
        '','解析失败 '+str(len(f['parse_errors']))+'；空白行 '+dump(f['blank_lines'])+'；重复 JSON 键 '+dump(f['duplicate_json_keys'])+'。',
        '','原始 id 重复组/多余记录：'+str(f['duplicate_ids']['group_count'])+'/'+str(f['duplicate_ids']['excess_records'])+'。id 类型（不同类型化 id 值的数量）：'+dump(f['id_types']),
        '','关注项字段事件计数（可重叠）：'+dump(f['warning_event_counts']),'','|字段|n|min|max|mean|P0/P25/P50/P75/P90/P95/P99/P100|','|---|---:|---:|---:|---:|---|']
        for k,v in f['lengths'].items():
            if v['n']: out.append('|'+k+'|'+str(v['n'])+'|'+str(v['min'])+'|'+str(v['max'])+'|'+format(v['mean'],'.6f')+'|'+' / '.join(format(x,'g') for x in v['quantiles'].values())+'|')
    out+=['','## 集合内重复与冲突','','|集合|口径|整条组/多余|同向组/多余|无向组/多余|同向语义冲突组|无向语义冲突组|','|---|---|---:|---:|---:|---:|---:|']
    for n,f in a['files'].items():
        for mode,d in f['duplicates'].items():
            cells=[str(d[k]['group_count'])+'/'+str(d[k]['excess_records']) for k in ('whole_record','directed_pair','unordered_pair')]
            out.append('|'+ '|'.join([n,mode]+cells+[str(d['directed_conflicts']['semantic_conflict_groups']),str(d['unordered_conflicts']['semantic_conflict_groups'])])+'|')
    out+=['','## 集合间重叠','','共享单句按不同文本数统计，不等同句子对重复或必须删除的数据。跨集合对数为不同键数，另报两侧涉及记录数及 JSON 中的记录组合数。','','|集合对|口径|同向对|无向对|无向左/右记录|共享单句|无向语义冲突组|','|---|---|---:|---:|---:|---:|---:|']
    for n,modes in a['cross_sets'].items():
        for mode,d in modes.items():
            u=d['unordered']
            out.append('|'+ '|'.join(map(str,[n,mode,d['directed']['shared_pair_groups'],u['shared_pair_groups'],str(u['left_records'])+'/'+str(u['right_records']),d['single_sentences']['distinct_shared_texts'],u['conflicts']['semantic_conflict_groups']]))+'|')
    out+=['','## 问题与后续约定核对','']+['- '+s for s in a['recommendations']]
    out+=['','## 尚无法确认','','- 下载来源、官方版本/真实性、许可授权和语义标签正确性无法仅凭本地内容确认。请补充原始下载链接、版本日期、许可或授权材料；本次不下载替换，也不据文件名或记录数断言为官方原版。','- 长度统计不是模型上下文长度；是否需要截断应待模型与 tokenization 确定后评估。','','## 来源追溯附录','','以下列出解析失败、异常、关注项、重复与冲突完整定位。共享单句和相同句子逐条定位见同目录 data_audit.json。']
    for n,f in a['files'].items():
        detail={k:f[k] for k in ('parse_errors','anomalies','warnings','duplicate_json_keys','duplicate_ids','duplicates')}
        out+=['','### '+n,'','~~~json',json.dumps(detail,ensure_ascii=False,indent=2),'~~~']
    for n,modes in a['cross_sets'].items():
        detail={mode:{k:v for k,v in d.items() if k!='single_sentences'} for mode,d in modes.items()}
        out+=['','### '+n,'','~~~json',json.dumps(detail,ensure_ascii=False,indent=2),'~~~']
    return '\n'.join(out)+'\n'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('train','dev','test'): p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('reports'))
    p.add_argument('--long-text-threshold',type=int,default=200)
    args=p.parse_args(); paths={n:getattr(args,n).resolve() for n in ('train','dev','test')}
    targets=[(args.output_dir/s).resolve() for s in ('data_audit.json','data_audit.md')]
    if any(t in paths.values() for t in targets): p.error('Output must not overwrite an input')
    files={}; data={}
    for n,path in paths.items():
        m,rows=read(path,n); data[n]=rows; files[n]=audit(m,rows,n,args.long_text_threshold)
    overlaps=cross(data)
    bad=any(f['anomalous_records'] or f.get('fatal_error') or f['duplicate_json_keys'] for f in files.values())
    rec=['结构检查'+('有异常，需先解决异常再划分。' if bad else '通过，具备进入下一阶段的基础条件；须先确定重复/冲突处理规则，再划分训练。')]
    for n,f in files.items():
        d=f['duplicates']['strip']; u=d['unordered_pair']
        rec.append(n+'：strip 无向句子对重复 '+str(u['group_count'])+' 组、多余 '+str(u['excess_records'])+' 条；语义标签冲突 '+str(d['unordered_conflicts']['semantic_conflict_groups'])+' 组。重复不等于本次删除授权；冲突需核查来源，不应直接多数表决。')
    u=overlaps['train--dev']['strip']['unordered']
    rec.append('Train–Dev strip 无向重叠 '+str(u['shared_pair_groups'])+' 组，涉及 Train '+str(u['left_records'])+' / Dev '+str(u['right_records'])+' 条。'+('下一阶段建议保持官方 Dev 不变，从 Train 排除重叠记录并保留来源映射；跨集合标签冲突需先核查，不能擅改 Dev。' if u['shared_pair_groups'] else '未检出同句子对泄漏；共享单句不直接删除。'))
    for n,f in files.items():
        for w in f['warnings']:
            rec.append(n+' 关注项来源 '+dump(w['source'])+'：'+dump(w['notes'])+'。下一阶段需核查成因后明确是否处理，当前未修改。')
    rec += ['仅从处理重复与冲突后的 Train，按标签比例约 90%/10% 进行分层分组划分；固定随机种子 42。建议确认后采用 strip 无向句子对键为组，保证交换句子顺序的同一对不跨内部训练/验证。组约束及整数取整可能无法精确达到比例，应记录实际比例、算法与版本。本阶段未执行。',
    '用户指定的官方 Dev 保留为最终评估集，不参与训练、提示词选择、调参或早停；本阶段不改动。其官方来源仍需外部材料核验。',
    'Test 已解析记录中缺 label '+str(files['test']['missing_label_records'])+' 条；无真实标签时暂不用于本地指标评估，不能用预测替代真实标签。',
    '下一阶段将精确字符串标签 "0"/"1" 显式转换为整数 0/1，原始文件保留不变。其他字段保留或另存来源映射，不未经说明丢弃。',
    '后续有标签数据每行采用 {"id":1,"sentence1":"示例句子一","sentence2":"示例句子二","label":1}。处理完成后统一分配全局唯一整数 id，保存原始文件/行号/原始 id 到新 id 的映射，各方法固定复用。Test 原始 id 不能直接当作跨集合全局 id。',
    '本次不添加 prediction、method、latency_ms；留待真实模型预测时填写。',
    '数字、标点、否定词、*** 等脱敏符号不作删除依据。空白与控制字符仅作关注项，任何可能影响语义的字符修改都需单独核查。']
    for n,path in paths.items():
        f=files[n]; f['sha256_after']=hashlib.sha256(path.read_bytes()).hexdigest(); f['unchanged']=f['sha256']==f['sha256_after']
    a={'schema_version':1,'run_time':dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),'command':'python '+' '.join('"'+s+'"' for s in sys.argv),
       'python_version':sys.version,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'long_text_threshold':args.long_text_threshold,
       'files':files,'cross_sets':overlaps,'recommendations':rec,'unknowns':['source','official version/authenticity','license','semantic label correctness']}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    targets[0].write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    targets[1].write_text(report(a),encoding='utf-8')
    print(json.dumps({'reports':[str(t) for t in targets],'summary':{n:{k:f[k] for k in ('total_records','valid_records','anomalous_records','unchanged')} for n,f in files.items()}},ensure_ascii=False,indent=2))

if __name__=='__main__': main()

