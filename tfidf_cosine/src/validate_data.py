"""Independent stage-3 acceptance; no preparation/audit helper imports."""
import argparse, base64, collections, hashlib, importlib.util, itertools, json, os, platform, random, shutil, subprocess, sys, tempfile, traceback
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,msg):
    if not ok: raise ValueError(msg)
def parse(s):
    def hook(pairs):
        o={}
        for k,v in pairs:
            need(k not in o,'duplicate JSON key '+k); o[k]=v
        return o
    def reject(v): raise ValueError('nonstandard JSON '+v)
    return json.loads(s,object_pairs_hook=hook,parse_constant=reject)
def read(p,formal=False):
    b=p.read_bytes(); need(not b.startswith(bytes([239,187,191])),str(p)+':1 BOM')
    if formal: need(bytes([13]) not in b,str(p)+': CR line ending')
    rows=[]
    for line,s in enumerate(b.decode('utf8').splitlines(),1):
        try:
            o=parse(s); need(type(o) is dict,'not object')
            if formal:
                need(set(o)=={'id','sentence1','sentence2','label'},'not exact four fields')
                need(type(o['id']) is int,'id not integer')
                need(type(o['label']) is int and o['label'] in (0,1),'invalid label')
                for k in ('sentence1','sentence2'): need(type(o[k]) is str and bool(o[k].strip()),'bad '+k)
            rows.append(o)
        except Exception as e: raise ValueError(f'{p}:{line}: {e}') from e
    return rows
def pair(r): return tuple(sorted((r['sentence1'],r['sentence2'])))
def dist(rows):
    c=collections.Counter(r['label'] for r in rows)
    return {'records':len(rows),'labels':{str(k):{'count':c[k],'ratio':c[k]/len(rows)} for k in (0,1)},'identical_sentence_records':sum(r['sentence1']==r['sentence2'] for r in rows)}

def accept(b,w,res):
    def passed(n,d): res['checks'].append({'name':n,'status':'passed','detail':d})
    ap=b/'reports/data_audit.json'; audit=parse(ap.read_text(encoding='utf8'))
    manifest=parse((b/'reports/data_manifest.json').read_text(encoding='utf8')); report=parse((b/'reports/data_preparation.json').read_text(encoding='utf8'))
    paths={n:Path(audit['files'][n]['path']) for n in ('train','dev','test')}
    protected=[*paths.values(),ap,b/'reports/data_audit.md',b/'src/inspect_data.py',b/'reports/data_manifest.json']+[b/p for p in manifest['outputs']]
    before={str(p):sha(p) for p in protected}
    for n,p in paths.items(): need(sha(p)==audit['files'][n]['sha256'],n+' raw hash changed')
    need(sha(ap)==manifest['phase1_audit_sha256'],'phase1 JSON changed since stage2')
    need(sha(b/'src/inspect_data.py')==audit['script_sha256'],'phase1 script changed')
    passed('原始哈希及第一阶段 JSON/脚本历史锚点',{n:sha(p) for n,p in paths.items()})
    res['checks'].append({'name':'第一阶段 Markdown 历史完整性','status':'limited','detail':'没有独立历史哈希；仅能验证本次验收前后未变及内容一致性，不能证明历史字节未变。'})
    raw={n:read(p) for n,p in paths.items()}; sets={n:read(b/'data/processed'/f'{n}.jsonl',True) for n in ('train','validation','evaluation')}
    ids=[r['id'] for rows in sets.values() for r in rows]; need(len(ids)==len(set(ids)),'global duplicate ID')
    pairs={n:{pair(r) for r in rows} for n,rows in sets.items()}
    for n,rows in sets.items():
        need([r['id'] for r in rows]==sorted(r['id'] for r in rows),n+' not ID sorted')
        need(len(pairs[n])==len(rows),n+' duplicate unordered pair')
    for a,c in itertools.combinations(sets,2): need(not pairs[a]&pairs[c],a+'--'+c+' pair overlap')
    passed('编码、四字段、类型、排序、全局 ID、无向重复与冲突','独立检查全部记录，无向句子对全局唯一。')
    src={}; converted={}; modifications=[]; conversions=collections.Counter()
    for n in ('train','dev'):
        rows=[]
        for line,o in enumerate(raw[n],1):
            need(set(o)=={'sentence1','sentence2','label'},f'{paths[n]}:{line} source fields')
            v=o['label']; need((type(v) is int and v in (0,1)) or (type(v) is str and v in ('0','1')),'bad source label')
            rid=line+(len(raw['train']) if n=='dev' else 0)
            src[rid]={'id':rid,'original_file':str(paths[n].resolve()),'original_line':line}
            r={'id':rid,'sentence1':o['sentence1'],'sentence2':o['sentence2'],'label':int(v)}
            if type(v) is str: conversions[n]+=1
            for f in ('sentence1','sentence2'):
                need(type(r[f]) is str and bool(r[f].strip()),'bad source sentence')
                if n=='train' and r[f].startswith('\ufeff'):
                    need(line==12078 and f=='sentence1','unapproved U+FEFF location')
                    text=r[f]; r[f]=text[1:]; modifications.append({'source':src[rid],'field':f,'before':text,'after':r[f]})
            rows.append(r)
        converted[n]=rows
    grouped=collections.defaultdict(list)
    for r in converted['train']: grouped[pair(r)].append(r)
    kept=[]; drops={}
    for members in grouped.values():
        need(len({r['label'] for r in members})==1,'source pair conflicting labels')
        winner=min(members,key=lambda x:x['id']); kept.append(winner)
        for r in members:
            if r['id']!=winner['id']: drops[r['id']]=winner['id']
    need({r['id']:r for n in ('train','validation') for r in sets[n]}=={r['id']:r for r in kept},'Train exact coverage/content differs')
    need(sets['evaluation']==converted['dev'],'Dev sentence/member/order/label/ID differs')
    passed('逐条来源、仅约定修改、最早行去重、ID 与完整覆盖',{'modifications':modifications,'excluded_to_retained':drops,'retained_train':len(kept)})
    rng=random.Random(42); expected_val=set(); allocations={}
    for lab in (0,1):
        members=sorted((r for r in kept if r['label']==lab),key=lambda x:x['id'])
        count=int((Decimal(len(members))*Decimal('0.1')).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
        rng.shuffle(members); expected_val.update(r['id'] for r in members[:count])
        allocations[str(lab)]={'total':len(members),'validation':count,'train':len(members)-count}
    need(expected_val=={r['id'] for r in sets['validation']},'documented split differs')
    need(report['config']['seed']==42 and Decimal(report['config']['validation_ratio'])==Decimal('0.1'),'wrong config')
    passed('独立重建 seed=42、比例=0.1 的划分成员',allocations)
    final={r['id']:n for n,rows in sets.items() for r in rows}; expected_prov=[]
    for rid,s in sorted(src.items()):
        p=dict(s,final_set=final.get(rid,'excluded_duplicate'))
        if rid in drops: p['retained_id']=drops[rid]
        expected_prov.append(p)
    provenance=read(b/'reports/data_provenance.jsonl'); need(provenance==expected_prov,'provenance differs')
    changes=read(b/'reports/processing_changes.jsonl'); mods=[c for c in changes if c['action']=='remove_leading_U+FEFF']; deletes=[c for c in changes if c['action']=='deduplicate_train']
    need(len(changes)==len(mods)+len(deletes),'unknown change')
    need([{k:c[k] for k in ('source','field','before','after')} for c in mods]==modifications,'text change log differs')
    need(len(deletes)==len(drops) and {c['removed_record']['id']:c['retained_record']['id'] for c in deletes}==drops,'deletion log differs')
    conv={r['id']:r for r in converted['train']}
    for c in deletes:
        a,z=c['removed_record']['id'],c['retained_record']['id']
        need(c['removed_record']==conv[a] and c['retained_record']==conv[z] and c['removed_source']==src[a] and c['retained_source']==src[z] and bool(c['reason']),'deletion contents/provenance differs')
    stats={n:dist(rows) for n,rows in sets.items()}
    need(report['outputs']==stats and report['retained_official_train']==dist(kept),'report distribution differs')
    need(report['input_counts']=={n:len(v) for n,v in raw.items()},'input counts differ')
    need(report['text_modifications']==len(mods) and report['deleted_records']==len(drops),'change counts differ')
    need(report['label_type_conversions']==dict(conversions) and report['allocations']==allocations,'allocation/conversions differ')
    need(report['config']==manifest['config'] and report['versions']==manifest['versions'],'manifest config differs')
    for p,v in manifest['outputs'].items(): need(sha(b/p)==v['sha256'] and (b/p).stat().st_size==v['bytes'],'manifest output differs '+p)
    for p,h in manifest['code_sha256'].items(): need(sha(b/'src'/p)==h,'manifest code differs '+p)
    for n,v in manifest['inputs'].items(): need(v==report['phase1_comparison'][n] and v['actual_sha256']==sha(paths[n]) and v['expected_sha256']==audit['files'][n]['sha256'],'manifest input differs')
    md=(b/'reports/data_preparation.md').read_text(encoding='utf8')
    for n,s in stats.items(): need(f'|{n}|{s["records"]}|{s["labels"]["0"]["count"]}|{s["labels"]["1"]["count"]}|' in md,'Markdown summary differs')
    passed('来源映射、修改日志、统计报告与 manifest',{'provenance_rows':len(provenance),'changes':len(changes),'hashed_outputs':len(manifest['outputs'])})
    res['counts']=stats; res['formal_sha256']={n:sha(b/'data/processed'/f'{n}.jsonl') for n in sets}
    def run(argv,cwd=None):
        env=dict(os.environ)
        env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
        env['PYTHONUTF8']='1'
        logical_argv=list(argv)
        if argv[0]=='powershell' and '-Command' in argv:
            argv=['powershell','-NoProfile','-EncodedCommand',base64.b64encode(argv[-1].encode('utf-16le')).decode('ascii')]
        p=subprocess.run(argv,cwd=cwd,env=env,capture_output=True,text=True,encoding='utf8',errors='replace')
        res['commands'].append({'argv':list(map(str,argv)),'readable_argv':logical_argv,'windows_command':subprocess.list2cmdline(list(map(str,argv))),'cwd':str(cwd) if cwd else None,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
        need(p.returncode==0,'command failed '+str(argv))
    compare={}
    for name in ('run1','run2'):
        dest=w/name
        run([sys.executable,'-B',str(b/'src/prepare_data.py'),'--audit',str(ap),'--output-root',str(dest),'--validation-ratio','0.1','--seed','42'])
        compare[name]={}
        for n in sets:
            p=dest/'data/processed'/f'{n}.jsonl'; equal=p.read_bytes()==(b/'data/processed'/f'{n}.jsonl').read_bytes()
            compare[name][n]={'sha256':sha(p),'equal_bytes_to_formal':equal}; need(equal,name+' differs '+n)
    passed('两次独立临时生成与正式文件逐字节、哈希一致',compare)
    portable=w/'portable'; (portable/'src').mkdir(parents=True); (portable/'reports').mkdir(); rawdir=w/'relocated_raw'; rawdir.mkdir()
    for n in paths: shutil.copyfile(paths[n],rawdir/f'{n}.json')
    for f in ('prepare_data.py','inspect_data.py','data_loader.py'): shutil.copyfile(b/'src'/f,portable/'src'/f)
    moved=json.loads(json.dumps(audit))
    for n in paths: moved['files'][n]['path']=str(w/'nonexistent_old_paths'/f'{n}.json')
    (portable/'reports/data_audit.json').write_text(json.dumps(moved,ensure_ascii=False),encoding='utf8')
    ps="$rawDir = Resolve-Path ..\\relocated_raw; python -B .\\src\\prepare_data.py --audit .\\reports\\data_audit.json --train (Join-Path $rawDir 'train.json') --dev (Join-Path $rawDir 'dev.json') --test (Join-Path $rawDir 'test.json') --output-root . --validation-ratio 0.1 --seed 42; exit $LASTEXITCODE"
    interpreter=Path(sys.executable).as_posix().replace("'", "''")
    python_function=""
    ps="$ErrorActionPreference = 'Stop'; "+ps.replace('python -B', "& '"+interpreter+"' -B")
    run(['powershell','-NoProfile','-Command',ps],portable)
    powershell_created=(portable/'data/processed/train.jsonl').exists()
    if not powershell_created:
        fallback=w/'portable_direct'
        shutil.copytree(portable/'src',fallback/'src')
        (fallback/'reports').mkdir()
        shutil.copyfile(portable/'reports/data_audit.json',fallback/'reports/data_audit.json')
        portable=fallback
        res['checks'].append({'name':'PowerShell README 命令宿主执行','status':'incomplete','detail':'本环境 PowerShell 调用 Python 未产生输出文件，尽管退出码为 0。不能据此宣称 PowerShell 示例通过；改用相同参数直接调用解释器完成路径迁移和读取逻辑检查。'})
        run([sys.executable,'-B','src/prepare_data.py','--audit','reports/data_audit.json','--train',str(rawdir/'train.json'),'--dev',str(rawdir/'dev.json'),'--test',str(rawdir/'test.json'),'--output-root','.','--validation-ratio','0.1','--seed','42'],portable)
    movedhash={}
    for n in sets:
        p=portable/'data/processed'/f'{n}.jsonl'; need(p.read_bytes()==(b/'data/processed'/f'{n}.jsonl').read_bytes(),'portable differs '+n); movedhash[n]=sha(p)
    passed('原始数据迁移与显式路径参数（直接解释器）',{'sha256':movedhash,'temp_audit_old_paths_intentionally_nonexistent':True,'copied_input_hashes':{n:sha(rawdir/f'{n}.json') for n in paths}})
    readme=(b/'README.md').read_text(encoding='utf8'); snippet=readme.split("@'",1)[1].split("'@ | python -B -",1)[0]
    run(['powershell','-NoProfile','-Command',"$ErrorActionPreference = 'Stop'\n@'"+snippet+"'@ | & '"+interpreter+"' -B -\nexit $LASTEXITCODE"],portable)
    run([sys.executable,'-B','-c',snippet],portable)
    passed('迁移输出目录执行 README Python 读取逻辑','直接解释器运行 README Python 代码；PowerShell 外壳执行不计为通过。')
    spec=importlib.util.spec_from_file_location('tested_loader',b/'src/data_loader.py'); loader=importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
    valid={'id':1,'sentence1':'句一','sentence2':'句二','label':0}
    cases={'invalid_json':'{bad','missing_field':{'id':1,'sentence1':'a','label':0},'empty_sentence':dict(valid,sentence1=''),'whitespace_sentence':dict(valid,sentence2='  '),'wrong_sentence_type':dict(valid,sentence1=123),'invalid_label':dict(valid,label=2),'string_label':dict(valid,label='0'),'float_id':dict(valid,id=1.0),'true_id':dict(valid,id=True),'false_id':dict(valid,id=False),'true_label':dict(valid,label=True),'false_label':dict(valid,label=False),'extra_field':dict(valid,extra=1)}
    errors=[]; t=w/'invalid_inputs'; t.mkdir()
    cases['duplicate_id']=[valid,valid]
    for name,value in cases.items():
        p=t/(name+'.jsonl')
        content=value if type(value) is str else '\n'.join(json.dumps(o,ensure_ascii=False) for o in (value if type(value) is list else [value]))
        p.write_text(content+'\n',encoding='utf8'); line=2 if name=='duplicate_id' else 1
        try: loader.load_data(p)
        except ValueError as e:
            need(str(p)+':'+str(line) in str(e),'error missing file/line '+name); errors.append({'case':name,'status':'passed','message':str(e)})
        else: raise ValueError('loader accepted '+name)
    passed('错误输入拒绝及文件名行号',errors)
    need(before=={str(p):sha(p) for p in protected},'protected files changed')
    passed('原始、正式、第一/二阶段文件验收前后未变',before)
    res['repairs']=['准备程序、loader、正式数据及第二阶段报告无需修复，未修改。','新增验收脚本与 README 验收说明，划分规则不变。','首次迁移测试的 PowerShell 子进程找不到 python，且非严格模式误返 0；验收运行器将当前解释器目录加入子进程 PATH、设置 PYTHONUTF8=1，并启用 ErrorActionPreference=Stop 后重新执行全部测试。未修改系统 PATH。']

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument('--work-dir',type=Path,default=Path(__file__).resolve().parents[2]/'work'); args=p.parse_args()
    b=args.output_root.resolve(); args.work_dir.mkdir(parents=True,exist_ok=True); w=Path(tempfile.mkdtemp(prefix='data-acceptance-',dir=args.work_dir))
    res={'status':'running','command':subprocess.list2cmdline([sys.executable,*sys.argv]),'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'dependencies':'standard library and installed Windows PowerShell'},'work_directory':str(w),'temporary_files_retained':True,'checks':[],'commands':[],'unverified':{'download_source':'待补充','official_version_authenticity':'待补充','license_authorization':'待补充','U+FEFF_origin_cause':'尚未确认'}}
    try: accept(b,w,res); res['status']='passed_with_incomplete_shell_check'
    except Exception as e:
        res.update(status='failed',error=str(e),traceback=traceback.format_exc()); res['checks'].append({'name':'后续检查','status':'incomplete','detail':'首次错误后停止，未列为通过的项目不得推断通过。'})
    report=b/'reports'; report.mkdir(exist_ok=True)
    (report/'data_validation.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    lines=['# 第三阶段：独立检查与复现验收','','总体状态：'+res['status'],'','独立检查不复用准备/审计程序的检查函数。仅复现时运行准备程序，错误测试时调用被测 loader。未训练、未预测、未改划分或正式数据。','','## 验收结果','']
    for c in res['checks']: lines+=['### '+c['name'],'','状态：'+c['status'],'','~~~json',json.dumps(c['detail'],ensure_ascii=False,indent=2),'~~~','']
    lines+=['## 数量与正式文件哈希','','~~~json',json.dumps({'counts':res.get('counts'),'sha256':res.get('formal_sha256')},ensure_ascii=False,indent=2),'~~~','','## 命令、退出状态与环境','','~~~json',json.dumps({'command':res['command'],'environment':res['environment'],'runs':res['commands']},ensure_ascii=False,indent=2),'~~~','','## 修复及尚未解决','']+['- '+s for s in res.get('repairs',[])]
    lines+=['- 第一阶段 Markdown 没有独立历史哈希锚点，历史字节完整性只可部分核实；本次前后未变单独验证。','- 下载来源、官方版本真实性、许可和授权：待补充；U+FEFF 成因尚未确认。','- 本次实际复现仅覆盖报告中的 Windows/CPython 环境，不代表跨所有 Python 版本都已验收。','- 临时测试材料保留在 '+str(w)+'，未混入正式数据。','']
    if res.get('error'): lines+=['## 失败与未完成','',res['error'],res['traceback']]
    (report/'data_validation.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({'status':res['status'],'counts':res.get('counts'),'checks':[{k:c[k] for k in ('name','status')} for c in res['checks']],'error':res.get('error')},ensure_ascii=False,indent=2)); return int(res['status']=='failed')

if __name__=='__main__': sys.exit(main())
