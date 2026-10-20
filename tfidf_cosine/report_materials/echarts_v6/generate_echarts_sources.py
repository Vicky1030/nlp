"""Generate the two offline ECharts figures from the frozen project results."""
from __future__ import annotations
import argparse, csv, json, re, base64, hashlib, shutil
from pathlib import Path

COLORS = {"char_1":"#4979CE", "char_1_2":"#079C88", "word_1":"#D88745"}
NAMES = {"char_1":"字符 1-gram", "char_1_2":"字符 1～2-gram", "word_1":"词级 1-gram"}

def rows(path):
    with path.open(encoding="utf-8", newline="") as f: return list(csv.DictReader(f))

def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]

def template_parts(template):
    if template.is_dir() and (template/'echarts.min.js').is_file():
        return (template/'echarts.min.js').read_text(encoding='utf-8'), (template/'ReportCJK.font').read_bytes()
    s = template.read_text(encoding="utf-8")
    start = s.index("<script>") + len("<script>")
    end = s.index("</script>", start)
    lib = s[start:end]
    # The provided HTML starts with ECharts and embeds a CJK font in CSS. Keep
    # the library and font bytes, but rebuild the chart option from source data.
    font_match = re.search(r"url\(data:font/[^;]+;base64,([^\)]+)\)", s)
    font = base64.b64decode(font_match.group(1)) if font_match else b""
    return lib, font

def shell(title, option, width=1280, height=820, line_chart=False):
    option_json = json.dumps(option, ensure_ascii=False, separators=(',', ':'))
    tooltip_js = """o.tooltip.formatter = params => {
      const items = Array.isArray(params) ? params : [params];
      const threshold = items[0] && Array.isArray(items[0].value) ? items[0].value[0] : (items[0] && items[0].axisValue);
      const head = `阈值：${Number(threshold).toFixed(2)}`;
      const body = items.map(item => {
        const value = Array.isArray(item.value) ? item.value[1] : item.value;
        return `${item.marker}${item.seriesName || 'Macro-F1'}：验证集 Macro-F1 ${Number(value).toFixed(2)}%`;
      }).join('<br/>');
      return `${head}<br/>${body}`;
    };""" if line_chart else ""
    best_labels_js = """const centers = [900, 210, 550];
    const callouts = [];
    o.series.forEach((series, index) => {
      const point = c.convertToPixel({ seriesIndex: index }, series.markPoint.data[0].coord);
      const center = centers[index], left = center - 115, top = 118;
      callouts.push({ type: 'line', shape: { x1: center, y1: top + 64, x2: point[0], y2: point[1] }, style: { stroke: series.lineStyle.color, lineWidth: 1.5, lineDash: [5, 4] }, silent: true });
      callouts.push({ type: 'rect', shape: { x: left, y: top, width: 230, height: 64, r: 4 }, style: { fill: '#FFFFFF', stroke: series.lineStyle.color, lineWidth: 1 }, z: 10, silent: true });
      callouts.push({ type: 'text', style: { x: center, y: top + 32, text: series.markPoint.label.formatter, font: '14px ReportCJK', lineHeight: 21, fill: series.lineStyle.color, textAlign: 'center', textVerticalAlign: 'middle' }, z: 11, silent: true });
    });
    c.setOption({ graphic: [...o.graphic, ...callouts] }, { replaceMerge: ['graphic'] });""" if line_chart else ""
    disable_markpoint_labels = "for (const series of o.series) { series.markPoint.label.show = false; if (series.markLine) series.markLine.data = series.markLine.data.filter(item => item.xAxis !== undefined); }" if line_chart else ""
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{title}</title>
<style>@font-face{{font-family:ReportCJK;src:url(ReportCJK.font)}}body{{margin:0;background:#eef3f8;display:flex;align-items:center;justify-content:center;min-height:100vh}}#chart{{width:{width}px;height:{height}px;max-width:100vw}}</style>
<div id="chart"></div><script src="echarts.min.js"></script><script>
window.__chartReady=false; window.__chartError=null;
document.fonts.ready.then(()=>{{
  try {{
    const c=echarts.init(document.getElementById('chart'),null,{{renderer:'svg'}});
    const o={option_json};
    if(o.series[0].type==='bar')o.series[0].label.formatter=p=>Number(p.value).toFixed(2)+'%';
    {tooltip_js}
    {disable_markpoint_labels} c.setOption(o); {best_labels_js} window.chart=c; window.addEventListener('resize',()=>c.resize());
    requestAnimationFrame(()=>{{if(!document.querySelector('#chart svg')) throw new Error('ECharts SVG未生成'); window.__chartReady=true;}});
  }} catch (e) {{ window.__chartError=String(e); }}
}}).catch(e=>{{window.__chartError=String(e);}});
</script></html>'''

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--package-root",type=Path,required=True)
    ap.add_argument("--template-dir",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--stage3-report",type=Path)
    ap.add_argument("--predictions",type=Path)
    ap.add_argument("--stage3-root",type=Path)
    args=ap.parse_args(); root=args.package_root.resolve(); out=args.output_dir.resolve()
    if not out.exists(): out.mkdir(parents=True,exist_ok=False)
    table=rows(root/'artifacts/tfidf_cosine_full_run_final/stage2/results/tfidf_cosine/threshold_comparison.csv')
    assert len(table)==306 and {r['scheme'] for r in table}==set(NAMES)
    for scheme in NAMES: assert sum(r['scheme']==scheme for r in table)==102
    stage3_root=args.stage3_root.resolve() if args.stage3_root else root/'artifacts/tfidf_cosine_full_run_final/stage3'
    report_path=args.stage3_report or stage3_root/'reports/tfidf_cosine_stage3.json'
    predictions_path=args.predictions or stage3_root/'results/tfidf_cosine/evaluation_predictions.jsonl'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    labels=[r['label'] for r in read_jsonl(predictions_path)]
    preds=[r['prediction'] for r in read_jsonl(predictions_path)]
    macro=report['metrics']['macro_f1']; allzero=sum(1 for a,b in zip(labels,[0]*len(labels)) if a==b)
    # Macro-F1 for all-zero baseline is calculated from the binary labels.
    tn=sum(a==0 for a in labels); pos=sum(a==1 for a in labels)
    baseline=(2*tn/(2*tn+pos)+0)/2
    macro_opt={}
    for r in table:
        if r['threshold']=='' : continue
        macro_opt.setdefault(r['scheme'],[]).append([float(r['threshold']),float(r['macro_f1'])*100])
    lib,font=template_parts(args.template_dir)
    bar={"animation":False,"backgroundColor":"#fff","textStyle":{"fontFamily":"ReportCJK","color":"#18324D"},"grid":{"left":108,"right":90,"top":180,"bottom":132},"tooltip":{"trigger":"axis","axisPointer":{"type":"shadow"}},"xAxis":{"type":"category","data":["TF-IDF＋余弦相似度","固定全部预测 0"],"axisTick":{"show":False},"axisLine":{"lineStyle":{"color":"#C7D2E0"}},"axisLabel":{"fontSize":22,"margin":25,"color":"#203B54"}},"yAxis":{"type":"value","min":0,"max":100,"interval":20,"name":"Macro-F1","nameTextStyle":{"fontSize":17,"color":"#738397"},"axisLabel":{"formatter":"{value}%","fontSize":16,"color":"#8290A2"},"splitLine":{"lineStyle":{"color":"#EAF0F6","type":"dashed"}},"axisLine":{"show":False}},"series":[{"type":"bar","barWidth":150,"data":[{"value":macro*100,"itemStyle":{"color":"#20A897","borderRadius":[12,12,0,0]}},{"value":baseline*100,"itemStyle":{"color":"#B5C5D8","borderRadius":[12,12,0,0]}}],"label":{"show":True,"position":"top","distance":18,"fontSize":33,"fontWeight":"bold","color":"#17384E","formatter":"{c}%"}}],"graphic":[{"type":"text","left":65,"top":36,"style":{"text":"最终评估 Macro-F1 对照","font":"bold 34px ReportCJK","fill":"#16314C"}},{"type":"text","left":67,"top":94,"style":{"text":"同一最终评估集 · 4,316 个句子对","font":"18px ReportCJK","fill":"#748398"}},{"type":"text","left":67,"bottom":35,"style":{"text":"配置：字符 1～2-gram，阈值 0.28    参照：固定全部预测为 0","font":"16px ReportCJK","fill":"#748398"}}]}
    line={"animation":False,"backgroundColor":"#fff","textStyle":{"fontFamily":"ReportCJK","color":"#17324D"},"grid":{"left":100,"right":70,"top":250,"bottom":185},"legend":{"bottom":54,"left":"center","itemWidth":32,"itemHeight":10,"itemGap":36,"textStyle":{"fontSize":18,"color":"#52677E"}},"tooltip":{"trigger":"axis","axisPointer":{"type":"line"}},"xAxis":{"type":"value","min":0,"max":1.01,"interval":0.1,"name":"分类阈值","nameLocation":"middle","nameGap":48,"nameTextStyle":{"fontSize":18,"color":"#62778C"},"axisLabel":{"fontSize":16,"color":"#8391A1"},"axisTick":{"show":False},"axisLine":{"lineStyle":{"color":"#C9D5E2"}},"splitLine":{"show":False}},"yAxis":{"type":"value","min":20,"max":60,"interval":10,"name":"验证集 Macro-F1","nameGap":22,"nameTextStyle":{"fontSize":17,"color":"#62778C"},"axisLabel":{"fontSize":16,"color":"#8391A1","formatter":"{value}%"},"splitLine":{"lineStyle":{"color":"#E7EDF4","type":"dashed"}},"axisLine":{"show":False}},"series":[] ,"graphic":[{"type":"text","left":65,"top":30,"style":{"text":"验证集阈值与 Macro-F1","font":"bold 34px ReportCJK","fill":"#16314C"}},{"type":"text","left":67,"top":82,"style":{"text":"3种表示 · 每种102个候选阈值 · 共306组比较","font":"18px ReportCJK","fill":"#748398"}}]}
    for scheme in NAMES:
        best_row=max((r for r in table if r['scheme']==scheme and r['threshold']!=''), key=lambda r:(float(r['macro_f1']), float(r['accuracy']), float(r['threshold']), -int(r['threshold_index'])))
        best=[float(best_row['threshold']), float(best_row['macro_f1'])*100]
        positions={"char_1":"top","char_1_2":"top","word_1":"top"}
        offsets={"char_1":[-180,-8],"char_1_2":[0,-8],"word_1":[180,-8]}
        label=f"最佳阈值：{best[0]:.2f}\nMacro-F1：{best[1]:.2f}%"
        guide_y=min(59.0, best[1]+4.0)
        series={"name":NAMES[scheme],"type":"line","data":macro_opt[scheme],"smooth":False,"showSymbol":False,"labelLayout":{"moveOverlap":"shiftY"},"lineStyle":{"width":4 if scheme=='char_1_2' else 3,"color":COLORS[scheme]},"itemStyle":{"color":COLORS[scheme]},"markPoint":{"symbol":"circle","symbolSize":12,"label":{"show":True,"formatter":label,"position":positions[scheme],"offset":offsets[scheme],"distance":24,"fontSize":14,"lineHeight":21,"color":COLORS[scheme],"backgroundColor":"#FFFFFF","borderColor":COLORS[scheme],"borderWidth":1,"borderRadius":4,"padding":[6,8]},"data":[{"coord":best}]},"markLine":{"silent":True,"symbol":["none","none"],"label":{"show":False},"lineStyle":{"color":COLORS[scheme],"width":1.2,"type":"dashed","opacity":0.8},"data":[[{"coord":best},{"coord":[best[0],guide_y]}]]}}
        if scheme=='char_1_2':
            series['markLine']['data'].append({"xAxis":0.28,"label":{"show":True,"formatter":"获选阈值 0.28","position":"insideStartTop","fontSize":14,"color":"#087E70"}})
        line['series'].append(series)
    (out/'MacroF1_ECharts.html').write_text(shell('最终评估 Macro-F1 对照',bar),encoding='utf-8')
    (out/'Threshold_MacroF1_LegendBottom.html').write_text(shell('验证集阈值与 Macro-F1',line,line_chart=True),encoding='utf-8')
    (out/'echarts.min.js').write_text(lib,encoding='utf-8'); (out/'ReportCJK.font').write_bytes(font)
    for name in ('LICENSE-Apache-2.0.txt','NOTICE-ECharts.txt','OFL-1.1.txt'):
        source=Path(__file__).with_name(name)
        if not source.is_file(): raise FileNotFoundError(f'缺少许可源文件: {source}')
        shutil.copyfile(source, out/name)
    resources={'echarts':{'name':'Apache ECharts','version':'5.6.0','source':'https://github.com/apache/echarts/releases/tag/5.6.0','license':'Apache-2.0','sha256':hashlib.sha256((out/'echarts.min.js').read_bytes()).hexdigest(),'modified':False},'font':{'name':'Noto Sans CJK SC','source':'https://github.com/notofonts/noto-cjk','license':'SIL Open Font License 1.1','sha256':hashlib.sha256((out/'ReportCJK.font').read_bytes()).hexdigest(),'modified':False}}
    (out/'THIRD_PARTY_RESOURCES.md').write_text('# 第三方资源\n\nECharts图表资源与字体许可独立于AFQMC数据许可。`echarts_v6`是本目录版本名，不表示ECharts 6.x。\n\n|资源|版本/身份|官方来源|许可|SHA-256|是否修改|\n|---|---|---|---|---|---|\n|Apache ECharts|5.6.0|https://github.com/apache/echarts/releases/tag/5.6.0|Apache-2.0|'+resources['echarts']['sha256']+'|否|\n|Noto Sans CJK SC|内部名称|https://github.com/notofonts/noto-cjk|SIL OFL 1.1|'+resources['font']['sha256']+'|否|\n\n完整文本见 `LICENSE-Apache-2.0.txt` 与 `OFL-1.1.txt`。来源核实为本轮记录，不推断最初获取历史。\n',encoding='utf-8')
    (out/'chart_data_summary.json').write_text(json.dumps({'source_csv':'artifacts/tfidf_cosine_full_run_final/stage2/results/tfidf_cosine/threshold_comparison.csv','rows':len(table),'rows_per_scheme':{s:len(macro_opt[s]) for s in NAMES},'final_macro_f1':macro,'all_zero_macro_f1':baseline,'all_zero_correct':allzero,'best':{s:max(macro_opt[s],key=lambda x:x[1]) for s in NAMES}},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'rows':len(table),'final_macro_f1':macro,'all_zero_macro_f1':baseline},ensure_ascii=False))
if __name__=='__main__':main()
