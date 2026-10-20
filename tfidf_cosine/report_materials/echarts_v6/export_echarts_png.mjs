import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL, fileURLToPath } from 'node:url';

function arg(name, fallback = null) {
  const i = process.argv.indexOf(name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

const here = path.dirname(fileURLToPath(import.meta.url));
const htmlDir = path.resolve(arg('--html-dir', here));
const outputDir = path.resolve(arg('--output-dir', path.join(htmlDir, 'png_export')));
const browserPath = arg('--browser-path', null);
const htmlFiles = ['MacroF1_ECharts.html', 'Threshold_MacroF1_LegendBottom.html'];
const outputTargets = [
  ...htmlFiles.map(file => path.join(outputDir, file.replace('.html', '.png'))),
  path.join(outputDir, 'export_audit.json')
];
let outputDirExists = true;
try { await fs.access(outputDir); } catch (e) { if (e.code === 'ENOENT') outputDirExists = false; else throw e; }
const conflicts = [];
for (const target of outputTargets) {
  try { await fs.access(target); conflicts.push(target); } catch (e) { if (e.code !== 'ENOENT') throw e; }
}
if (outputDirExists && conflicts.length === 0) conflicts.push(outputDir);
if (conflicts.length) throw new Error(`拒绝覆盖以下路径:\n${conflicts.join('\n')}\n请指定新的 --output-dir。`);
for (const file of [...htmlFiles.map(f => path.join(htmlDir, f)), path.join(htmlDir, 'echarts.min.js'), path.join(htmlDir, 'ReportCJK.font')]) {
  try { await fs.access(file); } catch (e) { if (e.code === 'ENOENT') throw new Error(`缺少HTML或必要资源: ${file}`); throw e; }
}
let chromium;
try { ({ chromium } = await import('playwright')); }
catch (e) { throw new Error(`无法加载项目依赖 playwright，请先在项目根目录执行 npm install: ${e.message}`); }
if (browserPath) {
  try { await fs.access(path.resolve(browserPath)); } catch (e) { throw new Error(`--browser-path 不存在或不可访问: ${path.resolve(browserPath)}`); }
}
await fs.mkdir(outputDir);
const browser = await chromium.launch(browserPath ? { headless: true, executablePath: path.resolve(browserPath) } : { headless: true });
const results = [];
const errors = [];
try {
  for (const file of htmlFiles) {
    const page = await browser.newPage({ viewport: { width: 1280, height: 820 }, deviceScaleFactor: 2 });
    page.on('console', msg => { if (msg.type() === 'error') errors.push(`${file}: ${msg.text()}`); });
    page.on('pageerror', err => errors.push(`${file}: ${err.message}`));
    const url = pathToFileURL(path.join(htmlDir, file)).href;
    await page.goto(url, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__chartReady === true, null, { timeout: 15000 });
    await page.evaluate(() => document.fonts.ready);
    await page.waitForFunction(() => document.fonts.status === 'loaded', null, { timeout: 15000 });
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const state = await page.evaluate(() => ({ ready: window.__chartReady, error: window.__chartError, svg: Boolean(document.querySelector('#chart svg')), fonts: document.fonts.status }));
    if (!state.ready || state.error || !state.svg || state.fonts !== 'loaded') throw new Error(`${file}渲染状态无效: ${JSON.stringify(state)}`);
    const tooltipChecks = [];
    if (file.startsWith('Threshold')) {
      for (const dataIndex of [10, 28, 50, 90]) {
        await page.evaluate((index) => window.chart.dispatchAction({ type: 'showTip', seriesIndex: 0, dataIndex: index }), dataIndex);
        await page.waitForTimeout(180);
        const text = await page.locator('body > div').allTextContents();
        const joined = text.join(' ');
        tooltipChecks.push({ dataIndex, text: joined });
        const expected = await page.evaluate(index => ({
          threshold: Number(window.chart.getOption().series[0].data[index][0]).toFixed(2),
          series: window.chart.getOption().series.map(item => ({ name: item.name, value: Number(item.data[index][1]).toFixed(2) }))
        }), dataIndex);
        if (!joined.includes(`阈值：${expected.threshold}`) || expected.series.some(item => !joined.includes(`${item.name}：验证集 Macro-F1 ${item.value}%`))) {
          throw new Error(`${file}悬浮提示与图表数据不符: ${JSON.stringify({ expected, text: joined })}`);
        }
      }
      await page.evaluate(() => window.chart.dispatchAction({ type: 'hideTip' }));
      await page.waitForTimeout(250);
      await page.evaluate(() => {
        window.chart.setOption({ tooltip: { show: false } });
        for (const node of document.querySelectorAll('#chart div')) {
          if (getComputedStyle(node).zIndex === '9999999') node.remove();
        }
      });
    }
    const svgBox = await page.locator('#chart svg').boundingBox();
    if (!svgBox || svgBox.width < 900 || svgBox.height < 600) throw new Error(`${file}图表尺寸异常`);
    const png = path.join(outputDir, file.replace('.html', '.png'));
    await page.locator('#chart svg').screenshot({ path: png });
    results.push({ file, png, svg: svgBox, tooltipChecks });
    await page.close();
  }
} finally { await browser.close(); }
if (errors.length) throw new Error(`浏览器控制台错误: ${errors.join('; ')}`);
await fs.writeFile(path.join(outputDir, 'export_audit.json'), JSON.stringify({ browser: { mode: browserPath ? 'explicit-path' : 'playwright-chromium', path: browserPath ? path.resolve(browserPath) : chromium.executablePath(), version: browser.version() }, results, errors }, null, 2), 'utf8');
console.log(JSON.stringify({ outputDir, results }, null, 2));
