"""下载官方 Qwen 模型；可提供 commit SHA 固定版本。"""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from common import MODEL_ID

p = argparse.ArgumentParser()
p.add_argument('--model', default=MODEL_ID)
p.add_argument('--revision', default='main', help='正式实验应改为两位同学统一的 commit SHA')
p.add_argument('--output', default='models/qwen3-1.7b')
a = p.parse_args()
path = snapshot_download(repo_id=a.model, revision=a.revision, local_dir=a.output)
Path(a.output, 'download_source.json').write_text(
    json.dumps({'model': a.model, 'requested_revision': a.revision}, indent=2), encoding='utf-8'
)
print('模型下载完成：', path)
print('正式对比前，B/C 必须固定模型的 commit SHA，不要长期使用浮动的 main。')
