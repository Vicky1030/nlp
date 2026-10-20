"""Write a fresh inventory of the public method payload, excluding generated data."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'reports/release_sha256.json'


def main():
    rows = []
    listed = subprocess.run(
        ['git', 'ls-files', '-z', '--cached', '--', 'tfidf_cosine'],
        cwd=ROOT.parent, check=True, capture_output=True, text=True,
        encoding='utf-8').stdout.split('\0')
    prefix = 'tfidf_cosine/'
    paths = sorted(item[len(prefix):] for item in listed if item.startswith(prefix))
    for rel in paths:
        if rel == MANIFEST:
            continue
        path = ROOT / Path(rel)
        if not path.is_file():
            raise SystemExit(f'Release file missing from worktree: {rel}')
        data = subprocess.run(['git', 'show', ':' + prefix + rel], cwd=ROOT.parent,
                              check=True, capture_output=True).stdout
        rows.append({'path': rel,
                     'sha256': hashlib.sha256(data).hexdigest()})
    (ROOT / MANIFEST).write_text(
        json.dumps({'files': len(rows), 'entries': rows},
                   ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(rows)} release file hashes')


if __name__ == '__main__':
    main()
