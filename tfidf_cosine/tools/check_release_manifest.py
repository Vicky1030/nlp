"""Verify the repository's selected release inventory without hashing outputs."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'reports/release_sha256.json').read_text(encoding='utf-8'))
    listed = subprocess.run(
        ['git', 'ls-files', '-z', '--cached', '--', 'tfidf_cosine'],
        cwd=ROOT.parent, check=True, capture_output=True, text=True,
        encoding='utf-8').stdout.split('\0')
    prefix = 'tfidf_cosine/'
    expected = {item[len(prefix):] for item in listed
                if item.startswith(prefix) and item != prefix + 'reports/release_sha256.json'}
    actual = {entry['path'] for entry in manifest['entries']}
    if actual != expected or manifest['files'] != len(actual):
        raise SystemExit('Release manifest does not match Git-tracked release files')
    for entry in manifest['entries']:
        path = ROOT / Path(entry['path'])
        if not path.is_file():
            raise SystemExit(f"Release file missing: {entry['path']}")
        data = path.read_bytes()
        text_attr = subprocess.run(['git', 'check-attr', 'text', '--', str(path)],
                                   cwd=ROOT, check=True, capture_output=True, text=True).stdout
        if not text_attr.rstrip().endswith('unset'):
            data = data.replace(b'\r\n', b'\n')
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry['sha256']:
            raise SystemExit(f"Release hash mismatch: {entry['path']}")
    print(f"Release manifest passed: {manifest['files']} files")


if __name__ == '__main__':
    main()
