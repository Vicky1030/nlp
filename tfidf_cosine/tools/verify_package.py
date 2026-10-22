"""Strict current delivery manifest check; paths resolve from package root."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    root = parser.parse_args().package_root.resolve()
    rows = json.loads((root / 'package_sha256.json').read_text(encoding='utf-8'))['files']
    expected = {r['path']: r for r in rows}
    if len(expected) != len(rows):
        raise SystemExit('duplicate manifest paths')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} - {'package_sha256.json'}
    errors = [f'missing: {p}' for p in expected.keys() - actual]
    errors += [f'unknown: {p}' for p in actual - expected.keys()]
    for name in expected.keys() & actual:
        path = (root / name).resolve()
        if root not in path.parents:
            errors.append(f'unsafe: {name}')
            continue
        row = expected[name]
        if path.stat().st_size != row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            errors.append(f'changed: {name}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(expected)} files; strict size, SHA-256 and file-set verification (manifest excludes itself)')


if __name__ == '__main__':
    main()
