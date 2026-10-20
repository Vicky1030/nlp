"""Verify the existing unified data, without normalizing or splitting it."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data_loader import load_data


def check(data_dir):
    expected = json.loads((ROOT / 'reports/processed_hashes.json').read_text(encoding='utf-8'))
    result = {}
    all_ids = set()
    for name, spec in expected.items():
        path = Path(data_dir) / (name + '.jsonl')
        if not path.is_file():
            raise FileNotFoundError(f'Missing external unified input: {path}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != spec['sha256']:
            raise ValueError(f'{name}: SHA-256 mismatch; use the existing unified processed data, do not re-split')
        rows = load_data(path)
        if len(rows) != spec['rows']:
            raise ValueError(f'{name}: row count mismatch')
        ids = {r['id'] for r in rows}
        if ids & all_ids:
            raise ValueError(f'{name}: IDs overlap another partition')
        all_ids |= ids
        result[name] = {'sha256': digest, 'rows': len(rows)}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    print(json.dumps(check(p.parse_args().data_dir), indent=2))
