"""Strict shared loader for four-field UTF-8/no-BOM JSONL."""
import json
from pathlib import Path

FIELDS = ('id', 'sentence1', 'sentence2', 'label')


def strict_json(text):
    def hook(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('duplicate JSON key: ' + key)
            obj[key] = value
        return obj
    def reject(value):
        raise ValueError('nonstandard JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=hook, parse_constant=reject)


def validate_record(obj, location):
    def fail(message):
        raise ValueError(f'{location}: {message}')
    if type(obj) is not dict or set(obj) != set(FIELDS):
        fail('expected exactly fields: ' + ', '.join(FIELDS))
    if type(obj['id']) is not int:
        fail('id must be integer, not bool')
    for field in ('sentence1', 'sentence2'):
        if type(obj[field]) is not str or not obj[field].strip():
            fail(field + ' must be a non-whitespace string')
    if type(obj['label']) is not int or obj['label'] not in (0, 1):
        fail('label must be integer 0/1, not bool')


def iter_data(path):
    """Yield validated records; fully consume iterator to check the whole file."""
    path = Path(path)
    seen = {}
    with path.open('rb') as stream:
        for line_no, raw in enumerate(stream, 1):
            loc = f'{path}:{line_no}'
            try:
                if line_no == 1 and raw.startswith(b'\xef\xbb\xbf'):
                    raise ValueError('UTF-8 BOM is not allowed')
                obj = strict_json(raw.decode('utf-8'))
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f'{loc}: {exc}') from exc
            validate_record(obj, loc)
            if obj['id'] in seen:
                raise ValueError(f'{loc}: duplicate id {obj["id"]}; first at line {seen[obj["id"]]}')
            seen[obj['id']] = line_no
            yield obj


def load_data(path):
    """Return all records after validation. No text normalization."""
    return list(iter_data(path))
