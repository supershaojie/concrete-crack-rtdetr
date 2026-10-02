"""Small, read-only input helpers. No detector or trainer imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def load_yaml(path):
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError('PyYAML unavailable; use the existing rtdetr Python, do not reinstall it') from e
    return yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))


def new_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path
