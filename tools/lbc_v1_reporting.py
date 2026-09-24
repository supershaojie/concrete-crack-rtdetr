"""Strict, lossless diagnostic JSON; never modifies computational values."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile


def report_json(value, **kwargs):
    locations = []

    def visit(item, path):
        if isinstance(item, float) and not math.isfinite(item):
            label = 'NaN' if math.isnan(item) else ('+Inf' if item > 0 else '-Inf')
            locations.append(dict(path=path, value=label))
            return {'__nonfinite_float__': label}
        if isinstance(item, dict):
            return {key: visit(child, path + '/' + str(key).replace('~', '~0').replace('/', '~1'))
                    for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(child, path + '/' + str(i)) for i, child in enumerate(item)]
        return item

    result = visit(value, '')
    if locations:
        # Paths use RFC 6901 JSON Pointer, relative to this report's root.
        result['nonfinite_fields'] = result.get('nonfinite_fields', []) + locations
    return json.dumps(result, ensure_ascii=False, allow_nan=False, **kwargs)


def write_json(path, value):
    payload = report_json(value, indent=2) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def append_jsonl(path, value):
    payload = report_json(value) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(payload)


def finalize_json(path, value):
    """In finally: a write error must not replace an in-flight computation error."""
    original = sys.exc_info()[1]
    try:
        write_json(path, value)
    except Exception as error:
        if original is None:
            raise
        # The original exception continues with its original traceback.
        try:
            print(f'Report write failed at {path}: {error!r}; original error preserved: {original!r}',
                  file=sys.stderr, flush=True)
        except Exception:
            pass
