"""Small reports and cleanup of exclusively owned BLC probe checkpoints."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile

from blc_common import sha256


@contextmanager
def temporary_probe(folder, purpose, report):
    """Only this invocation's TemporaryDirectory is removed, including on interruption."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(purpose=purpose, checkpoints=[], cleaned=False)
    report.setdefault("temporary_artifacts", []).append(record)
    owned = None
    try:
        with tempfile.TemporaryDirectory(prefix=purpose + "-", dir=folder) as tmp:
            owned = Path(tmp)
            record["directory"] = str(owned)
            try:
                yield owned
            finally:
                for path in sorted(owned.rglob("*.pt")):
                    record["checkpoints"].append(dict(path=path.relative_to(owned).as_posix(),
                                                       bytes=path.stat().st_size, sha256=sha256(path)))
    finally:
        record["cleaned"] = owned is not None and not owned.exists()


def retained_size(folder):
    """Observe retained bytes; never delete reports to meet the 10 MiB target."""
    return sum(p.stat().st_size for p in Path(folder).rglob("*") if p.is_file())


def compact_audit(value):
    """Keep state-inventory counts/hashes and failing samples, not repeated key lists."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "COMMON" and isinstance(item, list):
                failures = [row for row in item if not row.get("equal", False)]
                result[key] = dict(count=len(item), unequal_count=len(failures), unequal_samples=failures[:5],
                                   inventory_sha256=hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest())
            else:
                result[key] = compact_audit(item)
        return result
    if isinstance(value, list):
        return [compact_audit(item) for item in value]
    return value
