"""Small atomic evidence helpers shared by QCC training and offline tooling."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path


def strict_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return dict(value=None, nonfinite=str(value))
    if isinstance(value, dict):
        return {str(k): strict_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [strict_value(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(strict_value(value), ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def aggregate(samples):
    """Add counts/sums/sumsq; never average batch means of unequal populations."""
    result = {}
    for sample in samples:
        for key, value in sample.get("moments", {}).items():
            dest = result.setdefault(key, dict(count=0, sum=0.0, sumsq=0.0, min=None, max=None))
            for stat in ("count", "sum", "sumsq"):
                dest[stat] += value[stat]
            for stat, fn in (("min", min), ("max", max)):
                if value[stat] is not None:
                    dest[stat] = value[stat] if dest[stat] is None else fn(dest[stat], value[stat])
    for value in result.values():
        n = value["count"]
        value["mean"] = value["sum"] / n if n else None
        value["second_moment"] = value["sumsq"] / n if n else None
    counts = {key: sum(s.get(key, 0) for s in samples) for key in
              ("M", "active_groups", "selected", "cap_active_groups", "competitor_higher_groups", "no_candidates", "excluded_better", "p_lt_q", "p_ge_q")}
    denom = counts["M"]
    measured = result.get("q", {}).get("count", 0)
    return dict(moments=result, counts=counts, grouped_batches=sum("moments" in s for s in samples),
                p_lt_q_fraction=counts["p_lt_q"] / measured if measured else None,
                p_ge_q_fraction=counts["p_ge_q"] / measured if measured else None,
                raw_qcc=sum(s.get("weighted_KL_sum", 0) for s in samples) / max(denom, 1))
