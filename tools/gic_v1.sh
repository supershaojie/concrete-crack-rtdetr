#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
exec /root/miniconda3/envs/rtdetr/bin/python -u "$ROOT/tools/gic_v1.py" "$@"
