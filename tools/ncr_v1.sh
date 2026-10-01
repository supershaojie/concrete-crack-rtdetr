#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=${NCR_PYTHON:-/root/miniconda3/envs/rtdetr/bin/python}
export YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1
cd -- "$ROOT"
exec "$PYTHON" -u "$ROOT/tools/ncr_v1.py" "$@"
