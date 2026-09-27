#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/root/miniconda3/envs/rtdetr/bin/python
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false
export PYTHONPATH="$ROOT/ultralytics-main:$ROOT/tools"
cd "$ROOT"
exec "$PYTHON" -u "$ROOT/tools/rmd_v1.py" "$@"
