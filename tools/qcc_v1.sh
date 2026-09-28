#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT/ultralytics-main:$ROOT/tools"
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false
cd -- "$ROOT"
exec /root/miniconda3/envs/rtdetr/bin/python -u "$ROOT/tools/qcc_v1.py" "$@"
