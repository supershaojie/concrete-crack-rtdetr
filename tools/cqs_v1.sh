#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONPATH="$ROOT/ultralytics-main"
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
PYTHON=/root/miniconda3/envs/rtdetr/bin/python
if [[ "${1:-}" == sync ]]; then
  shift
  exec bash "$ROOT/tools/sync_cqs_v1.sh" "$@"
fi
exec "$PYTHON" -u "$ROOT/tools/cqs_v1.py" "$@"
