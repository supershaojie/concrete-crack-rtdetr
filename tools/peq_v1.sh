#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON=/root/miniconda3/envs/rtdetr/bin/python
export PYTHONPATH="$ROOT/ultralytics-main:$ROOT/tools"
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false
cd "$ROOT"
if [[ "${1:-}" == sync ]]; then
    shift
    exec bash "$ROOT/tools/sync_peq_v1.sh" "$@"
fi
exec "$PYTHON" "$ROOT/tools/peq_v1.py" "$@"
