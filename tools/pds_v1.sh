#!/usr/bin/env bash
set -Eeuo pipefail
PDS_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
export PDS_MAIN="${PDS_MAIN:-/root/autodl-tmp/projects/Crack_RTDETR}"
export PYTHONPATH="$PDS_ROOT/ultralytics-main:$PDS_ROOT/tools"
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
cd -- "$PDS_ROOT"
if [[ "${1:-}" == sync ]]; then
    shift
    exec bash "$PDS_ROOT/tools/sync_pds_v1.sh" "$@"
fi
exec /root/miniconda3/envs/rtdetr/bin/python -u "$PDS_ROOT/tools/pds_v1.py" "$@"
