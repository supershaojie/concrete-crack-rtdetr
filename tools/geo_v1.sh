#!/usr/bin/env bash
set -Eeuo pipefail
if [[ ${1:-} == --help ]]; then
  echo 'Usage: bash geo_v1.sh {prepare|preflight|status|start|resume|val|test|finish|pack} [options]'
  echo 'Subcommand help: bash geo_v1.sh preflight --help (fixed server Python required)'
  exit 0
fi
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$HERE"
export PYTHONPATH="$HERE/ultralytics-main:$HERE/tools${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false
exec /root/miniconda3/envs/rtdetr/bin/python "$HERE/tools/geo_v1.py" "$@"
