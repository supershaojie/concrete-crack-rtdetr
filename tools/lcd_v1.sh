#!/usr/bin/env bash
set -euo pipefail
lcd_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$lcd_root/ultralytics-main${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_AUTOINSTALL=false
exec /root/miniconda3/envs/rtdetr/bin/python -u "$lcd_root/tools/lcd_v1.py" "$@"
