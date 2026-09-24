#!/usr/bin/env bash
set -Eeuo pipefail
lbc_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
lbc_python="${LBC_PYTHON:-/root/miniconda3/envs/rtdetr/bin/python}"
export PYTHONPATH="$lbc_root/ultralytics-main:$lbc_root/tools${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$lbc_root"
exec "$lbc_python" "$lbc_root/tools/lbc_v1.py" "$@"
