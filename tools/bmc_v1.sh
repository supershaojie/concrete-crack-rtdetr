#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
[[ "$SCRIPT_DIR" != "${BASH_SOURCE[0]}" ]] || SCRIPT_DIR=.
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${BMC_PYTHON:-/root/miniconda3/envs/rtdetr/bin/python}"
export PYTHONPATH="$ROOT/ultralytics-main:$ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false
cd "$ROOT"
if [[ "${1:-}" == "_session" ]]; then
    [[ $# == 2 ]] || { echo 'usage: bmc_v1.sh _session DISPATCH_ID' >&2; exit 2; }
    dispatch="$2"
    [[ "$dispatch" =~ ^[0-9TZ.]+_[a-f0-9]{8}$ ]] || { echo 'Invalid dispatch ID' >&2; exit 2; }
    log="$ROOT/outputs/bmc_v1/console_${dispatch}.log"
    set +e
    "$PYTHON" "$ROOT/tools/bmc_v1.py" _worker --dispatch "$dispatch" 2>&1 | tee -a "$log"
    codes=("${PIPESTATUS[@]}")
    set -e
    # Save the Python code immediately; tee success must never mask failure.
    "$PYTHON" "$ROOT/tools/bmc_v1.py" _exit --dispatch "$dispatch" --code "${codes[0]}" --tee-code "${codes[1]}"
    exit "${codes[0]}"
fi
exec "$PYTHON" "$ROOT/tools/bmc_v1.py" "$@"
