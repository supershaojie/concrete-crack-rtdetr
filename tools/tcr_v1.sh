#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/root/miniconda3/envs/rtdetr/bin/python
export PYTHONPATH="$ROOT/ultralytics-main"
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
cd "$ROOT"
if [[ "${1:-}" == sync ]]; then
  shift
  exec bash "$ROOT/tools/sync_tcr_v1.sh" "$@"
fi
if [[ "${1:-}" != _dispatch ]]; then
  exec "$PYTHON" -u "$ROOT/tools/tcr_v1.py" "$@"
fi
dispatch="${2:?Missing dispatch}"
mode="${3:?Missing start/resume}"
[[ "$dispatch" =~ ^[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$ ]] || exit 64
[[ "$mode" == start || "$mode" == resume ]] || exit 64
log="$ROOT/outputs/tcr_v1/console_${dispatch}.log"
[[ ! -e "$log" ]] || { printf '%s\n' 'Existing console preserved'; exit 1; }
args=(_worker --dispatch "$dispatch")
[[ "$mode" != resume ]] || args+=(--resume)
set +e
"$PYTHON" -u "$ROOT/tools/tcr_v1.py" "${args[@]}" 2>&1 | tee "$log"
codes=("${PIPESTATUS[@]}")
set -e
"$PYTHON" -u "$ROOT/tools/tcr_v1.py" _exit --dispatch "$dispatch" --python-code "${codes[0]}" --tee-code "${codes[1]}"
if (( codes[0] != 0 )); then exit "${codes[0]}"; fi
exit "${codes[1]}"
