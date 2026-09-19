#!/usr/bin/env bash
# Every invocation activates the existing environment and asserts import identity.
set -Eeuo pipefail
if [[ "${1:-}" == "--help" || $# -eq 0 ]]; then
  echo 'Usage: bash tools/blc_server.sh environment|init|init-preflight|preflight|plan|start|resume|val|test|pack [--both]'
  echo 'BLC_VARIANT=cbr_lif_blc_v1 (default) or blc_v1. Only start/resume train.'
  exit 0
fi
BLC_SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
BLC_ROOT="$(cd -- "$BLC_SCRIPT_DIR/.." && pwd)"
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
set -u
cd "$BLC_ROOT"
export PYTHONPATH="$BLC_ROOT/ultralytics-main:$BLC_ROOT/tools"
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false
export BLC_VARIANT="${BLC_VARIANT:-cbr_lif_blc_v1}"
python -c 'from pathlib import Path; import ultralytics; root=Path.cwd(); assert Path(ultralytics.__file__).resolve()==root/"ultralytics-main/ultralytics/__init__.py"; print("ultralytics:",ultralytics.__file__)'
mkdir -p "outputs/blc_v1/$BLC_VARIANT/logs"
BLC_LOG="$(mktemp "outputs/blc_v1/$BLC_VARIANT/logs/${1}_$(date +%Y%m%dT%H%M%S)_XXXXXX.log")"
set +e
python -u tools/blc_server.py "$@" --variant "$BLC_VARIANT" 2>&1 | tee "$BLC_LOG"
BLC_CODES=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${BLC_CODES[0]}" > "$BLC_LOG.exit_code.txt"
printf 'BLC exit=%s, tee exit=%s, log=%s/%s\n' "${BLC_CODES[0]}" "${BLC_CODES[1]}" "$BLC_ROOT" "$BLC_LOG"
[[ "${BLC_CODES[0]}" -eq 0 ]] || exit "${BLC_CODES[0]}"
exit "${BLC_CODES[1]}"
