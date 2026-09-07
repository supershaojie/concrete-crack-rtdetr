#!/usr/bin/env bash
# Read-only diagnosis. No installation, optimizer, training, prepare/start/smoke.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-help}"
if [[ "$MODE" == help || "$MODE" == --help ]]; then
    echo 'Usage: bash tools/autodl_cbr_rescue.sh run --output DIR [--main DIR] [--evidence FILE] [--c17 FILE ...]'
    echo '       bash tools/autodl_cbr_rescue.sh pack --input DIR --output FILE.tar.gz'
    exit 0
fi
case "$MODE" in run|pack) ;; *) echo "Unknown diagnosis command: $MODE" >&2; exit 2 ;; esac
shift
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    CONDA_BASE=/root/miniconda3
else
    echo 'Existing conda/rtdetr environment not found; no dependency installation attempted.' >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate rtdetr
cd "$ROOT"
export PYTHONPATH="$ROOT/ultralytics-main"
export YOLO_AUTOINSTALL=false
if [[ "$MODE" == run ]]; then
    python - <<'PY'
import sys, torch
print('Python:', sys.executable, 'Torch:', torch.__version__, 'CUDA:', torch.version.cuda, flush=True)
assert torch.__version__.split('+')[0] == '2.1.2', 'Expected existing server PyTorch 2.1.2; do not upgrade dependencies'
assert torch.cuda.is_available(), 'Server CUDA unavailable'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY
fi
exec python tools/diagnose_cbr_rescue.py "$MODE" "$@"
