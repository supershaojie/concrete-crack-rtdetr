#!/usr/bin/env bash
# Explicit start-direct trains; all other commands are read-only or evaluation/export.
set -Eeuo pipefail
MODE="${1:-status}"
VARIANT="${2:-c24}"
VARIANT="${VARIANT,,}"
case "$MODE" in start-direct|status|val|test|pack|diagnose|check) ;; *) echo 'Unknown command'; exit 2 ;; esac
case "$VARIANT" in c24|c25) ;; *) echo 'Expected c24 or c25'; exit 2 ;; esac
SCCA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v conda >/dev/null 2>&1; then
    SCCA_CONDA="$(conda info --base)"
elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    SCCA_CONDA=/root/miniconda3
else
    echo 'Existing rtdetr conda environment not found'; exit 1
fi
source "$SCCA_CONDA/etc/profile.d/conda.sh"
conda activate rtdetr
cd "$SCCA_ROOT"
export PYTHONPATH="$SCCA_ROOT/ultralytics-main"
export YOLO_AUTOINSTALL=false
case "$MODE" in
    start-direct|status) python tools/train_scca.py "$MODE" "$VARIANT" ;;
    val|test|pack|diagnose) python tools/scca_results.py "$MODE" "$VARIANT" "${@:3}" ;;
    check) python tools/check_scca.py --source "${SCCA_MAIN:-/root/autodl-tmp/projects/Crack_RTDETR}/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" "${@:3}" ;;
esac
