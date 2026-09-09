#!/usr/bin/env bash
# EVC is independent of other experiments. Only start-direct starts formal training.
set -Eeuo pipefail
MODE="${1:-status}"
case "$MODE" in start-direct|status|val|test|pack-complete) ;; *) echo 'Expected start-direct/status/val/test/pack-complete'; exit 2 ;; esac
EVC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v conda >/dev/null 2>&1; then
    EVC_CONDA="$(conda info --base)"
elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    EVC_CONDA=/root/miniconda3
else
    echo 'Existing rtdetr conda environment not found'; exit 1
fi
source "$EVC_CONDA/etc/profile.d/conda.sh"
conda activate rtdetr
cd "$EVC_ROOT"
export PYTHONPATH="$EVC_ROOT/ultralytics-main"
export YOLO_AUTOINSTALL=false
case "$MODE" in
    status) python -u tools/train_evc.py status evc_deform ;;
    start-direct|val|test|pack-complete)
        mkdir -p "$EVC_ROOT/outputs/evc_command_logs"
        EVC_LOG="$(mktemp "$EVC_ROOT/outputs/evc_command_logs/${MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
        set +e
        if [[ "$MODE" == start-direct ]]; then
            python -u tools/train_evc.py start-direct evc_deform 2>&1 | tee "$EVC_LOG"
            EVC_CODES=("${PIPESTATUS[@]}")
        else
            python -u tools/evc_results.py "$MODE" "${@:2}" 2>&1 | tee "$EVC_LOG"
            EVC_CODES=("${PIPESTATUS[@]}")
        fi
        set -e
        printf '%s\n' "${EVC_CODES[0]}" > "${EVC_LOG}.exit_code.txt"
        [[ "${EVC_CODES[0]}" -eq 0 ]] || exit "${EVC_CODES[0]}"
        exit "${EVC_CODES[1]}"
        ;;
esac
