#!/usr/bin/env bash
# LCR-AIFI uses its own worktree and run. Only start-direct starts formal training.
set -Eeuo pipefail
MODE="${1:-status}"
case "$MODE" in start-direct|status|val|test|pack-complete) ;; *) echo 'Expected start-direct/status/val/test/pack-complete'; exit 2 ;; esac
LCR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${LCR_PYTHON:-}" ]]; then
    [[ -x "$LCR_PYTHON" ]] || { echo 'LCR_PYTHON is not executable'; exit 1; }
    export CONDA_DEFAULT_ENV=rtdetr
else
    if command -v conda >/dev/null 2>&1; then LCR_CONDA="$(conda info --base)"
    elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then LCR_CONDA=/root/miniconda3
    else echo 'Existing rtdetr environment not found; set LCR_PYTHON'; exit 1; fi
    source "$LCR_CONDA/etc/profile.d/conda.sh"
    conda activate rtdetr
    LCR_PYTHON="$(command -v python)"
fi
cd "$LCR_ROOT"
export PYTHONPATH="$LCR_ROOT/ultralytics-main"
export YOLO_AUTOINSTALL=false
case "$MODE" in
    status) "$LCR_PYTHON" -u tools/train_lcr.py status lcr_aifi ;;
    start-direct|val|test|pack-complete)
        mkdir -p "$LCR_ROOT/outputs/lcr_command_logs"
        LCR_LOG="$(mktemp "$LCR_ROOT/outputs/lcr_command_logs/${MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
        set +e
        if [[ "$MODE" == start-direct ]]; then
            "$LCR_PYTHON" -u tools/train_lcr.py start-direct lcr_aifi 2>&1 | tee "$LCR_LOG"
            LCR_CODES=("${PIPESTATUS[@]}")
        else
            "$LCR_PYTHON" -u tools/lcr_results.py "$MODE" "${@:2}" 2>&1 | tee "$LCR_LOG"
            LCR_CODES=("${PIPESTATUS[@]}")
        fi
        set -e
        printf '%s\n' "${LCR_CODES[0]}" > "${LCR_LOG}.exit_code.txt"
        [[ "${LCR_CODES[0]}" -eq 0 ]] || exit "${LCR_CODES[0]}"
        exit "${LCR_CODES[1]}"
        ;;
esac
