#!/usr/bin/env bash
# DRA-AIFI uses its own worktree and run. Only start-direct starts formal training.
set -Eeuo pipefail
MODE="${1:-status}"
case "$MODE" in start-direct|status|val|test|pack-complete) ;; *) echo 'Expected start-direct/status/val/test/pack-complete'; exit 2 ;; esac
DRA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v conda >/dev/null 2>&1; then
    DRA_CONDA="$(conda info --base)"
elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    DRA_CONDA=/root/miniconda3
else
    echo 'Existing rtdetr conda environment not found'; exit 1
fi
source "$DRA_CONDA/etc/profile.d/conda.sh"
conda activate rtdetr
cd "$DRA_ROOT"
export PYTHONPATH="$DRA_ROOT/ultralytics-main"
export YOLO_AUTOINSTALL=false
case "$MODE" in
    status) python -u tools/train_dra.py status dra_aifi ;;
    start-direct|val|test|pack-complete)
        mkdir -p "$DRA_ROOT/outputs/dra_command_logs"
        DRA_LOG="$(mktemp "$DRA_ROOT/outputs/dra_command_logs/${MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
        set +e
        if [[ "$MODE" == start-direct ]]; then
            python -u tools/train_dra.py start-direct dra_aifi 2>&1 | tee "$DRA_LOG"
            DRA_CODES=("${PIPESTATUS[@]}")
        else
            python -u tools/dra_results.py "$MODE" "${@:2}" 2>&1 | tee "$DRA_LOG"
            DRA_CODES=("${PIPESTATUS[@]}")
        fi
        set -e
        printf '%s\n' "${DRA_CODES[0]}" > "${DRA_LOG}.exit_code.txt"
        [[ "${DRA_CODES[0]}" -eq 0 ]] || exit "${DRA_CODES[0]}"
        exit "${DRA_CODES[1]}"
        ;;
esac
