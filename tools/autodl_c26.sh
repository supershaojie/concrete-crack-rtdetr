#!/usr/bin/env bash
# C26 is independent of C25. Only start-direct starts formal training.
set -Eeuo pipefail
MODE="${1:-status}"
case "$MODE" in start-direct|status|val|test|pack-complete) ;; *) echo 'Expected start-direct/status/val/test/pack-complete'; exit 2 ;; esac
C26_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v conda >/dev/null 2>&1; then
    C26_CONDA="$(conda info --base)"
elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    C26_CONDA=/root/miniconda3
else
    echo 'Existing rtdetr conda environment not found'; exit 1
fi
source "$C26_CONDA/etc/profile.d/conda.sh"
conda activate rtdetr
cd "$C26_ROOT"
export PYTHONPATH="$C26_ROOT/ultralytics-main"
export YOLO_AUTOINSTALL=false
case "$MODE" in
    status) python -u tools/train_c26.py status c26 ;;
    start-direct|val|test|pack-complete)
        mkdir -p "$C26_ROOT/outputs/c26_command_logs"
        C26_LOG="$(mktemp "$C26_ROOT/outputs/c26_command_logs/${MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
        set +e
        if [[ "$MODE" == start-direct ]]; then
            python -u tools/train_c26.py start-direct c26 2>&1 | tee "$C26_LOG"
            C26_CODES=("${PIPESTATUS[@]}")
        else
            python -u tools/c26_results.py "$MODE" "${@:2}" 2>&1 | tee "$C26_LOG"
            C26_CODES=("${PIPESTATUS[@]}")
        fi
        set -e
        printf '%s\n' "${C26_CODES[0]}" > "${C26_LOG}.exit_code.txt"
        [[ "${C26_CODES[0]}" -eq 0 ]] || exit "${C26_CODES[0]}"
        exit "${C26_CODES[1]}"
        ;;
esac
