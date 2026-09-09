#!/usr/bin/env bash
# FSA is independent of other experiments. Only start-direct starts formal training.
set -Eeuo pipefail
MODE="${1:-status}"
case "$MODE" in start-direct|status|val|test|pack-complete) ;; *) echo 'Expected start-direct/status/val/test/pack-complete'; exit 2 ;; esac
FSA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v conda >/dev/null 2>&1; then
    FSA_CONDA="$(conda info --base)"
elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    FSA_CONDA=/root/miniconda3
else
    echo 'Existing rtdetr conda environment not found'; exit 1
fi
source "$FSA_CONDA/etc/profile.d/conda.sh"
conda activate rtdetr
cd "$FSA_ROOT"
export PYTHONPATH="$FSA_ROOT/ultralytics-main"
export YOLO_AUTOINSTALL=false
case "$MODE" in
    status) python -u tools/train_fsa_deform.py status fsa_deform ;;
    start-direct|val|test|pack-complete)
        mkdir -p "$FSA_ROOT/outputs/fsa_deform_command_logs"
        FSA_LOG="$(mktemp "$FSA_ROOT/outputs/fsa_deform_command_logs/${MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
        set +e
        if [[ "$MODE" == start-direct ]]; then
            python -u tools/train_fsa_deform.py start-direct fsa_deform 2>&1 | tee "$FSA_LOG"
            FSA_CODES=("${PIPESTATUS[@]}")
        else
            python -u tools/fsa_deform_results.py "$MODE" "${@:2}" 2>&1 | tee "$FSA_LOG"
            FSA_CODES=("${PIPESTATUS[@]}")
        fi
        set -e
        printf '%s\n' "${FSA_CODES[0]}" > "${FSA_LOG}.exit_code.txt"
        [[ "${FSA_CODES[0]}" -eq 0 ]] || exit "${FSA_CODES[0]}"
        exit "${FSA_CODES[1]}"
        ;;
esac
