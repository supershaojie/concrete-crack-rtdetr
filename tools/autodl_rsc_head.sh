#!/usr/bin/env bash
# RSC-Head uses its own worktree and run. Only start-direct starts formal training.
set -Eeuo pipefail
MODE="${1:-status}"
case "$MODE" in start-direct|status|val|test|pack-complete) ;; *) echo 'Expected start-direct/status/val/test/pack-complete'; exit 2 ;; esac
RSC_HEAD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${RSC_HEAD_PYTHON:-}" ]]; then
    [[ -x "$RSC_HEAD_PYTHON" ]] || { echo 'RSC_HEAD_PYTHON is not executable'; exit 1; }
    export CONDA_DEFAULT_ENV=rtdetr
else
    if command -v conda >/dev/null 2>&1; then RSC_HEAD_CONDA="$(conda info --base)"
    elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then RSC_HEAD_CONDA=/root/miniconda3
    else echo 'Existing rtdetr environment not found; set RSC_HEAD_PYTHON'; exit 1; fi
    source "$RSC_HEAD_CONDA/etc/profile.d/conda.sh"
    conda activate rtdetr
    RSC_HEAD_PYTHON="$(command -v python)"
fi
cd "$RSC_HEAD_ROOT"
export PYTHONPATH="$RSC_HEAD_ROOT/ultralytics-main"
export YOLO_AUTOINSTALL=false
case "$MODE" in
    status) "$RSC_HEAD_PYTHON" -u tools/train_rsc_head.py status rsc_head ;;
    start-direct|val|test|pack-complete)
        mkdir -p "$RSC_HEAD_ROOT/outputs/rsc_head_command_logs"
        RSC_HEAD_LOG="$(mktemp "$RSC_HEAD_ROOT/outputs/rsc_head_command_logs/${MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
        set +e
        if [[ "$MODE" == start-direct ]]; then
            "$RSC_HEAD_PYTHON" -u tools/train_rsc_head.py start-direct rsc_head 2>&1 | tee "$RSC_HEAD_LOG"
            RSC_HEAD_CODES=("${PIPESTATUS[@]}")
        else
            "$RSC_HEAD_PYTHON" -u tools/rsc_head_results.py "$MODE" "${@:2}" 2>&1 | tee "$RSC_HEAD_LOG"
            RSC_HEAD_CODES=("${PIPESTATUS[@]}")
        fi
        set -e
        printf '%s\n' "${RSC_HEAD_CODES[0]}" > "${RSC_HEAD_LOG}.exit_code.txt"
        [[ "${RSC_HEAD_CODES[0]}" -eq 0 ]] || exit "${RSC_HEAD_CODES[0]}"
        exit "${RSC_HEAD_CODES[1]}"
        ;;
esac
