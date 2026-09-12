#!/usr/bin/env bash
# C17 + LIF v1 uses its own worktree and run. Only start-direct starts formal training.
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo 'Usage: bash tools/autodl_c17_lif_v1.sh start-direct|status|val|test|pack-complete'; exit 2; }
MODE="$1"
case "$MODE" in start-direct|status|val|test|pack-complete) ;; *) echo 'Expected start-direct/status/val/test/pack-complete'; exit 2 ;; esac
C17_LIF_V1_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${C17_LIF_V1_PYTHON:-}" ]]; then
    [[ -x "$C17_LIF_V1_PYTHON" ]] || { echo 'C17_LIF_V1_PYTHON is not executable'; exit 1; }
    export CONDA_DEFAULT_ENV=rtdetr
else
    if command -v conda >/dev/null 2>&1; then C17_LIF_V1_CONDA="$(conda info --base)"
    elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then C17_LIF_V1_CONDA=/root/miniconda3
    else echo 'Existing rtdetr environment not found; set C17_LIF_V1_PYTHON'; exit 1; fi
    source "$C17_LIF_V1_CONDA/etc/profile.d/conda.sh"
    conda activate rtdetr
    C17_LIF_V1_PYTHON="$(command -v python)"
fi
cd "$C17_LIF_V1_ROOT"
export PYTHONPATH="$C17_LIF_V1_ROOT/ultralytics-main"
export YOLO_AUTOINSTALL=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
case "$MODE" in
    status) "$C17_LIF_V1_PYTHON" -u tools/train_c17_lif_v1.py status ;;
    start-direct|val|test|pack-complete)
        mkdir -p "$C17_LIF_V1_ROOT/outputs/c17_lif_v1_command_logs"
        C17_LIF_V1_LOG="$(mktemp "$C17_LIF_V1_ROOT/outputs/c17_lif_v1_command_logs/${MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
        set +e
        if [[ "$MODE" == start-direct ]]; then
            "$C17_LIF_V1_PYTHON" -u tools/train_c17_lif_v1.py start-direct 2>&1 | tee "$C17_LIF_V1_LOG"
            C17_LIF_V1_CODES=("${PIPESTATUS[@]}")
        else
            "$C17_LIF_V1_PYTHON" -u tools/c17_lif_v1_results.py "$MODE" "${@:2}" 2>&1 | tee "$C17_LIF_V1_LOG"
            C17_LIF_V1_CODES=("${PIPESTATUS[@]}")
        fi
        set -e
        printf '%s\n' "${C17_LIF_V1_CODES[0]}" > "${C17_LIF_V1_LOG}.exit_code.txt"
        [[ "${C17_LIF_V1_CODES[0]}" -eq 0 ]] || exit "${C17_LIF_V1_CODES[0]}"
        exit "${C17_LIF_V1_CODES[1]}"
        ;;
esac
