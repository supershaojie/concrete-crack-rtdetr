#!/usr/bin/env bash
# LIF-Down uses its own worktree and run. Only start-direct starts formal training.
set -Eeuo pipefail
MODE="${1:-status}"
case "$MODE" in start-direct|status|val|test|pack-complete) ;; *) echo 'Expected start-direct/status/val/test/pack-complete'; exit 2 ;; esac
LIF_DOWN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${LIF_DOWN_PYTHON:-}" ]]; then
    [[ -x "$LIF_DOWN_PYTHON" ]] || { echo 'LIF_DOWN_PYTHON is not executable'; exit 1; }
    export CONDA_DEFAULT_ENV=rtdetr
else
    if command -v conda >/dev/null 2>&1; then LIF_DOWN_CONDA="$(conda info --base)"
    elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then LIF_DOWN_CONDA=/root/miniconda3
    else echo 'Existing rtdetr environment not found; set LIF_DOWN_PYTHON'; exit 1; fi
    source "$LIF_DOWN_CONDA/etc/profile.d/conda.sh"
    conda activate rtdetr
    LIF_DOWN_PYTHON="$(command -v python)"
fi
cd "$LIF_DOWN_ROOT"
export PYTHONPATH="$LIF_DOWN_ROOT/ultralytics-main"
export YOLO_AUTOINSTALL=false
case "$MODE" in
    status) "$LIF_DOWN_PYTHON" -u tools/train_lif_down.py status lif_down ;;
    start-direct|val|test|pack-complete)
        mkdir -p "$LIF_DOWN_ROOT/outputs/lif_down_command_logs"
        LIF_DOWN_LOG="$(mktemp "$LIF_DOWN_ROOT/outputs/lif_down_command_logs/${MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
        set +e
        if [[ "$MODE" == start-direct ]]; then
            "$LIF_DOWN_PYTHON" -u tools/train_lif_down.py start-direct lif_down 2>&1 | tee "$LIF_DOWN_LOG"
            LIF_DOWN_CODES=("${PIPESTATUS[@]}")
        else
            "$LIF_DOWN_PYTHON" -u tools/lif_down_results.py "$MODE" "${@:2}" 2>&1 | tee "$LIF_DOWN_LOG"
            LIF_DOWN_CODES=("${PIPESTATUS[@]}")
        fi
        set -e
        printf '%s\n' "${LIF_DOWN_CODES[0]}" > "${LIF_DOWN_LOG}.exit_code.txt"
        [[ "${LIF_DOWN_CODES[0]}" -eq 0 ]] || exit "${LIF_DOWN_CODES[0]}"
        exit "${LIF_DOWN_CODES[1]}"
        ;;
esac
