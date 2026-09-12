#!/usr/bin/env bash
# C19 + LIF-Down uses its own worktree and run. Only start-direct starts formal training.
set -Eeuo pipefail
MODE="${1:-status}"
case "$MODE" in start-direct|status|val|test|pack-complete) ;; *) echo 'Expected start-direct/status/val/test/pack-complete'; exit 2 ;; esac
C19_LIF_V1_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -eq 1 ]] || { echo 'Usage: bash tools/autodl_c19_lif_v1.sh start-direct|status|val|test|pack-complete'; exit 2; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
C19_LIF_V1_PYTHON="$(command -v python)"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$C19_LIF_V1_ROOT"
export PYTHONPATH="$C19_LIF_V1_ROOT/ultralytics-main"
export YOLO_AUTOINSTALL=false
case "$MODE" in
    status) "$C19_LIF_V1_PYTHON" -u tools/train_c19_lif_v1.py status ;;
    start-direct|val|test|pack-complete)
        mkdir -p "$C19_LIF_V1_ROOT/outputs/c19_lif_v1_command_logs"
        C19_LIF_V1_LOG="$(mktemp "$C19_LIF_V1_ROOT/outputs/c19_lif_v1_command_logs/${MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
        set +e
        if [[ "$MODE" == start-direct ]]; then
            "$C19_LIF_V1_PYTHON" -u tools/train_c19_lif_v1.py start-direct 2>&1 | tee "$C19_LIF_V1_LOG"
            C19_LIF_V1_CODES=("${PIPESTATUS[@]}")
        else
            "$C19_LIF_V1_PYTHON" -u tools/c19_lif_v1_results.py "$MODE" "${@:2}" 2>&1 | tee "$C19_LIF_V1_LOG"
            C19_LIF_V1_CODES=("${PIPESTATUS[@]}")
        fi
        set -e
        printf '%s\n' "${C19_LIF_V1_CODES[0]}" > "${C19_LIF_V1_LOG}.exit_code.txt"
        [[ "${C19_LIF_V1_CODES[0]}" -eq 0 ]] || exit "${C19_LIF_V1_CODES[0]}"
        exit "${C19_LIF_V1_CODES[1]}"
        ;;
esac
