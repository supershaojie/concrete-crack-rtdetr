#!/usr/bin/env bash
set -Eeuo pipefail
MINCOMPAT_VARIANT="${1:?Supply a registered variant}"
MINCOMPAT_MODE="${2:-status}"
case "$MINCOMPAT_VARIANT" in cscef_v52_compat|cscef_v52_scca_compat|triad_mincompat_v2) ;; *) echo 'Unknown variant'; exit 2 ;; esac
case "$MINCOMPAT_MODE" in start-direct|status|val|test|pack-complete) ;; *) echo 'Unknown mode'; exit 2 ;; esac
MINCOMPAT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
cd "$MINCOMPAT_ROOT"
export PYTHONPATH="$MINCOMPAT_ROOT/ultralytics-main"
export YOLO_AUTOINSTALL=false
if [[ "$MINCOMPAT_MODE" == status ]]; then
    python -u tools/train_mincompat_v2.py status "$MINCOMPAT_VARIANT"
else
    mkdir -p "$MINCOMPAT_ROOT/outputs/mincompat_v2_command_logs/$MINCOMPAT_VARIANT"
    MINCOMPAT_LOG="$(mktemp "$MINCOMPAT_ROOT/outputs/mincompat_v2_command_logs/$MINCOMPAT_VARIANT/${MINCOMPAT_MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
    set +e
    if [[ "$MINCOMPAT_MODE" == start-direct ]]; then
        python -u tools/train_mincompat_v2.py start-direct "$MINCOMPAT_VARIANT" 2>&1 | tee "$MINCOMPAT_LOG"
        MINCOMPAT_CODES=("${PIPESTATUS[@]}")
    else
        python -u tools/mincompat_v2_results.py "$MINCOMPAT_MODE" "$MINCOMPAT_VARIANT" "${@:3}" 2>&1 | tee "$MINCOMPAT_LOG"
        MINCOMPAT_CODES=("${PIPESTATUS[@]}")
    fi
    set -e
    printf '%s\n' "${MINCOMPAT_CODES[0]}" > "${MINCOMPAT_LOG}.exit_code.txt"
    [[ "${MINCOMPAT_CODES[0]}" -eq 0 ]] || exit "${MINCOMPAT_CODES[0]}"
    exit "${MINCOMPAT_CODES[1]}"
fi
