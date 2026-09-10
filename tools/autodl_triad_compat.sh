#!/usr/bin/env bash
set -Eeuo pipefail
TRIAD_VARIANT="${1:?Supply a registered variant}"
TRIAD_MODE="${2:-status}"
case "$TRIAD_VARIANT" in cscef_v6|scca_v2|cbr_v2|cscef_v6_scca_v2|cscef_v6_cbr_v2|scca_v2_cbr_v2|triad_v1) ;; *) echo 'Unknown variant'; exit 2 ;; esac
case "$TRIAD_MODE" in start-direct|status|val|test|pack-complete) ;; *) echo 'Unknown mode'; exit 2 ;; esac
TRIAD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
cd "$TRIAD_ROOT"
export PYTHONPATH="$TRIAD_ROOT/ultralytics-main"
export YOLO_AUTOINSTALL=false
if [[ "$TRIAD_MODE" == status ]]; then
    python -u tools/train_triad_compat.py status "$TRIAD_VARIANT"
else
    mkdir -p "$TRIAD_ROOT/outputs/triad_compat_command_logs/$TRIAD_VARIANT"
    TRIAD_LOG="$(mktemp "$TRIAD_ROOT/outputs/triad_compat_command_logs/$TRIAD_VARIANT/${TRIAD_MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
    set +e
    if [[ "$TRIAD_MODE" == start-direct ]]; then
        python -u tools/train_triad_compat.py start-direct "$TRIAD_VARIANT" 2>&1 | tee "$TRIAD_LOG"
        TRIAD_CODES=("${PIPESTATUS[@]}")
    else
        python -u tools/triad_compat_results.py "$TRIAD_MODE" "$TRIAD_VARIANT" "${@:3}" 2>&1 | tee "$TRIAD_LOG"
        TRIAD_CODES=("${PIPESTATUS[@]}")
    fi
    set -e
    printf '%s\n' "${TRIAD_CODES[0]}" > "${TRIAD_LOG}.exit_code.txt"
    [[ "${TRIAD_CODES[0]}" -eq 0 ]] || exit "${TRIAD_CODES[0]}"
    exit "${TRIAD_CODES[1]}"
fi
