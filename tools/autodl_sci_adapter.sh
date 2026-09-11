#!/usr/bin/env bash
set -Eeuo pipefail
SCI_ADAPTER_VARIANT="${1:?Supply a registered variant}"
SCI_ADAPTER_MODE="${2:-status}"
case "$SCI_ADAPTER_VARIANT" in cscef_v51_sci_control|scca_sci_cscef_v51|scca_sci_cscef_v51_cbr) ;; *) echo 'Unknown variant'; exit 2 ;; esac
case "$SCI_ADAPTER_MODE" in start-direct|status|val|test|pack-complete) ;; *) echo 'Unknown mode'; exit 2 ;; esac
SCI_ADAPTER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
cd "$SCI_ADAPTER_ROOT"
export PYTHONPATH="$SCI_ADAPTER_ROOT/ultralytics-main"
export YOLO_AUTOINSTALL=false
if [[ "$SCI_ADAPTER_MODE" == status ]]; then
    python -u tools/train_sci_adapter.py status "$SCI_ADAPTER_VARIANT"
else
    mkdir -p "$SCI_ADAPTER_ROOT/outputs/sci_adapter_command_logs/$SCI_ADAPTER_VARIANT"
    SCI_ADAPTER_LOG="$(mktemp "$SCI_ADAPTER_ROOT/outputs/sci_adapter_command_logs/$SCI_ADAPTER_VARIANT/${SCI_ADAPTER_MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
    set +e
    if [[ "$SCI_ADAPTER_MODE" == start-direct ]]; then
        python -u tools/train_sci_adapter.py start-direct "$SCI_ADAPTER_VARIANT" 2>&1 | tee "$SCI_ADAPTER_LOG"
        SCI_ADAPTER_CODES=("${PIPESTATUS[@]}")
    else
        python -u tools/sci_adapter_results.py "$SCI_ADAPTER_MODE" "$SCI_ADAPTER_VARIANT" "${@:3}" 2>&1 | tee "$SCI_ADAPTER_LOG"
        SCI_ADAPTER_CODES=("${PIPESTATUS[@]}")
    fi
    set -e
    printf '%s\n' "${SCI_ADAPTER_CODES[0]}" > "${SCI_ADAPTER_LOG}.exit_code.txt"
    [[ "${SCI_ADAPTER_CODES[0]}" -eq 0 ]] || exit "${SCI_ADAPTER_CODES[0]}"
    exit "${SCI_ADAPTER_CODES[1]}"
fi
