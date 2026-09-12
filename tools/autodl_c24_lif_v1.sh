#!/usr/bin/env bash
set -Eeuo pipefail
C24_ACTION="${1:-status}"
case "$C24_ACTION" in start-direct|preflight-only|status|val|test|pack-complete|pack-light|_worker) ;;
 *) echo 'Usage: autodl_c24_lif_v1.sh start-direct|preflight-only|status|val|test|pack-complete|pack-light'; exit 2 ;;
esac
C24_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# This same script executes inside the actual tmux worker; no inherited activation assumptions.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
cd "$C24_ROOT"
export PYTHONPATH="$C24_ROOT/ultralytics-main:$C24_ROOT/tools"
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
case "$C24_ACTION" in
 start-direct|preflight-only|status|_worker) exec python -u tools/train_c24_lif_v1.py "$C24_ACTION" "${@:2}" ;;
 val|test) exec python -u tools/c24_lif_v1_results.py "$C24_ACTION" "${@:2}" ;;
 pack-light|pack-complete) exec python -u tools/c24_lif_v1_pack.py "$C24_ACTION" "${@:2}" ;;
esac
