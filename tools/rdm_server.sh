#!/usr/bin/env bash
set -Eeuo pipefail
RDM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export RDM_MAIN="${RDM_MAIN:-/root/autodl-tmp/projects/Crack_RTDETR}"
export RDM_DATA="${RDM_DATA:-$RDM_MAIN/configs/crack_autodl.yaml}"
export RDM_VARIANT="${RDM_VARIANT:-cbr_lif_rdm_v1}"
next_variant=0
for arg in "$@"; do
  if (( next_variant )); then export RDM_VARIANT="$arg"; next_variant=0; continue; fi
  case "$arg" in --variant) next_variant=1;; --variant=*) export RDM_VARIANT="${arg#*=}";; esac
done
case "$RDM_VARIANT" in cbr_lif_rdm_v1|rdm_v1) ;; *) echo 'Unknown RDM_VARIANT' >&2; exit 2;; esac
source "${RDM_CONDA_BASE:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate rtdetr
export PYTHONPATH="$RDM_ROOT/ultralytics-main:$RDM_ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$RDM_ROOT"
action="${1:---help}"
folder="$RDM_ROOT/outputs/rdm_v1/$RDM_VARIANT"
mkdir -p "$folder"
log="$folder/${action//[^a-zA-Z0-9_-]/_}_$(date -u +%Y%m%dT%H%M%S)_$$.log"
set +e
python -u tools/rdm.py "$@" --variant "$RDM_VARIANT" 2>&1 | tee "$log"
code=${PIPESTATUS[0]}
set -e
printf 'RDM exit=%s log=%s\n' "$code" "$log"
exit "$code"
