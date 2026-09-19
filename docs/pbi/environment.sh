#!/usr/bin/env bash
# Source separately in every command. Conda activation may reference unset vars.
PBI_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PBI_HAD_NOUNSET=0
[[ $- != *u* ]] || PBI_HAD_NOUNSET=1
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
[[ "$PBI_HAD_NOUNSET" == 0 ]] || set -u
unset PBI_HAD_NOUNSET
export PBI_WORKTREE="$PBI_ENV_ROOT"
export PBI_MAIN="${PBI_MAIN:-/root/autodl-tmp/projects/Crack_RTDETR}"
export PBI_VARIANT="${PBI_VARIANT:-cbr_lif_pbi_v1}"
case "$PBI_VARIANT" in cbr_lif_pbi_v1|pbi_v1) ;; *) printf 'Unknown PBI_VARIANT: %s\n' "$PBI_VARIANT" >&2; return 2 ;; esac
export PBI_META="$PBI_WORKTREE/outputs/pbi/$PBI_VARIANT"
export PBI_INIT="$PBI_WORKTREE/weights/${PBI_VARIANT}_controlled_init.pt"
export PBI_DATA="${PBI_DATA:-$PBI_MAIN/configs/crack_autodl.yaml}"
export PYTHONPATH="$PBI_WORKTREE/ultralytics-main${PYTHONPATH:+:$PYTHONPATH}"
cd "$PBI_WORKTREE"
python - <<'PY'
import os
from pathlib import Path
import ultralytics
expected = Path(os.environ['PBI_WORKTREE']) / 'ultralytics-main'
actual = Path(ultralytics.__file__).resolve()
assert actual.is_relative_to(expected.resolve()), (str(actual), str(expected))
print('PBI worktree import:', actual)
PY
