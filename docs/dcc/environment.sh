# Source independently in every server command block. No GPU-idle gate.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
export DCC_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
export DCC_WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$DCC_WORKTREE/ultralytics-main${PYTHONPATH:+:$PYTHONPATH}"
export DCC_VARIANT=cbr_lif_dcc_v1
export DCC_META="$DCC_WORKTREE/outputs/dcc/$DCC_VARIANT"
export DCC_INIT="$DCC_WORKTREE/weights/${DCC_VARIANT}_controlled_init.pt"
export DCC_DATA="$DCC_MAIN/configs/crack_autodl.yaml"
cd "$DCC_WORKTREE"
