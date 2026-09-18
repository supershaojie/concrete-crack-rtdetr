#!/usr/bin/env bash
# Separate, explicit server actions. This script never chains preflight into start.
set -Eeuo pipefail
ACTION=${1:?Usage: bash tools/gra_server.sh environment SHA | init-preflight | plan | start | resume | val | test | pack}
WT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
ENV_FILE="$WT/outputs/gra/server.env"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
if [[ "$ACTION" == environment ]]; then
    SHA=${2:?Full delivery SHA required}
    [[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
    [[ $(git -C "$WT" rev-parse HEAD) == "$SHA" ]]
    git -C "$WT" merge-base --is-ancestor a0459d6a652cb702699087c88fa39a3e4c4087ec "$SHA"
    [[ -f "$WT/.git" ]]
    mkdir -p "$WT/outputs/gra"
    if [[ -e "$ENV_FILE" ]]; then
        source "$ENV_FILE"
        [[ "$GRA_SHA" == "$SHA" && "$GRA_WORKTREE" == "$WT" ]]
    else
        (set -o noclobber
         printf 'export GRA_MAIN=%q\nexport GRA_WORKTREE=%q\nexport GRA_SHA=%q\nexport PYTHONPATH=%q\n' \
             "$MAIN" "$WT" "$SHA" "$WT/ultralytics-main" > "$ENV_FILE")
    fi
fi
source "$ENV_FILE"
[[ "$GRA_WORKTREE" == "$WT" ]]
[[ $(git -C "$WT" rev-parse HEAD) == "$GRA_SHA" ]]
export PYTHONPATH="$WT/ultralytics-main"
cd "$WT"
V=cbr_lif_gra_v1
META="$WT/outputs/gra/$V"
INIT="$WT/weights/${V}_controlled_init.pt"
SOURCE="$MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
DATA="$MAIN/configs/crack_autodl.yaml"
CHECKS="$WT/outputs/gra/server_checks/checks.json"
PREFLIGHT="$META/server_preflight/preflight.json"
RUN="$MAIN/runs/c_series/cbr_lif_gra_v1_rtdetr_r18_lite_e200_b16_onlineaug"
COMMON=(--variant "$V" --initialized "$INIT" --audit "$META/init_audit.json" --source "$SOURCE" \
        --data "$DATA" --checks "$CHECKS" --preflight "$PREFLIGHT" --expected-sha "$GRA_SHA")
mkdir -p "$WT/outputs/gra/logs"
# A subshell and pipefail preserve the actual action failure through tee.
(
case "$ACTION" in
environment)
    python - <<'PY'
import os,sys,torch,ultralytics
from pathlib import Path
assert Path(ultralytics.__file__).resolve().is_relative_to(Path(os.environ['GRA_WORKTREE']).resolve())
print('Python:',sys.version,'Torch:',torch.__version__,'CUDA:',torch.version.cuda)
print('Import:',ultralytics.__file__,'SHA:',os.environ['GRA_SHA'])
print('CUDA available:',torch.cuda.is_available())
PY
    nvidia-smi || true
    ;;
init-preflight)
    for variant in cbr_lif_gra_v1 gra_v1; do
        init="$WT/weights/${variant}_controlled_init.pt"
        audit="$WT/outputs/gra/$variant/init_audit.json"
        if [[ -e "$init" ]]; then
            python - "$variant" "$init" "$audit" "$SOURCE" <<'PY'
import sys
sys.path.insert(0,'tools')
from train_gra import verify_initialization
verify_initialization(*sys.argv[1:])
print('Existing controlled initialization verified:',sys.argv[1])
PY
        else
            python tools/init_gra.py "$variant" --source "$SOURCE" --output "$init" --report "$audit"
        fi
    done
    python tools/check_gra.py --source "$SOURCE" --dataset "$MAIN/datasets/crack_det" \
        --output "$WT/outputs/gra/server_checks" --device all
    python tools/check_gra_native.py --checks "$WT/outputs/gra/server_checks" --dataset "$MAIN/datasets/crack_det" \
        --output "$WT/outputs/gra/server_native"
    python tools/preflight_gra.py --variant "$V" --initialized "$INIT" --audit "$META/init_audit.json" \
        --source "$SOURCE" --data "$DATA" --output "$META/server_preflight" --max-batches 16
    python tools/train_gra.py plan --variant "$V" --initialized "$INIT" --data "$DATA" --output "$META/formal_plan.json"
    ;;
plan)
    python tools/train_gra.py plan --variant "$V" --initialized "$INIT" --data "$DATA"
    ;;
start)
    python tools/train_gra.py start "${COMMON[@]}"
    ;;
resume)
    python tools/train_gra.py resume "${COMMON[@]}" --checkpoint "$RUN/weights/last.pt"
    ;;
val)
    BEST_SHA=$(sha256sum "$RUN/weights/best.pt" | cut -d ' ' -f1)
    python tools/eval_gra.py evaluate --variant "$V" --weights "$RUN/weights/best.pt" --weights-sha "$BEST_SHA" \
        --data "$DATA" --split val --device 0 --output "$META/eval/val"
    ;;
test)
    BEST_SHA=$(python - "$META/eval/val/metrics.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['checkpoint_sha256'])
PY
)
    python tools/eval_gra.py evaluate --variant "$V" --weights "$RUN/weights/best.pt" --weights-sha "$BEST_SHA" \
        --data "$DATA" --split test --device 0 --val-report "$META/eval/val/metrics.json" --output "$META/eval/test"
    ;;
pack)
    python tools/pack_gra_light.py --variant "$V" --run "$RUN" --evidence "$WT/outputs/gra/server_checks" \
        --output "$WT/outputs/gra/GRA_LIGHT_$(date -u +%Y%m%d_%H%M%S).tar.gz"
    ;;
*) printf 'Unknown action: %s\n' "$ACTION" >&2; exit 2;;
esac
) 2>&1 | tee "$WT/outputs/gra/logs/${ACTION}_$(date -u +%Y%m%d_%H%M%S).log"
