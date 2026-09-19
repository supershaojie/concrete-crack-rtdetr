#!/usr/bin/env bash
# Explicit, independent DPR lifecycle actions. Preflight never starts training.
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKTREE=$(cd -- "$SCRIPT_DIR/.." && pwd)
ACTION=${1:-help}
if [ "$#" -gt 0 ]; then shift; fi
VARIANT=${DPR_VARIANT:-cbr_lif_dpr_v1}
case "$VARIANT" in cbr_lif_dpr_v1|dpr_v1) ;; *) echo "Unknown DPR_VARIANT: $VARIANT" >&2; exit 2 ;; esac

if [ "$ACTION" = help ] || [ "$ACTION" = --help ] || [ "$ACTION" = -h ]; then
  cat <<'HELP'
Usage: bash tools/dpr_server.sh ACTION [tool arguments]
Actions: environment | init | init-preflight | plan | start | resume | val | test | pack
Set DPR_VARIANT=cbr_lif_dpr_v1 (default) or dpr_v1.
Set DPR_MAIN only to an explicitly verified equivalent main repository/data root.
init-preflight performs initialization, CPU mathematical/structure checks,
CPU/CUDA lifecycle checks, and native B16/640/AMP capacity (<=16 batches).
It never starts formal training, evaluation, or the other variant automatically.
resume requires --checkpoint PATH; val/test require --weights PATH (test also
requires --val-report PATH). pack forwards --run-dir/--output/--metadata.
HELP
  exit 0
fi

# Existing conda activation scripts can read unset variables. Enable nounset only
# after activation, so running from an existing tmux shell remains safe.
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
set -u
cd -- "$WORKTREE"
export PYTHONPATH="$WORKTREE/ultralytics-main${PYTHONPATH:+:$PYTHONPATH}"
export DPR_MAIN=${DPR_MAIN:-/root/autodl-tmp/projects/Crack_RTDETR}
python -c 'import pathlib,ultralytics; expected=pathlib.Path("ultralytics-main/ultralytics/__init__.py").resolve(); actual=pathlib.Path(ultralytics.__file__).resolve(); assert actual == expected,(actual,expected); print("ultralytics:",actual)'

META="$WORKTREE/outputs/dpr/$VARIANT"
SOURCE="$DPR_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
INITIALIZED="$WORKTREE/weights/${VARIANT}_controlled_init.pt"
DATA="$DPR_MAIN/configs/crack_autodl.yaml"
mkdir -p -- "$META"

case "$ACTION" in
  environment)
    python tools/train_dpr.py environment --variant "$VARIANT" "$@"
    ;;
  plan)
    python tools/train_dpr.py plan --variant "$VARIANT" --source "$SOURCE" --initialized "$INITIALIZED" --data "$DATA" "$@"
    ;;
  init|init-preflight)
    if [ "$#" -ne 0 ]; then echo "$ACTION takes no extra arguments; preserves the bounded contract." >&2; exit 2; fi
    STAMP=$(date -u +%Y%m%dT%H%M%S)_$$
    SESSION="$META/check_$STAMP"
    # Revoke before any dependency/check. A failed rerun cannot leave an old permit.
    if [ -f "$META/start_permit.json" ]; then
      mv -- "$META/start_permit.json" "$META/start_permit.revoked.$STAMP.json"
    fi
    mkdir -- "$SESSION"
    initialize_controlled() {
      if [ -f "$INITIALIZED" ]; then
        python tools/init_dpr.py --variant "$VARIANT" --source "$SOURCE" --output "$INITIALIZED" --report "$SESSION/init.json" --verify-existing
      else
        python tools/init_dpr.py --variant "$VARIANT" --source "$SOURCE" --output "$INITIALIZED" --report "$SESSION/init.json"
      fi
    }
    bounded_checks() {
      if [ "$ACTION" = init ]; then return 0; fi
      python tools/preflight_dpr.py --variant "$VARIANT" --source "$SOURCE" --initialized "$INITIALIZED" --data "$DATA" \
        --init-report "$SESSION/init.json" --math-report "$SESSION/math.json" --output "$SESSION/bounded" --device all --capacity
    }
    final_verify() {
      if [ "$ACTION" = init ]; then return 0; fi
      python tools/train_dpr.py verify --variant "$VARIANT" --source "$SOURCE" --initialized "$INITIALIZED" --data "$DATA" --report "$SESSION/bounded/preflight.json"
    }
    set +e
    {
      python tools/train_dpr.py environment --variant "$VARIANT" &&
      initialize_controlled &&
      python tools/check_dpr.py --variant "$VARIANT" --output "$SESSION/math.json" --imgsz 640 &&
      bounded_checks && final_verify
    } 2>&1 | tee "$SESSION/$ACTION.log"
    PIPE_CODES=("${PIPESTATUS[@]}")
    COMMAND_RC=${PIPE_CODES[0]}
    TEE_RC=${PIPE_CODES[1]}
    RC=$COMMAND_RC
    if [ "$RC" -eq 0 ]; then RC=$TEE_RC; fi
    set -e
    printf '%s\n' "$RC" > "$SESSION/exit_code.txt"
    printf '{"command_exit":%s,"tee_exit":%s,"effective_exit":%s}\n' "$COMMAND_RC" "$TEE_RC" "$RC" > "$SESSION/exit_status.json"
    exit "$RC"
    ;;
  start|resume)
    LOG="$META/${ACTION}_$(date -u +%Y%m%dT%H%M%S)_$$.log"
    set +e
    python tools/train_dpr.py "$ACTION" --variant "$VARIANT" --source "$SOURCE" --initialized "$INITIALIZED" --data "$DATA" "$@" \
      2>&1 | tee "$LOG"
    PIPE_CODES=("${PIPESTATUS[@]}")
    COMMAND_RC=${PIPE_CODES[0]}
    TEE_RC=${PIPE_CODES[1]}
    RC=$COMMAND_RC
    if [ "$RC" -eq 0 ]; then RC=$TEE_RC; fi
    set -e
    printf '%s\n' "$RC" > "$LOG.exit_code"
    printf '{"command_exit":%s,"tee_exit":%s,"effective_exit":%s}\n' "$COMMAND_RC" "$TEE_RC" "$RC" > "$LOG.exit_status.json"
    exit "$RC"
    ;;
  val|test)
    LOG="$META/${ACTION}_$(date -u +%Y%m%dT%H%M%S)_$$.log"
    set +e
    python tools/eval_dpr.py "$ACTION" --variant "$VARIANT" --data "$DATA" "$@" \
      2>&1 | tee "$LOG"
    PIPE_CODES=("${PIPESTATUS[@]}")
    COMMAND_RC=${PIPE_CODES[0]}
    TEE_RC=${PIPE_CODES[1]}
    RC=$COMMAND_RC
    if [ "$RC" -eq 0 ]; then RC=$TEE_RC; fi
    set -e
    printf '%s\n' "$RC" > "$LOG.exit_code"
    printf '{"command_exit":%s,"tee_exit":%s,"effective_exit":%s}\n' "$COMMAND_RC" "$TEE_RC" "$RC" > "$LOG.exit_status.json"
    exit "$RC"
    ;;
  pack)
    python tools/pack_dpr_light.py --variant "$VARIANT" "$@"
    ;;
  *) echo "Unknown action: $ACTION (use --help)" >&2; exit 2 ;;
esac
