#!/usr/bin/env bash
# Called in a detached tmux setup worker by the fixed-SHA .server.sh launcher.
set -Eeuo pipefail
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$WT"
source "$WT/scripts/yolov13l_flash_network.sh"
yolov13l_flash_network
BASE_PY=${YOLOV13L_BASE_PYTHON:-/root/miniconda3/envs/rtdetr/bin/python}
RUN_ID=${YOLOV13L_RUN_ID:-v13l_aug_x13_flash_01}
[[ "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$ && "$RUN_ID" != SMOKE_* ]] || exit 64
[[ ! -e "$WT/outputs/yolov13l-configurable/$RUN_ID" ]] || { printf 'Protected existing run: %s\n' "$RUN_ID" >&2; exit 73; }
[[ "$(git rev-parse HEAD)" == "${YOLOV13L_CODE_SHA:?Fixed implementation SHA required}" && -z "$(git status --porcelain --untracked-files=normal)" ]] || { printf 'Fixed clean implementation identity required\n' >&2; exit 65; }
LAUNCH="$WT/outputs/yolov13l-flash-preparation/${RUN_ID}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$RANDOM"
mkdir -p "$(dirname "$LAUNCH")"
mkdir "$LAUNCH"
finish_prepare() {
    local rc=$?
    trap - EXIT INT TERM
    printf '{"exit_code":%d,"formal_start_dispatched":%s}\n' "$rc" "${START_DISPATCHED:-false}" > "$LAUNCH/preparation_exit.json"
    printf 'Preparation exit=%d Logs=%s\n' "$rc" "$LAUNCH"
    exit "$rc"
}
trap finish_prepare EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
stage() {
    local name=$1
    shift
    "$BASE_PY" "$WT/benchmarks/comparison/yolov13l/stage.py" --directory "$LAUNCH" --name "$name" -- "$@"
}
"$BASE_PY" -c 'import yaml,sys; assert sys.version_info[:2]==(3,10); print(sys.executable,yaml.__version__)'
bootstrap_args=(--base-python "$BASE_PY" --env-dir "$WT/.envs/yolov13l-configurable-flash")
OLD_ASSET='/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov13l-configurable/.runtime/yolov13l-configurable/assets/yolov13l.pt'
if [[ -f "$OLD_ASSET" ]]; then bootstrap_args+=(--reuse-asset "$OLD_ASSET"); fi
stage bootstrap bash scripts/autodl_yolov13l_configurable.sh bootstrap "${bootstrap_args[@]}"
PY=$(cat "$WT/.runtime/yolov13l-configurable/python_path.txt")
CONFIG="$WT/runtime_configs/$RUN_ID.yaml"
if [[ ! -e "$CONFIG" ]]; then
    stage config bash scripts/autodl_yolov13l_configurable.sh config --python "$PY" --out "$CONFIG"
fi
stage check_config bash scripts/autodl_yolov13l_configurable.sh check-config --python "$PY" --config "$CONFIG"
# Compare two COMPLETE outputs of the same resolver, not a partial YAML to expanded args.
stage compare_preset "$PY" -c 'import sys;sys.path.insert(0,sys.argv[1]);from configuration import candidate;assert candidate(sys.argv[2])[0]==candidate(sys.argv[3])[0], "Candidate differs from first Flash preset"' \
    "$WT/benchmarks/comparison/yolov13l" "$CONFIG" "$WT/benchmarks/comparison/yolov13l/configs/v13l_aug_x13_flash_01.yaml"
stage flash_640_batch16 "$PY" benchmarks/comparison/yolov13l/flash_checks.py --config "$CONFIG" --out "$LAUNCH/flash_readiness.json"
# Reaching this stage means install/operator/parity/640-batch16/native-val/FP32 passed.
stage start bash scripts/autodl_yolov13l_configurable.sh start --python "$PY" --run-id "$RUN_ID" --config "$CONFIG" --detach
START_DISPATCHED=true
printf 'Training session: comparison-yolov13l-%s\n' "$RUN_ID"
