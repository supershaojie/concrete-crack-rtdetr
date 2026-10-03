#!/usr/bin/env bash
# Fixed tool: user-invoked start/resume/finalize, one protected tmux session per run.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${YOLOV5_PYTHON:-/root/autodl-tmp/envs/comparison-yolov5m-coco-b19-pilot/bin/python}"
OUTPUT_ROOT="${YOLOV5_OUTPUT_ROOT:-$ROOT/outputs/yolov5m-coco-native-ft-v1/runs}"
ACTION="${1:?Usage: script start|resume|finalize --run-id ID [--config /absolute/candidate.yaml] [--foreground]}"
shift
case "$ACTION" in start|resume|finalize) ;; *) printf 'Unsupported action: %s\n' "$ACTION"; exit 2;; esac
RUN_ID=""
CONFIG=""
FOREGROUND=0
EXTRA=()
while (( $# )); do
    case "$1" in
        --run-id) RUN_ID="${2:?--run-id requires a value}"; shift 2;;
        --config) CONFIG="${2:?--config requires a value}"; shift 2;;
        --foreground) FOREGROUND=1; shift;;
        --assets|--asset-cache|--source-project|--data|--data-root|--gt-cache|--dataset-cache)
            EXTRA+=("$1" "${2:?path option requires a value}"); shift 2;;
        *) printf 'Unknown launcher argument: %s\n' "$1"; exit 2;;
    esac
done
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$ ]] || { printf '%s\n' 'Invalid/missing --run-id'; exit 2; }
[[ -x "$PYTHON" ]] || { printf 'Existing interpreter unavailable: %s\n' "$PYTHON"; exit 2; }
SESSION="comparison-yolov5m-$RUN_ID"
RUN="$OUTPUT_ROOT/$RUN_ID"
[[ ! -e "$RUN.active.lock" ]] || { printf 'Active/stale run lock protected: %s\n' "$RUN.active.lock"; exit 2; }
if [[ "$ACTION" == start ]]; then
    [[ "$CONFIG" == /* || "$CONFIG" =~ ^[A-Za-z]:/ ]] || { printf '%s\n' 'start requires --config with an absolute path'; exit 2; }
    [[ -f "$CONFIG" ]] || { printf 'Config missing: %s\n' "$CONFIG"; exit 2; }
    [[ ! -e "$RUN" && ! -e "$RUN.launcher.log" ]] || { printf 'Existing run protected: %s\n' "$RUN"; exit 2; }
else
    [[ -z "$CONFIG" ]] || { printf '%s\n' 'resume/finalize use the run snapshot; --config is forbidden'; exit 2; }
    [[ -f "$RUN/frozen.json" ]] || { printf 'Run snapshot missing: %s\n' "$RUN"; exit 2; }
fi
mkdir -p -- "$OUTPUT_ROOT"
if (( ! FOREGROUND )); then
    command -v tmux >/dev/null || { printf '%s\n' 'tmux unavailable'; exit 2; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        printf 'Existing tmux session protected: %s\n' "$SESSION"; exit 2
    fi
    ARGS=("$ACTION" --run-id "$RUN_ID" --foreground "${EXTRA[@]}")
    if [[ "$ACTION" == start ]]; then ARGS+=(--config "$CONFIG"); fi
    printf -v CMD 'env YOLOV5_PYTHON=%q YOLOV5_OUTPUT_ROOT=%q bash %q ' \
        "$PYTHON" "$OUTPUT_ROOT" "$ROOT/scripts/autodl_yolov5m_coco_native_ft_v1.sh"
    printf -v REST '%q ' "${ARGS[@]}"
    GATE="yolov5m-native-$RUN_ID-$$"
    WINDOW=$(tmux new-session -d -P -F '#{window_id}' -s "$SESSION" "tmux wait-for '$GATE'; exec $CMD$REST")
    tmux set-option -w -t "$WINDOW" remain-on-exit on
    tmux wait-for -S "$GATE"
    printf 'Started %s (%s). Attach: tmux attach -t %s\n' "$SESSION" "$ACTION" "$SESSION"
    exit 0
fi
LOG="$RUN.launcher.log"
if [[ "$ACTION" != start ]]; then LOG="$RUN.$ACTION.$(date -u +%Y%m%dT%H%M%SZ).$$.log"; fi
(set -o noclobber; : > "$LOG") 2>/dev/null || { printf 'Log already exists: %s\n' "$LOG"; exit 2; }
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
printf '%s\n' 'GPU sharing is allowed; YAML batch/AMP are used exactly. OOM fails visibly.'
stop_own_pipeline() {
    local SIGNAL="$1" EXIT_CODE="$2"
    if [[ -n "${PIPE_PID:-}" ]]; then
        kill -s "$SIGNAL" -- "-$PIPE_PID" 2>/dev/null || true
        wait "$PIPE_PID" 2>/dev/null || true
        kill -s "$SIGNAL" -- "-$PIPE_PID" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 -- "-$PIPE_PID" 2>/dev/null || break
            sleep 0.5
        done
        kill -KILL -- "-$PIPE_PID" 2>/dev/null || true
    fi
    printf '{"status":"interrupted","signal":"%s"}\n' "$SIGNAL" > "$LOG.exit.json"
    exit "$EXIT_CODE"
}
trap 'stop_own_pipeline INT 130' INT
trap 'stop_own_pipeline TERM 143' TERM
run_pipeline() {
    set +m
    trap - INT TERM
    set +e
    ARGS=("$ACTION" --run-id "$RUN_ID" --output-root "$OUTPUT_ROOT" "${EXTRA[@]}")
    if [[ "$ACTION" == start ]]; then ARGS+=(--config "$CONFIG"); fi
    "$PYTHON" -u "$ROOT/benchmarks/comparison/yolov5m/run.py" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
    CODES=("${PIPESTATUS[@]}")
    set -e
    RC="${CODES[0]}"
    TEE_RC="${CODES[1]}"
    STATUS=failed
    if (( RC == 0 && TEE_RC == 0 )); then STATUS=completed; fi
    if (( RC == 130 || RC == 143 )); then STATUS=interrupted; fi
    printf '{"status":"%s","pipeline_exit_code":%d,"tee_exit_code":%d}\n' "$STATUS" "$RC" "$TEE_RC" > "$LOG.exit.json"
    printf '\nStatus=%s; pipeline=%s; tee=%s; output=%s\n' "$STATUS" "$RC" "$TEE_RC" "$RUN"
    if (( RC != 0 )); then exit "$RC"; fi
    exit "$TEE_RC"
}
set -m
run_pipeline &
PIPE_PID=$!
set +m
set +e
wait "$PIPE_PID"
exit "$?"
