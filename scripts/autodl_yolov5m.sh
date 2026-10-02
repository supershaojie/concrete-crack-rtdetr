#!/usr/bin/env bash
# User-invoked only. Never touches other tmux sessions or experiments.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${YOLOV5_PYTHON:-/root/miniconda3/envs/rtdetr/bin/python}"
SOURCE="${YOLOV5_SOURCE_PROJECT:-/root/autodl-tmp/projects/Crack_RTDETR}"
SESSION="comparison-yolov5m-scratch"
FOREGROUND=0
if [[ "${1:-}" == "--foreground" ]]; then FOREGROUND=1; shift; fi
RUN_NAME="${1:?Usage: autodl_yolov5m.sh [--foreground] UNIQUE_RUN_NAME}"
[[ $# == 1 && "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { printf '%s\n' 'Invalid run name'; exit 2; }
[[ -x "$PYTHON" ]] || { printf 'Interpreter unavailable: %s\n' "$PYTHON"; exit 2; }
RUN="$ROOT/outputs/yolov5m-scratch/runs/$RUN_NAME"
ASSETS="$ROOT/outputs/yolov5m-scratch/assets"
LOG="$RUN.launcher.log"
[[ ! -e "$RUN" && ! -e "$LOG" && ! -e "$RUN.active.lock" ]] || { printf 'Existing output protected: %s\n' "$RUN"; exit 2; }
mkdir -p -- "$(dirname -- "$RUN")"
if (( ! FOREGROUND )); then
    command -v tmux >/dev/null || { printf '%s\n' 'tmux unavailable'; exit 2; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        printf 'Existing tmux session protected: %s\n' "$SESSION"; exit 2
    fi
    printf -v CMD 'env YOLOV5_PYTHON=%q YOLOV5_SOURCE_PROJECT=%q bash %q --foreground %q' \
        "$PYTHON" "$SOURCE" "$ROOT/scripts/autodl_yolov5m.sh" "$RUN_NAME"
    GATE="yolov5m-scratch-launch-$RUN_NAME-$$"
    WINDOW=$(tmux new-session -d -P -F '#{window_id}' -s "$SESSION" "tmux wait-for '$GATE'; exec $CMD")
    tmux set-option -w -t "$WINDOW" remain-on-exit on
    tmux wait-for -S "$GATE"
    printf 'Started %s. Attach: tmux attach -t %s\n' "$SESSION" "$SESSION"
    exit 0
fi
# Exclusive logfile reservation; outputs are protected independently by Python's run lock.
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
printf '%s\n' 'GPU sharing is allowed. Batch16/640/AMP remain fixed; insufficient VRAM fails visibly.'
(set -o noclobber; : > "$LOG") 2>/dev/null || exit 2
stop_own_pipeline() {
    local SIGNAL="$1" EXIT_CODE="$2"
    # This group is created below solely for this launch, including its tee/workers.
    if [[ -n "${PIPE_PID:-}" ]]; then
        kill -s "$SIGNAL" -- "-$PIPE_PID" 2>/dev/null || true
        wait "$PIPE_PID" 2>/dev/null || true
        # Cover a child fork racing the first signal while its shell exits.
        kill -s "$SIGNAL" -- "-$PIPE_PID" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 -- "-$PIPE_PID" 2>/dev/null || break
            sleep 0.5
        done
        kill -KILL -- "-$PIPE_PID" 2>/dev/null || true
    fi
    printf '{"status":"interrupted","signal":"%s"}\n' "$SIGNAL" > "$RUN.launcher_exit.json"
    exit "$EXIT_CODE"
}
trap 'stop_own_pipeline INT 130' INT
trap 'stop_own_pipeline TERM 143' TERM
run_pipeline() {
set +m
trap - INT TERM
set +e
"$PYTHON" -u "$ROOT/benchmarks/comparison/yolov5m/run.py" pipeline \
    --run "$RUN" --assets "$ASSETS" --source-project "$SOURCE" 2>&1 | tee -a "$LOG"
CODES=("${PIPESTATUS[@]}")
set -e
RC="${CODES[0]}"
TEE_RC="${CODES[1]}"
STATUS=failed
if (( RC == 0 && TEE_RC == 0 )); then STATUS=completed; fi
printf '{"status":"%s","pipeline_exit_code":%d,"tee_exit_code":%d}\n' "$STATUS" "$RC" "$TEE_RC" > "$RUN.launcher_exit.json"
printf '\nStatus=%s; pipeline=%s; tee=%s; output=%s\n' "$STATUS" "$RC" "$TEE_RC" "$RUN"
if (( RC != 0 )); then exit "$RC"; fi
exit "$TEE_RC"
}
# Job control gives this function's pipeline a private process group. Signals to
# the launcher terminate only this experiment's children, never another session.
set -m
run_pipeline &
PIPE_PID=$!
set +m
set +e
wait "$PIPE_PID"
exit "$?"
