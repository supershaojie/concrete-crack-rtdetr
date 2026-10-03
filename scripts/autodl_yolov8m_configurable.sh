#!/usr/bin/env bash
# One independent tmux/session and process lock per candidate; reuse pilot assets read-only.
set -Eeuo pipefail

run_stage() {
    local stage=$1
    shift
    "$PY" "$(dirname "$ENTRY")/stage.py" --directory "$LAUNCH" --name "$stage" -- "$@"
}

finish() {
    local rc=$?
    trap - EXIT INT TERM
    local state=failed
    if (( rc == 0 )); then state=completed; fi
    if (( rc == 130 || rc == 143 )); then state=interrupted; fi
    printf '{"status":"%s","mode":"%s","exit_code":%d}\n' "$state" "$MODE" "$rc" > "$LAUNCH/pipeline_status.json"
    "$PY" "$ENTRY" summary --run "$RUN" > "$LAUNCH/final_summary.log" 2>&1 || true
    cat "$LAUNCH/final_summary.log"
    printf '\nYOLOv8m %s status=%s exit=%d\nRun: %s\nLogs: %s\n' "$MODE" "$state" "$rc" "$RUN" "$LAUNCH"
    exit "$rc"
}

main() {
    local mode=${1:---help}
    if [[ "$mode" == --help || "$mode" == -h ]]; then
        printf '%s\n' \
            'Usage: bash scripts/autodl_yolov8m_configurable.sh start --run-id ID --config /absolute/candidate.yaml' \
            '       bash scripts/autodl_yolov8m_configurable.sh resume|finalize --run-id ID' \
            'start freezes YAML before tmux dispatch; start/resume train and report public val only.' \
            'finalize requires legally completed training; reuses the same best/val and adds public test.' \
            'Optional start paths: --python PY --source DIR --weights PT --data YAML --data-root DIR --public-coco DIR --reuse-run DIR' \
            'resume/finalize read all paths and parameters from the run snapshot; --python may select its frozen interpreter.' \
            'Session: comparison-yolov8m-<run-id>. No bootstrap, downloads, GPU-idle gate or batch fallback.'
        return
    fi
    shift
    [[ "$mode" =~ ^(start|resume|finalize|_pipeline)$ ]] || { printf 'Unknown mode\n' >&2; return 64; }
    local script root run_id='' config='' pipeline_mode='' python_explicit=0
    script=$(realpath "${BASH_SOURCE[0]}")
    root=$(dirname "$(dirname "$script")")
    local pilot=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-coco-b19-pilot
    PY="$pilot/.envs/yolov8m-coco-b19-pilot/bin/python"
    local paths=()
    while (( $# )); do
        [[ $# -ge 2 ]] || { printf 'Missing option value\n' >&2; return 64; }
        case "$1" in
            --run-id) run_id=$2;;
            --config) config=$2;;
            --python) PY=$2; python_explicit=1;;
            --source|--weights|--data|--data-root|--public-coco|--reuse-run) paths+=("$1" "$2");;
            --mode) pipeline_mode=$2;;
            *) printf 'Unknown option: %s\n' "$1" >&2; return 64;;
        esac
        shift 2
    done
    [[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$ ]] || { printf 'Invalid or missing run-id\n' >&2; return 64; }
    cd "$root"
    RUN="$root/outputs/yolov8m-configurable/$run_id"
    ENTRY="$root/benchmarks/comparison/yolov8m/run.py"
    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    if [[ "$mode" == start ]]; then
        [[ -n "$config" && "$config" == /* ]] || { printf 'start requires an absolute --config path\n' >&2; return 64; }
        [[ ! -e "$RUN" ]] || { printf 'Protected existing run: %s\n' "$RUN" >&2; return 73; }
    else
        [[ -z "$config" && ${#paths[@]} == 0 ]] || { printf 'Existing runs use frozen config/paths; changes require a new run-id\n' >&2; return 64; }
        [[ -d "$RUN" ]] || { printf 'Run missing: %s\n' "$RUN" >&2; return 66; }
        if (( ! python_explicit )); then
            # Read only the frozen runtime path with the already available pilot Python.
            PY=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$RUN/runtime_paths.json")
        fi
    fi
    [[ -x "$PY" ]] || { printf 'Reused Python unavailable: %s\n' "$PY" >&2; return 69; }
    local session="comparison-yolov8m-$run_id"
    if [[ "$mode" != _pipeline ]]; then
        command -v tmux >/dev/null || { printf 'tmux unavailable\n' >&2; return 69; }
        local live=0
        if tmux has-session -t "=$session" 2>/dev/null; then
            live=1
            if tmux list-panes -s -t "=$session" -F '#{pane_dead}' | command grep -qx 0; then
                printf 'Protected active run/session: %s\n' "$session" >&2
                return 73
            fi
        fi
        if [[ "$mode" == start ]]; then
            # Exclusive mkdir and YAML byte-copy occur before asynchronous launch.
            "$PY" "$ENTRY" prepare --run "$RUN" --run-id "$run_id" --config "$config" "${paths[@]}"
        else
            "$PY" "$ENTRY" guard --run "$RUN"
        fi
        local command window gate="yolov8m-${run_id}-launch-${RANDOM}"
        printf -v command '%q ' "$BASH" "$script" _pipeline --run-id "$run_id" --python "$PY" --mode "$mode"
        if (( live )); then
            window=$(tmux new-window -d -P -F '#{window_id}' -t "=$session" "tmux wait-for '$gate'; exec $command")
        else
            window=$(tmux new-session -d -P -F '#{window_id}' -s "$session" "tmux wait-for '$gate'; exec $command")
        fi
        tmux set-option -w -t "$window" remain-on-exit on
        tmux select-window -t "$window"
        tmux wait-for -S "$gate"
        printf 'Started %s (%s). Attach: tmux attach -t %s\nRun: %s\n' "$session" "$mode" "$session" "$RUN"
        return
    fi
    [[ "$pipeline_mode" =~ ^(start|resume|finalize)$ ]] || { printf 'Invalid internal mode\n' >&2; return 64; }
    MODE=$pipeline_mode
    command -v flock >/dev/null || { printf 'flock unavailable\n' >&2; return 69; }
    exec 9>"$RUN/.pipeline.lock"
    flock -n 9 || { printf 'Protected active pipeline: %s\n' "$RUN" >&2; return 73; }
    export YOLOV8M_PIPELINE_LOCKED="$RUN"
    LAUNCH="$RUN/launch/${MODE}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p "$(dirname "$LAUNCH")"
    mkdir "$LAUNCH"
    trap finish EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
    printf '%s\n' 'GPU0 sharing allowed. Physical batch16/640/AMP fixed; OOM fails with its real exit status.'
    if [[ "$MODE" == start ]]; then
        run_stage preflight "$PY" "$ENTRY" preflight --run "$RUN"
        run_stage train "$PY" "$ENTRY" train --run "$RUN"
    elif [[ "$MODE" == resume ]]; then
        local completed
        completed=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status") == "completed")' "$RUN/train_status.json")
        if [[ "$completed" != True ]]; then
            run_stage train "$PY" "$ENTRY" train --run "$RUN" --resume
        fi
    fi
    run_stage export_val "$PY" "$ENTRY" export --run "$RUN" --split val
    run_stage evaluate_val "$PY" "$ENTRY" evaluate --run "$RUN" --split val
    if [[ "$MODE" == finalize ]]; then
        run_stage export_test "$PY" "$ENTRY" export --run "$RUN" --split test
        run_stage evaluate_test "$PY" "$ENTRY" evaluate --run "$RUN" --split test
    fi
    run_stage summary "$PY" "$ENTRY" summary --run "$RUN"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
