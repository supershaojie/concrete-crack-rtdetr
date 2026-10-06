#!/usr/bin/env bash
# Isolated environment, immutable recipe, and one tmux/process lock per run.
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
    printf '\nYOLOv13-L %s status=%s exit=%d\nRun: %s\nLogs: %s\n' "$MODE" "$state" "$rc" "$RUN" "$LAUNCH"
    exit "$rc"
}

main() {
    local mode=${1:---help}
    if [[ "$mode" == --help || "$mode" == -h ]]; then
        printf '%s\n' \
          'Usage: bash scripts/autodl_yolov13l_configurable.sh bootstrap [--base-python PY] [--reuse-source DIR] [--reuse-asset PT]' \
          '       ... check-config [--config /absolute/file.yaml | --clone-config-from RUN] [--set key=value ...]' \
          '       ... config --out runtime_configs/ID.yaml [--set key=value ...]' \
          '       ... start --run-id ID [--config /absolute/file.yaml | --clone-config-from RUN] [--set key=value ...]' \
          '       ... status|preflight|resume|finalize|pack|archive-config --run-id ID [mode options]' \
          'pack defaults to review evidence excluding .pt; --include-weights includes best/last after legal completion.' \
          'resume/finalize use only frozen configuration, paths, code and environment; no overrides/downloads.' \
          'Each start/resume/finalize uses comparison-yolov13l-ID; GPU sharing allowed, no parameter fallback.' \
          'bootstrap/check-config/config/check never start training. Interactive launch attaches/switches tmux; --detach returns.'
        return
    fi
    shift
    local script root base_python
    script=$(realpath "${BASH_SOURCE[0]}")
    root=$(dirname "$(dirname "$script")")
    cd "$root"
    ENTRY="$root/benchmarks/comparison/yolov13l/run.py"
    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    if [[ "$mode" == bootstrap ]]; then
        base_python=${YOLOV13L_BASE_PYTHON:-/root/miniconda3/envs/rtdetr/bin/python}
        local bootstrap_args=("$@")
        while (( $# )); do
            if [[ "$1" == --fresh-torch ]]; then shift; continue; fi
            [[ $# -ge 2 ]] || { printf 'Missing option value\n' >&2; return 64; }
            if [[ "$1" == --base-python ]]; then base_python=$2; fi
            shift 2
        done
        "$base_python" "$root/benchmarks/comparison/yolov13l/bootstrap.py" "${bootstrap_args[@]}"
        return
    fi
    PY="$root/.envs/yolov13l-configurable-flash/bin/python"
    if [[ -f "$root/.runtime/yolov13l-configurable/python_path.txt" ]]; then
        PY=$(cat "$root/.runtime/yolov13l-configurable/python_path.txt")
    fi
    local run_id='' pipeline_mode='' python_explicit=0 detach=0 option
    local forwarded=() candidates=() paths=() extras=()
    while (( $# )); do
        if [[ "$1" == --detach ]]; then detach=1; shift; continue; fi
        if [[ "$1" == --include-weights ]]; then extras+=("$1"); forwarded+=("$1"); shift; continue; fi
        [[ $# -ge 2 ]] || { printf 'Missing option value: %s\n' "$1" >&2; return 64; }
        option=$1
        case "$option" in
            --python) PY=$2; python_explicit=1;;
            --run-id) run_id=$2; forwarded+=("$1" "$2");;
            --mode) pipeline_mode=$2;;
            --config) [[ "$2" == /* ]] || { printf '%s\n' '--config requires an absolute path' >&2; return 64; }; candidates+=("$1" "$2"); forwarded+=("$1" "$2");;
            --set|--clone-config-from) candidates+=("$1" "$2"); forwarded+=("$1" "$2");;
            --source|--weights|--data|--data-root|--public-coco|--reuse-run) paths+=("$1" "$2"); forwarded+=("$1" "$2");;
            --out|--output|--split) extras+=("$1" "$2"); forwarded+=("$1" "$2");;
            *) printf 'Unknown option: %s\n' "$1" >&2; return 64;;
        esac
        shift 2
    done
    if [[ "$mode" =~ ^(config|check-config|check|measure|status|summary|preflight|pack|archive-config|prepare)$ ]]; then
        [[ -x "$PY" ]] || { printf 'Private Python missing; run bootstrap or specify --python\n' >&2; return 69; }
        "$PY" "$ENTRY" "$mode" "${forwarded[@]}"
        return
    fi
    [[ "$mode" =~ ^(start|resume|finalize|_pipeline)$ ]] || { printf 'Unknown mode: %s\n' "$mode" >&2; return 64; }
    [[ ${#extras[@]} == 0 ]] || { printf 'Unsupported option for this lifecycle mode\n' >&2; return 64; }
    [[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$ && "$run_id" != SMOKE_* ]] || { printf 'Invalid or missing formal run-id\n' >&2; return 64; }
    RUN="$root/outputs/yolov13l-configurable/$run_id"
    if [[ "$mode" == start ]]; then
        [[ ! -e "$RUN" ]] || { printf 'Protected existing run: %s\n' "$RUN" >&2; return 73; }
    else
        [[ ${#candidates[@]} == 0 && ${#paths[@]} == 0 ]] || { printf 'Frozen run: changed parameters/paths require a new run-id\n' >&2; return 64; }
        [[ -d "$RUN" ]] || { printf 'Run missing: %s\n' "$RUN" >&2; return 66; }
        if (( ! python_explicit )); then
            PY=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$RUN/runtime_paths.json")
        fi
    fi
    [[ -x "$PY" ]] || { printf 'Frozen/private Python unavailable: %s\n' "$PY" >&2; return 69; }
    export YOLOV13L_LAUNCH_COMMAND
    printf -v YOLOV13L_LAUNCH_COMMAND '%q ' "$BASH" "$script" "$mode" "${forwarded[@]}" --python "$PY"
    local session="comparison-yolov13l-$run_id"
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
            "$PY" "$ENTRY" prepare --run "$RUN" --run-id "$run_id" "${candidates[@]}" "${paths[@]}"
        else
            "$PY" "$ENTRY" guard --run "$RUN" --operation "$mode"
        fi
        local command window gate="yolov13l-${run_id}-launch-${RANDOM}"
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
        if (( ! detach )) && [[ -t 0 && -t 1 ]]; then
            if [[ -n "${TMUX:-}" ]]; then tmux switch-client -t "=$session"; else tmux attach -t "=$session"; fi
        fi
        return
    fi
    [[ "$pipeline_mode" =~ ^(start|resume|finalize)$ ]] || { printf 'Invalid internal mode\n' >&2; return 64; }
    source "$root/scripts/yolov13l_flash_network.sh"
    yolov13l_flash_network
    MODE=$pipeline_mode
    command -v flock >/dev/null || { printf 'flock unavailable\n' >&2; return 69; }
    exec 9>"$RUN/.pipeline.lock"
    flock -n 9 || { printf 'Protected active pipeline: %s\n' "$RUN" >&2; return 73; }
    export YOLOV13L_PIPELINE_LOCKED="$RUN"
    LAUNCH="$RUN/launch/${MODE}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p "$(dirname "$LAUNCH")"
    mkdir "$LAUNCH"
    trap finish EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv || true
    printf '%s\n' 'GPU sharing allowed. Training parameters remain frozen; OOM fails with the real exit status.'
    if [[ "$MODE" == start ]]; then
        run_stage preflight "$PY" "$ENTRY" preflight --run "$RUN"
        run_stage train "$PY" "$ENTRY" train --run "$RUN"
    elif [[ "$MODE" == resume ]]; then
        run_stage train "$PY" "$ENTRY" train --run "$RUN" --resume
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
