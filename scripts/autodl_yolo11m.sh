#!/usr/bin/env bash
# Independent single-GPU experiment; no GPU-idle checks, global locks or process killing.
set -Eeuo pipefail
export PYTHONUNBUFFERED=1

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
    printf '{"status":"%s","exit_code":%d}\n' "$state" "$rc" > "$LAUNCH/pipeline_status.json"
    if [[ -d "$RUN" ]]; then
        "$PY" "$ENTRY" summary --run "$RUN" > "$LAUNCH/final_summary.log" 2>&1 || true
        cat "$LAUNCH/final_summary.log"
    fi
    printf '\nYOLO11m status=%s exit=%d\nOutput: %s\nLogs: %s\n' "$state" "$rc" "$RUN" "$LAUNCH"
    exit "$rc"
}

main() {
    local mode=${1:---help}
    if [[ "$mode" == --help || "$mode" == -h ]]; then
        printf '%s\n' 'Usage: bash scripts/autodl_yolo11m.sh start|run|start-resume|resume [options]' \
            'Options: --run-id ID --data YAML --data-root ROOT --public-coco DIR --base-python PYTHON' \
            'start uses tmux comparison-yolo11m-scratch; run stays in this shell. Resume requires an explicit run-id.' \
            'Runs bootstrap -> light preflight -> train -> export/evaluate val -> export/evaluate test -> summary.'
        return
    fi
    shift
    [[ "$mode" =~ ^(start|run|start-resume|resume)$ ]] || { printf 'Unknown mode\n' >&2; return 64; }
    local script root run_id explicit_id=0 data data_root='' public_coco='' base
    script=$(realpath "${BASH_SOURCE[0]}")
    root=$(dirname "$(dirname "$script")")
    run_id="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_${RANDOM}"
    data=/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml
    base=/root/miniconda3/envs/rtdetr/bin/python
    while (( $# )); do
        [[ $# -ge 2 ]] || { printf 'Missing option value\n' >&2; return 64; }
        case "$1" in
            --run-id) run_id=$2; explicit_id=1;;
            --data) data=$2;;
            --data-root) data_root=$2;;
            --public-coco) public_coco=$2;;
            --base-python) base=$2;;
            *) printf 'Unknown option: %s\n' "$1" >&2; return 64;;
        esac
        shift 2
    done
    [[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$ ]] || { printf 'Invalid run-id\n' >&2; return 64; }
    if [[ "$mode" == *resume && "$explicit_id" != 1 ]]; then printf 'Resume needs --run-id\n' >&2; return 64; fi
    cd "$root"
    RUN="$root/outputs/yolo11m-scratch/$run_id"
    ENTRY="$root/benchmarks/comparison/yolo11m/run.py"
    PY="$base"
    local options=(--run-id "$run_id" --data "$data" --base-python "$base")
    [[ -z "$data_root" ]] || options+=(--data-root "$data_root")
    [[ -z "$public_coco" ]] || options+=(--public-coco "$public_coco")
    if [[ "$mode" == start* ]]; then
        command -v tmux >/dev/null || { printf 'tmux unavailable\n' >&2; return 69; }
        if tmux has-session -t '=comparison-yolo11m-scratch' 2>/dev/null; then
            printf 'Protected existing comparison-yolo11m-scratch session; inspect it before starting another run.\n' >&2
            return 73
        fi
        local next=run command window gate="yolo11m-scratch-launch-${run_id}-${RANDOM}"
        [[ "$mode" != start-resume ]] || next=resume
        printf -v command '%q ' "$BASH" "$script" "$next" "${options[@]}"
        # Gate the command so remain-on-exit is set before even a fast failure can occur.
        window=$(tmux new-session -d -P -F '#{window_id}' -s comparison-yolo11m-scratch "tmux wait-for '$gate'; exec $command")
        tmux set-option -w -t "$window" remain-on-exit on
        tmux wait-for -S "$gate"
        printf 'Started comparison-yolo11m-scratch. Attach: tmux attach -t comparison-yolo11m-scratch\nRun: %s\n' "$RUN"
        return
    fi
    if [[ "$mode" == run ]]; then
        [[ ! -e "$RUN" ]] || { printf 'Protected existing run: %s\n' "$RUN" >&2; return 73; }
    else
        [[ -d "$RUN" ]] || { printf 'Resume run missing: %s\n' "$RUN" >&2; return 66; }
    fi
    local attempt=$run_id
    [[ "$mode" != resume ]] || attempt="${run_id}_resume_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_${RANDOM}"
    LAUNCH="$root/outputs/yolo11m-scratch-launch/$attempt"
    mkdir -p "$(dirname "$LAUNCH")"
    mkdir "$LAUNCH"  # exclusive: no previous logs can be overwritten
    trap finish EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
    printf '%s\n' 'GPU sharing is allowed. Batch16/640/AMP remain fixed; insufficient VRAM fails visibly.'
    run_stage bootstrap "$base" "$root/benchmarks/comparison/yolo11m/bootstrap.py" --base-python "$base"
    PY=$(<"$root/.runtime/yolo11m-scratch/python_path.txt")
    local inputs=(--data "$data")
    [[ -z "$data_root" ]] || inputs+=(--data-root "$data_root")
    [[ -z "$public_coco" ]] || inputs+=(--public-coco "$public_coco")
    if [[ "$mode" == run ]]; then
        run_stage preflight "$PY" "$ENTRY" preflight --run "$RUN" "${inputs[@]}"
        "$PY" -c 'import pathlib,shutil,sys; src,dst=map(pathlib.Path,sys.argv[1:]); shutil.copytree(src,dst)' "$root/.runtime/yolo11m-scratch" "$RUN/bootstrap"
        printf '%s\n' "$LAUNCH" > "$RUN/launch_directory.txt"
        run_stage train "$PY" "$ENTRY" train --run "$RUN"
    else
        run_stage train "$PY" "$ENTRY" train --run "$RUN" --resume
    fi
    local split
    for split in val test; do
        run_stage "export_$split" "$PY" "$ENTRY" export --run "$RUN" --split "$split"
        run_stage "evaluate_$split" "$PY" "$ENTRY" evaluate --run "$RUN" --split "$split"
    done
    run_stage summary "$PY" "$ENTRY" summary --run "$RUN"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
