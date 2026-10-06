#!/usr/bin/env bash
# AutoDL launcher, fixed to the VERIFIED implementation commit. Run from SSH.
set -Eeuo pipefail
CODE_SHA='6673ac43fa270c711a3bcf41a8f3c5bc88c80efb'
REPO='/root/autodl-tmp/projects/Crack_RTDETR'
WT='/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov13l-flash-configurable'
BASE_PY='/root/miniconda3/envs/rtdetr/bin/python'
RUN_ID='v13l_aug_x13_flash_01'

# The accelerator can refer to unset variables; restore all old shell options.
enable_network() {
    local old_flags=$- old_options rc=0
    old_options=$(set +o)
    set +eu
    if [[ -r /etc/network_turbo ]]; then source /etc/network_turbo; rc=$?; fi
    eval "$old_options"
    [[ "$old_flags" == *e* ]] && set -e
    [[ "$old_flags" == *u* ]] && set -u
    (( rc == 0 )) || return "$rc"
    local bypass='localhost,127.0.0.1,::1,pypi.org,.pypi.org,files.pythonhosted.org,.pythonhosted.org,pypi.python.org,download.pytorch.org,.pytorch.org'
    export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$bypass"
    export no_proxy="${no_proxy:+$no_proxy,}$bypass"
}
git_network() {
    local attempt
    for attempt in 1 2 3; do
        if git -c http.version=HTTP/1.1 "$@"; then return 0; fi
    done
    return 1
}
enable_network
git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null || { printf 'Mother repository missing: %s\n' "$REPO" >&2; exit 66; }
command -v tmux >/dev/null || { printf 'tmux missing\n' >&2; exit 69; }
"$BASE_PY" -c 'import sys,yaml;assert sys.version_info[:2]==(3,10); print("Bootstrap caller:",sys.executable,"PyYAML",yaml.__version__)'
git_network -C "$REPO" fetch origin bench/yolov13l-flash-configurable
git -C "$REPO" cat-file -e "${CODE_SHA}^{commit}"
if [[ ! -e "$WT" ]]; then
    git -C "$REPO" worktree add --detach "$WT" "$CODE_SHA"
fi
[[ "$(git -C "$WT" rev-parse HEAD)" == "$CODE_SHA" && -z "$(git -C "$WT" status --porcelain --untracked-files=normal)" ]] || { printf 'Protected existing path; requires clean fixed implementation: %s\n' "$WT" >&2; exit 65; }
[[ ! -e "$WT/outputs/yolov13l-configurable/$RUN_ID" ]] || { printf 'Protected existing Flash run; use its status/resume/finalize commands\n' >&2; exit 73; }
export YOLOV13L_CODE_SHA="$CODE_SHA" YOLOV13L_BASE_PYTHON="$BASE_PY" YOLOV13L_RUN_ID="$RUN_ID"
SESSION="setup-yolov13l-flash-$RUN_ID"
gate="yolov13l-flash-setup-$RUN_ID-$RANDOM"
printf -v worker '%q ' env "YOLOV13L_CODE_SHA=$CODE_SHA" "YOLOV13L_BASE_PYTHON=$BASE_PY" "YOLOV13L_RUN_ID=$RUN_ID" bash "$WT/scripts/autodl_yolov13l_flash_prepare.sh"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    if tmux list-panes -s -t "=$SESSION" -F '#{pane_dead}' | command grep -qx 0; then
        printf 'Protected live setup session: %s\n' "$SESSION" >&2; exit 73
    fi
    window=$(tmux new-window -d -P -F '#{window_id}' -t "=$SESSION" "tmux wait-for '$gate'; exec $worker")
else
    window=$(tmux new-session -d -P -F '#{window_id}' -s "$SESSION" "tmux wait-for '$gate'; exec $worker")
fi
tmux set-option -w -t "$window" remain-on-exit on
tmux select-window -t "$window"
tmux wait-for -S "$gate"
printf 'Setup: %s\nTraining after checks: comparison-yolov13l-%s\nFixed code: %s\n' "$SESSION" "$RUN_ID" "$CODE_SHA"
if [[ -t 0 && -t 1 ]]; then
    if [[ -n "${TMUX:-}" ]]; then tmux switch-client -t "=$SESSION"; else tmux attach -t "=$SESSION"; fi
fi
