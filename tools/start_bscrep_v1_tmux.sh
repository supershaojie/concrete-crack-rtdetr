#!/usr/bin/env bash
# Explicitly invoked only after server preflight; no implicit initialization.
set -Eeuo pipefail
WT=${BSCREP_V1_WORKTREE:-/root/autodl-tmp/projects/Crack_RTDETR-bscrep-v1}
source "$WT/outputs/bscrep_env.sh"
VARIANT=${1:-cbr_lif_bscrep_v1}
case "$VARIANT" in cbr_lif_bscrep_v1|bscrep_v1) ;; *) exit 2 ;; esac
SESSION="${VARIANT}-training"
PREFLIGHT="$WT/outputs/${VARIANT}_server_preflight/checks.json"
[[ -f "$PREFLIGHT" ]]
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "Existing tmux session protected: $SESSION" >&2; exit 3
fi
WORKER="$WT/outputs/${VARIANT}_worker.sh"
[[ ! -e "$WORKER" ]] || { echo "Existing worker protected: $WORKER" >&2; exit 4; }
(
  set -o noclobber
  {
    echo '#!/usr/bin/env bash'
    echo 'set -Eeuo pipefail'
    printf 'source %q\n' "$WT/outputs/bscrep_env.sh"
    printf 'trap '\''rc=$?; echo "$rc" > %q'\'' EXIT\n' "$WT/outputs/${VARIANT}_worker_exit.txt"
    printf 'python -u tools/train_bscrep_v1.py start --variant %q --preflight %q 2>&1 | tee %q\n' \
      "$VARIANT" "$PREFLIGHT" "$WT/outputs/${VARIANT}_training.log"
  } > "$WORKER"
)
tmux new-session -d -s "$SESSION" "bash $(printf '%q' "$WORKER")"
echo "Dispatched $SESSION; inspect $WT/outputs/${VARIANT}_training.log"
