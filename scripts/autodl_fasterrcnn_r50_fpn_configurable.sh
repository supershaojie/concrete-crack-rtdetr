#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
RUNTIME="$PROJECT/.runtime/fasterrcnn-r50-fpn-configurable"
MODULE="$PROJECT/benchmarks/comparison/fasterrcnn_r50_fpn"
mkdir -p "$RUNTIME/logs"
STAGE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$STAGE" == help ]]; then
  echo 'Usage: bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh bootstrap|print-config|preflight|full|train|export|evaluate|redraw|pack|summary [implemented CLI options]'
  echo 'bootstrap: --base-python PATH --cuda-wheel cu118|cu121; remaining stages: see run.py --help'
  exit 0
fi
BASE_PYTHON="${FRCNN_BASE_PYTHON:-$(command -v python || command -v python3)}"
if [[ "$STAGE" == bootstrap ]]; then
  exec "$BASE_PYTHON" "$MODULE/bootstrap.py" "$@"
fi
if [[ "$STAGE" == print-config && ! -f "$RUNTIME/python_path.txt" ]]; then
  PYTHON="$BASE_PYTHON"
else
  [[ -f "$RUNTIME/python_path.txt" ]] || { echo 'Missing bootstrap receipt. Run bootstrap first.' >&2; exit 2; }
  IFS= read -r PYTHON < "$RUNTIME/python_path.txt"
fi
[[ -x "$PYTHON" ]] || { echo "Recorded interpreter unavailable: $PYTHON" >&2; exit 2; }
case "$STAGE" in
  full|preflight|train|export)
    [[ -n "${TMUX:-}" ]] || { echo 'Run this stage inside its own tmux session. See server commands document.' >&2; exit 2; }
    SESSION="$(tmux display-message -p '#{session_name}')"
    WINDOW="$(tmux display-message -p '#{window_id}')"
    PANE="$(tmux display-message -p '#{pane_id}')"
    echo "Actual tmux session=$SESSION window=$WINDOW pane=$PANE"
    tmux set-window-option -t "$WINDOW" remain-on-exit on
    "$PYTHON" -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); p.write_text(json.dumps(dict(session=sys.argv[2],window=sys.argv[3],pane=sys.argv[4],remain_on_exit=True))+"\n")' "$RUNTIME/tmux_latest.json" "$SESSION" "$WINDOW" "$PANE"
    ;;
  print-config|evaluate|redraw|pack|summary) ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
printf -v FRCNN_LAUNCH_COMMAND '%q ' bash "$PROJECT/scripts/autodl_fasterrcnn_r50_fpn_configurable.sh" "$STAGE" "$@"
export FRCNN_LAUNCH_COMMAND PYTHONUNBUFFERED=1
LOG="$RUNTIME/logs/${STAGE}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
echo "LOG=$LOG"
set +e
"$PYTHON" "$MODULE/run.py" "$STAGE" "$@" 2>&1 | tee "$LOG"
PIPE_CODES=("${PIPESTATUS[@]}")
set -e
CODE="${PIPE_CODES[0]}"
if [[ "$CODE" == 0 && "${PIPE_CODES[1]}" != 0 ]]; then CODE="${PIPE_CODES[1]}"; fi
echo "Stage=$STAGE Python_exit=${PIPE_CODES[0]} Tee_exit=${PIPE_CODES[1]} Final_exit=$CODE LOG=$LOG"
exit "$CODE"
