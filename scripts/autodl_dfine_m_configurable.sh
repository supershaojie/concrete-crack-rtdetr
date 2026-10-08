#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/autodl_dfine_m_configurable.sh"
ENTRY="$ROOT/benchmarks/comparison/dfine_m/run.py"
CACHE="$ROOT/.runtime/dfine-m-configurable"
mode="${1:-preview}"; if (($#)); then shift; fi
args=("$@"); run_id=""; run_dir=""; resume=""
for ((i=0;i<${#args[@]};i++)); do
  case "${args[i]}" in
    --run-id) run_id="${args[i+1]}" ;;
    --run-dir) run_dir="${args[i+1]}" ;;
    --resume) resume="${args[i+1]}" ;;
  esac
done
if [[ -n "$run_id" && ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]]; then
  printf 'Invalid run-id\n' >&2; exit 2
fi
if [[ "$mode" == tmux ]]; then
  [[ -n "$run_id" ]] || { printf 'tmux needs --run-id\n' >&2; exit 2; }
  session="${DFINE_TMUX_SESSION:-comparison-dfine-m-configurable}"
  command -v tmux >/dev/null || { printf 'tmux unavailable\n' >&2; exit 2; }
  if tmux has-session -t "=$session" 2>/dev/null; then
    printf 'Session already exists: %s. Inspect it or set another DFINE_TMUX_SESSION.\n' "$session" >&2; exit 2
  fi
  tmux new-session -d -s "$session" -n "$run_id" 'bash --noprofile --norc'
  pane="$(tmux display-message -p -t "$session:$run_id.0" '#{pane_id}')"
  tmux set-window-option -t "$session:$run_id" remain-on-exit on
  printf -v command '%q ' bash "$SELF" all "${args[@]}"
  command+='; exit $?'
  tmux send-keys -t "$pane" -l "$command"
  tmux send-keys -t "$pane" Enter
  printf 'Started session=%s window=%s pane=%s\nAttach: tmux attach -t %q\nExit: tmux display-message -p -t %q "#{pane_dead_status}"\n' "$session" "$run_id" "$pane" "$session" "$pane"
  exit 0
fi
mkdir -p "$CACHE/logs" "$CACHE/runs"
export DFINE_ACCELERATION_STATUS=entry_not_available
if [[ -r /etc/network_turbo && "${DFINE_ENABLE_ACCELERATION:-1}" == 1 ]]; then
  # Server-owned AutoDL entry only; outcome recorded, credentials never printed here.
  set +u
  if source /etc/network_turbo >/dev/null 2>&1; then export DFINE_ACCELERATION_STATUS=entry_sourced; else export DFINE_ACCELERATION_STATUS=entry_failed; fi
  set -u
fi
base="${DFINE_BASE_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
if [[ -z "$base" || ! -x "$base" ]]; then base="$(command -v python3 || command -v python)"; fi
python="${DFINE_PYTHON:-$base}"
if [[ -z "${DFINE_PYTHON:-}" && -f "$CACHE/python.path" && "$mode" != bootstrap ]]; then
  IFS= read -r python < "$CACHE/python.path"
fi
[[ -x "$python" ]] || { printf 'Recorded Python unavailable: %s\n' "$python" >&2; exit 2; }
export DFINE_LAUNCH_COMMAND
printf -v DFINE_LAUNCH_COMMAND '%q ' bash "$SELF" "$mode" "${args[@]}"
if [[ -n "${DFINE_EXPECTED_CODE_SHA:-}" && "$(git -C "$ROOT" rev-parse HEAD)" != "$DFINE_EXPECTED_CODE_SHA" ]]; then
  printf 'Implementation commit differs from DFINE_EXPECTED_CODE_SHA\n' >&2; exit 2
fi
label="${run_id:-$(basename -- "${run_dir:-preview}")}"
[[ "$label" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] || label=standalone
exec 9>"$CACHE/runs/$label.launch.lock"
flock -n 9 || { printf 'This D-FINE run already has an active launcher: %s\n' "$label" >&2; exit 3; }
run_stage() {
  local stage="$1"; shift
  local logfile="$CACHE/logs/${label}_${stage}_$(date -u +%Y%m%dT%H%M%S).log"
  printf '\nD-FINE stage=%s Python=%s log=%s\n' "$stage" "$python" "$logfile"
  set +e
  "$python" -u "$ENTRY" "$stage" "$@" 2>&1 | tee "$logfile"
  local codes=("${PIPESTATUS[@]}"); local rc="${codes[0]}"
  set -e
  printf '%s\n' "$rc" > "$logfile.exit_code"
  printf 'D-FINE %s real Python exit_code=%s\n' "$stage" "$rc"
  if [[ -z "$run_dir" && -n "$run_id" && -f "$CACHE/runs/$run_id.path" ]]; then IFS= read -r run_dir < "$CACHE/runs/$run_id.path"; fi
  if [[ -n "$run_dir" && -d "$run_dir" ]]; then
    mkdir -p "$run_dir/logs"
    cp -- "$logfile" "$logfile.exit_code" "$run_dir/logs/"
  fi
  if [[ "$rc" == 0 && "${codes[1]}" != 0 ]]; then rc="${codes[1]}"; fi
  return "$rc"
}
if [[ "$mode" == all ]]; then
  if [[ -z "$resume" ]]; then
    [[ -n "$run_id" ]] || { printf 'New all requires --run-id\n' >&2; exit 2; }
    run_stage bootstrap "${args[@]}"
    IFS= read -r python < "$CACHE/python.path"
    run_stage prepare "${args[@]}"
    run_stage preflight --run-dir "$run_dir"
    run_stage train --run-dir "$run_dir"
  else
    [[ -n "$run_dir" ]] || run_dir="$ROOT/outputs/dfine-m-configurable/$run_id"
    run_stage train --run-dir "$run_dir" --resume "$resume"
  fi
  run_stage final --run-dir "$run_dir"
  run_stage pack --run-dir "$run_dir"
else
  run_stage "$mode" "${args[@]}"
fi
