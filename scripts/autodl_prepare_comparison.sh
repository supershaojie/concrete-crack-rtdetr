#!/usr/bin/env bash
# Existing Python only; no installation, training, inference, GPU checks, or session cleanup.
set -u -o pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 2
python=${COMPARISON_PYTHON:-/root/miniconda3/envs/rtdetr/bin/python}
output=${COMPARISON_OUTPUT:-$repo/outputs/comparison_prepare/$(date -u +%Y%m%dT%H%M%SZ)_$$}
if [[ ${COMPARISON_OWN_SESSION:-0} == 1 && -n ${TMUX_PANE:-} ]]; then
  tmux set-option -w -t "$TMUX_PANE" remain-on-exit on || exit 2
fi
if [[ ${1:-} == --tmux ]]; then
  shift
  session=${COMPARISON_SESSION:-comparison-prepare}
  command -v tmux >/dev/null || { echo 'tmux unavailable; run without --tmux'; exit 2; }
  if tmux has-session -t "=$session" 2>/dev/null; then
    echo "Existing session preserved: $session. Set COMPARISON_SESSION to a new name."; exit 2
  fi
  tmux new-session -d -s "$session" -c "$repo" env "COMPARISON_PYTHON=$python" "COMPARISON_OUTPUT=$output" COMPARISON_OWN_SESSION=1 bash "$repo/scripts/autodl_prepare_comparison.sh" "$@" || exit 2
  printf 'Preparation session: %s\nView: tmux attach -t %q\n' "$session" "$session"
  exit 0
fi
[[ -x $python ]] || { echo "Existing Python unavailable: $python"; exit 2; }
[[ ! -e $output && ! -e $output.console.log ]] || { echo "Output already exists; preserved: $output"; exit 2; }
mkdir -p -- "$(dirname -- "$output")" || exit 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd -- "$repo" || exit 2
"$python" benchmarks/comparison/prepare.py --profile server --project /root/autodl-tmp/projects/Crack_RTDETR --output "$output" "$@" 2>&1 | tee "$output.console.log"
codes=("${PIPESTATUS[@]}")
rc=${codes[0]}
printf '%s\n' "$rc" > "$output.process_exit_code.txt"
printf '\nPREPARATION FINISHED\nPython process exit code: %s\ntee exit code: %s\nSummary: %s/summary.json\nLog: %s.console.log\n' "$rc" "${codes[1]}" "$output" "$output"
[[ $rc -eq 0 && ${codes[1]} -ne 0 ]] && exit 2
exit "$rc"
