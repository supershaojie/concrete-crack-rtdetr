#!/usr/bin/env bash
set -Eeuo pipefail
CH4_SCRIPT_DIR=${BASH_SOURCE[0]%/*}
if [[ "$CH4_SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]]; then CH4_SCRIPT_DIR=.; fi
CH4_CODE=$(cd -- "$CH4_SCRIPT_DIR/.." && pwd -P)
CH4_PY=${CH4_VIS_PY:-"$CH4_CODE/.venv-ch4/bin/python"}
action=${1:-help}
if (($#)); then shift; fi
if [[ "$action" == setup ]]; then
  CH4_BASE_PY=${CH4_BASE_PY:-/root/miniconda3/envs/rtdetr/bin/python}
  if [[ ! -x "$CH4_PY" ]]; then "$CH4_BASE_PY" -m venv "$CH4_CODE/.venv-ch4"; fi
  "$CH4_PY" -m pip install --disable-pip-version-check -r "$CH4_CODE/benchmarks/visualization/chapter4/requirements-common.txt"
  "$CH4_PY" -c 'import sys,numpy,PIL,reportlab,tqdm; print(sys.executable,numpy.__version__,PIL.__version__,reportlab.Version,tqdm.__version__)'
  exit 0
fi
if [[ "$action" == tmux ]]; then
  stage=${1:-full}; if (($#)); then shift; fi
  args=("$@"); out=""
  for ((i=0;i<${#args[@]};i++)); do
    if [[ "${args[i]}" == --out ]]; then out=${args[i+1]:-}; fi
  done
  [[ -n "$out" ]] || { printf '%s\n' 'tmux requires --out DIRECTORY' >&2; exit 64; }
  mkdir -p -- "$out/logs"
  out=$(cd -- "$out" && pwd -P)
  session=${CH4_TMUX_SESSION:-ch4-vis-eight}
  base_session=$session
  suffix=1
  while tmux has-session -t "=$session" 2>/dev/null; do
    session="$base_session-$suffix"; suffix=$((suffix+1))
  done
  stub="$out/start_${session}.sh"
  {
    printf '%s\n' '#!/usr/bin/env bash' 'set -uo pipefail'
    printf 'cd -- %q\n' "$CH4_CODE"
    printf 'export CH4_VIS_PY=%q\n' "$CH4_PY"
    printf 'bash %q %q ' "$CH4_CODE/scripts/autodl_chapter4_visualization.sh" "$stage"
    printf '%q ' "${args[@]}"
    printf '2>&1 | tee -a %q\n' "$out/logs/tmux.log"
    printf '%s\n' 'result=${PIPESTATUS[0]}'
    printf 'printf "%%s\\n" "$result" > %q\n' "$out/exit_code.txt"
    printf '%s\n' 'printf "\nChapter4 finished with exit code %s; inspect logs and pack_receipt.json\n" "$result"' 'exec bash'
  } > "$stub"
  printf -v command 'bash %q' "$stub"
  tmux new-session -d -s "$session" -c "$CH4_CODE" "$command"
  printf '%s\n' "$session" > "$out/tmux_session.txt"
  printf 'Started %s\nAttach: tmux attach -t %q\nLog: %s/logs/tmux.log\n' "$session" "$session" "$out"
  exit 0
fi
if [[ "$action" == help ]]; then
  printf '%s\n' 'setup | tmux full --handoff ZIP --out DIR --device cuda:0 | prepare/preflight/smoke/predictions/candidates/validate-layers/cam/render/full/review/add-model/pack/status --out DIR [options]'
  exit 0
fi
[[ -x "$CH4_PY" ]] || { printf 'Missing isolated common Python: %s; run this script with setup first.\n' "$CH4_PY" >&2; exit 69; }
cd -- "$CH4_CODE"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec "$CH4_PY" -u -m benchmarks.visualization.chapter4.cli "$action" "$@"
