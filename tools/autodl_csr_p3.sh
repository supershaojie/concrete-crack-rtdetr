#!/usr/bin/env bash
# Each invocation initializes the existing environment and this worktree.
set -Eeuo pipefail
if [[ "${1:-}" == "--help" || $# -eq 0 ]]; then
  printf '%s\n' 'Usage: bash tools/autodl_csr_p3.sh COMMAND [ARGUMENTS...]
Commands:
  plan|start|resume|status  -> tools/train_csr_p3.py COMMAND
  init                    -> tools/init_csr_p3.py
  preflight               -> tools/preflight_csr_p3.py
  val|test                -> tools/csr_p3_results.py COMMAND
  pack                    -> tools/pack_csr_p3.py
Every entry activates existing Conda rtdetr and the current worktree PYTHONPATH.
Use COMMAND --help for the actual Python options.'
  exit 0
fi
csr_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
export PYTHONPATH="$csr_root/ultralytics-main"
export YOLO_AUTOINSTALL=false
export CSR_P3_MAIN="${CSR_P3_MAIN:-/root/autodl-tmp/projects/Crack_RTDETR}"
cd -- "$csr_root"
command="$1"
shift
case "$command" in
  plan|start|resume|status) exec python tools/train_csr_p3.py "$command" "$@" ;;
  init) exec python tools/init_csr_p3.py "$@" ;;
  preflight) exec python tools/preflight_csr_p3.py "$@" ;;
  val|test) exec python tools/csr_p3_results.py "$command" "$@" ;;
  pack) exec python tools/pack_csr_p3.py "$@" ;;
  *) printf 'Unknown command: %s\n' "$command" >&2; exit 2 ;;
esac
