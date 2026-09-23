#!/usr/bin/env bash
# RDL 独立操作入口；每次显式定位源码、环境，不依赖上一段临时变量。
set -euo pipefail
rdl_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || $# == 0 ]]; then
  echo '用法: bash tools/rdl_v1.sh prepare|preflight|diagnose|start|resume|val|test|pack [选项]'
  echo 'prepare/preflight/diagnose 不会启动训练。start 必须显式调用。'
  exit 0
fi
# 路径已由成功母版 metadata/launch/plan.json 和 AUTODL.md 核验；每次使用前检查。
test -f /root/miniconda3/etc/profile.d/conda.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
test "$(command -v python)" = /root/miniconda3/envs/rtdetr/bin/python
cd -- "$rdl_root"
export PYTHONPATH="$rdl_root/ultralytics-main"
export PYTHONUNBUFFERED=1
exec python "$rdl_root/tools/rdl_v1.py" "$@"
