#!/usr/bin/env bash
# 中文：每次独立定位脚本所在工作树并使用已核验的历史 Python；不依赖上段 shell 变量。
set -euo pipefail
ror_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ror_python='/root/miniconda3/envs/rtdetr/bin/python'
[[ -x "$ror_python" ]] || { echo "历史解释器不存在: $ror_python；请提供实际环境路径，不自动安装。"; exit 1; }
cd "$ror_root"
export YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1
export PATH="$(dirname "$ror_python"):$PATH"
exec "$ror_python" tools/ror_v1.py "$@"
