# TRC-AIFI v1 服务器交接

本页只交接命令，本次没有登录、同步服务器或启动训练/test。首轮主组合是 **CBR + LIF + TRC**，不要改成单模块 `trc_v1`。不停止其他实验，不为并行而调低 batch。建议等现有实验释放显存后预检及训练。

将提交外交付文件 `trc_v1_delivery.json` 上传到 `/root/autodl-tmp/trc_v1_delivery.json`。它的 `commit` 必须是本次最终完整 40 位 SHA；`base` 应为 `a0459d6a652cb702699087c88fa39a3e4c4087ec`。网络备用小包上传到 `/root/autodl-tmp/trc_v1.bundle`，并与交付记录的 SHA256 核对。不要把本页 base 当作最终交付 SHA。

## 1. 获取固定提交，创建独立 worktree

以下代码先查本地对象，缺失才 fetch 指定分支。每次网络尝试 120 秒、最多 5 次，启用 HTTP/1.1、progress、no-tags、禁止凭证提示、低速超时。已有正确对象时不重复联网；网络失败使用包含同一完整 SHA 的小 bundle。bundle 以成功组合 base 为 prerequisite，服务器主仓库应已有该对象；缺少时 `bundle verify` 会明确失败，保留现场。

```bash
set -Eeuo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
ROOT=/root/autodl-tmp/projects/Crack_RTDETR-trc-v1
DELIVERY=/root/autodl-tmp/trc_v1_delivery.json
BUNDLE=/root/autodl-tmp/trc_v1.bundle
BRANCH=exp-rtdetr-r18-lite-trc-v1
SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$DELIVERY")"
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
export GIT_TERMINAL_PROMPT=0
if ! git -C "$MAIN" cat-file -e "$SHA^{commit}" 2>/dev/null; then
  for ATTEMPT in 1 2 3 4 5; do
    echo "Fetch $BRANCH attempt $ATTEMPT/5"
    if timeout 120s git -C "$MAIN" -c http.version=HTTP/1.1 \
      -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 fetch --progress --no-tags \
      --no-write-fetch-head origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"; then
      git -C "$MAIN" cat-file -e "$SHA^{commit}" 2>/dev/null && break
    fi
  done
fi
if ! git -C "$MAIN" cat-file -e "$SHA^{commit}" 2>/dev/null; then
  python3 - "$DELIVERY" "$BUNDLE" <<'PY'
import hashlib, json, pathlib, sys
record = json.loads(pathlib.Path(sys.argv[1]).read_text())
actual = hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest()
assert actual == record['bundle_sha256'], (actual, 'bundle SHA mismatch')
PY
  git -C "$MAIN" bundle verify "$BUNDLE"
  [[ "$(git bundle list-heads "$BUNDLE" "refs/heads/$BRANCH" | cut -d' ' -f1)" == "$SHA" ]]
  git -C "$MAIN" fetch --no-tags --no-write-fetch-head "$BUNDLE" \
    "refs/heads/$BRANCH:refs/codex-delivery/trc-v1/bundle-$SHA"
fi
git -C "$MAIN" cat-file -e "$SHA^{commit}"
SCRIPT="$(mktemp /tmp/sync-trc-v1.XXXXXX.sh)"
trap 'rm -f -- "$SCRIPT"' EXIT
git -C "$MAIN" show "$SHA:tools/sync_trc_v1.sh" > "$SCRIPT"
bash "$SCRIPT" "$SHA" "$MAIN" "$ROOT" "$BUNDLE"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$SHA" ]]
```

`sync_trc_v1.sh` 不 reset/clean/stash，不修改其他 worktree。已存在目标仅接受同库 linked worktree、同 SHA 且 tracked clean，其他情况拒绝并保留。它先 `git worktree add`、确认 HEAD，再创建 weights/outputs 目录。交付审计落在新工作树 `outputs/trc_v1_delivery.json`，该文件在 Git 之外。

[sync_checks.json](sync_checks.json) 记录真实本地 Git/linked-worktree 检查，包括精确对象不联网、重入、保护主 checkout、拒绝脏目标/普通目录/短 SHA、五次有界失败和同 SHA 小 bundle 导入。测试仓库与提交均为隔离 fixture，使用 `GIT_SSH_COMMAND=false` 注入离线失败，不代表已同步真实服务器。脚本和本页六个 bash 块均通过 `bash -n`。

## 2. 创建统一环境文件、核对实际 import

每个后续代码块显式 source 同一环境文件，不依赖另一终端的残留变量。仅激活已有 rtdetr 环境，不安装/升级依赖。SOURCE 指向服务器现有固定源权重，不使用训练完成的 best/last，不需要 factors 文件。

```bash
set -Eeuo pipefail
ENVFILE=/root/autodl-tmp/trc_v1.env
[[ ! -e "$ENVFILE" ]] || { echo "Existing $ENVFILE preserved; inspect and reuse it if identical."; exit 1; }
cat > "$ENVFILE" <<'ENV'
set -Eeuo pipefail
export MAIN=/root/autodl-tmp/projects/Crack_RTDETR
export ROOT=/root/autodl-tmp/projects/Crack_RTDETR-trc-v1
export DELIVERY=/root/autodl-tmp/trc_v1_delivery.json
export SOURCE="$MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
export INIT="$ROOT/weights/cbr_lif_trc_v1_controlled_init.pt"
export PREFLIGHT="$ROOT/outputs/trc_v1/cbr_lif_trc_v1/preflight_server"
export PLAN="$ROOT/outputs/trc_v1/cbr_lif_trc_v1/plan.json"
export DATA="$MAIN/configs/crack_autodl.yaml"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
export PYTHONPATH="$ROOT/ultralytics-main:$ROOT/tools"
cd "$ROOT"
export SHA="$(python -c 'import json,os; print(json.load(open(os.environ["DELIVERY"]))["commit"])')"
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$(git rev-parse HEAD)" == "$SHA" ]]
ENV
source "$ENVFILE"
python - <<'PY'
import inspect, os, pathlib, sys, torch, ultralytics
from ultralytics.nn.modules.trc_aifi import AIFI_TRC
root = pathlib.Path(os.environ['ROOT']).resolve()
for path in (pathlib.Path(ultralytics.__file__).resolve(), pathlib.Path(inspect.getfile(AIFI_TRC)).resolve()):
    assert root in path.parents, path
    print(path)
print(sys.executable, sys.version, torch.__version__, torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA required for the server preflight'
print(torch.cuda.get_device_name(0))
PY
sha256sum "$SOURCE"
# Must equal fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e.
```

## 3. 初始化、AMP 资源、服务器补检、plan

初值工具拒绝覆盖已有正式初值。预检用隔离副本检查真实 train 在线增强、B16/640/native AMP、原 loss/DN/反向，记录 GT/DN、GradScaler 实际 scale 和 CUDA 峰值显存。当前服务器项为 **PENDING**；只有服务器报告真实 **PASSED** 才能 start。若 OOM/非有限值/资源缺失，保留失败记录并排查，不修改 batch、关闭 AMP 或伪造 PASS。

```bash
set -Eeuo pipefail
source /root/autodl-tmp/trc_v1.env
python tools/init_trc_v1.py --variant cbr_lif_trc_v1 --source "$SOURCE" \
  --output "$INIT" --report "$ROOT/outputs/trc_v1/cbr_lif_trc_v1/initialization.json"
python tools/train_trc_v1.py prepare-amp --main "$MAIN" \
  --output "$ROOT/outputs/trc_v1/amp_resources.json"
python tools/preflight_trc_v1.py --variant cbr_lif_trc_v1 --source "$SOURCE" \
  --initialized "$INIT" --data "$DATA" --output "$PREFLIGHT" --server \
  2>&1 | tee "$ROOT/outputs/trc_v1/preflight-console.log"
python tools/train_trc_v1.py plan --variant cbr_lif_trc_v1 --source "$SOURCE" \
  --initialized "$INIT" --preflight "$PREFLIGHT/preflight.json" --data "$DATA" \
  --c2-args "$ROOT/docs/trc_v1/cbr_lif_args.yaml" --project "$MAIN/runs/c_series" --plan "$PLAN"
```

AMP 资源准备复用原入口，必须有实际 assets/bus.jpg 和参考权重，原 `check_amp` 照常执行。plan 只写计划和参数差异，不占正式 run 目录。主 run 名是 `cbr_lif_trc_v1_rtdetr_r18_lite_e200_b16_onlineaug`。独立单模块以后使用 `--variant trc_v1`，并更换对应 INIT/PREFLIGHT/PLAN 路径；不要复用组合初值或预检。

## 4. 单独执行正式 start（本次未执行）

确认预检通过且资源空闲后，才手动执行这一段。它不会终止其他 session，已有 session/log/正式 run 会被保留。训练入口核对计划绑定的 SHA、源码、模型配置、源权重、初值及预检证据。终端退出后 tmux 继续运行。

```bash
set -Eeuo pipefail
source /root/autodl-tmp/trc_v1.env
SESSION=trc-v1-training
RUNNER="$ROOT/outputs/trc_v1/start.sh"
[[ ! -e "$RUNNER" && ! -e "$ROOT/outputs/trc_v1/train-console.log" ]]
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists; preserved."; exit 1
fi
cat > "$RUNNER" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
source /root/autodl-tmp/trc_v1.env
python tools/train_trc_v1.py start --plan "$PLAN" 2>&1 | tee "$ROOT/outputs/trc_v1/train-console.log"
SH
printf -v COMMAND 'bash %q' "$RUNNER"
tmux new-session -d -s "$SESSION" "$COMMAND"
python tools/train_trc_v1.py status --plan "$PLAN"
# Observe later: tmux attach -t trc-v1-training
```

状态与恢复命令：

```bash
set -Eeuo pipefail
source /root/autodl-tmp/trc_v1.env
python tools/train_trc_v1.py status --plan "$PLAN"
# Only after confirming the prior training process stopped and a valid last.pt exists:
# python tools/train_trc_v1.py resume --plan "$PLAN"
```

resume 沿用 last.pt 的已学习 TRC/EMA/优化器，不能再次执行 init 当作恢复。完整 200 轮、patience 早停、失败和中断分别由生命周期状态记录；tmux 存在不等于成功。

## 5. 后续 test 与轻量包（本次未执行）

```bash
set -Eeuo pipefail
source /root/autodl-tmp/trc_v1.env
python tools/test_trc_v1.py --plan "$PLAN" --split test
python tools/pack_trc_v1_light.py --plan "$PLAN" --output "$ROOT/outputs/trc_v1_LIGHT.tar.gz"
sha256sum "$ROOT/outputs/trc_v1_LIGHT.tar.gz"
```

test 沿用原 best/EMA 与原评估设置，保存 P/R/AP50/AP75/AP50:95、PR/混淆矩阵和速度。包默认包含 args、results.csv、metrics、曲线、必要源码、Git/初始化/预检审计与 manifest/SHA256；不包含权重、数据、参考包或巨量逐图预测。现有包拒绝覆盖。
