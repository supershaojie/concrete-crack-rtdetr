# LSRT v1 服务器交接

本轮仅交付和补检；正式训练 **NOT_STARTED**，最终 test **NOT_RUN**。首轮模型为 **original CBR＋原 LIF-Down＋LSRT**；另一个 `lsrt_v1` 仅准备单模块配置，不含 CBR/LIF。两者均不含 RCS-Q，不压缩或冻结 R18。不要自动登录服务器或操作其他实验进程。

分支：`exp-rtdetr-r18-lite-lsrt-v1`。固定基点：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。交付 SHA 模板：`__DELIVERY_SHA__`；提交外的 `DELIVERY_SHA.txt`、`SERVER_HANDOFF_DELIVERED.md` 将填入真实 40 位 SHA，避免提交自包含 SHA 的循环。以下按顺序操作，所有新终端先 source 同一个环境文件。

## 1. 上传交付小文件并同步到新工作树

先把交付目录里的 `DELIVERY_SHA.txt`、`sync_lsrt_v1.sh`、可选 `lsrt_v1.bundle` 上传到 `/root/autodl-tmp/lsrt_v1_delivery/`。不上传数据集、源权重或参考 ZIP。从已有主仓库导入同一 SHA；脚本先查对象，缺失才对指定分支做 HTTP/1.1 fetch，单次 120 秒，最多 5 次；设置 `GIT_TERMINAL_PROMPT=0`、`lowSpeedLimit=1`、`lowSpeedTime=60`、`--progress --no-tags`。网络失败后可导入 bundle，同样核验固定 SHA 与基点祖先关系。已有分支/工作树/未提交内容全部保留。

```bash
set -Eeuo pipefail
LSRT_DELIVERY_DIR=/root/autodl-tmp/lsrt_v1_delivery
LSRT_DELIVERY_SHA="$(tr -d '\r\n' < "$LSRT_DELIVERY_DIR/DELIVERY_SHA.txt")"
test "$LSRT_DELIVERY_SHA" = '__DELIVERY_SHA__'
bash "$LSRT_DELIVERY_DIR/sync_lsrt_v1.sh" "$LSRT_DELIVERY_SHA" \
  /root/autodl-tmp/projects/Crack_RTDETR \
  /root/autodl-tmp/projects/Crack_RTDETR-lsrt-v1 \
  "$LSRT_DELIVERY_DIR/lsrt_v1.bundle"
```

若默认目标已经属于其他 SHA，另选一个全新路径并同步修改下段 `LSRT_WORKTREE`；不要删旧目录。脚本成功创建工作树、核验 HEAD 后才创建 `weights/outputs`。

## 2. 建立可跨终端使用的环境文件

环境文件只使用现有 Conda `rtdetr`，不安装或升级任何包。下列 `set -C` 防止覆盖已存在文件；如曾创建，直接使用并检查其内容。

```bash
set -Eeuo pipefail
LSRT_ENV=/root/autodl-tmp/lsrt_v1_delivery/lsrt_v1.env
( set -C; cat > "$LSRT_ENV" <<'BASH'
set -Eeuo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
export LSRT_V1_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
export LSRT_WORKTREE=/root/autodl-tmp/projects/Crack_RTDETR-lsrt-v1
export LSRT_DELIVERY_SHA=__DELIVERY_SHA__
export PYTHONPATH="$LSRT_WORKTREE/ultralytics-main"
export YOLO_AUTOINSTALL=false
export PYTHONUNBUFFERED=1
export LSRT_SOURCE="$LSRT_V1_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
export LSRT_DATA="$LSRT_V1_MAIN/configs/crack_autodl.yaml"
export LSRT_AUDIT="$LSRT_WORKTREE/outputs/lsrt_v1/audit"
export LSRT_INIT="$LSRT_WORKTREE/weights/cbr_lif_lsrt_v1_controlled_init.pt"
export LSRT_PLAN="$LSRT_WORKTREE/outputs/lsrt_v1/plan/main/plan.json"
export LSRT_PREFLIGHT="$LSRT_WORKTREE/outputs/lsrt_v1/preflight/main/report.json"
cd "$LSRT_WORKTREE"
test "$(git rev-parse HEAD)" = "$LSRT_DELIVERY_SHA"
BASH
)
source "$LSRT_ENV"
mkdir -p "$LSRT_AUDIT"
python - <<'PY'
import sys, pathlib, torch, ultralytics, os
print('Python:', sys.version, sys.executable)
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('ultralytics:', ultralytics.__file__)
assert pathlib.Path(ultralytics.__file__).resolve() == pathlib.Path(os.environ['LSRT_WORKTREE'])/'ultralytics-main/ultralytics/__init__.py'
assert torch.cuda.is_available()
print('GPU:', torch.cuda.get_device_name(0))
PY
nvidia-smi
sha256sum "$LSRT_SOURCE"
# 必须等于 fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e
```

正式入口固定原成功环境 Python 3.10.13 / torch 2.1.2+cu121；差异应记录并核查，不能直接升级。GPU、空闲显存和其他实验占用以实际输出为准；不杀进程，不自行减 batch。

## 3. 服务器生成初值、准备 AMP 资源、执行必要补检

这是本轮允许的服务器后续补检步骤；不包含 optimizer.step 的正式训练。CPU/小图预检和本地不同版本 CUDA 结果不能替代服务器真实 B16/640。

```bash
source /root/autodl-tmp/lsrt_v1_delivery/lsrt_v1.env
python tools/init_lsrt_v1.py --variant cbr_lif_lsrt_v1 \
  --source "$LSRT_SOURCE" --output "$LSRT_INIT" \
  --report "$LSRT_AUDIT/initialization-main.json"
python tools/train_lsrt_v1.py resources --main "$LSRT_V1_MAIN" \
  --output "$LSRT_AUDIT/amp-resources.json"
python tools/preflight_lsrt_v1.py --variant cbr_lif_lsrt_v1 \
  --source "$LSRT_SOURCE" --initialized "$LSRT_INIT" \
  --device cuda --data "$LSRT_DATA" --capacity \
  --output "$LSRT_WORKTREE/outputs/lsrt_v1/preflight/main" \
  2>&1 | tee "$LSRT_AUDIT/preflight-main.log"
python - <<'PY'
import json, os
r=json.load(open(os.environ['LSRT_PREFLIGHT']))
print(r['status'], r.get('pending'), r['capacity'])
assert r['status']=='PASSED', 'Stop: pending/failed preflight is not permission to start'
PY
```

补检覆盖：真实 nc=1 `RTDETR.train` 构建、AdamW 参数恰好一次、公共权重/初始输出一致、LSRT 数学与梯度、零/非零状态重载与 EMA、原生融合、CUDA FP32/native AMP/half，以及真实 train 的 B16/640 原在线增强/DN 前向与反向。容量同时看固定顺序与 GT 较多的 train 批次，记录 GT/DN/显存；不从 test 选样本。失败或 OOM 原样记录，不改 AMP、batch、imgsz、queries 或验收容差。AMP 资源仍使用原 `bus.jpg`、`yolo26n.pt` 的正常检查路径。

预检目录必须全新。若此前某次失败，保留其目录，用新的 `--output` 路径并在环境文件中准确更新 `LSRT_PREFLIGHT`；不得只改 JSON 状态。源码、配置、固定源、初始化、数据字节变化后重新生成 plan 和预检。

## 4. 生成完整计划与配方差异

```bash
source /root/autodl-tmp/lsrt_v1_delivery/lsrt_v1.env
python tools/train_lsrt_v1.py plan --variant cbr_lif_lsrt_v1 \
  --source "$LSRT_SOURCE" --initialized "$LSRT_INIT" --data "$LSRT_DATA" \
  --c2-args "$LSRT_WORKTREE/docs/c19_lif_v1/c2_args.yaml" \
  --baseline-inventory "$LSRT_WORKTREE/docs/c19_lif_v1/checks.json" \
  --project "$LSRT_WORKTREE/outputs/lsrt_v1/train" \
  --output "$LSRT_WORKTREE/outputs/lsrt_v1/plan/main"
```

plan 不创建训练目录，完整保存 109 字段和逐项差异。仅模型、名称、输出位置、核验过的环境路径允许不同；200 epochs、patience 50、batch 16、640、seed 42、workers 8、AdamW、lr0 0.0005、AMP、在线增强等全部继承。三 split 名称/类别与原配置完全一致，图像/标签内容逐字节冻结。历史 `checks.json` 可核对全部路径、标签指纹和计数；该历史记录没有全量图像内容哈希，所以不能声称补出了历史图像字节证据。本次新记录的全量图像内容哈希严格绑定预检与 start。

## 5. 正式主实验命令：仅后续明确授权后执行

本轮停在上一步，不执行本段。先检查 `nvidia-smi` 的空闲显存；只启动主 variant。新 tmux 内重新激活环境，避免继承其他实验的 PATH/PYTHONPATH。

```bash
source /root/autodl-tmp/lsrt_v1_delivery/lsrt_v1.env
nvidia-smi
( set -C; cat > "$LSRT_AUDIT/start-main.sh" <<'BASH'
#!/usr/bin/env bash
set -Eeuo pipefail
source /root/autodl-tmp/lsrt_v1_delivery/lsrt_v1.env
python -u tools/train_lsrt_v1.py start \
  --plan "$LSRT_PLAN" --preflight "$LSRT_PREFLIGHT" \
  --output "$LSRT_AUDIT/launch-main" 2>&1 | tee "$LSRT_AUDIT/train-main.log"
BASH
)
if tmux has-session -t '=lsrt-v1-main' 2>/dev/null; then
  echo 'Existing tmux session preserved'; exit 1
fi
tmux new-session -d -s lsrt-v1-main "bash '$LSRT_AUDIT/start-main.sh'"
python tools/train_lsrt_v1.py status --launch "$LSRT_AUDIT/launch-main"
```

start 再次核对所有源码/配置/源/初值指纹、完整图像标签字节、全部必要预检状态、真实容量记录和实际 Trainer 参数；只接受受控零更新初值。训练过程仍由原 Trainer 执行，所有 requires_grad 参数只进入 AdamW 一次，原始 CBR/LIF 与所有 LSRT 参数均可训练。目录 `exist_ok=False`，共享运行锁拒绝重复启动；不自动 name2/name3、不 OOM 自动降 batch。结果状态分为 200 轮完成、patience 早停、其他停止、异常、OOM、手动中断。

中断后的原生 resume 是独立命令，保留已学习的非零 LSRT，不生成新初值、不要求与零初值等价。仅恢复本计划真实 `last.pt`；checkpoint 必须保留 optimizer/EMA/scaler。确认旧进程已退出后，新终端可用：

```bash
source /root/autodl-tmp/lsrt_v1_delivery/lsrt_v1.env
python -u tools/train_lsrt_v1.py resume \
  --plan "$LSRT_PLAN" --preflight "$LSRT_PREFLIGHT" \
  --checkpoint "$LSRT_WORKTREE/outputs/lsrt_v1/train/cbr_lif_lsrt_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/last.pt" \
  --previous-launch "$LSRT_AUDIT/launch-main" \
  --output "$LSRT_AUDIT/resume-main-01" 2>&1 | tee "$LSRT_AUDIT/resume-main-01.log"
```

resume 也应放到独立 tmux。若锁显示旧 PID 仍存活或 PID 已复用，保留证据并人工核对，不自行删除锁或杀进程。后续第二次恢复应使用最近一次 launch/resume 目录作为 `--previous-launch`；每次使用全新审计目录。

## 6. 单模块：只准备，不默认启动

```bash
source /root/autodl-tmp/lsrt_v1_delivery/lsrt_v1.env
python tools/init_lsrt_v1.py --variant lsrt_v1 --source "$LSRT_SOURCE" \
  --output "$LSRT_WORKTREE/weights/lsrt_v1_controlled_init.pt" \
  --report "$LSRT_AUDIT/initialization-single.json"
python tools/preflight_lsrt_v1.py --variant lsrt_v1 --source "$LSRT_SOURCE" \
  --initialized "$LSRT_WORKTREE/weights/lsrt_v1_controlled_init.pt" \
  --device cuda --data "$LSRT_DATA" --capacity \
  --output "$LSRT_WORKTREE/outputs/lsrt_v1/preflight/single" \
  2>&1 | tee "$LSRT_AUDIT/preflight-single.log"
python tools/train_lsrt_v1.py plan --variant lsrt_v1 --source "$LSRT_SOURCE" \
  --initialized "$LSRT_WORKTREE/weights/lsrt_v1_controlled_init.pt" --data "$LSRT_DATA" \
  --baseline-inventory "$LSRT_WORKTREE/docs/c19_lif_v1/checks.json" \
  --project "$LSRT_WORKTREE/outputs/lsrt_v1/train" \
  --output "$LSRT_WORKTREE/outputs/lsrt_v1/plan/single"
```

## 7. 以后评估与轻量打包

本轮不执行 val/test。以后训练成功完成或 patience 正常早停后，使用真实训练选择的 `best.pt`，沿用原 EMA 加载与独立 FP32 协议：640、B16、workers 0、conf .001、iou .7、max_det 300、half=False、augment=False、rect=False、seed42；继承原已纠正的 sorted-confidence mask。test 必须对应同 SHA、同 checkpoint、同数据和同配置的完成 val。保存 P/R/AP50/AP75/AP50:95、各 IoU AP、原协议逐图平均速度、PR/混淆矩阵和配置。默认不导出逐图完整预测。

```bash
source /root/autodl-tmp/lsrt_v1_delivery/lsrt_v1.env
python tools/lsrt_v1_results.py val --plan "$LSRT_PLAN" \
  --launch "$LSRT_AUDIT/launch-main" --device 0 \
  --output "$LSRT_WORKTREE/outputs/lsrt_v1/eval/main-val"
python tools/lsrt_v1_results.py test --plan "$LSRT_PLAN" \
  --launch "$LSRT_AUDIT/launch-main" --device 0 \
  --val-report "$LSRT_WORKTREE/outputs/lsrt_v1/eval/main-val/metrics.json" \
  --output "$LSRT_WORKTREE/outputs/lsrt_v1/eval/main-test"
```

如最终完成的是 resume，请将评估/打包的 `--launch` 改为最终成功的 resume 审计目录。打包本身不触发训练或评估，可用于当前尚未训练的工程证据：

```bash
source /root/autodl-tmp/lsrt_v1_delivery/lsrt_v1.env
python tools/lsrt_v1_results.py pack --plan "$LSRT_PLAN" \
  --preflight "$LSRT_WORKTREE/outputs/lsrt_v1/preflight/main" \
  --initialization-report "$LSRT_AUDIT/initialization-main.json" \
  --launch "$LSRT_AUDIT/launch-main" \
  --val "$LSRT_WORKTREE/outputs/lsrt_v1/eval/main-val" \
  --test "$LSRT_WORKTREE/outputs/lsrt_v1/eval/main-test" \
  --output "$LSRT_WORKTREE/outputs/lsrt_v1/lsrt_v1_LIGHT.tar.gz"
```

包中 train/val/test/审计/源码分目录，生成逐成员 manifest 和外部 SHA256，回读核验每个成员。默认排除数据集、权重 best/last、参考模块包、逐图预测和大规模 batch 图片；未完成项目明确写 NOT_STARTED/NOT_RUN，不凭打包成功宣称实验完成。
