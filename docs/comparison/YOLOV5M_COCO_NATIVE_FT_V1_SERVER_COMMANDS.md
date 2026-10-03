# YOLOv5m 固定训练工具：服务器命令

只用于 `bench/yolov5m-coco-native-ft-v1`。旧 pilot、scratch、v8 的入口和输出保持各自历史身份。本次交付不包含服务器操作或正式训练。运行时保持第一次选定的代码 SHA，后续候选只写 YAML、换 run-id。

## 1. 首次拉取和独立工作树

下面的代码 SHA 在交付后的身份补充中固定为实际实施提交；当前完整命令也可直接从已推送分支解析一次并冻结。**已启动训练的工作树不执行 checkout/pull/reset。** 已有工作树 HEAD 不同就清楚停止，另行检查，不能覆盖它。

```bash
set -euo pipefail
SOURCE=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-native-ft-v1
BRANCH=bench/yolov5m-coco-native-ft-v1
git -C "$SOURCE" remote get-url origin
git -C "$SOURCE" fetch origin "$BRANCH"
CODE_SHA="$(git -C "$SOURCE" rev-parse FETCH_HEAD)"
[[ "$CODE_SHA" =~ ^[0-9a-f]{40}$ ]]
printf 'Fixed code SHA: %s\n' "$CODE_SHA"
if [[ -e "$WT" ]]; then
    [[ "$(git -C "$WT" rev-parse HEAD)" == "$CODE_SHA" ]] || { printf '%s\n' 'Existing worktree HEAD differs; do not switch an active worktree'; exit 2; }
    [[ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ]] || { printf '%s\n' 'Existing tracked changes protected'; exit 2; }
else
    git -C "$SOURCE" worktree add --detach "$WT" "$CODE_SHA"
fi
cd "$WT"
mkdir -p outputs/yolov5m-coco-native-ft-v1 runtime_configs
if [[ -f outputs/yolov5m-coco-native-ft-v1/fixed_code_sha.txt ]]; then
    [[ "$(cat outputs/yolov5m-coco-native-ft-v1/fixed_code_sha.txt)" == "$CODE_SHA" ]]
else
    printf '%s\n' "$CODE_SHA" > outputs/yolov5m-coco-native-ft-v1/fixed_code_sha.txt
fi
```

## 2. 环境和官方资产一次准备

复用已有解释器，只读检查，不创建 venv、不 pip install。检查若发现缺依赖则停止；确需补依赖时另建独立环境并通过 `YOLOV5_PYTHON` 指向它，不能修改旧任务环境。原始官方权重必须是 42,806,829 bytes / SHA256 `61d933360ba5a7733a36764996c800287d973889d875227f5beedd2473a97a56`。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-native-ft-v1
export YOLOV5_PYTHON=/root/autodl-tmp/envs/comparison-yolov5m-coco-b19-pilot/bin/python
PYTHON="$YOLOV5_PYTHON"
ASSET_CACHE=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-b19-pilot/outputs/yolov5m-coco-b19-pilot/assets
ASSETS="$PWD/outputs/yolov5m-coco-native-ft-v1/assets"
test -x "$PYTHON"
"$PYTHON" - <<'PY'
import sys, torch, torchvision, yaml, cv2, numpy, scipy, pandas, seaborn, matplotlib, tqdm, thop, IPython, git
assert sys.version_info >= (3, 9)
torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([.5]), .7)
print('Reuse interpreter:', sys.executable, 'prefix:', sys.prefix)
print('Torch/torchvision:', torch.__version__, torchvision.__version__)
PY
"$PYTHON" benchmarks/comparison/yolov5m/assets.py --assets "$ASSETS" --asset-cache "$ASSET_CACHE"
"$PYTHON" benchmarks/comparison/yolov5m/assets.py --assets "$ASSETS" --verify-only
```

上述入口只从缓存 Git 对象克隆固定 v7.0 SHA `915bbf294bb74c859f0b41f1c23bc395014ea679`，在新目录应用一次原生补丁。旧已补丁工作树不变，权重直接读旧官方文件、不复制或下载。运行目录存在就严格核验，没有自动重打补丁或覆盖缓存。

数据默认读取 `/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml` 和 `datasets/crack_det`。准备自动寻找该主项目和同级 `Crack_RTDETR-comparison-base` 下 `outputs/comparison_prepare/*/dataset_manifest.json` 和 `coco/val.json`/`test.json`。标签、路径、文件大小和 ID 轻量核对，已有尺寸清单只读复用，再抽查3个 train 图片头。不会全库图片哈希/解码、复制、重新划分或增强。

若尺寸缓存位于其他已有目录，把实际路径作为启动器的 `--dataset-cache /absolute/existing/dataset_manifest.json`；公共 GT 目录可用 `--gt-cache /absolute/existing/coco`。这些是已实现的路径参数，之后记录到 run 快照，resume/finalize 不必再指定。没有可用尺寸缓存会明确停止；不能为绕过检查重新扫描43GB图片。

## 3. 创建首组完整候选

这是整体配方候选，不是已验证最优或单变量消融。这里省略的原生 loss/固定 COCO/无冻结/公共评估设置，由代码默认值补全到 `resolved_config.yaml`。也可以直接复制仓库 `benchmarks/comparison/yolov5m/v5_ft_01.yaml`。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-native-ft-v1
test ! -e runtime_configs/v5_ft_01.yaml
cat > runtime_configs/v5_ft_01.yaml <<'YAML'
epochs: 200
patience: 50
batch: 16
imgsz: 640
workers: 8
device: 0
seed: 42
deterministic: true
amp: true
cache: false
rect: false
multi_scale: false
nbs: 64
optimizer: SGD
lr0: 0.003
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
warmup_momentum: 0.8
warmup_bias_lr: 0.01
cos_lr: true
hsv_h: 0.015
hsv_s: 0.5
hsv_v: 0.35
degrees: 5
translate: 0.1
scale: 0.4
shear: 0
perspective: 0
flipud: 0
fliplr: 0.5
mosaic: 0.5
mixup: 0
cutmix: 0
copy_paste: 0
close_mosaic: 10
YAML
bash scripts/autodl_yolov5m_coco_native_ft_v1.sh start --run-id v5_ft_01 --config "$PWD/runtime_configs/v5_ft_01.yaml"
```

start 自动完成轻量检查 → 原生训练 → best 的公共 val → summary。调参阶段不跑 test。物理 batch16/imgsz640/device0 为本候选配置值，与现有 v8 共用 GPU0，不检查空闲门槛，不自动缩小 batch 补偿 OOM。OOM 的真实退出状态和已保存轮数留在 run 中；本机小样本检查不代表服务器双任务显存充足。

## 4. 第二组接口示例

示例改 lr0/mosaic，仍未验证优劣；候选应逐组评估，不需修改/提交代码、重装环境或下载权重。可在第一组结束后再启动，也可由使用者自行决定 GPU 并行容量。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-native-ft-v1
"${YOLOV5_PYTHON:-/root/autodl-tmp/envs/comparison-yolov5m-coco-b19-pilot/bin/python}" - <<'PY'
from pathlib import Path
import yaml
source=Path('runtime_configs/v5_ft_01.yaml')
target=Path('runtime_configs/v5_ft_02.yaml')
config=yaml.safe_load(source.read_text())
config.update(lr0=0.0015, mosaic=0.3)
with target.open('x') as stream:
    yaml.safe_dump(config,stream,sort_keys=False)
PY
bash scripts/autodl_yolov5m_coco_native_ft_v1.sh start --run-id v5_ft_02 --config "$PWD/runtime_configs/v5_ft_02.yaml"
```

YAML 支持 config.py DEFAULTS 中的字段：训练时长、batch/imgsz/workers/device/seed、确定性/AMP/RAM cache/rect/multi_scale、nbs、SGD/Adam/AdamW、lr/decay/warmup/余弦或线性调度、原生 loss 与 label_smoothing、原生增强数值、close_mosaic、plots/save_period。`momentum` 对 SGD 是动量，对 Adam/AdamW 是原生 beta1；beta2=.999/eps=1e-8、SGD Nesterov 和原生3组更新保留。未开放上游不存在的字段；命令行没有另一套数值覆盖或 --set。

imgsz 必须是32的倍数；不做 autobatch。`rect: true` 必须把 mosaic/mixup 设0；原生 MixUp 只在 Mosaic 分支执行，mixup>0 要求 mosaic>0。CutMix/CopyPaste 非0、未知/重复字段、错误类型/范围、CPU+AMP 会清楚拒绝。`close_mosaic: 0` 表示不关闭，其余按 `epochs-close_mosaic` 的索引关闭并重建 worker。新数值必须新 run，不能带新 YAML 恢复旧 run。

## 5. tmux、日志、退出与同 run 恢复

```bash
tmux ls
tmux attach -t comparison-yolov5m-v5_ft_01
# 离开并保持运行：按 Ctrl-b，再按 d。
# 如需中断本组：在该组终端按 Ctrl-c；保留原生 last/best 和实际轮数。
RUN=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-native-ft-v1/outputs/yolov5m-coco-native-ft-v1/runs/v5_ft_01
tail -f "$RUN.launcher.log"
cat "$RUN/status.json"
cat "$RUN/summary.json"
cat "$RUN.launcher.log.exit.json"
tmux display-message -p -t '=comparison-yolov5m-v5_ft_01' '#{pane_dead} #{pane_dead_status}'
```

每组独立 `comparison-yolov5m-<run-id>`，保留原生每轮实时进度和原始 CR 控制字符。同名活动会话、run 输出和 `.active.lock` 都保护。结束的 tmux 会话以 remain-on-exit 保留；恢复前只清理**本组已经退出**的会话：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-native-ft-v1
SESSION=comparison-yolov5m-v5_ft_01
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    [[ "$(tmux display-message -p -t "=$SESSION" '#{pane_dead}')" == 1 ]] || { printf '%s\n' 'Session still active; do not duplicate'; exit 2; }
    tmux kill-session -t "=$SESSION"
fi
bash scripts/autodl_yolov5m_coco_native_ft_v1.sh resume --run-id v5_ft_01
```

不需要候选源 YAML 仍然存在。恢复严格检查同一代码/上游补丁/权重/数据/环境/run 快照，读取自己的 last 完整恢复 optimizer、EMA、scaler、scheduler、累计步数/decay、RNG、dataloader generator 和早停状态。原生 opt/hyp 原始增益保存并重新构建一次有效增益，不能从已缩放值再缩放。沿用原生 half checkpoint、epoch 边界清残余梯度和 worker 随机流重启语义，不承诺逐位连续等价。训练已完成时 resume 只补未完成的公共 val 步骤，不追加训练。

SIGKILL/断电可能留下锁；先读 `$RUN.active.lock`、核实 PID/命令和本组所有进程已退出，再手工保留一份 stale lock 并移开，才能恢复。不能把仍活跃的锁删掉。checkpoint/epoch_state/best 不一致或 partial 预测缓存会清楚报告；不能自动覆写已有 best 或失败缓存。恢复日志独立保存为 `<run>.resume.<UTC时间>.<PID>.log` 及其 `.exit.json`。

## 6. 确认选定完整 run 后补 test

先根据同一 val 的 mAP50–95 和收敛表现选定完整 run。其 native/training_complete.json 可以是实际早停完成，无需冒称200轮。按上节方法关闭已经退出的本组 tmux 会话，再执行：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-native-ft-v1
bash scripts/autodl_yolov5m_coco_native_ft_v1.sh finalize --run-id v5_ft_01
tmux attach -t comparison-yolov5m-v5_ft_01
```

finalize 要求训练完整和已有身份一致的公共 val，复用 val，只给同一 best 补公共 test/summary，不训练。重复 finalize 只复用一致缓存。native best/早停采用已有完整精度 mAP50–95、latest-tie 规则；native val 的 batch 由 run 的 batch 派生，默认16、rect=True、NMS=.6，实际 model/input dtype、形状和调用条件都有记录。

公共 val/test 固定同一 best/EMA、FP32、640、batch16、conf=.001、类内 NMS=.7、max_det300、无TTA、corrected_sorted_conf_mask_v1。YOLO crack id0→category_id1，包含全部图像和空预测，浮点坐标按真实 x/y resize gain 和整数 padding 逆变换，无额外裁框。checkpoint、配置、GT、预测及 metrics 身份/哈希不一致会拒绝缓存。

## 7. 选定配置与小结果归档，普通提交推送

在独立归档工作树提交；训练工作树 HEAD 和历史 run 不变。归档只包含冻结 YAML、代码/资产/数据/配置身份、初始化/AutoAnchor/有效参数、原生 CSV 和结果摘要；不复制权重、完整日志或预测缓存。测试/调参中的 run 不提前归档为正式结果。

```bash
set -euo pipefail
SOURCE=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-native-ft-v1
BRANCH=bench/yolov5m-coco-native-ft-v1
CODE_SHA="$(cat "$WT/outputs/yolov5m-coco-native-ft-v1/fixed_code_sha.txt")"
[[ "$(git -C "$WT" rev-parse HEAD)" == "$CODE_SHA" ]]
git -C "$SOURCE" fetch origin "$BRANCH"
ARCHIVE_BASE="$(git -C "$SOURCE" rev-parse FETCH_HEAD)"
git -C "$SOURCE" merge-base --is-ancestor "$CODE_SHA" "$ARCHIVE_BASE"
ARCHIVE_WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-native-archive-v5_ft_01
test ! -e "$ARCHIVE_WT"
git -C "$SOURCE" worktree add --detach "$ARCHIVE_WT" "$ARCHIVE_BASE"
ARCHIVE_REL=docs/comparison/archive/yolov5m-coco-native-ft-v1/v5_ft_01
PYTHON="${YOLOV5_PYTHON:-/root/autodl-tmp/envs/comparison-yolov5m-coco-b19-pilot/bin/python}"
"$PYTHON" "$WT/benchmarks/comparison/yolov5m/run.py" archive --run-id v5_ft_01 --archive-dir "$ARCHIVE_WT/$ARCHIVE_REL"
git -C "$ARCHIVE_WT" add -- "$ARCHIVE_REL"
git -C "$ARCHIVE_WT" commit -m 'docs(comparison): archive selected YOLOv5m native v5_ft_01'
git -C "$ARCHIVE_WT" push origin "HEAD:refs/heads/$BRANCH"
git -C "$ARCHIVE_WT" rev-parse HEAD
git -C "$WT" rev-parse HEAD
```

这是普通快进推送。如果远端在此期间前进，push 会拒绝，应在归档工作树检查/整合远端后再推，不能 force push；不合并 main。归档中的 `training_code_sha` 始终是实际训练 run 的 SHA，不改成归档提交。
