# AutoDL 操作顺序

使用现有 `rtdetr` conda 环境、PyTorch 2.1.2 和 CUDA；脚本不安装/升级共享依赖。主仓库、数据及 C2 原始 args/初始化都从 `/root/autodl-tmp/projects/Crack_RTDETR` 读取。请先以最终交付回复的完整 SHA 核对远端。

## 同步和新 worktree

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR
git fetch origin codex/acr-query-attention
git rev-parse origin/codex/acr-query-attention
git ls-remote origin refs/heads/codex/acr-query-attention
test ! -e /root/autodl-tmp/projects/Crack_RTDETR-acr
git worktree add --detach /root/autodl-tmp/projects/Crack_RTDETR-acr origin/codex/acr-query-attention
cd /root/autodl-tmp/projects/Crack_RTDETR-acr
git rev-parse HEAD
```

使用 detached worktree 执行已提交版本，避免服务器同名本地分支冲突；不会切换主仓库分支。已有该 worktree 时先用 `git worktree list`、其中的 `git status` 和 `git rev-parse HEAD` 核对，不删除或覆盖。

## 单模块（默认 C22）

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-acr
bash tools/autodl_acr.sh prepare c22
```

prepare 校验干净源码、环境、C2 全部109字段、资源、数据和源权重 hash，生成独立 FP32 初始化；检查 CPU/CUDA FP32/AMP/half、三层接线、全部状态映射、nc=1 实际 train API、optimizer/EMA，并用独立模型、目录、optimizer 执行 **3 个真实 batch16**。含原生 AMP helper、真实 loss/DN/匹配对齐、FP32 backward、AMP 更新、保存/重载及真实单图末批。结束时没有进入正式训练 epoch 循环。

本地 audit 使用 batch2 是 RTX2060 6GiB 下的独立 smoke 设置；服务器 prepare 固定 smoke batch16，不用本机报告替代 AutoDL 验证。smoke 初始化 scaler=128 只帮助检查三个确实执行的更新；正式 start 保持 C2 原生 GradScaler 设置。显存不足时 prepare 失败并保留日志，不会自动调正式 batch、关闭 AMP 或放宽精度容差。

prepare 成功后，由你手动执行：

```bash
bash tools/autodl_acr.sh start c22
bash tools/autodl_acr.sh status c22
tmux attach -t acr_c22_rtdetr_r18_lite_acr_e200_b16_onlineaug
# 从 tmux 离开而保留训练：Ctrl-b，然后 d
tail -F outputs/c22/launch/console.log
```

start 重新检查提交、代码/初始化/数据/args/audit hash、实际 train 参数、nc=1 公共状态、全部 optimizer 分组及初始 EMA。原子启动锁和 tmux/process/退出记录防止重复启动。log、bootstrap、退出码保留；已有 run/锁/历史 dispatch 时拒绝覆盖，不自动 resume。

训练成功结束后做完整独立 val：

```bash
bash tools/autodl_acr.sh val c22
```

主指标来自 `outputs/c22/evaluation_val/metrics_summary.json`，含 mAP50–95、AP75、P/R、原生速度、排序 mask 问题是否触发、固定样本预测/GT（不包含图片）。训练过程中不允许此入口读取仍在更新的 best.pt。

## 用同一口径重评 C2 / C17

先定位真实历史 best 文件，不猜测 C17 的运行目录。下面两行会让你输入**实际存在的完整路径**，其余命令可以直接复制。C2/C17 评估可以在 C22 开训前完成。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-acr
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
export PYTHONPATH="$PWD/ultralytics-main"
export YOLO_AUTOINSTALL=false
read -r -p 'C2 best.pt 完整路径: ' C2_BEST
read -r -p 'C17 best.pt 完整路径: ' C17_BEST
test -f "$C2_BEST" && test -f "$C17_BEST"
python tools/acr_results.py evaluate --weights "$C2_BEST" --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml --split val --device 0 --batch 16 --policy corrected --label C2 --output outputs/reference_c2_val
python tools/acr_results.py evaluate --weights "$C17_BEST" --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml --split val --device 0 --batch 16 --policy corrected --label C17 --output outputs/reference_c17_val
```

相同设置：640、batch16、FP32、conf=.001、iou=.7、max_det300、不加 NMS、不改预测分数。原公共排序 mask 问题已确认；统一入口默认 corrected，不改旧 run 的任何文件。需要复现旧规则时另给新 `--output` 并指定 `--policy historical`；不能混用两种口径。

## 组合 C23

可提前完成以下 prepare，它只验证组合，不训练：

```bash
bash tools/autodl_acr.sh prepare c23
```

C22 完整 corrected val 支持独立收益，并且你已比较 C2/C17/C22 后，才执行：

```bash
export ACR_COMBINATION_REVIEWED=1
export ACR_C2_VAL_REPORT="$PWD/outputs/reference_c2_val/metrics_summary.json"
export ACR_C17_VAL_REPORT="$PWD/outputs/reference_c17_val/metrics_summary.json"
bash tools/autodl_acr.sh start c23
bash tools/autodl_acr.sh status c23
# 训练成功结束后
bash tools/autodl_acr.sh val c23
```

脚本检查三份完整 val 的数据和评估口径相同，且 C22 mAP50–95>C2；评审证据路径/hash 会保留。这个单次比较仅用于排期，不证明统计显著性。组合最终必须同时对照 C17 和 C22，不能仅因超过 C2 宣称成功。

## Test 与下载包

固定结构和实验选择后，手动 test；不得以 test 调上限或选模块。test 必须使用已完整 val 的同一权重、数据和评估设置。

```bash
bash tools/autodl_acr.sh test c22
bash tools/autodl_acr.sh pack c22
# 组合若已决定开展且完成 val，同理：
# bash tools/autodl_acr.sh test c23
# bash tools/autodl_acr.sh pack c23
```

pack 也允许 test 尚未运行，此时明确加入 `evaluation/test/NOT_RUN.txt`，不会为了打包自动执行 test。包保存在主仓库的 `downloads/acr/c22_<时间戳>.tar.gz` 或 C23 同类路径；旁边有 `.sha256`、`.inventory.json`，包内含每个文件的 SHA256 清单。最多20MiB，超限报错，不静默删证据，不覆盖旧包。

包含模型 YAML、源码快照/相对 C2 diff、完整提交 SHA、初始化/optimizer/EMA/运行审计、实际训练/评估参数、results.csv、results.png 训练曲线、val/test 指标、少量数值诊断和环境。排除权重、原图、标注数据集及训练样本拼图，仅允许 `results.png` 这一个训练曲线图片。

## 路径和失败处理

- C22 正式 run：`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c22_rtdetr_r18_lite_acr_e200_b16_onlineaug`。
- C23 正式 run：`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c23_rtdetr_r18_lite_cscef_v51_acr_e200_b16_onlineaug`。
- 两套报告：新 worktree 的 `outputs/c22/launch`、`outputs/c23/launch`；两套初始化位于 `weights/c22_acr_controlled_init.pt`、`weights/c23_acr_controlled_init.pt`。
- bus.jpg 缺失时优先复制主仓库可信副本，否则下载官方文件并验证固定 hash；yolo26n.pt 优先复制主仓库 helper，否则原生 AMP 自检按官方逻辑获取。必须真正 passed，skipped 会阻止 start。
- prepare 失败看对应带时间戳日志；修复后可再次 prepare，保留旧审计和 smoke。已成功启动/失败退出的正式 run 不自动覆盖或重启。所有独立 evaluate/pack 的目标路径必须不存在，重试时指定新的 `--output`。
- 没有任何远程任务由本次开发机自动启动。服务器运行到哪一步，以服务器新生成的报告为准。
