# C17：CSCEF-v5.1 / mean_structure

本轮从 C15 实际训练提交 `fad58b01eb813c5cea7f9fc2c36c93a5a4d388e7` 建立独立分支 `exp-rtdetr-r18-lite-cscef-v51`。C2 基点为 `67c3078e54a657fd96d65fee657a75fbb1dae0d6`。本地 worktree：`D:\MyProjects\Crack_RTDETR\outputs\worktrees\cscef-v51`。没有重置、清理、合并或修改其他实验 worktree。

## 附件核对和解释边界

已读取 `c15_cscef_v5_complete_20260906_022343_832775.tar.gz` 中的 C2/C15 `metrics_summary.json`、C2 `training/args.yaml`、`source/source_commit.txt`，并用包内 manifest 核对这些提取文件及 `source/c15_source.bundle`。`git bundle verify` 通过，bundle HEAD 为上述 `fad58b0`；本地已经包含该提交，无需再次导入。

| 官方 test | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|
| C2 | 0.858162985651444 | 0.4583604005521489 | 0.46963623190802783 |
| C15 | 0.8276554181816267 | 0.4769045979534264 | 0.46770864203611556 |
| C15 − C2，百分点 | −3.050757 | +1.854420 | −0.192759 |

已读取 `c15_v5_val_20260906_030708_872996.tar.gz` 中 `summary.json` 和四组 `metrics.json`。四组均为同一个 C15 checkpoint 的 val 推理干预，没有重新训练。checkpoint SHA256：`099212cf5a03cfb6c3066044f3dee373e465f7d8e8308c29c97e7451eadc99d5`。

| val 干预 | P | R | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| original | 0.798834 | 0.796885 | 0.841499 | 0.488802 | 0.476741 |
| bypass | 0.742671 | 0.771651 | 0.800718 | 0.480605 | 0.460884 |
| half_residual | 0.778991 | 0.787570 | 0.828586 | 0.506769 | 0.483723 |
| mean_structure | 0.785502 | 0.793731 | 0.835325 | 0.508805 | 0.487115 |

bypass 支持“当前已训练网络使用了该分支”。mean_structure 是这四组中 mAP50-95 最好的候选，且五项指标均优于 half_residual。相对 original，其 mAP50-95 提升 1.037471 个百分点，但 P、R、mAP50 仍下降。

空间均值保留的是每张图原始置信度的均值；实际 delta RMS / lateral RMS 的数据集均值从 0.0859071553 降至 0.0630912520。空间分布和实际残差幅度同时改变，未单独证明某个因素是唯一原因。这些数值不是 C17 重新训练的结果，也不能跨 val/test 与 C2 比较后宣称超过 baseline。C17 要验证从 C2 相同干净初始化训练后能否适应该修改。

## 实现与模块包参考

模块包目录：`D:\7.1 rtdetr改\魔鬼面具_RTDETR\RTDETR-main`。

对应压缩包：`D:\7.1 rtdetr改\魔鬼面具_RTDETR\RTDETR-20260623.zip`。两者均可访问；本轮直接查阅目录中的源码，没有解压或复制整个模块包。

实际读取 `ultralytics/nn/extra_modules/block.py` 中：

- `ScharrConv`（8172 行起）：独立逐通道 X/Y Scharr 卷积、零 padding、`0.5*grad_x + 0.5*grad_y` 输出。仅核对核系数与方向，未复用其实现；它的边界和输出语义不等同于项目 V5 的 `/32` Scharr、reflect/replicate、detached FP32 结构张量置信度。
- `Scharr`（11541 行起）：Scharr 边缘幅值、BatchNorm 和额外卷积。未采用，会改变既定内容/结构分支和初始化。
- `CGAFusion`（4768 行起）：空间、通道、像素注意力融合和带 bias 的 1×1 卷积。未采用，它引入本轮未授权的额外注意力与融合机制。

本轮实际复用项目 `cscef_v5.py::CSCEFv5` 的全部构造、内容投影、GroupNorm/SiLU、depthwise convolution、Scharr/结构张量公式、输入检查、尺寸对齐、dtype 转换、残差相加和零初始化。唯一修改为新增子类 `CSCEFv51` 的置信度方法：

```python
@torch.no_grad()
def _compute_structure_confidence(self, gradient_x, gradient_y):
    c = super()._compute_structure_confidence(gradient_x, gradient_y)
    return c.mean(dim=(-2, -1), keepdim=True).expand_as(c)
```

顺序是完整非线性 V5 置信度 → 每图 H/W 平均 → 广播 → 继承 forward 内转换为 h.dtype → `output_projection(c_mean * h)` → 原残差位置相加。无 batch 平均、固定常数、半残差、alpha、RMS 对齐或上限；不增加 state_dict 前缀。原 V5 和原始 AIFI 未修改，也未触碰 GSDR-AIFI 实验。

文件：

- `ultralytics-main/ultralytics/nn/modules/cscef_v51.py`：上述子类。
- `ultralytics-main/ultralytics/nn/modules/__init__.py`、`ultralytics-main/ultralytics/nn/tasks.py`：导出、导入，仅接入已有 CSCEF parser 分支。
- `ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cscef-v51.yaml`：以 V5 YAML 为基础，只改变第 18 层类名；输入 `[17,16]`、后续 Concat `[16,18]`、decoder `[20,23,26]` 均不变。
- `tools/init_rtdetr_r18_lite_cscef_v51_controlled.py`：复用 V5 受控初始化和原层索引映射，不复制训练过的 C15 状态。
- `tools/audit_rtdetr_r18_lite_cscef_v51.py`：继承已修复 V5 审计路径，检查 V5 完整置信度平均、非零残差一致性、Trainer 初始化、CUDA 和真实优化器分组。
- `tools/train_rtdetr_r18_lite_cscef_v51.py`：完整 C2 配方锁定、默认准备、显式启动、复用不可覆盖的 plan、日志/退出码、实际 Trainer 入口检查。
- `ultralytics-main/tests/test_cscef_v51.py`、`tests/fixtures/c17_original_c2_args.yaml`：有针对性的回归及附件 C2 配方测试副本；生产工具不回退到该 fixture。
- 本文和 `C17_CSCEF_V51_C2_FIELDS.md`：依据、验证、完整字段对照与服务器命令。

## 已执行验证

本地环境：Python 3.9.25、PyTorch 2.7.1+cu118、当前 worktree Ultralytics 8.4.21、NVIDIA GeForce RTX 2060。均使用合成张量，没有正式训练或数据集 val/test。

- 原始 C2 初始化实际 SHA256 匹配 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。新增 checkpoint 保持 FP32、epoch=-1、EMA/optimizer/scaler/updates/train_metrics/train_results 清空。
- 受控初始化：映射 533 个共同状态；raw、`RTDETR(checkpoint)`、`RTDETR(YAML).load(checkpoint)` 三种读取均为 540 个状态精确一致。
- 实际 `RTDETRTrainer.get_model`：通过 `__new__` 仅绕过数据集/目录初始化，保持实际模型构建和加载方法；nc=1、80 各 533/533 共同状态相同，5 个分类权重相同，构建/加载后的 CPU RNG 相同，无事后头部复制。CPU FP32、640×640 输入的完整嵌套输出逐 tensor 精确相同，max_abs=0。
- 非零输出投影与其余完全相同权重：v51 与 V5 mean_structure 干预完整 forward 精确一致，max_abs=0；矩形、1×1、单行、单列、零/常量/随机输入均有限，形状和 dtype 正确，且随机边界案例确实激活残差。
- 结构分支保持 detached FP32；固定梯度和 Scharr 边界检查通过；同一张图在单独输入和不同 batch 中的结构均值精确一致。
- 第一次 backward 只有输出投影获得非零梯度；开启输出投影后所有内容权重和 semantic 输入获得有限非零梯度，输出依赖 semantic 内容。
- CUDA FP32、AMP FP16、显式 half：模块前向、反向、非零干预对照，以及整模型 640 前向均通过；整模型输出 `[1,300,5]` 有限。AMP 干预对照包含 lateral FP16 / semantic FP32 的混合类型输入。
- 参数实测：新增 26,912，5 个可训练权重及 2 个 Scharr buffer；nc=1 整模型 unfused 20,109,684。通过真实 `Trainer.build_optimizer` 构造 AdamW，122 个衰减权重（0.0001）、81 个 norm 无衰减权重、128 个 bias 无衰减参数；每个参数恰好覆盖一次。
- C17 有针对性单元测试、原 V5 18 项回归、baseline/V5 parser smoke、Python 3.10 语法解析、Git diff 检查。非 AMP 审计使用 `nullcontext()`，模拟 PyTorch 2.1 CPU FP16 autocast 构造限制的测试通过。
- 实际 Windows 准备调用在原服务器 data 路径不可用处明确失败，没有修改 data、配方或生成虚假的服务器准备结果。

本地完整审计保存在 worktree 的 `outputs/cscef_v51/initialization.json`、`audit.json`，测试/审计日志也在该目录；它们和权重均不提交。报告包含 LF 规范化源码 SHA256，覆盖 v51、继承的 V5、YAML、注册/parser、初始化/审计/训练工具、共用映射工具、原始 Trainer/API/配置入口。服务器将从已提交源码重新生成初始化和审计。

待服务器完成：Python 3.10.13 / PyTorch 2.1.2+cu121 / RTX 4090 的实际重验；访问原 C2 data 配置的准备步骤；用户执行下面的显式 tmux 启动命令后才开始正式 C17 训练。没有新增 holdout、多种子、交叉验证、显存门槛或停止其他实验进程的要求。

## C2 全字段锁定

[完整 109 字段对照](C17_CSCEF_V51_C2_FIELDS.md)：106 项值和类型不变，只有 `model`、`name`、`save_dir` 允许变化。`project` 固定为原始 `/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series`。保留原始 data、batch、优化器、增强、损失、训练轮数和其他全部字段，缺失/新增字段不靠默认值补齐。

默认调用只生成 `launch_plan.json` 和 `train_args.yaml`；`--execute` 或 `--tmux` 显式启动后保存 `console.log` 和 `exit_code.json`。`launch_claim.json` 是互斥启动标记。`--plan` 复用已准备内容，不重写已有 plan；执行前复查原 C2 args、权重、审计、data 配置、源码哈希和当前提交。RTDETR API 传给真实 Trainer 的完整字段也会再次核对。

正式 test 以后仍读取 C2 原始评估记录，已知固定项：split=test、imgsz=640、batch=16、workers=8、device=0、seed=42、conf=0.001、iou=0.7、max_det=300、half=False、augment=False。此轮不执行 test。

## AutoDL 完整命令

以下首段整体粘贴到服务器终端即可：从 Git 同步、建立/检查独立 worktree，生成初始化、审计和准备，再显式创建独立 tmux 会话启动 C17。现有 v51 worktree 必须属于本分支且无未提交修改；不会覆盖已有初始化、实验目录或日志。无需另传脚本，也不使用 Windows 的本地代理地址。

```bash
bash <<'C17'
set -euo pipefail
C17_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
C17_WT=/root/autodl-tmp/projects/Crack_RTDETR_cscef_v51
C17_BRANCH=exp-rtdetr-r18-lite-cscef-v51

git -C "$C17_MAIN" fetch origin "refs/heads/$C17_BRANCH:refs/remotes/origin/$C17_BRANCH"
if [ -e "$C17_WT/.git" ]; then
  test "$(git -C "$C17_WT" rev-parse --show-toplevel)" = "$C17_WT"
  test "$(git -C "$C17_WT" branch --show-current)" = "$C17_BRANCH"
  test -z "$(git -C "$C17_WT" status --porcelain)"
elif [ -e "$C17_WT" ]; then
  printf '%s\n' 'v51 path already exists and is not the expected worktree; preserving it.' >&2
  exit 1
elif git -C "$C17_MAIN" show-ref --verify --quiet "refs/heads/$C17_BRANCH"; then
  git -C "$C17_MAIN" worktree add "$C17_WT" "$C17_BRANCH"
else
  git -C "$C17_MAIN" worktree add -b "$C17_BRANCH" "$C17_WT" "origin/$C17_BRANCH"
fi
git -C "$C17_WT" merge --ff-only "origin/$C17_BRANCH"
test "$(git -C "$C17_WT" rev-parse HEAD)" = "$(git -C "$C17_WT" rev-parse "origin/$C17_BRANCH")"
git -C "$C17_WT" merge-base --is-ancestor fad58b01eb813c5cea7f9fc2c36c93a5a4d388e7 HEAD
git -C "$C17_WT" log -1 --oneline

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
cd "$C17_WT"
export PYTHONPATH="$C17_WT/ultralytics-main"
export PYTHONUNBUFFERED=1

python tools/init_rtdetr_r18_lite_cscef_v51_controlled.py \
  --source "$C17_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" \
  --output "$C17_WT/weights/rtdetr_r18_lite_cscef_v51_imagenet_backbone_init.pt" \
  --report "$C17_WT/outputs/cscef_v51/initialization.json"

python tools/audit_rtdetr_r18_lite_cscef_v51.py \
  --source "$C17_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" \
  --initialized "$C17_WT/weights/rtdetr_r18_lite_cscef_v51_imagenet_backbone_init.pt" \
  --report "$C17_WT/outputs/cscef_v51/audit.json"

python -m unittest discover -s ultralytics-main/tests -p test_cscef_v51.py -v
python -m unittest discover -s ultralytics-main/tests -p test_cscef_v5.py -v

python tools/train_rtdetr_r18_lite_cscef_v51.py \
  --c2-args "$C17_MAIN/runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml" \
  --initialized "$C17_WT/weights/rtdetr_r18_lite_cscef_v51_imagenet_backbone_init.pt" \
  --audit-report "$C17_WT/outputs/cscef_v51/audit.json" \
  --name c17_rtdetr_r18_lite_cscef_v51_e200_b16_onlineaug \
  --report-dir "$C17_WT/outputs/cscef_v51/launch_c17"

python tools/train_rtdetr_r18_lite_cscef_v51.py \
  --plan "$C17_WT/outputs/cscef_v51/launch_c17/launch_plan.json" --tmux
tmux list-sessions
C17
```

后续在终端按需执行：

```bash
# 查看实时日志；Ctrl-C 仅退出 tail，不停止训练
tail -n 80 -F /root/autodl-tmp/projects/Crack_RTDETR_cscef_v51/outputs/cscef_v51/launch_c17/console.log
```

```bash
# 进入训练会话；Ctrl-B 然后 D 可脱离会话
tmux attach -t cscef_v51_c17_rtdetr_r18_lite_cscef_v51_e200_b16_onlineaug
```

```bash
# 正常结束 exit_code=0；非零表示失败/中断。文件尚不存在时查看日志和 tmux 状态。
if [ -f /root/autodl-tmp/projects/Crack_RTDETR_cscef_v51/outputs/cscef_v51/launch_c17/exit_code.json ]; then
  cat /root/autodl-tmp/projects/Crack_RTDETR_cscef_v51/outputs/cscef_v51/launch_c17/exit_code.json
else
  printf '%s\n' 'No exit status yet; inspect console.log and tmux list-sessions.'
fi
```
