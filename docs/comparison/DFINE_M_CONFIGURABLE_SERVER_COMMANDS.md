# D-FINE-M：服务器命令与可配置配方

本分支 `bench/dfine-m-configurable` 的父提交为公共 YOLO26 实现 `a25e6379be06970f1aa02fbb85d2c9a614a6436f`。官方 D-FINE 锁定 `956d1709314c2c6a4df6f34de232054578a7449f`，不追踪 master。实际项目提交由下方 `git rev-parse HEAD` 显示并写入 run 回执。本机 smoke 与服务器正式训练分开，本文未宣称服务器数据、batch16 或论文指标已通过。

作者 README 的 COCO M 资产是 [dfine_m_coco.pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth)，79,108,938 字节，SHA256 `b44a7586bf490858c7b8bce9e44bd025cb88724df9a07a8deb3ae1c12e608195`。实际 checkpoint 仅含 `model`，初始化保留1042兼容键，仅重建11个分类/去噪键。新 run 连续更新 EMA，best 保存真正参与 FP32 公共 val 的 `selected_state_dict`，默认原始来源 `ema`。没有 Objects365、ImageNet 追加下载或随机初始化回退。

## 1. 独立取得代码

建议目录未在服务器现场核验；若已存在，先核对分支和未提交修改，不覆盖其他工作树。

```bash
PROJECT=/root/autodl-tmp/projects/Crack_RTDETR-bench-dfine-m-configurable
BRANCH=bench/dfine-m-configurable
if [ ! -e "$PROJECT" ]; then
  git clone --single-branch --branch "$BRANCH" https://github.com/supershaojie/concrete-crack-rtdetr.git "$PROJECT"
fi
cd "$PROJECT"
[ "$(git branch --show-current)" = "$BRANCH" ] || { echo "分支不同，请检查"; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "存在未提交修改，请检查"; exit 1; }
git fetch origin "$BRANCH"
git merge --ff-only "origin/$BRANCH"
export DFINE_EXPECTED_CODE_SHA="$(git rev-parse HEAD)"
printf '项目提交：%s\n' "$DFINE_EXPECTED_CODE_SHA"
```

## 2. 环境、权重和配置预览

先激活可用的已有 Torch 环境；`DFINE_BASE_PYTHON` 取当前真实解释器。脚本记录绝对解释器、sys.prefix、关键版本、CUDA/设备、导入路径和 pip freeze 身份。兼容则只读复用，不在原环境 pip 升级；不兼容则创建本模型 `.envs/dfine-m-configurable`，`venv --copies` 和固定 requirements-lock。新建参考环境为 Python3.10.14、Torch2.1.2、Torchvision0.16.2、NumPy1.26.4；已有环境以实际原生算子检查为准。私有最小路径不导入官方 profiler/额外 backbone/solver，因而不需 calflops/transformers/loguru/tensorboard；完整官方 requirements 和文件哈希仍锁定。无 MMDetection/MMCV/FlashAttention/部署前置依赖。

```bash
export DFINE_BASE_PYTHON="$(command -v python)"
bash scripts/autodl_dfine_m_configurable.sh bootstrap
mkdir -p runtime_configs
cp benchmarks/comparison/dfine_m/default_config.yaml runtime_configs/dfine_m.yaml
bash scripts/autodl_dfine_m_configurable.sh preview --config runtime_configs/dfine_m.yaml
```

存在可读 `/etc/network_turbo` 时默认 source 服务器已有 AutoDL 入口，记录入口结果和 proxy 环境键，不据此声称已加速。可用 `DFINE_ENABLE_ACCELERATION=0` 关闭。Git/下载/pip 有阶段、进度、超时和有界重试；相同源码/权重校验后复用。下载失败不切换来源。`bootstrap --new-environment` 可明确要求独立环境，已有环境目录不重装。

YAML 可调 epochs、batch_size、workers、增强、lr0/backbone_lr、betas/weight_decay、AMP、EMA、warmup、lrf、梯度累积/裁剪、推理 batch/workers/device 和数据/cache/source/weights 路径。优先级为提交默认值 → YAML 或既有配方 → CLI。未知或未接入字段报错，当前仅支持已实现 AdamW。单类、固定640及公共 conf0.001/max300/NMS-free 为协议固定项。两个 LR 直接执行，不按 batch 暗缩放。

```bash
bash scripts/autodl_dfine_m_configurable.sh preview --config runtime_configs/dfine_m.yaml \
  --set lr0=0.00008 --set backbone_lr=0.000008 --set mixup=0.25 --set train_workers=4
```

默认200轮，第196轮（零基195）关闭 Mosaic/MixUp/CutMix，其余增强保留。每轮重建非 persistent workers，旧预取不跨边界；没有官方120/132分阶段加载、EMA重启或随机 collate resize。FDR/GO-LSD、matcher、原生 auxiliary/denoising loss 和权重不变。

## 3. 独立 tmux 完整执行

先检查 YAML 的 data_yaml。默认历史配置是 `/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml`，data_root=null，解析实际原 YAML。不重划分、不改原标签、不复制整库。

```bash
RUN=dfine_m_coco_e200_b16_s42_20261009
bash scripts/autodl_dfine_m_configurable.sh tmux --run-id "$RUN" --config runtime_configs/dfine_m.yaml
# 路径覆盖可加 --set data_yaml=/实际/config.yaml --set data_root=/实际/crack_det
# 匹配缓存可加 --set reuse_manifest=/实际/已验证run 或 --set public_coco=/实际/gt

tmux attach -t comparison-dfine-m-configurable
# 脱离：Ctrl-b，然后 d；重连使用相同 attach。
tmux display-message -p -t "comparison-dfine-m-configurable:$RUN.0" '#{pane_dead_status}'
tail -n 80 .runtime/dfine-m-configurable/logs/${RUN}_train_*.log
```

实际 session/window/pane 保留 remain-on-exit，完成/失败均可检查。已有同名 session 会报错，不杀旧会话；另一个本模型 run 可设置 `DFINE_TMUX_SESSION=comparison-dfine-m-run2`。只锁本 run/本模型 bootstrap cache，允许其他模型同跑 GPU0，不设整卡锁。

完整流程：bootstrap → prepare → 隔离进程最终 batch16/640 forward/native loss/backward/optimizer/EMA 预检 → fresh continuous train → 同一 selected best 的完整 FP32 val/test → CPU公共评价/曲线 → 小包。预检不污染正式模型/EMA/BN/RNG，不能保证同 GPU 两组全过程无 OOM。非有限 loss/gradient、OOM、标签/来源错误直接失败，保留日志，不自动缩 batch/输入，不停其他进程。终端逐轮显示 batch、损失、组 LR、EMA updates、val 摘要。真实 Python `PIPESTATUS[0]` 保存为 `.log.exit_code`，不以 tee 成功冒充训练成功。

## 4. 独立阶段、恢复、新配方

```bash
RUN=dfine_m_coco_e200_b16_s42_20261009
RUNDIR="$PWD/outputs/dfine-m-configurable/$RUN"
bash scripts/autodl_dfine_m_configurable.sh prepare --run-id "$RUN" --config runtime_configs/dfine_m.yaml
bash scripts/autodl_dfine_m_configurable.sh preflight --run-dir "$RUNDIR"
bash scripts/autodl_dfine_m_configurable.sh train --run-dir "$RUNDIR"

# 中断后显式恢复本 run last；COCO、其他 run、其他模型、smoke 不能作 formal resume。
bash scripts/autodl_dfine_m_configurable.sh train --run-dir "$RUNDIR" \
  --resume "$RUNDIR/train/weights/last.pth"
# 恢复后自动继续 final/pack：
bash scripts/autodl_dfine_m_configurable.sh all --run-dir "$RUNDIR" \
  --resume "$RUNDIR/train/weights/last.pth"

# 复制 frozen recipe 到新 run，不继承 optimizer/权重/best/RNG，不改 Python。
bash scripts/autodl_dfine_m_configurable.sh prepare --run-id dfine_m_lr08_new \
  --clone-config-from "$RUNDIR" --set lr0=0.00008 --set backbone_lr=0.000008
```

freeze 后配置不能编辑；任何变参创建新 run。last 完整保存 model、EMA module/updates、optimizer、update schedule、GradScaler、criterion、Python/NumPy/Torch/CUDA/loader RNG、轮数和真实 best 身份。resume 校验 UUID、模型/上游/源码/配方/环境/数据/COCO初始化、best SHA和已完成状态。

## 5. 完整 FP32 val/test、CPU复用与仅打包

```bash
RUNDIR="$PWD/outputs/dfine-m-configurable/$RUN"
bash scripts/autodl_dfine_m_configurable.sh final --run-dir "$RUNDIR" --split val
bash scripts/autodl_dfine_m_configurable.sh final --run-dir "$RUNDIR" --split test
# 默认连续两个 split：
bash scripts/autodl_dfine_m_configurable.sh final --run-dir "$RUNDIR"
# 完整合法 cache 可直接 CPU 重算/重绘/打包，final 不再 GPU 推理。
bash scripts/autodl_dfine_m_configurable.sh evaluate --run-dir "$RUNDIR"
bash scripts/autodl_dfine_m_configurable.sh pack --run-dir "$RUNDIR"
```

默认 batch16/workers0、固定640 letterbox、FP32/eval，无 autocast/half/TTA/TF32/NMS。官方300 query/topk300、sigmoid一次；postprocessor 先使用640输入尺寸，然后按 round 后 gain_x/gain_y 与整数 left/top inverse一次。YOLO0/派生COCO1显式转 native0，预测只转一次 public1。预测按 image_id 升序含全部图像/空预测、原图浮点 xyxy 和 letterbox 参数，带 best/配置/代码/数据/GT/公共评价身份及完整 receipt。

主表 `corrected_sorted_conf_mask_v1`：P/R 是公共最大F1点，AP50/AP75/mAP50–95 复用现有 evaluator；原值未舍入0–1，终端显示百分数。最终VAL/TEST两行并排，含图像/实例/batch、耗时/速度和五项指标，打包后仍显示。test不参与选 best/调参。原生 COCO AP 未替代主表。train/results.csv、results.png 显示真实 native losses/LR/public val；PR/P/R/F1图及公共函数原始曲线数值保留，无伪造epoch或平滑覆盖。

包为 `outputs/dfine-m-configurable/${RUN}_results.zip`，严格 `<100,000,000` 字节。白名单包含配置/完整include、加载/身份/环境/源码/许可证回执、全量高精度预测/GT/映射/指标、训练表图/曲线、日志、真实命令、小脚本和 visualization_handoff.json、大小/SHA清单。排除所有 COCO/best/last 权重、原图、环境/vendor，拒绝symlink。仅缩减可选大日志，核心超限则报错，服务器原文件/GT哈希不变。

## 6. 可求梯度的 selected EMA 恢复接口

使用 handoff 中真实解释器/项目/数据/结果绝对路径。仅提供恢复接口和候选层/形状，未实施 Grad-CAM++/选图/部署。

```python
import sys
sys.path.insert(0, "/实际项目/benchmarks/comparison/dfine_m")
from model import load_for_visualization
model, metadata = load_for_visualization(
    "/实际run/resolved_config.yaml", "/实际run/train/weights/best.pth", "cuda:0"
)
# model(rgb_float32_tensor) -> pred_logits[B,300,1], pred_boxes[B,300,4] normalized cxcywh
# 原生模型允许 backward；推理 runner 才使用 no_grad，未自动deploy/fusion。
```

visualization_handoff.json 包含项目/解释器/sys.prefix/上游/code commit/config/best路径与SHA、selected来源和键、实际候选backbone/encoder形状、数据/GT/预测/指标绝对路径及核验状态。

## 7. 本机验证与服务器边界

```bash
"$(cat .runtime/dfine-m-configurable/python.path)" benchmarks/comparison/dfine_m/verification.py
"$(cat .runtime/dfine-m-configurable/python.path)" benchmarks/comparison/dfine_m/run.py smoke
```

smoke 使用新建独立 synthetic train/val/test，固定640、batch1、2轮，保存/恢复、同一 EMA 导出、公共评价/曲线、缓存复用、梯度恢复和小包。不能当正式 batch16 容量或论文指标。服务器待验证历史 train6048/45573、val1728/12840、test864/6663、identity `3401e485b40398ae096e8fd00101cae38b607ee1d6d1b5d5abba5c443e273483` 及 GT实际 SHA。接入只读取标签/必要尺寸头/stat，不重哈希43GB原图或全解码审计。正式200轮和真实完整val/test仅由用户在服务器运行。

AMP 补充：项目显式增加可配置 `amp_init_scale=1.0`，`amp_growth_factor=2.0`、`amp_backoff_factor=0.5`、`amp_growth_interval=2000` 保留 GradScaler 的增长/回退语义。RTX2060 synthetic batch1 的原生损失在65536和1024起始scale下出现非有限梯度，scale1的相同forward/backward/clip/optimizer/EMA通过；因此起始scale作为公开项目参数记录，未改变AMP精度、FDR/GO-LSD公式，也不在失败后自动调整正式run。真实batch16仍需服务器独立预检。

缓存/环境补充：默认仅在已知 YOLO26/v5/v8 对比项目的有限 run 清单中查找匹配 inventory，核验原图stat、标签字节、实际YAML split/order及身份后只读复用；否则建立本模型轻量 inventory。显式 reuse_manifest 不匹配则失败。gt字节直接保留，public_gt_identity.json 记录实际与历史 SHA。缺少轻量依赖但 Torch/Torchvision/NumPy 兼容时，创建 `--system-site-packages --copies` 小 overlay，并用实际核心版本约束安装缺失包；只有核心冲突或显式新环境才创建全量隔离环境。实际解释器及版本在 bootstrap/freeze 回执记录，未验证的服务器安装不冒称通过。
