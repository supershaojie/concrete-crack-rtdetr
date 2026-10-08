# Faster R-CNN ResNet-50-FPN configurable：服务器命令

日期：2026-10-09（北京时间）。分支 `bench/fasterrcnn-r50-fpn-configurable`，实际父提交 `a25e6379be06970f1aa02fbb85d2c9a614a6436f`。上游参考 `torchvision v0.16.2` / `c6f39778e636ec40a69bdbc74386818c57a65af3`；正式环境 torch2.1.2 + torchvision0.16.2 + NumPy1.26.4。

默认：标准 Faster R-CNN ResNet50-FPN，官方 COCO detection pretrained 后替换背景0/crack1两类头；SGD、lr0=0.02、200轮、batch16、640、seed42、无EMA、不早停。这是项目建议配方，学习率与增强不声称是完整官方默认，收敛需由真实val验证。公共主表保持 `corrected_sorted_conf_mask_v1`，test不参与选best。

服务器目录/环境/数据/4090容量尚未由本次 Codex 登录核验。Codex 不SSH、不启动正式长训练或全量test、不停止其他实验。一次容量预检通过不保证两组并发全过程不OOM。所有实际服务器身份由执行回执提供。

## 1. 独立项目

新目录不存在则clone；已有目录只检查身份并快进，不reset或覆盖。完成run冻结后不要pull改代码继续原run，新提交需新项目/run。

```bash
export FRCNN_PROJECT=/root/autodl-tmp/projects/Crack_RTDETR-bench-fasterrcnn-r50-fpn-configurable
export FRCNN_BRANCH=bench/fasterrcnn-r50-fpn-configurable
if [ ! -e "$FRCNN_PROJECT" ]; then
  git clone --branch "$FRCNN_BRANCH" --single-branch https://github.com/supershaojie/concrete-crack-rtdetr.git "$FRCNN_PROJECT"
else
  test "$(git -C "$FRCNN_PROJECT" remote get-url origin)" = https://github.com/supershaojie/concrete-crack-rtdetr.git || exit 1
  test "$(git -C "$FRCNN_PROJECT" branch --show-current)" = "$FRCNN_BRANCH" || exit 1
  test -z "$(git -C "$FRCNN_PROJECT" status --porcelain)" || exit 1
  git -C "$FRCNN_PROJECT" fetch origin "$FRCNN_BRANCH"
  git -C "$FRCNN_PROJECT" merge --ff-only "origin/$FRCNN_BRANCH"
fi
cd "$FRCNN_PROJECT"
git log -1 --format='%H %s'
```

## 2. 自动配置及依赖/权重

先检查当前Python/驱动。脚本优先只读复用完全匹配环境，依赖缺失用本模型隔离overlay；Torch/vision不匹配才新建 `--copies` venv，绝不改正在训练的母环境。新venv要求基础Python3.9–3.11，默认cu118，也支持显式cu121。若当前Python更新，先激活服务器已有合适环境再执行，不升级正在训练的环境。

```bash
cd "$FRCNN_PROJECT"
python --version
nvidia-smi
mkdir -p runtime_configs
if [ ! -e runtime_configs/frcnn_coco_sgd_200.yaml ]; then
  cp benchmarks/comparison/fasterrcnn_r50_fpn/default_config.yaml runtime_configs/frcnn_coco_sgd_200.yaml
fi
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh print-config --config runtime_configs/frcnn_coco_sgd_200.yaml
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh bootstrap --base-python "$(command -v python)" --cuda-wheel cu118
cat .runtime/fasterrcnn-r50-fpn-configurable/python_path.txt
```

`/etc/network_turbo`存在时自动尝试AutoDL学术加速并继承代理，记录成功与否；依赖有下载进度、超时及有界重试。解释器用绝对路径，不resolve成母环境。bootstrap比较实际wheel模型源码与固定官方commit，记录wheel元数据、CUDA构建、真实NMS/ROIAlign扩展路径和完整SHA。无FlashAttention/TensorRT/ONNX安装。

COCO由实际官方枚举解析URL，预检自动下载到本模型缓存，完整SHA校验后复用。此文件已在本次实施实际下载测得：

```text
fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
167502836 bytes
SHA256 258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384
```

默认保留预训练backbone/FPN/RPN/ROI特征，只有类别输出头重新初始化；初始化回执记录全部加载/排除/新建键、官方旧FPN/RPN键迁移、实际参数量、训练层和FrozenBatchNorm。全骨干可训练不表示FrozenBatchNorm统计更新。加载失败不会退回随机权重。

## 3. tmux 预检→训练→同一best最终评估

首次创建新会话；同名已存在不要重复新建，使用attach。进入会话后执行预检，成功exit0后再训练。只按run锁定重复启动，不使用整卡排他锁，不要求另一组停止。

```bash
cd "$FRCNN_PROJECT"
tmux new-session -s comparison-fasterrcnn-r50-fpn-configurable -c "$FRCNN_PROJECT"
```

在tmux里面粘贴：

```bash
export FRCNN_PROJECT=/root/autodl-tmp/projects/Crack_RTDETR-bench-fasterrcnn-r50-fpn-configurable
cd "$FRCNN_PROJECT"
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh preflight --run-id frcnn_coco_sgd_200_s42 --config runtime_configs/frcnn_coco_sgd_200.yaml
```

exit0并有 `PASSED_ACTUAL_CONFIGURED_CAPACITY` 后：

```bash
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh train --run-id frcnn_coco_sgd_200_s42
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh export --run-id frcnn_coco_sgd_200_s42 --split both
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh evaluate --run-id frcnn_coco_sgd_200_s42 --split both
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh pack --run-id frcnn_coco_sgd_200_s42
```

也可用一条完整链，新run-id必须不存在，不能用它隐式恢复：

```bash
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh full --run-id frcnn_coco_full_01 --config runtime_configs/frcnn_coco_sgd_200.yaml
```

预检读取实际YAML split、图像stat/标签/必要尺寸头，核验train6048/45573框、val1728/12840框、test864/6663框及轻量数据身份；不对43GB原图全内容hash/解码审计。按实际batch16/640执行一次前向/反向/SGD，独立子进程退出后重建正式模型，保护正式seed/BN/optimizer状态。OOM/非有限损失或梯度真实失败、留回执，不自动降配方。

脚本先记录真实session/window/pane，再设置window `remain-on-exit`。每阶段stdout/stderr进入 `.runtime/fasterrcnn-r50-fpn-configurable/logs/`，结尾输出Python/tee/最终退出码。脱离Ctrl-b后d，重连/查看：

```bash
tmux attach-session -t comparison-fasterrcnn-r50-fpn-configurable
tmux list-panes -t comparison-fasterrcnn-r50-fpn-configurable -F '#{session_name} #{window_id} #{pane_id} dead=#{pane_dead} exit=#{pane_dead_status}'
tmux capture-pane -p -t comparison-fasterrcnn-r50-fpn-configurable -S -100
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-fasterrcnn-r50-fpn-configurable
ls -lt .runtime/fasterrcnn-r50-fpn-configurable/logs
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh summary --run-id frcnn_coco_sgd_200_s42
```

## 4. YAML/CLI调参，新run和缓存

优先级：提交默认 < 用户YAML < `--set key=value`。未知字段、错误类型/范围和未支持的组合报错，不吞参数。默认配置中的训练、增强、模型后处理和路径均进入展开值/实际身份；EMA、累积、多尺度等未实现选项明示固定禁用。每个新run保存原YAML、CLI overrides、resolved_config与哈希、代码/依赖/数据/初始化身份。不得修改冻结run，只能新建。

编辑 `runtime_configs/frcnn_coco_sgd_200.yaml` 后换run-id；以下CLI真实支持：

```bash
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh print-config --config runtime_configs/frcnn_coco_sgd_200.yaml --set lr0=0.01 --set mixup=0.20 --set degrees=18.0
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh preflight --run-id frcnn_lr001_mix020_01 --config runtime_configs/frcnn_coco_sgd_200.yaml --set lr0=0.01 --set mixup=0.20 --set degrees=18.0
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh preflight --run-id frcnn_clone_lr005_01 --clone-config-from frcnn_coco_sgd_200_s42 --set lr0=0.005
```

克隆只复制配方，不继承权重/优化器/历史。默认 `data_yaml` 是历史 `/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml`；`data_root: null`按实际YAML.path解析，不硬覆盖split。路径不存在时报具体缺项。可通过YAML或 `--set data_yaml=/真实路径.yaml` 覆盖。

可选 `public_coco: /既有GT目录`、`reuse_manifest: /匹配manifest.json`、`weights_file: /已下载官方COCO文件`、`weights_cache: /本模型缓存目录`；使用前填真实存在路径。缓存缺失自动在本run派生，显示进度。GT仅身份/标签/尺寸/image-id吻合才按原字节复制，来源另存，不重写GT改变hash。空图、浮点精度、稳定image_id保留；源图/标签目录只读，不写.cache/.npy。

默认无早停，等分较晚epoch更新best，test仅最终评估。早停需在新run显式 `--set early_stopping=true --set patience=50`。200轮最后5轮从零基epoch195关闭Mosaic/MixUp/CutMix；消耗完整epoch，nonpersistent workers退出，下轮复制新变换树，丢弃旧预取。HSV、仿射、翻转保留，无额外隐式增强。

默认6048图/batch16时每轮378步，计划75600步；step0=0.00002，warmup第1889步=0.02，第1890步=0.02，最后step75599=0.0002。无batch自动LR缩放、无YOLO nbs/bias warmup、无官方首轮warmup叠加。参数组使用固定官方检测参考 `norm_weight_decay=None` 的所有trainable参数单组SGD，并记录参数键、步数、有效batch/各组LR。cuDNN确定性/worker seed/算法warn_only策略如实记录，不承诺逐位重现。

## 5. 显式恢复、独立FP32 val/test、CPU重算和只打包

在同一项目原代码/配置/环境/数据/初始化下恢复last，并检查best/last完整配对。只从最后完整保存epoch继续，未保存的中途epoch重放，恢复SGD/schedule/scaler/RNG与历史best；COCO/旧scratch/另一run checkpoint拒绝。

```bash
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh train --run-id frcnn_coco_sgd_200_s42 --resume
# 或训练未完成时继续完整后处理链，不能附加config/set：
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh full --run-id frcnn_coco_sgd_200_s42 --resume
```

训练已经完成时不再train/full，单独导出同一best：

```bash
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh export --run-id frcnn_coco_sgd_200_s42 --split val
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh export --run-id frcnn_coco_sgd_200_s42 --split test
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh evaluate --run-id frcnn_coco_sgd_200_s42 --split both
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh redraw --run-id frcnn_coco_sgd_200_s42 --split both
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh pack --run-id frcnn_coco_sgd_200_s42
```

首次导出前可显式调整推理batch并记录，与训练冻结配置分开；已有不同设置缓存保护报错：

```bash
bash scripts/autodl_fasterrcnn_r50_fpn_configurable.sh export --run-id frcnn_coco_sgd_200_s42 --split both --inference-batch 8
```

默认全量val1728/test864、batch16/workers0、FP32/eval、无autocast/half/TTA。数据640letterbox auto=False/scaleup=True，内部transform min=max=640、保留官方mean/std；模型返回输入坐标后只按实际gain_x/gain_y及整数left/top逆变换一次。原生RPN及ROI NMS保留，ROI默认score0.001/NMS0.7/max300内部生效，外层/公共评估不新增NMS。

完整高精度jsonl.gz按image_id升序并保留空记录，同分保留原导出顺序；metadata绑定best SHA、代码/数据/GT/评估配置及实际后处理，配套完成回执。完整身份一致缓存只CPU验证/评估/重绘/打包，不重复GPU推理；半套缓存/旧best拒绝。每轮与最终公共P/R/AP50/AP75/mAP50–95保持0–1原值，终端并排显示VAL/TEST百分数，P/R来自公共最大F1点。原生COCOeval只列辅助，不用于选best。真实CSV/results图逐epoch生成，PR/P/R/F1数值及定义公开，不平滑覆盖、不补造训练点；混淆矩阵未计算，不声称存在。

run：`outputs/fasterrcnn-r50-fpn-configurable/frcnn_coco_sgd_200_s42/`；权重留服务器 `checkpoints/best.pt/last.pt`；包在 `outputs/fasterrcnn-packages/`。严格 `<100000000 bytes`，显式白名单含配置/脚本快照/身份、完整val/test指标/预测/GT/image映射、日志回执、CSV/results与曲线、可视化交接、大小/SHA manifest。排除COCO/best/last/optimizer/scaler、原图、环境/vendor/cache、大量逐batch图，拒绝symlink。超限先减可选展示PNG/冗余辅助日志，核心数值/预测不删除；仍超限真实报错并保留原文件。

## 6. 后续检测图和Grad-CAM++交接

使用bootstrap回执解释器，轻量恢复接口如下（直接使用库接口时无需手写上传Python文件；代码已经在仓库）：

```python
import sys
from pathlib import Path
project = Path('/root/autodl-tmp/projects/Crack_RTDETR-bench-fasterrcnn-r50-fpn-configurable')
sys.path.insert(0, str(project / 'benchmarks/comparison/fasterrcnn_r50_fpn'))
from visualization import load_for_visualization
run = project / 'outputs/fasterrcnn-r50-fpn-configurable/frcnn_coco_sgd_200_s42'
restored = load_for_visualization(run, run / 'checkpoints/best.pt', 'cuda:0')
model = restored['model']
image, geometry = restored['preprocess_rgb'](image_rgb)  # HWC RGB uint8 NumPy
# 检测调用可用torch.no_grad；Grad-CAM++调用使用torch.enable_grad。
# 原图坐标：restored['inverse_boxes'](output['boxes'], geometry)
```

模型严格恢复正式best，背景0/crack1/public1，没有全局inference_mode/no_grad。`visualization_layers.json`登记真实命名backbone/FPN层和输出形状；`visualization_handoff.json`保存真实项目/解释器/代码/配置/best绝对路径、model而非EMA、数据/推理/结果路径与验证范围。本次只交接，不实现整套Grad-CAM++或自动选论文图片。

## 7. 验证范围

本地已有torch2.7.1+cu118/torchvision0.22.1+cu118、RTX2060用于明确的SMOKE_ONLY兼容验证，不等于正式目标版本。可复现程序 `benchmarks/comparison/fasterrcnn_r50_fpn/test_core.py` 和 `smoke.py`；smoke仅独立合成数据3轮训练→保存/恢复→FP32 val/test→求梯度恢复→打包，指标不得进入论文主表。

服务器待实测：bootstrap固定源码/wheel/CUDA、原始6048/1728/864数据/GT身份、batch16/640 AMP容量、200轮val收敛、同一best最终完整val/test。具体本地已通过项目和证据见仓库交付验证记录。本次对未现场执行的服务器检查不填“通过”。
