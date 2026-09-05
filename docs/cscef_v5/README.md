# CSCEF-v5：实现、验证与服务器入口

本次实现对应用户正式授权的修订版 `C13_CSCEF_v5_Codex_prompt.md`。起点为 V4 提交
`f34502c43a88edc9cb1076bd996b9c3fc2d23e31`，目标分支为 `exp-rtdetr-r18-lite-cscef-v5`。
C2 提交为 `67c3078e54a657fd96d65fee657a75fbb1dae0d6`。未发现适用的 `AGENTS.md`。
原 V4 主 worktree 及其未跟踪文件保留，C14 worktree 未改动。

本轮没有启动正式训练，没有运行 val/test 数据集评估，没有性能增益结论。

## 结构与变更范围

```text
l = GN_l(P_l(L))
s = GN_s(P_s(S))
h = SiLU(GN_h(DW3(P_mix(concat(l,s)))))
c = V4_structure_confidence(Scharr(l))  # FP32 / no_grad / detach
L_out = L + P_out(c * h)
```

`P_l/P_s: 256→32`，`P_mix: 64→32`，`DW3: 32通道 depthwise 3×3`，`P_out: 32→256`。
卷积无 bias；GN 为 8 组、affine=False、eps=1e-6。输出投影精确置零，其余卷积保留 PyTorch 默认
Kaiming-uniform 初始化。所有随机层构造在 `torch.random.fork_rng(devices=[])` 中完成。
没有余弦带通、alpha、RMS matcher、learned gate 或 attention。

结构置信度保留 V4 的 `/32` Scharr 核、reflect/单轴退化时 replicate、3×3 平均池化及其边界行为：

```text
energy = jxx+jyy
coherence = clamp(sqrt(clamp((jxx-jyy)^2+4*jxy^2,min=0))/(energy+eps),0,1)
reference = mean_spatial(energy).detach()
reliability = clamp(energy/(energy+reference+eps),0,1)
c = sqrt(clamp(coherence*reliability,0,1)).detach()
```

只在层18输入 `[17,16]` 插入 V5，Concat 保持 `[16,18]`，decoder 输入保持 `[20,23,26]`。
真实 neck 在 AMP 下可能给出不同 dtype 的 L/S，各自投影遵循 autocast，最终输出跟随 L 的 dtype。

| 文件 | 作用 |
|---|---|
| `ultralytics-main/ultralytics/nn/modules/cscef_v5.py` | V5 模块 |
| `ultralytics-main/ultralytics/nn/modules/__init__.py` | 新增导出 |
| `ultralytics-main/ultralytics/nn/tasks.py` | 仅新增 import 和 parser 识别 |
| `ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cscef-v5.yaml` | 单插入点模型图 |
| `tools/init_rtdetr_r18_lite_cscef_v5_controlled.py` | SHA256 约束的 nc=80 初始化与重载审计 |
| `tools/audit_rtdetr_r18_lite_cscef_v5.py` | 真实 trainer 建模、CPU/GPU、梯度和复杂度审计 |
| `tools/train_rtdetr_r18_lite_cscef_v5.py` | C2 配方锁定、默认只准备、显式启动与日志 |
| `ultralytics-main/tests/test_cscef_v5.py` | 模块与训练入口回归测试 |
| `docs/cscef_v5/local_validation.json` | 本机验证摘要、源文件与日志哈希 |

旧 CSCEF 模块、YAML、权重、测试、工具及公共 trainer/optimizer/loss/decoder/AIFI/data/default 均未改动。

## 已实际完成的验证

本机环境：Windows，Python 3.9.25，PyTorch 2.7.1+cu118，Ultralytics 8.4.21（本 worktree），RTX 2060。
没有安装或升级依赖；测试使用已有 Python 的 unittest。

- C2 初始化 SHA256：`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
- nc=80 共同状态 533/533 精确映射；新增状态7项；backbone 参数69/69键、11,199,968 numel完整覆盖。
- 保存为 FP32，C2 原 half 数值精确提升，新增权重不做 half 量化；epoch=-1，无 EMA/optimizer/scaler/updates。
- 原始读取、`RTDETR(checkpoint)`、`RTDETR(YAML).load(checkpoint)` 三条路径540/540状态精确重载。
- nc=1 与 nc=80 均实际调用未改的 `RTDETRTrainer.get_model`，所有533个共同状态及构建/加载后的CPU RNG精确一致。
  5个分类相关权重最大绝对差均为0。无 nc=1 建模后的手工 head 复制。
- 为避免读取数据集，审计以 `RTDETRTrainer.__new__` 提供该真实方法读取的 `data[nc/channels]`，
  使用真实 `init_seeds(42, deterministic=True)`，对应单进程 RANK=-1。
  没有声称运行完整 Trainer 初始化或正式训练生命周期。
- nc=1/80：CPU FP32、随机640×640输入、完整返回树逐张量精确相等，最大绝对差0。
- V4/V5相同 gx/gy 下结构公式及 Scharr 边界精确一致；零图、常数图、随机图、非方形、1×1/单轴1均有限。
- 第一轮反向只有输出投影非零梯度；测试用更新后全部上游卷积获得非零有限梯度，改变 S 会改变残差。
- CUDA FP32、AMP FP16、显式 half：模块前向/反向通过，完整640×640单类模型输出均为 `(1,300,5)` 且有限。
- V5 15项测试通过；V4原有21项测试通过（包括CUDA和旧文件保护）；baseline smoke通过；三个CLI的 `--help` 通过。
- 训练入口的逐字段锁定、缺参拒绝、旧实验保护、checkpoint与审计不匹配拒绝，以及 console/退出码记录均以
  **明确标识的合成配置和 mock API** 测试。它们不构成服务器原始C2配置核对或真实训练证据。

| nc=1 | unfused 参数 | fused 参数 | unfused GFLOPs | fused GFLOPs |
|---|---:|---:|---:|---:|
| C2 | 20,082,772 | 19,877,716 | 58.2766080 | 57.1657728 |
| V5 | 20,109,684 | 19,904,628 | 58.6210816 | 57.5102464 |

新增可训练参数精确为26,912（8,192+8,192+2,048+288+8,192），比V4多10,239。
GFLOPs为本项目model.info/THOP统计，**不完整计入函数式Scharr、平均池化和逐点运算**。

完整逐张量审计位于本机 worktree 的 `outputs/cscef_v5/audit.json`，初始化审计为
`outputs/cscef_v5/initialization.json`。权重为 `weights/rtdetr_r18_lite_cscef_v5_imagenet_backbone_init.pt`；
权重和运行输出按原仓库规则忽略，不推送Git。已提交摘要记录其SHA256。
审计在提交前执行，报告中的Git HEAD是V4来源提交，具体审计对象由工作区状态与代码规范化LF的SHA256绑定。

## C13 证据边界

诊断压缩包SHA256为 `455bcc87dfd70fcda3b5d38e4345f8883f6a009d8744ac44c57bb9e838161d8f`；
已核对两个JSON，独立初始化审计与summary内嵌内容一致。
bandpass均值0.993260、移除结构项综合AP下降、原nc=1重建5个分类权重不同支持本次候选设计；
这些证据不证明V5训练后更优。neck双尺度内容及结构引导是本模块设计重点；零初始化、残差和1×1卷积是通用手段。
C14/GSDR-AIFI未在本轮分析或改动。

## 服务器补充命令

服务器原始C2 `args.yaml` 在本机不可用，真实配方对照、Linux/服务器环境复验、库自带的训练AMP预检和tmux真实启动
尚未执行。本机调用准备入口时已确认缺少该文件会报错退出，不回落到默认配置或C13记录。
以下先完成服务器初始化、审计与准备；最后的训练命令留到用户明确启动时使用。

若服务器尚无V5分支/worktree，从推送分支建立；若已存在先检查其工作区状态并复用，不强制重置：

```bash
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR_cscef_v5
git -C "$MAIN" fetch origin exp-rtdetr-r18-lite-cscef-v5
git -C "$MAIN" worktree add -b exp-rtdetr-r18-lite-cscef-v5 "$WT" origin/exp-rtdetr-r18-lite-cscef-v5
```

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
cd /root/autodl-tmp/projects/Crack_RTDETR_cscef_v5
export PYTHONPATH="$PWD/ultralytics-main"
export YOLO_CONFIG_DIR="$PWD/outputs/runtime"
export YOLO_AUTOINSTALL=false
mkdir -p "$YOLO_CONFIG_DIR/Ultralytics"

python tools/init_rtdetr_r18_lite_cscef_v5_controlled.py \
  --source /root/autodl-tmp/projects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt
python tools/audit_rtdetr_r18_lite_cscef_v5.py \
  --source /root/autodl-tmp/projects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt
python -m unittest discover -s ultralytics-main/tests -p test_cscef_v5.py -v
python -m unittest discover -s ultralytics-main/tests -p test_cscef_v4.py -v
python tools/test_rtdetr_r18_lite_model.py
python tools/train_rtdetr_r18_lite_cscef_v5.py
```

初始化拒绝覆盖任何已存在的输出checkpoint；已生成时先运行审计，需重新生成则以 `--output` 指定新的明确路径，
并向后续审计/准备命令传入对应的 `--initialized`。
准备工具默认读取：
`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml`。
它完整读取全部字段，只允许model/project/name/save_dir改变，pretrained仍为True，data和其余设置逐字段保留。
默认200epochs、batch16、seed42、AdamW、lr0=0.0005等仅用于检测错误，不用于补齐缺失字段。
`--help` 可查看路径参数；没有任意超参数覆盖选项。

若新worktree缺少 `ultralytics-main/ultralytics/assets/bus.jpg`，仅在旧主worktree该文件已确认正常的前提下执行：

```bash
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
ASSET=ultralytics-main/ultralytics/assets/bus.jpg
if [ ! -f "$ASSET" ]; then
  test -f "$MAIN/$ASSET" || exit 1
  mkdir -p "$(dirname "$ASSET")" outputs/cscef_v5
  cp -n "$MAIN/$ASSET" "$ASSET"
  sha256sum "$MAIN/$ASSET" "$ASSET" > outputs/cscef_v5/amp_asset_source.sha256
fi
```

本轮不执行下列训练命令。未来明确启动后，它会重新核对C2原始args、checkpoint、审计代码哈希和干净工作树，
在独立tmux会话中启动，保留真实console与退出码；不关闭AMP或修改check_amp：

```bash
python tools/train_rtdetr_r18_lite_cscef_v5.py --tmux
tail -f outputs/cscef_v5/launch_c15/console.log
cat outputs/cscef_v5/launch_c15/exit_code.json
```

默认实验目录：`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c15_rtdetr_r18_lite_cscef_v5_e200_b16_onlineaug`。
准备输出：`outputs/cscef_v5/launch_c15/launch_plan.json`（Git commit、初始化/C2 args/审计/data配置哈希和逐字段比较）、
`train_args.yaml`。训练后同目录增加 `console.log`、`exit_code.json`。
实验名或日志目录已存在时，用新的 `--name` 和 `--report-dir`，不覆盖旧结果。

正式test仍为split=test、640、batch16、workers8、device0、seed42、conf0.001、iou0.7、max_det300、
half=False、augment=False；准备报告仅记录这份参考，不触发评估。
