# 本地验证

环境为 Windows、Python 3.9.25、PyTorch 2.7.1+cu118、RTX 2060 6GB。没有安装/升级依赖，没有正式 epoch loop，没有完整数据集 val/test。报告 runtime 的 commit 是测试时的 C2 父提交、dirty=true；最终代码由提交及 `source_files_sha256.json` 定位，不能把父提交误认为交付代码版本。

## 实测参数与公共映射

| Variant | nc=1 unfused 参数 | 公共 state | 新 state |
|---|---:|---:|---:|
| C2 | 20,082,772 | 533 | 0 |
| cscef_v6 | 20,109,684 | 533 | 7 |
| scca_v2 | 20,148,312 | 533 | 5 |
| cbr_v2 | 20,128,661 | 533 | 14 |
| cscef_v6_scca_v2 | 20,175,224 | 533 | 12 |
| cscef_v6_cbr_v2 | 20,155,573 | 533 | 21 |
| scca_v2_cbr_v2 | 20,194,201 | 533 | 19 |
| triad_v1 | 20,221,113 | 533 | 26 |

全部 7 个模型的公共 tensor max_abs=0，MISSING/UNEXPECTED/SHAPE_MISMATCH 为空。唯一初始化来源 SHA256 为 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。原生 Trainer nc80→nc1 只重建预期的 9 个分类状态。每个 variant 的逐 tensor 映射、初始化文件重载及 YAML API 加载检查见 [evidence](evidence/) 和 [汇总](validation_summary.json)。参数量与规范理论值全部一致。

## Forward / backward

- 全部 7 variant：FP32 640×640、160×192，P3/P4/P5、Decoder、box、score 和全部嵌套输出对 C2 **max_abs=0、max_rel=0**；640 三尺度为 80²/40²/20²。
- C2/C17/C19/C24 原 YAML 构建与矩形 forward 有限。C2 公共受保护文件无差异。
- CSCEF 非零输出干预只改变 Decoder P3，不改变原 PAN P4/P5。额外改变 AIFI/top-down、固定 P3_base 后，stable refs 和 CSCEF residual 保持不变。
- SCCA v1/v2 同参数、非零输出、pre/post-norm forward 误差均为 0。v1 对 src/s 有梯度，v2 无该额外边；v2 MHA/FFN 和支路参数仍有梯度。
- CSCEF refs、CBR ref/query 无支路梯度；CSCEF 对 base 和 CBR 对 original_box 的直接导数误差均为 0。非零激活后各创新参数梯度非零且有限。
- 每个 variant 在 CPU FP32 与 CUDA AMP 各执行两个 synthetic 原生 RT-DETR loss/DN/backward/AdamW 更新；batch2→batch1、GT 数变化，保持常规300 query，DN 总查询数动态变化。第二步全部新参数获得非零梯度，主干/neck reference 主路径梯度存在。
- 使用真实 `BaseTrainer.build_optimizer`，所有新旧参数各覆盖一次，bias/norm/weight decay 分组与原规则一致。AMP retained feature 梯度是在 loss scaling 下观察其存在性；参数梯度在 unscale 后检查。
- 全部 variant 执行非零整网真实 `.half()` 推理与 AMP/FP32 对比，输出有限。CBR 的最终几何运算保持 FP32。
- 两步更新后完整模型保存/重载的 tensor 和输出误差为 0；原生 Trainer 再重建保留非零新参数。未在加载后再次 zero-init。

模块/路由/流程共 21 项单元测试通过；实际 Bash 语法和显式 Git stub 的同步场景通过。启动流程以 mock 验证先预检再 tmux 派发、重复启动拒绝和 token 保护，没有启动服务器进程。参见 [ops_validation.json](ops_validation.json)。

## 工具链与诊断

用原本已有的小样本副本，每个原 split 复制两张图及标签、逐文件 SHA 核对，独立运行统一 val/test（共4张评估图）。保留原 split 身份、640/batch16/conf=.001/iou=.7/max_det300/FP32/seed42/workers0。检查 AP75、曲线、混淆矩阵、有效配置、完整查询/GT 导出和相同 checkpoint 门禁。这是未训练模型的工具链检查，指标不用于模型选择或论文。见 [evaluation_pipeline.json](evaluation_pipeline.json)。

完整 pack 使用显式 synthetic 训练状态夹具和上述真实微型评估输出，验证完整性/manifest/SHA256、拒绝覆盖、拒绝损坏 test 证据、禁止隐式补跑评估。打包按最近 C26 complete 政策包含 best/last，权重仅留 outputs/服务器下载包，不进入 Git。硬链接权重必须写成普通归档成员，已用该夹具发现并修复。见 [pack_validation.json](pack_validation.json)。

[diagnostics.json](diagnostics.json) 包含非零诊断干预下的 feature shape/dtype/RMS、CSCEF residual ratio、SCCA delta/src 与 delta/S、CBR 修正和饱和率、同图 batch 误差、AMP 差异及本机短计时。仅为本机 synthetic B1 160×192 粗测，不代表4090正式训练速度。

跨精度的原生 encoder top-k 排序会变化。本次固定 seed73 非零诊断：Triad 原始逐行 AMP/FP32 最大误差0.833283，共享299个 encoder 索引，按相同索引对齐后最大误差0.000520229；C2 原始误差0.75，共享300个索引，对齐误差0.000325263。没有强制固定 top-k，也没有把这项检查宣称为 AMP/FP32 等价。正式零初始化等价比较是同精度父子模型，其最大误差仍为0。

## 未验证

AutoDL Python3.10/PyTorch2.1.2+cu121/RTX4090 的实际运行、640 batch16 训练容量、正式200 epoch、完整 val/test、多种子、真实训练 checkpoint 梯度夹角及最终组合涨点均未验证。服务器 start-direct 会在原环境重新执行所选 variant 的 AMP/half、原生 loss/DN 与梯度预检，失败时保留日志并停止，不自动调整 batch。

## 复现

```bash
python tools/check_triad_compat.py --source weights/rtdetr_r18_lite_imagenet_backbone_init.pt --output outputs/local_check_new
python tools/check_triad_compat_ops.py --bash bash --report outputs/ops_check_new.json
python tools/check_triad_compat_gradients.py --report outputs/gradient_check_new.json
python tools/audit_triad_compat.py --report outputs/audit_new.json
```

所有输出目录/初始化文件应使用新的路径，避免覆盖先前证据。失败日志保留在本地 outputs：第一次诊断日志的 half 梯度求和溢出已改为 FP32 归约；这不是模型 tensor 非有限。后续7模型完整验证通过，没有放宽零初始化相等阈值。
