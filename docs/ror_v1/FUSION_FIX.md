# ROR 融合失败：检查修复与有界母版对照

本次起点：`378c6772ac7d884c3909b4a61e266e4d5d49f066`。母版固定为 `a0459d6a652cb702699087c88fa39a3e4c4087ec`。

## 已确认的问题与修复范围

服务器提供的失败发生在 `lif_input`、FP32、pre-selection，max_abs≈9.76156e-5，30051 个元素超出原 `atol=2e-5 / rtol=2e-4`。这个位置在 top-k 之前，候选排序不能解释它。

源码检查：比较的是 `model.float().eval()` 与它的 `deepcopy(model).fuse()`，两者来自同一份完成 optimizer/空与单 GT 测试后的权重和 BN 状态。原 optimizer 测试确实更新了模型；这不是“其中一方被污染”。resume 使用另一份 rebuilt 模型。原 `BaseModel.fuse` 对 LIFDown 的排除条件、LIF BN 和残差路径全部保留。ROR 在正常 eval 中不执行损失包装路径。

已修复两项检查问题：

- 融合抛异常时 `integration()` 无返回值，`run()` 与 `preflight()` 丢失先前已完成的检查摘要。现在逐阶段持久化，外层共享报告；失败保留阶段、结构化异常、traceback、融合报告路径、梯度和更新的逐参数证据。CUDA 更新报告先于后续检查保存，解析式检查失败也不会丢掉直接更新结果。
- 原 RepConv 融合新建的 Conv2d 带默认 `training=True`。检查端在原生 fuse 后再次 `eval()`，并核验所有子模块。Conv2d 无 train/eval 数学分支，**不能把该标志问题说成服务器数值误差的已证实根因**。

增加了 state/BN 内容哈希、拷贝前后精确一致、存储不共享、原对象未被 fuse 改写、LIF 状态保留、FP32、输入和后端核验。resume 前后另核对原对象 state 哈希。既有融合协议只增加可选的 JSON-only 失败输出；原母版调用默认行为、容差、候选审查与接受规则不变。

没有改生产模型、ROR 公式、初始化、LIF/CBR、训练 Trainer、正式超参数或 start 门禁。没有将 native 融合失败自动改成通过。

## 实际验证

本地 Python3.9.25 / torch2.7.1+cu118 / RTX2060，与服务器 Python3.10.13 / torch2.1.2+cu121 / RTX4090 不同。

| 受影响检查 | 本地结果 |
| --- | --- |
| 同输入、同初始 state 的母版/ROR pre-selection 对照 | PASS；两者 unfused/fused 分别逐位一致 |
| native 融合 `lif_input` | 两者 max_abs=9.685754776000977e-8；失败元素 0/204800 |
| 误差分布 | mean_abs=8.7152529e-9，p99=3.2596290e-8，最大容差占比≈0.00269 |
| TF32 隔离 | 前后相同结果；后端完整恢复；不能外推至 RTX4090 |
| 状态隔离和 LIF BN 保护 | PASS；未改变 unfused 对象；全部模块 eval |
| CUDA 独立一步更新复测 | **REVIEW_REQUIRED**，4 个 state 超出原界限 |
| 异常与清理回归 | 5 项通过：内层/外层报告保留、后端异常恢复、原融合隔离与 JSON-only 输出、失败 fixture 自动选择与 SHA 核验 |
| 编译、诊断 CLI | PASS |
| 服务器同现场复现、正式训练 | **NOT_RUN** |

本地融合用的是统一源生成的受控初始状态，不是服务器失败时经过 optimizer/BN 更新的状态。母版类直接执行固定 SHA 的原 tasks.py；共享生产模块要求与母版无差异，eval 不调用当前训练 criterion。证据见 `fusion_fix/local_cuda_fusion.json`。

独立 CUDA 复测：ROR/母版梯度最大差 0.000244140625；母版重复反向最大差 0.0002994537353515625，逐参数梯度界限通过。直接更新最大差 0.0002355724573135376；4 个 state 的直接更新超出各自 `max(2e-7,4×母版重复更新差)`。例如 `model.14.m.1.conv1.conv.weight` 差 6.2547624e-5，界限 4.7593145e-5。每套实际梯度对应的首步 AdamW 解析式通过。原 CUDA grid-sampler 反向发出非确定性警告；Adam 首步 `g/(abs(g)+eps)` 对近零梯度敏感。这些证据支持存在原生反向波动，**尚不足以宣称直接更新一致性通过或所有超界均已解释**。其门禁继续为 REVIEW_REQUIRED。见 `fusion_fix/local_cuda_update.json` 和两个逐参数 variation JSON。

## 服务器只执行一次有界诊断

先用交付消息中的完整 SHA 更新同一分支（普通 fast-forward；保留主工作区 HEAD）：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-ror_v1/tools/sync_ror_v1.sh <交付完整SHA>
```

然后执行下列一个诊断命令，**无需重跑 prepare 或完整 preflight**：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-ror_v1
PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false timeout 900 \
  /root/miniconda3/envs/rtdetr/bin/python tools/ror_v1_fusion_diagnostic.py \
  --fixture auto --device cuda:0 --check-update \
  --update-source /root/autodl-tmp/projects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt
```

`--fixture auto` 选择上次检查已保存的、同设备 FP32 pre-selection 失败的最新 fixture，并核验原 JSON 记录的 SHA。恢复精确 unfused 权重/BN、B2/160 输入和 RNG，分别运行同初始状态的原母版和 ROR。不会重新保存巨大权重。也可用 `--fixture /完整路径/fixture.pt` 精确指定。没有 fixture 时直接报错，不偷偷改用不同输入；`--source` 仅是显式的初始化对照，不能冒充失败现场。

固定工作量：两种后端设置，每种母版/ROR各做 unfused/fused 一次前向，共 8 次 B2/160 前向；随后独立跑一次原零权重 optimizer 对照（两套更新及一次母版重复反向），不跑 AMP/capacity/resume/数据评估或 epoch 循环。外部 timeout 上限 900 秒。

输出到新的 `outputs/ror_v1/fusion_control_<UTC>/diagnostic.json`，独立更新的逐参数 JSON 在该目录的 `update/` 下。保留每个 pre-selection key 的失败数、最大绝对/相对误差、均值/RMS/分位数、最大容差占比及最大失败样例，同时记录 TF32、cuDNN 版本/开关、deterministic/warn-only、matmul precision、相关环境变量、源文件哈希及比较状态哈希。

判读依据：

1. 看 `saved_fused_state_reproduced` 和 `original_failure_reproduced`，确认是否复现原保存的 fused state 和同位置失败统计。
2. 看 native 下母版与 ROR 是否均失败，以及 `ror_vs_mother_exact` 的融合前/后对照。若两者不同，不能归因于共有平台差异。
3. 只有共同 native 误差在 TF32 隔离后消失，才报告该对照中的后端数值效应。隔离只临时关闭 matmul/cuDNN TF32，包含 fuse 本身，`finally` 恢复；不改变 benchmark/deterministic 配置。没有修改训练进程设置。环境强制覆盖也会记录，避免只看一个布尔值。
4. native 或独立更新仍失败时退出码为 2，完整 JSON 保留；表示 REVIEW_REQUIRED。查看具体子项，不把隔离通过、pre-selection 通过或解析式通过等同于完整融合/全预检通过。

本地无法确认 RTX4090 上母版本身是否重现 9.76e-5 差异。因此服务器数值根因仍待以上同现场对照，TF32/cuDNN 目前只是待检验假设；没有据此扩大容差。

## start 与已有 prepare 状态

本次没有调用 start，没有创建 tmux 会话。prepare/preflight 本身不启动 tmux；只有显式 start/resume 才创建 `ror-v1-training`。诊断命令只读查询服务器现有会话，把结果写到 `tmux` 字段，不会创建/停止它。

诊断不改 `plan.json`、旧预检或训练身份。原 prepare 的 code SHA 仍是旧版；原 start 的 SHA 和 PASS 门禁保持有效。此次交付只处理故障定位与相关检查，不把旧报告或新诊断改写成可启动长训的授权。服务器对照结论与独立 CUDA 更新问题解决后，才能另行处理正式预检身份。
