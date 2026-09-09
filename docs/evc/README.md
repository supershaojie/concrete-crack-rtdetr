# EVC-Deform v1：C2 单模块实验

交付分支 `codex/evc-deform`；稳定别名 `evc_deform`。本次没有启动正式训练或完整数据集 val/test。`full_server_preflight=NOT_RUN`，4090 / PyTorch 2.1.2 环境未实测，延迟未测，不以参数少推断低延迟或精度提升。

本地独立 worktree：`D:/MyProjects/Crack_RTDETR/outputs/worktrees/evc`。Git 工具基础为 `beedcfa307e250fb2de47587097c51f9c141123b`，目的是复用已有直接启动与同次评估导出工具；**模型结构从纯 C2 `rtdetr-resnet18-lite.yaml` 构造**，没有加载 C26、C19、C24 或其他训练好的 best。旧候选源码因仓库历史仍存在，但不在本实验模型中启用。

仓库工具接受稳定别名，不要求数字 C 编号，因此本轮不占用 C27/C28，避免与并行 DRA 分配冲突。检查过本地分支、工作树与结果目录，现有编号到 C26。原工作区未跟踪 bundle/图稿保持原状；未找到适用的 AGENTS.md。

## 结构与公式

新类：`ultralytics-main/ultralytics/nn/modules/evc_deform.py:EVCMSDeformAttn`。
最小解析入口：`RTDETRDecoderEVC`，继承完整原生 RTDETRDecoder，仅在公共初始化结束后替换 `decoder.layers.2.cross_attn`。
新 YAML：`ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-evc-deform.yaml`。

核对实际 C2：3 层 decoder、256 通道、8 heads、3 levels、每头每尺度 4 points；常规 query 300，DN 动态前缀保持原义。前两层仍是原 MSDeformAttn，self-attention、AIFI、骨干、FFN、norm、box/class heads、DN、Hungarian 匹配与 loss 均沿用 C2。C17/CSCEF、C19/CBR、C24/SCCA、ACR、DRA、RCA 均不启用。不增加采样点、二次采样、特征增强分支或修框头。

原交叉注意力的权重 logit 为 query 的线性预测。query 本身可以间接包含先前图像信息，本研究增加的是对**本次采样内容**的显式证据校验。参考框宽高缩放原本就存在，不作为创新。

沿用原 offset、reference box、padding mask 得到坐标和投影 value。独立 helper `sample_deformable_values` 使用原有 PyTorch `grid_sample`：坐标从 `[0,1]` 转为 `[-1,1]`，xy 顺序，`bilinear / padding_mode=zeros / align_corners=False`。每尺度一次采样，共 3 次，得到

```text
V = sampled_value: [B,Q,H,S,K,D]，当前 D=32。
q_h: 传入 cross_attn 的 query 按头拆分，包含原生位置编码。
Uq_h, Uv_h: 各头独立的无 bias 32→8 线性映射，尺度/点共享该头参数。
e_sk = dot(normalize(Uq_h(q_h)), normalize(Uv_h(V_hsk)))
```

两个投影均为非零 Xavier 初始化。评分、L2 normalize（epsilon=1e-6）、logsumexp 和权重 softmax 在关闭 autocast 的 FP32 区域计算。零向量给出有限的零证据；不增加有效点筛选、不另加越界权重规则、不 detach query/value/offset。原 helper 的代码及返回类型保持不变。

省略 batch/query/head 下标，`z_sk` 是原始 attention logit：

```text
p0_sk = softmax_k(z_sk)
u_s = logsumexp_k(z_sk)
e_mean_s = mean_k(e_sk)
beta_p_h = tanh(a_p_h)       a_p_h 初始化 0
beta_s_h = tanh(a_s_h)       a_s_h 初始化 0
p_sk = softmax_k(z_sk + beta_p_h * (e_sk - e_mean_s))
t_s = sum_k(p0_sk * e_sk)    必须使用原 p0
pi_s = softmax_s(u_s + beta_s_h * t_s)
w_sk = pi_s * p_sk
o_h = sum_s,k(w_sk * V_hsk)
```

权重最后转回采样结果 dtype，按旧 helper 的维度及归约顺序合并 heads，再进入原 output_proj。最终 value 始终是原 32 维采样值，不是归一化后的 8 维评分向量。正式 forward 不存大张量、quantile 或逐批诊断状态。

当两个系数为零，`exp(logsumexp_k z_s) / sum_s exp(logsumexp_k z_s)` 乘以 `exp(z_sk) / sum_k exp(z_sk)`，等于对全部 `S×K` 的统一 softmax。固定本次 query/z 时，只调整点内校验不会因归一化改变尺度份额；端到端训练仍会改变 query/z。同尺度减去均值不改变 softmax，是相对证据的表达方式。校验系数绝对值上限为 1；不偏置 P3，不加 entropy/diversity/contrastive loss。普通 QK 内容注意力、softmax 分解和中心化均不宣称为新发明；证据分数不是校准后的真实正确率，涨点尚未得到实验支持。

## 公共初始化与配方

权威公共来源：`weights/rtdetr_r18_lite_imagenet_backbone_init.pt`，SHA256：

```text
fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e
```

该 C2 初始化来自项目既有 torchvision ResNet18 ImageNet backbone 适配流程，不是训练完成的裂缝模型。`init_evc.py` 校验 SHA 和无训练状态，严格检查 C2 键/shape，identity 映射全部 533 个公共 state；只新增 4 个 state。随后按固定 seed42 的原生 `RTDETRTrainer.get_model` 做 nc=80→1 类别适配，记录 9 项分类 shape 跳过（DN 类别 embedding、encoder score、3 个 decoder score 的 weight/bias）。适配后公共状态与独立 C2 nc=1 重建逐项一致。保存的是 **nc=1、FP32、无 optimizer/EMA/训练状态** 的新初始化；训练 API 再构造 nc=1 时 537 个 state 全部精确加载。

`mapping.json` 含全部公共 source→target 映射、shape、equal、missing/unexpected 和分类适配报告。新增参数为最后一层的 `evc_q[8,8,32]`、`evc_v[8,8,32]`、`evc_a_p[8]`、`evc_a_s[8]`，共 **4,112**。实际未融合 nc=1 参数量：C2 **20,082,772**；EVC **20,086,884**。独立评估会执行框架原生融合，日志中融合后参数量较小，不与这里的未融合口径混用。

实际权威 args 路径：`D:/rtdetr跑结果/c2 200e在线/c2_rtdetr_r18_lite_e200_b16_onlineaug_20260830_215053/train_run/args.yaml`。归档为 `c2_args.yaml`，逐字段/类型核对共 109 项：200 epochs、640、batch16、seed42、AdamW、lr0=0.0005、lrf=0.01、weight_decay=0.0001、warmup5、cos_lr=True、AMP=True、close_mosaic10；其余增强、workers8、deterministic、patience50、nbs64、loss 均保留。

默认服务器路径下仅 model/name/save_dir 改为本实验身份。本地 `parameter_diff.json` 中另有 data/project 的 Windows 路径表示差异；`server_parameter_diff.json` 用 POSIX 路径构造验证默认服务器只有上述 3 个差异。显式 `EVC_MAIN` 覆盖仅允许 data/project/save_dir 路径迁移，仍须使用同一数据配置内容和 109 项 C2 配方。启动后再次核对实际 trainer args。原 AdamW 对所有公共/新参数各收录一次，分组沿用原逻辑（新增 4 项均属原 weight decay 组）。小样本更新/保存的文件均为临时验证文件，已清理，绝不作为正式初始化。

## 验证记录

- `report.json` / `validation.console.log`：Windows、Python 3.9.25、PyTorch 2.7.1+cu118、RTX2060。
- 零系数分解最大误差 `2.98e-7`；点内更新的尺度份额误差 `2.38e-7`；权重非负且和为 1。
- 真正 grid_sample 路径：CPU/CUDA、2D/4D reference、padding、越界、Q=17；新 helper 每次只调用 3 次 grid_sample。最大输出误差 `3.88e-7`，公共梯度误差 `3.82e-6`，输入/reference 梯度误差 `1.53e-5`。梯度容差 atol=5e-5、rtol=3e-4，前向 atol=2e-6、rtol=2e-5。
- 固定 query/坐标，改变单个 sampled_value 后证据改变约 0.585、权重改变约 0.00244；query-only 原权重不变。额外验证尺度项严格使用 p0；零向量/大 logit 有限。
- 公共源初始化的整网 CPU FP32，输入 `[1,3,640,640]`，零系数输出最大误差 0。
- CPU FP32、CUDA FP32、CUDA AMP 分别执行 3 次原生 loss/DN/Hungarian 更新（160 输入，batch 2/1/2）；300 常规 query 与动态 DN 保留，梯度有限。系数首步非零梯度，Uq/Uv 首步梯度为 0、后续非零，验证了模块能够开启；不要求随机 loss 单调。
- 实际 CUDA half 整网前向返回有限的 `[1,300,5]` float16；更新后保存重载状态/输出一致；公共 RTDETR.train 重建验证在训练 setup 前主动结束。
- 各 2 张**合成图**经过独立 val/test 的真实入口，使用相同 checkpoint SHA、640/FP32、全部 300 预测和全部 GT 同次导出；此为链路验证，不是精度结果。测试临时文件已清理，报告保留路径用于追溯，不能误认为正式实验文件。
- `operations_report.json` / `operations.console.log`：显式 MOCKED tmux/worker 控制测试；无 audit marker 可投递、重复预留拒绝、模拟 OOM 写 exit_code=1、状态分类与 PID 复用保护；真实文件 IO 验证 >20 MiB 包、SHA、inventory、回读，以及不完整证据标记和拒绝覆盖。
- 原 transformer.py、utils.py、head.py 未修改，其规范化 LF SHA 在 `source_trace.json`；仅增加新类和解析注册。

未验证：AutoDL 4090 / PyTorch 2.1.2、真实 tmux 投递及 200e 训练、完整数据集 val/test、服务器 batch16 显存、CPU half、部署导出、延迟。必要验证不作为 start-direct 的门槛；没有伪造服务器 passed 标记。

## 来源与后续路线

模块包实际路径：`D:/7.1 rtdetr改/魔鬼面具_RTDETR/RTDETR-20260623.zip`，SHA256 与附件指定的 `b91dbaf5f74e35ff23079fe0854140e7fd82207c21455b564abc594636aba2cc` 一致。已完整读取并核对压缩包内/解压后的 `RTDETR-main/ultralytics/nn/extra_modules/SFS_MSDeformAttn/ops/modules/ms_deform_attn.py:MSDeformAttn_for_sfs` 内容一致。借鉴接口、head/level/point/value 组织与几何对照；没有引入其 MSDeformAttnFunction、自定义 CUDA、第二次 key 采样或整包依赖。完整追溯见 `source_trace.json`。

项目后续路线仍是：两个精度模块各自及组合有效后，再设计真正减参的骨干轻量卷积。当前只记录，第三阶段未实施。

服务器操作见 [AUTODL.md](AUTODL.md)。本次最终完整 commit SHA 与普通 push 结果以交付消息为准；验证时 HEAD 是工具基础提交，测试针对当时 worktree 新实现，未把旧 HEAD 伪称为最终提交。
