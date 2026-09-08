# SCCA-AIFI：C24 / C25

> 本文保留最初 C24/C25 交付记录。C25 已按用户要求进入独立组合实验交付，当前入口、完整结果包和验证记录见 [C25 文档](../c25/README.md)；以下“待选”等表述属于历史阶段。

SCCAAIFI（Spatially Conditioned Channel Attention AIFI）在第 9 层替换原 AIFI。
C24 为 C2＋SCCA，先由用户独立训练；C25 为 C17 原 CSCEFv51＋SCCA，仅交付待选组合。
没有正式训练成绩，不宣称稳定涨点、删除背景或恢复已经丢失的像素细节。

## 来源与范围

- 仓库 `supershaojie/concrete-crack-rtdetr`，新分支 `codex/scca-aifi`。
- 独立 worktree：`D:/MyProjects/Crack_RTDETR/outputs/worktrees/scca`。
- 分支基于已提交 ACR `ed366ebdb514a83a5504243213a1580bf74a1a48`，其父提交即 C17 `0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139`。
- C2 来源 `67c3078e54a657fd96d65fee657a75fbb1dae0d6`。C17 的 cscef_v5.py、cscef_v51.py 和 YAML 原封保留。
- 已读实验记录、C17/ACR/SALA 文档、全部本地分支/worktree、远端分支及本地结果目录；C21 为 SALA，C22/C23 为 ACR，未发现 C24/C25 占用。
- 祖先和仓库中未找到适用 AGENTS.md；`.agents` 没有文件。其他分支、未提交 ACR 后续修改、SALA、CBR 和历史结果保持原处；新模型不启用这些候选。
- 本机服务器相对位置的 C2 args 不存在，使用真实归档 `D:/rtdetr跑结果/c2 200e在线/c2_rtdetr_r18_lite_e200_b16_onlineaug_20260830_215053/train_run/args.yaml`，原始 SHA256 `ab0594ac3758b53421dc0a5adbea693505fc9e2591d8e668a25ba950d70234fd`。

设计动机与实现来源有明确边界：

1. [RT-DETR](https://arxiv.org/abs/2304.08069) 的混合编码器将尺度内交互与跨尺度融合分开。SCCA 保持 AIFI 位置，CSCEF 保持原融合位置。
2. [XCiT](https://arxiv.org/abs/2106.09681) 提供沿通道建立关系的思想。这里依用户规定使用空间条件 Q、内容 K/V、中心化、归一化和有界温度，不是完整 XCiT 复现。
3. [DAT，ICCV 2023 图像超分辨率](https://arxiv.org/abs/2308.03364) 提供空间与通道信息交互的动机，其超分辨率实验不能证明裂缝检测收益。
4. 实际读取 `D:/7.1 rtdetr改/魔鬼面具_RTDETR/RTDETR-main/ultralytics/nn/extra_modules/CAFM.py` 的 CAFM_Fusion。文件 SHA256 `74377b3d2eb98b9902fb19dbfe6e7990dfd40df459f96910b6d5518ce8504158`。参考压缩包 `D:/7.1 rtdetr改/魔鬼面具_RTDETR/RTDETR-20260623.zip`，实测 SHA256 与要求一致：`b91dbaf5f74e35ff23079fe0854140e7fd82207c21455b564abc594636aba2cc`。CAFM 两路池化描述相乘构造通道矩阵并再处理空间权重；SCCA 使用一次空间 MHA 输出生成 Q。没有复制整个包、依赖 einops 或增加 CUDA 扩展。

三篇论文的 arXiv 摘要/元信息于本次任务核对；具体算法遵循用户完整公式，未声称复现论文全模型。

## 计算与位置

可编辑结构图见 [architecture.mmd](architecture.mmd)，可在 Mermaid 编辑器修改节点和连线。图展示默认 post-norm，继承的 AIFI.forward 保持原 flatten、token、位置编码 W/H 顺序和恢复形状。

输入 `F[B,C,H,W] → X[B,N,C]`，`N=HW`。当前 640 输入为 `[B,256,20,20]`。
空间 MHA 为 `S=MHA(X+P,X+P,X)`，沿用 256 宽、8 头、FFN 1024、GELU、原 dropout。S 是最终输出投影后的值，尚未 dropout1/残差相加；只计算一次。

新增 `Xn=LN_C(X)`、`Sn=LN_C(S)` 均为 `eps=1e-5, elementwise_affine=False`。
`Q=Linear_Q(Sn), K=Linear_K(Xn), V=Linear_V(Xn)` 为无 bias 的 `C→64`，重排到 `[B,4,16,N]`。

仅 Q/K 沿 N 去均值：`Q0=Q-mean_N(Q), K0=K-mean_N(K)`。
`Qhat=Q0/max(norm_2,N(Q0),1e-6)`，K 同理，保留广播维度；V 不中心化/归一化。
每头 `tau=exp(ln(4)*tanh(t))`，t 初始 0，数学范围 `(1/4,4)`，浮点饱和时可能达到边界。
`Ac=softmax_last(tau*(Qhat @ Khat.transpose(-2,-1)))`，严格 `[B,4,16,16]`，不额外除 sqrt(N)、sqrt(d) 或 N−1。
`Uc=Ac@V`，逆重排 `U[B,N,64]`，`D=Linear_O(U)[B,N,C]`。相关性是每头潜在通道间关系；去均值不等价于检测背景。

post-norm：`Y=norm1(X+dropout1(S)+D)`；`T=fc2(dropout(act(fc1(Y))))`；`Z=norm2(Y+dropout2(T))`。
pre-norm：`Xp=norm1(X)`；空间 MHA 和通道计算的输入改为 `(Xp,S)`；`Y=X+dropout1(S)+D`；`Z=Y+dropout2(fc2(dropout(act(fc1(norm2(Y))))))`。
没有额外 gate、dropout、局部采样或卷积。

Q/K 的中心化、范数、温度、矩阵、softmax、Ac@V 都在显式关闭 autocast 的 FP32 区域中；四个线性投影保留正常 AMP，U 转成 O 权重 dtype 后再投影，支持实际 model.half()。模型计算不 detach、不缓存激活、不重复 MHA、不统计分位数。N=1/常量输入得到有限均匀注意力。

新增参数 `3*256*64+64*256+4=65,540`。Q/K/V Xavier uniform，O 全零；构造新增线性层时隔离恢复 CPU RNG，不影响后续公共层初始化。O 为零时仍执行新增计算：首次 O 获得梯度，后续 Q/K/V/温度才能获得非零梯度。AdamW 的权重衰减可能在任务梯度为零时也改变 Q/K/V，这不代表任务梯度已生效。

| 配置 | nc=1 参数 | nc=80 参数 | 原 Decoder 输入 |
|---|---:|---:|---|
| C2 原基线 | 20,082,772 | 20,184,208 | `[19,22,25]` |
| C24 SCCA | 20,148,312 | 20,249,748 | `[19,22,25]` |
| C17 CSCEFv51 | 20,109,684 | 20,211,120 | `[20,23,26]` |
| C25 CSCEFv51＋SCCA | 20,175,224 | 20,276,660 | `[20,23,26]` |

两份新 YAML 分别来自 C2/C17，仅替换原第9层 `AIFI` 为 `SCCAAIFI`，保留 `[1024,8]`，不插入 YAML 层。
C25 的 CSCEF 在 layer18，输入 `[17,16]`，Concat `[16,18]`，hidden=32/group=8/eps=1e-6，Scharr buffer 和每图空间平均置信机制、零输出初始化不变。SCCA 影响下游 CSCEF，互补只是待验证假设。

## 统一初始化与配方

唯一源权重：`weights/rtdetr_r18_lite_imagenet_backbone_init.pt`，SHA256 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
`init_scca.py` 校验 SHA、无训练状态、C2 完整 533 个状态和所有形状；C25 在原 C17 插层偏移下映射，额外保留 5 个 CSCEF 参数张量和 2 个 Scharr buffer。最后完整 strict load 并逐键比值，无意外缺失、额外项和 shape mismatch。

SCCA 新状态仅 5 个：scca_q/k/v/o.weight、scca_temperature_raw；两个无仿射 LN 无 state。公共 `ma/fc1/fc2/norm1/norm2` 名称保留。
nc80→nc1 原生 train API 重建时，C24 精确加载529/538，C25 精确加载536/545；各9个分类相关状态因 nc 改变重新初始化（denoising embed、enc score weight/bias、3组 dec score weight/bias）。工具逐一核对这些跳过项，拒绝其他缺失，不以 strict=False 无报错认定正确。
正式初始化来自新建模型，绝不从 smoke 模型、optimizer、EMA 或 C17 best 复用。

[c2_args.yaml](c2_args.yaml) 是109字段真实归档副本；服务器必须读取主仓库原 C2 args 并核对全部字段及类型。
生成的新旧全字段差异分别在 `c24_parameter_diff.json`、`c25_parameter_diff.json`，只改变 model/name/save_dir；project/data/增强/AMP/200e/640/batch16/AdamW/.0005/seed42 均不变。

## 评估、诊断与交付

独立评估默认完整 val，主指标 mAP50–95，辅以 AP75、Precision、Recall、参数和实测耗时。
公共原 RTDETRValidator 存在“排序后使用排序前置信度 mask”的问题。本任务保留原公共代码及训练 best 选择，独立 `scca_results.py` 使用 `corrected_sorted_conf_mask_v1`，记录受影响图数。C2、C17 对照必须同入口同设置重评，不能直接与旧历史指标相减。结构和选择固定后手动 test；test 检查同权重、数据和设置的已完成 val。

按需 diagnose 使用固定排序前4张 val 图、batch1，记录准确 RMS(D)/(RMS(X)+1e-8)、RMS(D)/(RMS(S)+1e-8)、四头温度/熵和 S 空间变化 RMS。分母<1e-6单独标记。统计在额外进程中用临时 hook，标量归约、只转 CPU 小通道矩阵，不缓存 GPU 激活或计算图、不使用随机抽样、不改 forward 返回。额外投影仅在这4图诊断启用，正常训练无统计。

pack 收集源码快照和差异、提交信息、YAML、完整参数差异、初始化映射、实际 args、日志、results.csv/曲线、val/test 指标和 PR/混淆矩阵以及有限诊断。未执行项目用 NOT_RUN 标明。只允许指定结果曲线 PNG；排除权重、图片、预测图片和大张量。压缩包≤20MiB，超限拒绝写包，附 SHA256 和逐文件清单；新时间戳文件名防覆盖。

完整操作见 [AUTODL.md](AUTODL.md)，实际验证见 [VALIDATION.md](VALIDATION.md)。没有任何200轮训练由本任务自动启动。
