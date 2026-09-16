# TRC-AIFI v1 — 特征重复度校准 AIFI

主实验为 **原 RT-DETR-R18-Lite + original CBR + 原 LIF-Down + TRC-AIFI**，variant 为 `cbr_lif_trc_v1`。`trc_v1` 是原基线 + TRC 的后续单模块消融。两者都只把原 layer 9 的 `AIFI [1024, 8]` 替换成 `AIFI_TRC [1024, 8]`。主组合的 layer 8 是 S5 投影，layer 19 是原 P3 RepC3，layer 20 是原 LIFDown，layer 26 是原 RTDETRDecoderCBR、from 为 `[19,22,25]`；单模块配置不含 CBR/LIF。

分支为 `exp-rtdetr-r18-lite-trc-v1`，base 为 `a0459d6a652cb702699087c88fa39a3e4c4087ec`。最终完整交付 SHA 在提交外 `trc_v1_delivery.json` 的 `commit` 字段，提交内不写自己的未知 SHA。正式训练 **NOT_STARTED**；最终 test **NOT_RUN**。

## 真实来源

[PROVENANCE.json](PROVENANCE.json) 记录 Git 状态、路径、完整 SHA、原始文件散列和源码对应关系。已实际读取成功组合包：

`D:/rtdetr跑结果/C19＋LIF：原 CBR＋LIF-Down v1/c19_lif_v1_complete_20260913_115700_919970.tar.gz`

SHA256 为 `e188e53593f61c16204c172a03b86439af13dd0fb5025bdbbdc8a661217d5129`。包内 delivery/source_record/package 均指向上述 base；exit code 为 0，results.csv 有 200 轮，完整包记录同一 best 的 val/test。历史 training_state/launch_state 保留了 training/dispatched，成功判断来自退出码、完整结果和打包审计，不能单看旧状态字符串。核验了包内 14 份关键证据与已解压副本逐字节一致，12 份父源码与 base 的 LF 内容一致。

[cbr_lif_args.yaml](cbr_lif_args.yaml) 是成功组合实际训练的完整 109 字段原件，SHA256 `31029c8d3bcf0e2dc3cb7107fa7db316f37cd954eba21bf4fc854e2070fc476b`。[c2_args.yaml](c2_args.yaml) 是同包的原 C2 权威配方，SHA256 `ab0594ac3758b53421dc0a5adbea693505fc9e2591d8e668a25ba950d70234fd`。两者只在 model/name/save_dir 三字段不同。TRC plan 逐字段继承成功组合，生成差异表；允许变更仅模型身份、名称/输出目录和核验后的环境路径。保留 200e、patience 50、B16、640、AdamW、lr0 0.0005、AMP、seed 42 和全部在线增强。

实际参考包为 `D:/7.1 rtdetr改/魔鬼面具_RTDETR/RTDETR-20260623.zip`，SHA256 与用户给定值 `b91dbaf5f74e35ff23079fe0854140e7fd82207c21455b564abc594636aba2cc` 完全相同。直接读了 ZIP 内 `RTDETR-main/ultralytics/nn/modules/transformer.py` 的 TransformerEncoderLayer/AIFI，以及 `nn/extra_modules/transformer.py` 的 DynamicPosBias（816–850）、DPB_Attention（852–941）、TransformerEncoderLayer_DPB（943–975）。DPB 用相对坐标 MLP 产生每头位置偏置，在 softmax 前加到 logits；它及其包装类自建 QKV、Conv FFN 和 LN。TRC 仅借鉴加性 logits 偏置的接入思路，使用原 self.ma 的 attn_mask，不复制 DPB 或导入 extra_modules。

## 数学与状态

640 输入对应 `X=[B,400,256]`；X 是 AIFI flatten 后、加入位置编码之前的内容。固定 d=32、heads=8、tau=8、lambda_bound=0.5、r_clip=2。TRC 的非仿射 functional LN 使用 eps=1e-5，L2 eps=1e-6。

```text
U = layer_norm(X.float())
Z = normalize(Linear(256,32,bias=False)(U), dim=-1)
S = clamp(Z @ Z.transpose(-1,-2), -1, 1)
rho[j] = sum_k exp(8 * (S[j,k] - 1))
r = clamp(log(rho.clamp_min(1e-6)) - mean_token(log(rho.clamp_min(1e-6))), -2, 2)
lambda[b,h,i] = (0.5 * tanh(Linear(256,8,bias=True)(U))).transpose(1,2)
bias[b,h,i,j] = -lambda[b,h,i] * r[b,j]
attention = softmax(original_QK_logit + bias)
```

包含 self 项，不对角线修补，不额外加 1，不剪枝或采样。零描述归一化后仍为零，此时 rho 可小于 1；零或相同描述的中心化 r 为零。正 lambda 能减弱高重复内容的数量效应，负 lambda 允许增强其贡献。重复度不是背景概率，负偏置不是裂缝分割 mask，也不保证消除背景或保留所有裂缝。

`AIFI_TRC` 继承 AIFI，只新增 `self.trc`，公共的 ma QKV/out_proj、fc1/fc2、norm1/norm2 路径保留。沿用原 flatten、正弦位置编码、pre/post norm、dropout、激活、缩放和 MHA need_weights 默认设置。TRC 不对原 Q/K/V 做描述分支的 L2 归一化。bias 按 batch-major/head-minor reshape 成 `[B*8,N,N]`；bool mask 的 True 仍禁止关注，float mask 仍加性合并，padding mask 保留。

新增参数仅为 `256*32 + 256*8 + 8 = 10,248`。首次构造 descriptor 使用 Xavier uniform，coefficient 的 weight/bias 为零，初始偏置为零；构造随机数隔离避免改变后续公共层初值。原始源始终是 `rtdetr_r18_lite_imagenet_backbone_init.pt`，SHA256 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`，先源 nc=80、后原 Trainer 的 nc=1 适配。初始化报告与预检核对公共参数、BN buffers、明确的分类 shape 白名单及 AdamW 每参数恰好一次；不以宽泛 strict=False 掩盖遗漏。resume/EMA/load/fuse 不重新清零 coefficient。

TRC 全部数学在局部关闭 autocast 的 FP32 区间完成，通过可微 `.float()` 转换兼容 half 参数；不在 forward 改变模块参数对象。传入原 MHA 的 mask 适配有效 QKV dtype，其余模型保持原 native AMP。初始数学等价使用明确容差，不承诺不同 kernel 逐 bit 一致。

## 容量与验证边界

B1、N400 的主要新增矩阵计算为 `400*256*32 + 400*256*8 + 400*400*32 = 9,216,000 MAC`，按 2 FLOPs/MAC 是 **0.018432 GFLOPs**。此值不包含 LN、L2、exp/log、tanh、逐元素 bias/mask，不是完整实测算量。B16 的单个 `[16,8,400,400]` FP32 bias 约 **81.92 MB**（十进制），autograd 和 attention 还会增加实际峰值。参数少不代表无延迟/显存代价。预检 THOP 只在副本执行，显式统计函数式 TRC 投影/相似度，并分别给出父/候选、融合/未融合、同 nc=1/640 口径和未覆盖运算；不能把 THOP 当作真实推理速度。

本地已有 Python 3.9.25 / PyTorch 2.7.1+cu118 / RTX 2060 6 GiB；不安装升级依赖。具体本地结果以本任务的 [VALIDATION.md](VALIDATION.md) 和 [local_checks.json](local_checks.json) 为准，旧实验成功不能作为 TRC PASS。服务器历史环境是 Python 3.10.13 / PyTorch 2.1.2+cu121 / RTX 4090，交接时重新核对实际环境。**服务器真实数据、train 在线增强、B16/640/native AMP、原 loss/DN/反向及峰值显存必须服务器补检，当前为 PENDING。** 预检使用隔离副本，正式初值和正式优化器更新数保持 0；GradScaler 按原生 scale/unscale/backoff 记录，不固定 scale=1，不关闭 AMP 规避故障。

不逐 batch 保存 400×400 注意力图。需要诊断时仅查看 r 分布、lambda 正负比例/均值、bias 幅度及梯度汇总。没有保证涨点或首创结论。

## 入口

初始化、预检、训练、后续 test 和轻量包分别见 `tools/init_trc_v1.py`、`tools/preflight_trc_v1.py`、`tools/train_trc_v1.py`、`tools/test_trc_v1.py`、`tools/pack_trc_v1_light.py`。plan 不占正式 run 目录；start 绑定预检、源码、数据、初始化及参数，拒绝旧 run，不生成 name2/name3。resume 保留已学习 TRC。test 沿用原 best/EMA 与评估设置，输出 P/R/AP50/AP75/AP50:95、PR/混淆矩阵和速度。轻量包默认无权重、数据、参考模块包和逐图预测，含小型结果、必要源码及审计、manifest 与 SHA256。

完整服务器命令在 [AUTODL.md](AUTODL.md)。同步和预检不会登录服务器；本次没有执行服务器命令、正式 start 或 test。
