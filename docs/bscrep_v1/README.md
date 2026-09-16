# BSC-Rep v1（双侧上下文支撑 RepC3）

主实验 `cbr_lif_bscrep_v1` = 原 R18-Lite + original CBR + 原 LIF-Down + BSC。
消融 `bscrep_v1` = 原 R18-Lite + BSC。正式训练 **NOT_STARTED**，最终 test **NOT_RUN**。

基点为 `a0459d6a652cb702699087c88fa39a3e4c4087ec`，已由成功组合的实际交付记录、退出码 0、200 轮 results.csv 和 test runtime 核对。独立分支为 `exp-rtdetr-r18-lite-bscrep-v1`，工作树在主目录的 `outputs/worktrees/Crack_RTDETR-bscrep-v1`。当前其他分支和未提交文件保持原样。来源与原文件逐项哈希见 `base_audit.json`；完整父配方是 `parent_args.yaml`（109 字段），另保留完整权威 `c2_args.yaml`。

## 实现与机制边界

仅 layer19 的 `[-1,3,RepC3,[256,0.5]]` 变为 `BSCRepC3`。层号、from、Concat、PAN 和 Decoder from=[19,22,25] 保留。640 输入的该层为 512→256、stride8，隐藏 128 通道，内部仍有 3 个原 RepConv，BSC 分支只有一个。单模块 layer20 是原 Conv、layer26 是原 RTDETRDecoder。

`BSCRepC3` 继承 RepC3，保留 cv1/cv2/m/cv3 的参数名称和原运算。令 `T=m(cv1(X))`，输出为 `cv3((T+bsc(T))+cv2(X))`。分支将 T 投影到 32 通道，对 `(0,1),(1,0),(1,1),(1,-1)` 各取同方向正负两侧距离 1、2、3 的均值。replicate padding=3 后显式切片，不使用 roll、图外零值或逐像素循环。共享门控读入 `[Z,M,abs(C_plus-C_minus)]`，每方向为 `sigmoid(gate(...))*(M-Z)`，四方向平均后投影回 128 通道。

新增参数精确为 `128*32 + 96*32 + 32 + 32*128 = 11,296`。没有新增归一化、额外激活、总门控、损失或外部依赖。四次共享 gate 由普通 Conv2d hook 分别计数；half 权重推理才用可微 FP32 权重转换。整个新分支局部 FP32，delta 转回 T.dtype；forward 不修改 Parameter 对象或重置已学习权重。

门控学习判断两端上下文是否适合补充中心，没有单调性约束，不能解释成“两端越一致门控必然越大”。这不是裂缝分割、拓扑重建或强制连接断裂。原 CBR 采样候选框内外边界，BSC 则在 Decoder 前处理 P3 特征网格。残差和局部差异公式本身不作为原创贡献；待验证的候选机制是同方向两侧几何关系参与 P3 上下文判断。不保证涨点，也不声称已证明学术首创。

指定参考包 SHA256 已匹配并实际阅读 RepC3、DilatedReparamBlock、UniRepLKNetBlock、CFBlock 条带上下文；仅比较接口/机制，未复制或导入 extra_modules。详见 `reference_audit.json`。

## 初值、训练和恢复

唯一源权重是 `rtdetr_r18_lite_imagenet_backbone_init.pt`，SHA256 为 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。通过原组合初始化器构造父模型，再 strict 迁移所有公共参数和 BN buffers。BSC 构造隔离 CPU RNG；in_proj/gate.weight Xavier uniform，gate.bias=0，仅 out_proj.weight=0。两配置 BSC 初值逐值相同。不使用训练完成的父权重或额外因子文件。

实际 RTDETRTrainer.get_model 按 nc=1 重建并调用原生 model.load。包装器核对完整键集、逐值相等和全部 9 项类别 shape 适配（denoising embedding、encoder 分类层、三个 decoder 分类层的 weight/bias），未让 strict=False 掩盖遗漏。`initialization_checks.json` 列出每项 shape。新增参数在原生 AdamW 构建前存在，所有可训练参数恰好进入一次。

`tools/train_bscrep_v1.py plan` 不训练、不创建正式 run。`start` 要求与当前 SHA、源码、YAML、源权重、初值、数据相符的服务器 PASS 预检，并拒绝已有 run。`resume` 只接受本实验 last.pt，保留已学习权重和原生 optimizer/scaler/EMA/epoch，不调用 initialize。运行记录区分完整 200 轮、早停、中断和异常；保留原 OOM 防护，不自动降低 B16 或产生 name2/name3。

完整 109 字段逐字段继承成功组合配方；只有模型身份、实验名称/输出位置、已核对环境路径变化。核心为 E200/P50/B16/640、seed42、原 AdamW/lr0=0.0005、原在线增强、原 native AMP。数据按成功归档的三划分路径清单与标签 SHA 核对，不重新划分。

## 验证与算量

`tools/check_bscrep_v1_math.py` 负责方向坐标、正反半径、replicate、小尺寸/矩形、朴素公式对照、两端交换、常量输入、零输出等价、首步 out_proj 梯度、非零诊断上游梯度、RNG、half 参数和标准 THOP。`tools/preflight_bscrep_v1.py` 负责真实 Trainer/优化器、保存/重载/EMA/resume、原 loss/反向及整网融合。

诊断使用隔离副本：为验证原生 AdamW 和恢复会执行有限诊断更新，正式初值文件和正式优化器更新次数均为 0。AMP 使用原生 GradScaler 默认初始 scale 和有限 backoff，逐次记录 unscale 后梯度；持续非有限会对同一批数据用父模型定位。CPU 不启用 float16 autocast。严格 FP32 融合关闭 TF32 后恢复设置，使用父项目候选索引对齐/固定索引重放协议及原容差；LIF 的共同 BN 保留。

同一次统计、nc=1、B1/640，以下是 THOP 已覆盖算子的 2 FLOPs/MAC 结果，不能据此推断速度。functional attention/grid_sample、逐元素运算、sigmoid/abs、padding/slicing/搬运等未完整计入，详见 `cost.json`。

| 模型 | 未融合参数 | 未融合 GFLOPs | 融合参数 | 融合 GFLOPs |
|---|---:|---:|---:|---:|
| 原基线 | 20,082,772 | 58.276608 | 19,877,716 | 57.165773 |
| 基线+BSC | 20,094,068 | 58.538752 | 19,889,012 | 57.427917 |
| 原 CBR+LIF | 20,149,765 | 58.672512 | 19,944,965 | 57.564954 |
| CBR+LIF+BSC | 20,161,061 | 58.934656 | 19,956,261 | 57.827098 |

BSC 卷积主项额外 `131,072,000 MAC = 0.262144 GFLOPs`。主模型不承担减少 10% 参数的目标。

本机 RTX2060/torch2.7.1+cu118 的小规模诊断不替代服务器：真实在线增强 B16/640/AMP、实际服务器 AMP 资源、峰值显存和 GT/DN 门禁必须通过 `--server --data ...` 执行。没有登录/同步服务器、修改正在运行的实验、启动正式训练或最终 test。最新实际结果和 PENDING 项见 `validation.json`。

## 后续入口与交接

- `tools/init_bscrep_v1.py`：固定源权重生成独占初值和审计。
- `tools/preflight_bscrep_v1.py`：有限预检，新输出目录；`--server` 必须全部通过才返回 PASS。
- `tools/train_bscrep_v1.py plan|start|resume`：默认主组合。
- `tools/test_bscrep_v1.py val|test`：固定训练所选 best/EMA，原 FP32/640/B16/conf=.001/iou=.7/max_det=300 和父修正置信度掩码。test 先核对同权重 val 记录。
- `tools/pack_bscrep_v1_light.py`：args/results/metrics/曲线/混淆矩阵/源码/审计、逐文件 SHA manifest。默认排除数据、权重、参考包和巨量预测。
- `tools/sync_bscrep_v1.sh`、`tools/start_bscrep_v1_tmux.sh`：独立服务器工作树和单独显式启动。

最终 40 位交付 SHA 不写入本提交，避免自引用。提交后生成的 `outputs/bscrep_v1_delivery/DELIVERY.md` 包含完整 SHA、远端核验、可复制的从新终端顺序执行命令及小型 bundle 导入方案；通用说明见 `AUTODL.md`。
