# LCD v1：原 LIF-Down＋原 CBR＋局部—上下文差分交互

这是固定候选的工程实现，没有启动正式训练，没有本实验的完整 val/test 或涨点结论。
分支 `exp-rtdetr-r18-lite-lcd-v1` 直接来自母版 `a0459d6a652cb702699087c88fa39a3e4c4087ec`。
本地 worktree 位于主仓库可写的 `outputs/worktrees/Crack_RTDETR-lcd_v1`；服务器按任务书使用同级 `Crack_RTDETR-lcd_v1`。

## 结构与来源

只把 YAML 第 5 层的 `Blocks` 改为稳定可导入的 `BlocksLCD`，完整原 `blocks` 计算之后执行唯一 LCD。公共参数路径不变；所有全局层号、from 和 save 不变。第 6 层和第 17 层同时读取增强输出 Y。640 输入的 S3 是 `[B,128,80,80]`，不是 Neck 第 19 层。

```
U = LN_channels(P(X))
A = conv2d(U, K, groups=32, dilation=1, padding=1)
B = conv2d(U, K, groups=32, dilation=2, padding=2)
T = tanh(G(B-A))
R = O(SiLU(A) * T)
Y = X + R
```

P:128→32、G:32→32、O:32→128，均 1×1、stride=1、bias=False；K 是唯一注册的 `[32,1,3,3]` Parameter。两次卷积都使用同一 K，第二次功能式卷积未 detach。LayerNorm 在每个 `(b,y,x)` 的 32 通道上计算，eps=1e-6，初始 affine=(1,0)。空间坐标不变，零 padding，d=1/d=2 分别取相邻 1/2 格。中心项只在 B−A 中抵消，边界常量输入的差分一般不为零。

只有 O 为零，P/K/G 沿用 PyTorch Conv2d 默认 Kaiming-uniform(a=√5)。新增 CPU 构造在 fork_rng 中使用 424002，不消耗公共随机序列。第一步内部梯度为零是预期；O 有效更新后再核对内部可学习性。首步全局裁剪范数可能改变，未要求整步参数更新等于母版。

[Mona 官方项目](https://github.com/Leiyi-HU/mona) 提供小适配分支的基础思想；[Rewrite the Stars](https://arxiv.org/abs/2403.19967) 提供乘性交互的基础思想。本实现按任务书单独实现共享核差分调制，没有串联两份完整模块，也没有整体复制模块包。材料目录只有执行文档，未找到指定迁移状态文档；这不影响源码定义核验。不能把 T 的有界性宣传成 R≤10%，也不能声称差分已经识别背景或保证裂缝保留、全球首创、必然涨点。

## 母版与初始化

LIF 第 20 层及 CBR 第 26 层源码保持原 LF 哈希；原 LIF 的 Conv-BN 融合排除、CBR 的 36 点/rho=.10/条件梯度/框修正均保留。decoder 仍为 3 层、300 query、输入 `[19,22,25]`。没有引入其他候选或额外 loss。loss、VFL、L1/GIoU、encoder/aux/DN、matcher 全部使用母版源码：class/bbox/giou=1/5/2，matcher=2/5/2。所有原路径正常联合反传，输入和联合梯度会因 LCD 而改变。

统一源 `weights/rtdetr_r18_lite_imagenet_backbone_init.pt` 的 SHA256 固定为 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。先运行未经放宽的母版 `controlled_models`/`build_training_model` 审计，再从同一随机状态构建 LCD，并用真实原生 Trainer 做 nc80→nc1 重建。最终初值为 nc1、epoch=-1，写 `outputs/lcd_v1/lcd_v1_controlled_init.pt`，不会覆盖母版初值。

552 项母版公共状态精确相同；只有原 9 项分类 keys 发生 nc80→nc1 形状适配，与 6 项 LCD 新 keys 分开列出。正式 `LCDTrainer.get_model` 重建核对全部 558 项 tensor 状态，O 保持零。没有使用任何训练完成的 best/last 作初值。

| 新状态 `model.5.lcd.*` | 参数量 | 原 AdamW 组 | 初始 LR / decay |
|---|---:|---|---|
| P.weight | 4096 | weight | .0005 / .0001 |
| norm.weight | 32 | bn（规范化无衰减） | .0005 / 0 |
| norm.bias | 32 | bias | .0005 / 0 |
| shared_dw.weight | 288 | weight，仅一次 | .0005 / .0001 |
| G.weight | 1024 | weight | .0005 / .0001 |
| O.weight | 4096 | weight | .0005 / .0001 |
| 合计 | **9568** | 无新 buffer | 原 warmup 动态 LR |

nc1 未融合总参数 **20,159,333**（母版 20,149,765）。80×80 时卷积 MAC=62,668,800，包括第二次功能式 DW 的 1,843,200 MAC；这是运算计数，不是实测延迟，不包括 LN/激活/乘法/访存。没有依赖漏计功能式卷积的 THOP 值。

## 配方与数据

`mother_args.yaml` 来自成功包 `metadata/launch/actual_train_args.yaml`，共 109 字段。与母版 c2_args 逐字段/类型对照，除历史身份字段外相同，覆盖本版本全部 DEFAULT_CFG。`resolved_formal_config.yaml` 和 `recipe_diff.json` 展示完整服务器配方，仅变更 model/name/save_dir；prepare 可根据经核验的等价绝对路径重新解析身份。

保留 epochs200/patience50/B16/640/nbs64/seed42/workers8、AdamW(.0005,.01,.937,.0001)、warmup5、cos_lr/deterministic/AMP，以及任务书所有增强。正式训练不冻结、不另设 LCD LR、optimizer 或裁剪、不自动减 batch、不关 AMP。旧 native OOM 重试通过实验回调禁止减 batch。

`dataset_inventory.json` 直接来自成功包；本地物理目录已逐项复核：6048/1728/864 图像，45573/12840/6663 框，nc1。指纹覆盖相对文件清单和标签内容，**不等于对所有图像内容做 SHA**；短检样本另外记录图像 SHA。已有增强同源图跨 split 的局限沿用，未移动文件/重划数据/改增强。

## 有界验证与已知边界

`local_checks.json` 是本地最终记录，`initialization.json`、`prepare_evidence.json`、`ops_checks.json`、`export_precision_checks.json` 提供分项证据。报告中的 runtime commit 是执行时母版 HEAD；实际改动版本由 source_signature 绑定，最终交付 SHA 由提交后远端核对给出。不能把这些本地记录当服务器 preflight。

本地 Python3.9.25/torch2.7.1+cu118/CUDA11.8/RTX2060 6GB。真实图像 B2/160 跑原 L0、native warmup、unscale/全模型 clip10/scaler.step/EMA，最多两次有效 optimizer 更新；最后允许一个不 step 的 backward 验证 O 更新后的内部梯度。CPU 和 CUDA AMP 的统计分开记录；scaler 溢出不算有效更新。源码和工程准备不能证明 mAP 提升。

融合对照仅在隔离 FP32/eval 模型上局部禁 TF32/autocast，随后再次 eval，正常和异常退出均恢复设置。沿用 RDL 修复提交 `91abf119f92a0b930fee933f30a2e6950143f023` 的精度作用域思想和母版候选探针，没有导入任何 RDL 训练/loss。原容差 atol=rtol=3e-5。保留原始逐行失败；只有集合一致、完整一一映射、全部中间量/boxes/scores 对齐通过且无非有限才报 `PASS_CANDIDATE_PERMUTATION`，生产顺序不变。

早期未固定短检随机序列的 CPU 样本曾出现 SET_DRIFT，保存在 `early_fusion_set_drift.json`，没有被改为通过。之后将诊断随机种子固定为 42 并保存/恢复调用方 RNG；固定 fixture 的 CPU/CUDA 均可通过候选置换验收。这表明近并列候选对数值变化敏感，**不构成任意输入融合输出等价保证**。服务器遇到集合漂移仍失败关闭，不放宽容差或改输出次序。

本地已检查模型/EMA/optimizer/scheduler/scaler/未完成累积梯度的保存恢复，独立 Python 进程 load、无标签 eval、全模型固定 B1/160 TorchScript 导出及融合后 FP16 AutoBackend。动态导出/640导出/ONNX/TensorRT 未验证、未安装依赖。服务器 Python3.10.13/torch2.1.2+cu121/RTX4090 的 B16/640、最多16 mini-batch、最多900秒、1–2有效更新短检必须另执行，不能用本地结果替代。

## 生命周期与命令

入口 `tools/lcd_v1.sh ACTION` 固定使用 `/root/miniconda3/envs/rtdetr/bin/python`，也可直接调用相同路径下 `tools/lcd_v1.py ACTION`。所有工程输出在实验 worktree 的 `outputs/lcd_v1/`，正式 run 为主仓库 `runs/c_series/lcd_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`。

1. `sync_lcd_v1.sh FULL_SHA`：固定 origin/分支 fetch 核对完整 SHA，首次创建或干净 worktree 正常前进；拒绝倒退、已跟踪修改、未跟踪/忽略冲突，不 reset/clean/force。脚本先用新 SHA 中的 sync helper 更新，再执行新代码，主/其他 worktree HEAD 保持不变。
2. `prepare`：核验源、完整配方、物理 split，受控初值和配置；不启动训练。重复 prepare 保存旧报告，保留相同初值。
3. `preflight`：完成针对性本地形状检查和服务器 B16/640 更新短检。短检是独立子进程，新 optimizer/scaler，固定真实训练 batch 可重用，不表示收敛；超时/未完成/AMP关闭均明确失败，模型不用于正式训练。
4. `start`：只核对有效报告身份，不重跑预检。固定 tmux `lcd-v1-training` 中执行绝对 Python 和本实验脚本。先报已派发，`status` 依据进程、日志和训练 setup 才报 RUNNING。日志 `console_<UTC>.log`；没有 tee 管道丢失退出码，shell/Python 各有结束记录。
5. `status`：读取 session 对应 worker PID/子进程、真实日志尾部、results.csv 路径、best/last 和退出状态。
6. `resume`：核验 last.pt 源码/初值/配置/数据身份，恢复 epoch、完整模型、EMA、FP32 optimizer、scaler、scheduler、early stopper、RNG、未完成的 scaled 梯度和 last_opt_step；仍用同一 tmux。只支持 epoch 边界，不能保证多进程增强或任意 batch 中断后的逐位复现。
7. `val`：成功结束后独立评估训练 val fitness 选择的 best，固定 SHA。
8. `test`：只对独立 val 已锁定同 SHA 的 best 做一次；已有评估目录拒绝覆盖。沿用 corrected_sorted_conf_mask_v1、640/B16/workers0/FP32/conf.001/iou.7/max_det300。输出 AP50/AP75、十阈值 AP、P/R/原最大F1工作点，固定精度召回只是辅助，不用于调公式或选权重。
9. `pack`：源码差异、身份、配方、小报告、日志尾部、CSV、评估指标组成 LIGHT，限制8MB，排除模型/数据/大fixture；保留本地 best/last/CSV。

原训练器已经保存 scaler。本分支在实验 Trainer 中补充完整训练模型、FP32 optimizer、scheduler、累积梯度、RNG 与 early-stopper，避免原 final_eval strip 掉恢复信息。共享 BaseTrainer 只增加 last_opt_step 的可选读/写，未启用 LCD 属性时行为不变。原评估函数只增加显式 verifier/metric_summary 扩展参数，默认母版行为保持。

正式训练过程中每 epoch 的少量 batch 汇总 residual RMS 比、tanh 饱和率、O范数与梯度有限性，输出 detached 标量，不增加监督。保留原AMP溢出处理并计数；连续64次无法执行更新或整epoch无更新会报错，绝不修改配方硬过。保持200e/patience50规则，不因前几十轮偏低停训，不自动搜索超参。

历史母版独立 val/test mAP50–95=52.454272%/52.200902%，仅为同协议参照，本轮未复测。完整训练后先看 val 全指标，再看一次锁定 test；若无综合改善应如实判定失败。

服务器短命令见交付回复及 [SERVER.md](SERVER.md)。本轮没有执行 start 或正式长训。
