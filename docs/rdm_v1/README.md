# RDM v1：原 CBR＋原 LIF-Down 的独立第三模块

母版 `a0459d6a652cb702699087c88fa39a3e4c4087ec`，分支 `exp-rtdetr-r18-lite-rdm-v1`。
本次正式训练 **NOT_STARTED**，独立完整 val / 最终 test **NOT_RUN**，没有登录服务器。

## 结构与来源

两份 YAML 分别从原组合和原 C2 配置派生，只替换 model.5 为
`[-1, 1, RDMBlocks, [128, BasicBlock, 2, 3, "relu", 32]]`。
`RDMBlocks` 继承 `Blocks`，保留两个 BasicBlock、原末端 ReLU 和 `.blocks.*` 键；
完整 stage 输出后仅调用一次 `.rdm`。640 输入产生 B×128×80×80，并同时被 model.6 和 model.17 消费。
原 model.20 LIFDown 和 model.26 CBR、所有其余 from/重复次数均保持。
单模块配置保持 C2 的下采样和 decoder。

`Z=Wd(X)`；局部分支 `L=SiLU(Dl(Z))`；区域分支固定 4×4 均值池化后
`SiLU(Pc(Dc(Q)))`，bilinear 上采样回原尺寸；
`G=sigmoid(Wg(cat(Z,C_region,abs(Z-C_region))))`；`Y=X+Wo(G*L)`。
只有 Wo.weight 置零，其余权重 Xavier uniform，Wg.bias 置零。
新增层在隔离 CPU RNG(seed=42) 下构造；不改变 CUDA RNG，不在加载或 forward 中重置。
无新增归一化、张量 buffer、可学习 alpha、额外激活或部署重参数化。
H/W≥4，矩形余数按 floor 池化，H/W<4 明确拒绝。

借鉴 [SMFANet ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf)
及[官方 SMFA/FMB 实现](https://github.com/Zheng-MJ/SMFANet/blob/main/basicsr/archs/SMFANet_arch.py)
的局部/区域调制思路。官方实现使用的全局方差、adaptive max pooling、GELU、nearest 等没有引入。
本设计使用差异条件门控和单处残差适配，不是 SMFA 改名；区域描述并非真实背景标签，绝对差并非纯高频或裂缝标签。
指定参考 ZIP 在附件目录、下载目录和项目的有界搜索中未找到，未解压/复制资源包。
实际搜索边界和来源见 `provenance.json`。不从超分论文推断检测涨点，不声称零开销或减参 10%。

| nc=1 配置 | 未融合实测 | 原生融合后实测 | 新增参数 |
|---|---:|---:|---:|
| C2＋RDM | 20,095,668 | 19,890,612 | 12,896 |
| CBR＋LIF＋RDM | 20,162,661 | 19,957,861 | 12,896 |

设计式 `2Cr+18r+4r²+r=12,896` 与实测一致，仅 7 项新状态、0 个新 tensor buffer。
主卷积 74,457,600 MACs，即 0.1489152 GFLOPs（2 FLOPs/MAC）。
一次现有 THOP 2.0.18、B1/640/nc1/未融合计数：C2 58.276608→58.4255488，
组合 58.672512→58.8214528 GFLOPs；两者增量均 0.1489408（包含计数器的池化项）。
functional 插值、abs、激活、乘加等未完整覆盖；这不是完整实测开销或 FPS。
母版原生 fuse 保留骨干 ConvNormLayer 的 BN 和 LIF 的 BN，所以 `is_fused()` 的阈值结果
仍可能为 False；本次按真实 model.8 Conv-BN 状态判断参数量，没有修改融合实现。

## 初始化、数据与配方

复用母版 `init_c19_lif_v1.py` / `init_lif_down.py` 的公共源验证及父模型构造。
源 SHA256 为 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`，nc80、epoch=-1。
COMMON 组合 552 项、单模块 533 项，NEW 各 7 项；公共 parameter/buffer 的 key、shape、value 全等。
实际 `RTDETRTrainer.get_model` nc1 重建核对两变体，允许原生 9 项分类适配，其余状态精确迁移。
两个变体 RDM 初值相同且不共享 storage；公共参数正常训练，新增状态仅进入原生优化器一次。

默认 `init` 只生成/核验 `weights/cbr_lif_rdm_v1_controlled_init.pt`。
存在时逐项核验复用，SHA 和 mtime 均保持；`init --both` 明确额外生成单模块初值。
受控初值是必要正式模型文件，约 77.66 MiB，不属于预检临时文件。

`parent_args.yaml` 来自实际成功结果 `C19＋LIF：原 CBR＋LIF-Down v1/training/args.yaml`。
全部 109 字段与附件一致，与母版已存配方仅父初始化的 gatefix 路径不同。
两份 `*_recipe.yaml` 保存完整服务器配方，`*_recipe_diff.json` 保存全部字段差异，
默认服务器路径下只改变 model/name/save_dir。无库默认值补配方。
200 epochs、patience50、B16/640、nbs64、seed42、workers8、AdamW/lr0=.0005/lrf=.01、
warmup5、cos_lr、AMP、原在线增强、EMA/梯度累计/loss/matcher/DN/原 epoch 验证与 best 选择均保持。

实际母版数据清单与本机清单、标签哈希完全一致：6048/1728/864 张，45573/12840/6663 个框。
`data_identity.json` 固定清单及标签身份；运行时还检查配置、类别、数据根路径和 GT 几何。
严格检查后才转换为 JSON 可往返结构，避免 names 的整数键造成假 stale。

## 验证范围和限制

`local_checks.json`：两配置结构、640 连接实测、参数量、公共初值、Trainer nc1 重建、RNG 隔离，
正常及非 4 整除矩形的恒等/非零分支/门控上下文/梯度/独立 CPU FP32 公式均通过。
最大前向差异 1.1920929e-7；参数梯度最大差异 1.4551915e-11；阈值固定 atol2e-5/rtol2e-4。
`ops_checks.json`：CPU 原生 checkpoint 半精度保存口径、原生加载/optimizer/EMA 恢复函数、
非零 RDM CPU FP32 AutoBackend 160 前向及拒绝缺失/过期/超预算证据的检查通过。
它没有执行真实数据训练，也不是服务器实际 resume 验收。

`server_pending.json`：本机 RTX2060、PyTorch2.7.1+cu118，与指定服务器不等价。
服务器 B16/640/native AMP、真实更新、实际 resume、1 batch half EMA val、CUDA FP32/half AutoBackend
均 **PENDING/NOT_RUN**；未开展本机容量尝试、整网 CPU 训练、完整 val/test。
开发中一次参数量检查误用了母版 `is_fused()`，已定位并修复检查，旧失败报告留在 outputs 中；
最终报告没有未解决的 FAILED 项。PASSED 仅表示所述工程范围，不表示精度提升或所有 dtype 路径等价。

服务器 `preflight` 默认主组合：首次最多 8 batch，至少 2 次 Wo 实际变化且梯度有限的原生 optimizer 更新；
随后原生保存、同一 half 保存源核验、原生 resume 至少 1 次有效更新，总计最多 16 batch（含 scaler 跳步）。
Wo 有效更新后检查上游梯度，验证真实 DN 和 RDM 调用。有限前向/loss/参数且由完整 optimizer.step 调用证据确认的
原生 AMP 溢出跳步及 scale 回退记为 AMP_BACKOFF，在既有预算内继续，不计有效更新；
预算不足标 PENDING，不自动扩预算、降 batch、重试、关闭 AMP。
随后只做一次原生 half EMA 真 val batch（母版原生 val loader 为 B32）、各一次 CUDA FP32/half AutoBackend B1/640。
原生自动完整 val/final_eval 被预检的有界退出截断。不会自动 start、消融或 test。

AMP 提前中止修复及小型 fixture 见 `amp_backoff_fix/README.md`。
本目录原有报告保留其原始身份和状态，不能作为修复后的服务器通过证据；修复后需要一次新的有界预检。

全部测试专用 checkpoint、optimizer/EMA 文件和运行目录放入有归属的 TemporaryDirectory；
正常、异常和可捕获中断清理，报告写在外部。不保留大型 tensor 或整仓副本。
SIGKILL/断电不能保证清理；本工具不扫描删除历史目录。若有遗留，仅在核实 owner.json PID 已退出后
人工检查本工具前缀目录，不能清理别的实验。action.lock 同样先检查 PID。
`storage.json` 记录交付时留存大小；每次预检新增小报告/日志目标≤10MiB，默认结果包硬限制20MiB。

## 入口

统一入口 `bash tools/rdm_server.sh ACTION`；`--help` 显示真实参数。
使用独立 RDM_MAIN/RDM_DATA/RDM_VARIANT；现有 Conda 环境 rtdetr，不安装/升级依赖。
环境必须从本 worktree 导入 ultralytics。仅显示 GPU 使用情况，不设空闲门槛，不终止别的任务。
原生 AMP 探测复用主目录的 yolo26n.pt 和 bus.jpg（可在 weights/ 和 ultralytics-main/ultralytics/assets/）；
在临时目录链接使用，缺失则 PENDING，不能静默下载探测 checkpoint 到工作目录。

- `environment`：环境、源码身份、nvidia-smi。
- `init [--both]`：默认主初值，已存在核验复用。
- `check`：本机可做的两配置 CPU 检查。
- `preflight`：当前内容未测时先 check，再执行本变体有界服务器链路。
- `plan`：只读显示完整配方、路径和阻塞项。
- `start`：要求本变体必要检查全部通过且代码内容/初值/数据/配方一致，重新加载未训练初值；拒绝已有 run。
- `resume --checkpoint PATH`：只接受本实验真实未完成且保有 optimizer/EMA 的原生 checkpoint；不重新 init。
- `val --checkpoint PATH`：显式独立完整 val。
- `test --checkpoint PATH --val-report PATH`：要求同 checkpoint、数据、代码和策略的已完成 val。
- `pack`：只打包已有轻量结果，排除权重/数据/参考 ZIP/大预测明细，不触发 train/eval。

旧 preflight.json 不会被覆写；需要显式重测时，先将该小报告移动为带时间戳的审阅文件。
文档提交仅改变 HEAD 不要求重测；代码、受控初值、数据或配方改变时由内容哈希识别，重新检查受影响项。
`start` 对 PENDING/NOT_RUN/FAILED 或 stale 给出具体原因。运行状态记录 RUNNING/COMPLETED_200/
EARLY_STOPPED/INTERRUPTED/FAILED；最终 eval 异常不会自动从头重训。

独立评估直接复用母版 `corrected_sorted_conf_mask_v1` 的 postprocess：
640/B16/workers0/FP32/conf.001/iou.7/max_det300/augmentFalse/rectFalse/seed42，
保留原一对一匹配，不增加 NMS；输出 P/R、AP50/AP75/mAP50–95、逐 IoU AP、PR/混淆矩阵和曲线。
训练 args.yaml/results.csv 留在原生 run，轻量包收集已有内容。
主组合有效后再安排 C2＋RDM 自身预检和重训消融，不以临时关支路替代消融。

完整固定 SHA 同步命令及未来 start/resume/val/test/pack 命令在提交确定后生成的 `RDM_HANDOFF.md` 中。
交接绑定最终代码提交，避免为在文档内填写自身 SHA 而制造提交循环。
