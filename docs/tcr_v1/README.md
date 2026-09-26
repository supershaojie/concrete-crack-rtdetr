# TCR v1：三带对比残差

本实验只包含 RT-DETR-R18-Lite + 原 LIF-Down + 原 CBR + TCR v1，直接派生自 `a0459d6a652cb702699087c88fa39a3e4c4087ec`。分支为 `exp-rtdetr-r18-lite-tcr-v1`。本地开发 worktree 为主仓库同级 `Crack_RTDETR-tcr_v1`。

## 实际机制和接入

YAML 只把原第 17 层的 `Conv` 改为 `ConvTCR`，仍从骨干节点 5 取输入，参数仍为 `[256,1,1,None,1,1,False]`。原 conv/bn/act 输出后执行 TCR，再进入第 18 层拼接。第 19 层仍是融合后的 Neck P3，原 LIF-Down 在 20，原 CBR 检测头在 26，输入仍为 `[19,22,25]`。

TCR 用无 bias 的 1×1 P 将 256 通道降到 16。对四个固定整数方向计算五点切向均值 C，再沿法向平移 ±1、±2 步取得两侧均值，逐元素计算 `relu(min(C-Bplus,C-Bminus))-relu(min(Bplus-C,Bminus-C))`。八组按方向、侧距、通道的顺序拼为 128 通道，经无 bias 的 O 投影回 256 通道并加到原 X。仅 O 零初始化，P 保持 Kaiming；没有额外 norm、gate、激活或损失。

只有共享 4 像素内区保留响应，H 或 W≤8 时残差为零。所有移位为 pad/slice，无循环绕边、可学习方向或双线性采样。条带求和、差分、minimum 使用局部 FP32，O 按原 AMP 执行，残差转回 X.dtype。对角整数一步长度为 √2，因此不宣称严格旋转不变或各方向完全等距。

新增参数 **36,864**；nc1 未融合模型实测 **20,186,629**，状态项 **554**。仅增加 `model.17.tcr.P.weight` 和 `model.17.tcr.O.weight`，公共状态路径保持原样。ConvTCR 同时实现 forward 与 forward_fuse。原 LIFDown 的 pre-BN 残差及融合保护保持不变。原 CBR 仍以 36 个采样点、rho=.10 仅修正最终框，分类分数仍来自原 decoder。

## 初始化、配方和证据

公共未训练源 SHA256 为 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。初始化复用固定母版的 `controlled_models`，先审计原 LIF/CBR 的公共源映射，再增加 TCR。额外模块在 CPU fork_rng 中构造；真实原生 Trainer.get_model 重建后，再逐 key 比较 552 个公共状态，并检查 9 个 nc80→nc1 类别适配键的明确 allowlist。所有 347 个可训练参数张量恰好进入 AdamW 一次。

`mother_args.yaml` 来自本机已解包的成功母版 `training/args.yaml`，与执行合同附录一致，共 109 项。它与母版仓库中的 `resolved_formal_config.yaml` 仅存在已核实的 model 路径 `-gatefix` 差异；其余 108 项一致。正式配置只改变 model/name/save_dir；`resolved_server_args.yaml` 和 `args_diff_109.json` 给出完整记录。研究参数单独放在 `research.yaml`。

没有改动 criterion、Hungarian matcher、DN/query、encoder topk、optimizer_step、梯度裁剪、EMA、scaler、保存或 resume 的原生实现。真实 criterion 的 loss_gain 为 class=1/bbox=5/giou=2；matcher cost 为 class=2/bbox=5/giou=2。母版 VFL 已经使用 detached 匹配框 IoU，不能把 TCR 描述为新增质量监督。

原模块 LF 规范化 SHA256：

```text
lif_down.py 26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7
cbr.py      d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787
```

`validation_summary.json` 汇总真实执行范围；详细本地记录位于 `outputs/tcr_v1/`，小型报告随本目录提交。开发阶段报告中的 HEAD 为当时检出的母版，额外保存了受测工作树文件内容哈希。最终交付 SHA 在提交后写入忽略目录中的 DELIVERY.md/server_commands.md，避免报告把自身 SHA 写进自身形成循环。

## 已执行验证与限制

本机环境：Python 3.9.25、PyTorch 2.7.1+cu118、RTX 2060 6GB、工作树 Ultralytics 8.4.21。CPU/CUDA 标量和独立逐坐标空间参考、非方形和小边界、亮暗线/仿射平面/理想单侧阶跃、粗线/弯曲/分叉诊断、零初始化、原生 loss 与公共梯度、后续 P/O 梯度、空 GT、EMA/save-load、融合/AutoBackend、AMP 与原生 FP16 epoch validator 均有相应本地证据。

CPU 合成数据夹具还执行了真实 Trainer setup、一次更新、原生 checkpoint 保存、原生 resume 恢复 optimizer/scaler/EMA/epoch、每轮 validator、独立 validator 和无标签 predict。夹具指标不是裂缝实验 AP。非零 TCR 完整模型的静态 FP32 TorchScript 导出/重载已验证；其余导出格式及 dynamic/half/int8/nms/optimize 导出明确报错。PyTorch 2.1 服务器执行情况需以服务器预检为准。

融合比较复用母版 `c19_lif_v1_diagnostic.fusion_protocol` 和 `fusion_accepted`，记录 encoder 候选身份、对齐比较和固定 query 回放。近似同分候选可在融合后换行，不能直接把未对齐的 300 行输出当作融合算子误差。`lif_input` 位于候选选择之前，其超差必须先追踪上游，不能用候选换行解释。正式推理不改 topk，也没有全局关闭 fuse。公共梯度采用报告中明确的 FP32 容差；CUDA grid_sample 的非确定性警告仍如实保留，不承诺逐位复现。

服务器融合精度修复见 [fusion_precision_fix.md](fusion_precision_fix.md)。预检在局部严格 FP32 作用域内创建融合模型并比较，关闭 matmul/cuDNN TF32 和实际 autocast，退出或异常时恢复。另以同一输入、权重、RNG 执行原运行精度对照；原容差保持 `atol=2e-5, rtol=2e-4`。新顺序追踪覆盖原骨干、节点 17 原 Conv 输出、TCR/P/O、节点 19、`lif_input` 及后续 neck。原 LIF/CBR、TCR 公式和正式配方未改。

预检 `PASS` 要求严格及原运行精度比较均通过；`PASS_STRICT_FP32_ONLY` 仅确认有界训练容量和严格 FP32 融合门槛，原运行精度的失败仍保留为失败，并在终端和报告中明确显示。任一严格 FP32 算子/候选门槛失败均阻止启动。这个状态不证明运行精度下的融合一致性，也不证明服务器尚未执行的新版预检已经通过。

本机数据身份核对为 train 6048/45573 GT、val 1728/12840 GT、test 864/6663 GT。这里只读取 test 路径和标签以核对划分身份，没有 test 推理。独立只读 probe 预先冻结字典序前 64 张 train 图、640、无增强，加载已训练母版公共权重，P 为 v1 随机初值、O=0。八组均出现正负响应，BN/参数未更新。随机投影 probe 只确认特征响应，不证明学到方向或涨点。

服务器真实 B16/640 预检、正式 200 轮训练、完整 val、锁定 test 均为 **PENDING/NOT_RUN**。未登录服务器、未启动正式训练，也未执行最终 test。

## 服务器操作

先按本次交付的完整 SHA 下载 `tools/sync_tcr_v1.sh`，再显式运行 `sync FULL_SHA` 建立交付身份。不能只下载脚本或直接拉取漂移的分支 head。同步入口只安全 fetch/快进独立 worktree，拒绝脏工作树、错误归属和本实验活跃进程。

`tools/tcr_v1.sh` 或绝对 Python 入口 `tools/tcr_v1.py` 支持：

| 动作 | 实际行为 |
|---|---|
| prepare | 核对导入、原模块、公共源、完整配方、数据；生成受控初值；准备并检查 check_amp 所需 bus.jpg/yolo26n.pt |
| preflight | 独立子进程，真实 B16/640 原生 AMP/AdamW/warmup 累积；最多 16 micro-batch、900 秒；至少两次实际改变参数的有效更新及 O 更新后 P 梯度；诊断 scaler=128；检查原生 FP16 validator、非零融合和 AutoBackend |
| probe | 最多 64 张固定 train 图的只读诊断；缺母版权重/数据写 PENDING；已有结果保留 |
| start | 要求当前预检通过，启动 `tcr-v1-training`；正式 scaler 使用母版默认，不继承诊断状态 |
| status | 展示当前 dispatch、真实进程及子进程、阶段、epoch、日志尾和独立 Python/tee 退出码 |
| resume | 只接受本 run 的有效 last，恢复原生 optimizer/scaler/EMA/epoch；核对原训练身份 |
| val | 本 run 正式 best 的独立 FP32 完整 val，并锁定权重/config/data/source/评分模式 |
| test | 仅此显式动作评估 val 锁定身份；相同已完成结果复用 |
| pack | 仅打包已有证据，缺项标记 NOT_RUN；不触发训练/评估 |
| archive-failed-start | 仅 args/空 CSV/无 checkpoint、无活跃进程的 setup 失败目录可重命名归档；另行 start |
| commands | 根据当前真实完整 HEAD 生成独立可复制的服务器命令文档 |
| diagnose-fusion --fixture PATH | 只读载入旧失败的 fixture.pt；新目录中比较严格 FP32、原运行精度及历史融合权重，不训练、不改旧证据 |

预检缓存基于相关源码、研究配置、109 项训练配置、初值/公共源、数据和环境身份，报告时间戳不参与。数据每次核对标签内容 SHA256、相对路径清单和图像 size/mtime；没有跨路径迁移。正式训练每 100 个 micro-batch 采样八组非零/正负计数、RMS、残差比及 P/O 权重和梯度，按 epoch 汇总，保留分母和采样数；持续零残差会记录事件。

正式 run：`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/tcr_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`。

工程证据：`/root/autodl-tmp/projects/Crack_RTDETR-tcr_v1/outputs/tcr_v1/`。tmux 同时显示 Python 输出和 tee 日志；Ctrl+B 后按 D 脱离。重复启动仅阻止本实验，不操作其他实验进程。完成状态记录实际 epoch，包含早停，不统一宣称 200 轮完成。

LIGHT 写入主仓库 `/root/autodl-tmp/projects/Crack_RTDETR/downloads/tcr_v1/`，命令输出 FileZilla 可用的完整绝对文件名及校验结果。它包含源码和 diff、完整配置及逐字段 diff、哈希、初始化/预检/状态、训练 CSV、所有机制 jsonl、日志摘要、现有 val/test 指标、遗漏大文件的路径/大小/SHA256。`pack --include-predictions` 另建 ERROR_ANALYSIS 包，只纳入已有 train/val 逐图记录，不自动加入 test 或重跑。

## 评估与研究边界

每轮 val 保持母版 validator 的精度与 fitness 选择，TCR 一直启用。独立 val/test 直接复用母版 `corrected_sorted_conf_mask_v1` 的真实排序后 mask 实现，FP32/640/B16/workers0/conf=.001/iou=.7/max_det300/augment=false/rect=false/seed42。逐图记录包含原图尺寸、坐标约定、原 query ID、最终框、分数和 GT。P/R 使用该模型自己的平滑平均 F1 最大工作点。

几何桶预先写入 research.yaml，只对 GT 框短边/面积/宽高比作辅助分析，使用明确的 score=.25、IoU=.5/.75 独立诊断匹配，不改主 AP。没有把框几何解释成裂缝粗细、交叉或分叉标签。若需比较变化，应对同协议母版预测用相同桶分析；本交付不预填任何新 AP 或语义子集收益。

粗裂缝、强弯曲、分叉可能不满足三带条件，部分接缝反而满足；残差主路不能保证不退化。既有数据含同源增强图跨 split 的限制，仅作同协议比较。历史母版/CQS/BMC 分数属于历史参照，不是本次实验成绩。

相关研究链接与本地模块包审计见 `research_context.md`。TCR 按给定公式独立实现，没有移植模块包或其他候选分支。
