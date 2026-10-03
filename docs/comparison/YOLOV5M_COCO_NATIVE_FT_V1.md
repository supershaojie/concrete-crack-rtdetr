# YOLOv5m 原生预处理与外部配置固定工具交付

2026-10-03，Asia/Shanghai。实施代码完整 SHA：`64da7e2f3041cb212f8c81f5e52bb68c457de8ea`，实际父提交：`07d77c16ec168bd9547cd5aac04817c1d5d46e39`。开发分支 `bench/yolov5m-coco-native-ft-v1` 从该指定、已验证 pilot 派生，后续提交只补身份文档，不改变运行代码。实施前本地/远端均没有该分支；远端通过本机既有代理只读核实。独立本机工作树位于 `D:/MyProjects/Crack_RTDETR/outputs/worktrees/Crack_RTDETR-bench-yolov5m-coco-native-ft-v1`，主项目 HEAD 和原未跟踪实验文件保留。

## 实现

固定 [YOLOv5 v7.0 源码](https://github.com/ultralytics/yolov5/tree/915bbf294bb74c859f0b41f1c23bc395014ea679)，完整上游 SHA `915bbf294bb74c859f0b41f1c23bc395014ea679`。恢复原生等比 resize、LetterBox、RandomPerspective、乘法 HSV、原生 Mosaic/MixUp 分支及框裁切/面积/比例过滤。固定版本的 load_image 使用 int 截断，不引入其他版本的 ceil 规则。旧 stretch、加法 hue、非 Mosaic MixUp 和 CutMix 链不进入新 runtime；旧 b19 文件只保留来源归档。Albumentations 不隐式执行；CutMix/CopyPaste 非0明确拒绝。保留 M 模型、原生 Detect/loss/assigner/EMA/AMP/optimizer 更新顺序，沿用已验证的 mAP50–95 选 best/latest-tie、早停、checkpoint 保存和公共评估。

`config.py` 定义固定默认值和支持字段，YAML 只覆盖已有数值/选项。支持严格未知/重复字段、类型、有限数值/范围和组合校验，batch→batch_size、cache true→只读源图的 RAM cache；拒绝 autobatch、磁盘源图缓存、多GPU/DDP、CPU+AMP、rect 与活动 Mosaic/MixUp、原生 MixUp 没有 Mosaic 等组合。SGD/Adam/AdamW 调用原生 smart_optimizer，momentum 同时是 Adam 系列原生 beta1，beta2/eps/Nesterov 保持原生。没有 --set、搜索器、调参平台或增强管线切换系统。

每个 run 保存原始 YAML、补全的 resolved_config.yaml、raw train_hyp.yaml、resolved_options/model_check、固定代码/上游/补丁/官方权重/数据/环境和配置哈希。原生训练 opt/hyp 从这个 run 派生；恢复验证 checkpoint 的 raw hyp，再由原生训练只缩放一次 nc/imgsz/nl/decay，effective_training 和 effective_resume 分别保留。每轮记录实际各组 LR/decay/动量或 betas、累计成功更新数和 sum(lr×decay)，保留 CSV、best/last、真实退出状态和实际已保存轮数；耦合 SGD/Adam 的累计量不冒称独立乘法衰减。close_mosaic 由 epochs-close_mosaic 计算，丢弃旧预取并重建真实 worker，0 表示不关闭。

固定代码身份与每 run 配置身份分开。外部 YAML 移动/编辑不影响旧 run，run 内快照编辑则拒绝。同一 run 的 last 要通过代码/数据/资产/环境/配置/UUID、初始化审计、optimizer/EMA/scaler/scheduler/RNG/generator、best/last/epoch_state 等校验。环境记录固定实际 site/dist-packages 根，避免 setuptools 导入私有 vendor 目录被误判为新安装依赖；解释器、真实导入路径和已安装版本继续严格校验。保持原生 half checkpoint 和 epoch 边界恢复语义，不承诺逐位连续等价。

一次准备的新源码目录从旧资产的**已提交原始 Git 对象**克隆并应用本次补丁，不覆盖旧已补丁源码。官方权重不复制、不下载，读取原文件前核对 42,806,829 bytes / SHA256 `61d933360ba5a7733a36764996c800287d973889d875227f5beedd2473a97a56`。新候选始终从官方 COCO 权重开始、resume=False、freeze=[0]（无冻结）；实际检查 475/481 全兼容迁移、参数20,871,318，不排除兼容 anchor/层。旧 run 的 best/last 只允许用于自己的恢复/导出。

数据保持固定 train6048/45573框、val1728/12840框、test864/6663框，YOLO crack0→公共 category1。沿用路径/标签字节/图像大小清单轻量身份，不做全库图像哈希或复制。已有尺寸清单核对路径/标签/大小/ID，抽查3个 train 图片头，在 run 内构建原生标签/shape 缓存；无可用清单就停止，缓存坏了不全库重扫。AutoAnchor 只看 train，逐项核对原始 wh shapes，并从原生真实 jittered targets 记录前后 anchors/BPR；保留原生连续尺度近似与实际 int resize 的亚像素取整差异。

`start --run-id ... --config /absolute/file.yaml` 完成准备、训练、best 公共 val 和汇总，test 为 NOT_EXECUTED。`resume --run-id` 不要求源 YAML；完整训练后仅续做公共 val。`finalize --run-id` 要求已完整训练且 val 身份一致，只补同一 best 的 test；已有一致缓存复用。GT 内容要与冻结标签/尺寸一致，导出/metrics 检查 checkpoint/config/GT/预测/metadata/metrics 哈希和公共 policy。中断、错误、partial、不一致缓存清楚报告，不自动覆盖。

公共 val/test 固定同一 best/EMA、FP32、640、batch16、conf=.001、类内 NMS=.7、max_det300、无TTA、corrected_sorted_conf_mask_v1；保留全部图像/空预测/浮点坐标，逆变换用实际 x/y gain 和整数 padding，无额外裁框。native val 默认batch16、rect=True、NMS=.6，调用条件、实际 dtype/shape 逐次记录。test 不参与选参。原生 loss raw box=.05、cls=.3、obj=.7、cls_pw/obj_pw=1、iou_t=.2、anchor_t=4、fl_gamma=0；单类 cls_loss=0 保留原生行为。

## 必要验证与边界

使用已有本机 pilot 解释器，只读复用 Python3.9.25、Torch2.7.1+cu118、torchvision0.22.1+cu118、NumPy1.26.4。没有环境安装或服务器操作。CPU batch2/imgsz64 的真实合成 native 训练：SGD 从第1轮中断恢复至3轮、AdamW 从第1轮中断恢复至2轮，初始化记录保留，optimizer/EMA/scaler/scheduler/RNG/generator 和累计更新正常，raw/effective loss 未重复缩放，243个有限非零梯度。另检查原生 Adam 三组 optimizer 的真实更新，以及本机实际 M 模型的私有 CUDA FP32/FP16 AMP 探针。

同一代码用三份配置准备 run，SGD/AdamW、lr0、weight_decay、nbs、degrees、HSV、mosaic/mixup 差异进入真实选项/dataset/optimizer；源 YAML 改写、移动/删除后旧 run 继续使用自己的快照，新 run 读取新值。未知字段 CLI 拒绝；修改 resume 选项、初始化/外来/已完成 checkpoint、固定代码身份、预测和 metrics 文件均拒绝。实际 CLI 的 resume/finalize、已完成 val/test 缓存复用和重复 finalize 通过；finalize 前后 val 文件哈希和 train stage 列表保持相同，start/恢复选参阶段 test 未执行。

原生几何检查：合成非方形图与标签对齐、乘法 HSV LUT、裁切/原生面积过滤、奇数尺寸导出逆变换；双 worker 在 epochs7/close3 的索引4关闭 Mosaic/MixUp，真实计数由 [Mosaic2,HSV1,RandomPerspective2] 变为 [0,1,1]，最后阶段恢复和 close0 通过。合成 train AutoAnchor shapes 与原始尺寸一致，实际 BPR=1.0（仅这4张合成图，不能代表完整 train BPR）。

另只读检查3张真实 train 图片，3072×4096→480×640→LetterBox640×640；每张3框均对齐，归一化坐标最大误差1.4603138e-6，导出逆变换误差0。没有全库图像哈希、划分、增强或复制。这3张旧固定 train 图含既有离线增强，只作为预处理 QA，不产生指标结论。

相关检查：配置5项、现有控制5项、现有中断报告1项、mock tmux/两组会话/同名保护/恢复/参数拒绝1项；Python 编译、Bash语法、Git diff whitespace 检查通过。完整真实合成 native best 的公共640 FP32 val/test 导出与冻结公共 evaluator、空预测/AP fixture 通过。检查在指定父提交加实施修改的工作树完成，随后核实规范化运行源代码指纹与实际提交逐文件一致；smoke 的历史 code SHA 保留为当时父 HEAD，不伪改为提交后的 SHA。详细证据为 [验证 JSON](evidence/yolov5m_coco_native_ft_v1_validation.json)，大权重/日志只在 ignored outputs。

未验证 AutoDL 实际环境、真实全数据训练/性能、GPU0 同时两个 batch16/imgsz640 的容量及真实 Linux tmux/POSIX 信号。所有 smoke 都有 SMOKE_ONLY_SYNTHETIC 身份，不可当正式实验、最优配置或新 run 初始化。首次候选同时改变优化、增强和训练预处理，应作为整体配方比较。

## 使用和正式归档

默认候选在 [v5_ft_01.yaml](../../benchmarks/comparison/yolov5m/v5_ft_01.yaml)。完整可执行服务器命令见 [服务器命令](YOLOV5M_COCO_NATIVE_FT_V1_SERVER_COMMANDS.md)，含首次固定代码、只读复用环境/资产、完整 YAML、两个 run、tmux/日志、同 run 恢复、选定 finalize 和独立归档工作树普通提交/推送。数值候选写 gitignored runtime_configs/，代码 HEAD 和 run 输出不随调参提交。

选择已完整训练并 finalize 的正式 run 后，archive 入口从 run 快照复制 YAML/小结果及实际 training_code_sha。在独立归档工作树普通提交并推同一个开发分支，运行工作树不切 HEAD；不上传权重/全量日志，不合并 main，不 force push。
