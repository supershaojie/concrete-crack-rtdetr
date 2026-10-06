# YOLOv13-L COCO 可配置对比实验

本文件保留原 native 实施提交 `64f6639847a0b6a1c18f8f246ba27e274a4de3ad` 的历史说明。当前 `bench/yolov13l-flash-configurable` 的后端和环境要求见 [Flash 实施说明](YOLOv13L_FLASH_CONFIGURABLE.md)。旧实验继续使用旧工作树和原冻结提交。


2026-10-05（Asia/Shanghai）。分支 `bench/yolov13l-configurable`。首轮 `v13l_aug_x13_01` 是 **NOT_RUN 计划**，本次交付不在服务器启动正式训练。训练代码 SHA 见同目录 `YOLOv13L_CONFIGURABLE_HANDOFF.md`；之后若只补文档，以该文件中的实施 SHA 创建训练 worktree，不以新的文档 HEAD 替换训练身份。

## 模型、初始化和配方身份

固定作者 [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13/tree/73289949533efac82bb5f72ec19b746618656bd2)，commit `73289949533efac82bb5f72ec19b746618656bd2`，分叉版本 `8.3.63`，`ultralytics/cfg/models/v13/yolov13.yaml`，L/l `[1.00,1.00,512]`，nc1、crack、YOLO class0。实际导入必须来自新 worktree 的 `.vendor/yolov13l-configurable/yolov13`，版本号本身不足以证明模型身份。保持作者 DSC3k2、DSConv、A2C2f、HyperACE、FullPAD_Tunnel、DownsampleConv、Detect、损失、分配器及 EMA。

从[作者官方 yolov13l.pt COCO80 资产](https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13l.pt)执行原生 nc80→nc1 `DetectionModel` + `model.load`。实测文件 111819790 字节，SHA256 `f95ad5bbf3aa80a3df2a28ff4c623582ded6e02bb43d7cea9063c2b248e316cb`，与 GitHub 发布资产摘要一致；校验先于 Torch 反序列化。实测 nc80 参数 27627783，nc1 未融合参数 27566874；迁移 1562/1568 个张量，只有 `model.32.cv3.{0,1,2}.2.{weight,bias}` 改变 80→1 输出维。773 个 gate/HyperACE/注意力相关张量逐项核验等于源值，7 个 FullPAD gate 是已学习的非零值。记录见 `initialization.lock.json`、`upstream.lock.json`，每个新 run 另存 `initialization.json` 及 optimizer 第一步前复核报告。迁移失败直接报错。

训练、AMP 对照和独立评估均固定 **native**。AAttn 实际执行 matmul 与减最大值后的 exp/sum 稳定 softmax；作者日志中的 “scaled_dot_product_attention” 不代表调用 Torch SDPA。训练默认 AMP=true，独立 val/test 强制 FP32，内部 mm/bmm/卷积/softmax/exp 经 `ArithmeticAudit` 检查，CPU/CUDA autocast 与 TF32 关闭。Flash/native parity=`NOT_VERIFIED`，本轮不安装 Flash wheel。

默认配置和 `configs/v13l_aug_x13_01.yaml` 精确采用用户给出的 `v8_aug_x13_01` 数值，不再次乘 1.3：200 轮上限、patience50、batch16、640、workers8、GPU0、seed42、SGD、lr0=.01、momentum=.937、warmup5、cosine、nbs64；增强详见提交的 YAML。nbs 是参考 batch；原生累积、Nesterov、衰减分组和缩放、预热、调度与 EMA 保持作者 trainer 语义。完整 native args 与每个 param group、实际 warmup iterations、LR、step 尝试/成功/AMP overflow 数均归档。Adam/AdamW 允许作为新候选，其 momentum 对应 beta1，beta2=.999、eps=1e-8。相同数值不代表架构、损失实现或运行依赖完全相同。

## 配置接口和冻结

`默认配置 < --config YAML 或 --clone-config-from 冻结配方 < 重复 --set key=value`。完整和部分 YAML 均接受。严格拒绝未知键、重复 YAML/CLI 键、字符串冒充 bool/int、NaN/Inf、AutoBatch、optimizer=auto、概率越界、非法跨字段组合。JSON schema 由同一校验器生成；模型、任务、类别、COCO 初始化、native 后端、公共协议不作为调参入口。训练尺寸640、device0、seed42固定；cache/rect/multi_scale固定false，bgr/copy_paste固定0。erasing=0、auto_augment=null 是检测任务不生效的分类字段。支持实际生效的 epochs/patience/batch/workers/nbs/AMP/deterministic、优化器、损失 gain、几何/HSV/翻转/Mosaic/MixUp/CutMix 等字段，范围及约束见 `config_schema.json`。

新 run 在 trainer 创建前存输入 YAML（默认模式有显式说明）、CLI 覆盖、完整 resolved/native args、输入来源、启动 argv/Bash 命令、runtime paths、唯一 run UUID、默认/输入/CLI/结果哈希及代码/上游/补丁/数据/环境/初始化身份。`--clone-config-from` 只继承配方，生成新 UUID，仍从官方 COCO 权重初始化。候选文件改动不会影响已冻结 run。resume/finalize 拒绝参数/路径覆盖，要求原实施 SHA、内容、数据与环境；不要在训练 worktree 拉取新版或修改代码。

`resume` 仅从该 run 的 `last.pt` 恢复，校验 best/last 哈希及成对保存身份，恢复 FP32 model/optimizer/EMA、AMP scaler、scheduler、Python/NumPy/Torch CPU/CUDA RNG、best/patience、增强关闭状态、worker 共享计数与 epoch_trace。已合法完成、到 epoch 上限或 patience 的 run 拒绝续训。恢复范围是原生 epoch 边界；预取 worker 批次和跨轮未更新梯度不会逐比特回放，不宣称逐比特复现。

## 数据、增强和公共评估

沿用固定 train/val/test，6048/45573、1728/12840、864/6663（图/框），轻量身份 `3401e485b40398ae096e8fd00101cae38b607ee1d6d1b5d5abba5c443e273483`。本机清单、标签字节、图片大小和划分路径已核验。服务器启动还会重新核验实际文件。此身份不等同完整图像内容哈希；保留历史同源跨划分审计限制，本任务不重建数据，也不宣称解决同源泄漏。

训练 square stretch、框随独立 x/y gain 变换。顺序为 Mosaic→CopyPaste(p0)→RandomPerspective→MixUp→锁定检测 CutMix→NoAlbumentations→MotherHSV→上下/左右翻转→Format RGB；MotherHSV 保持 BGR→HSV→BGR、Hue 加性 LUT、S/V 乘性 LUT、sat[0]=0。CutMix adapter 消费 cutmix，字段不传入旧作者 get_cfg；beta1、3 个候选区域、主框碰撞拒绝、供体 IOA≥.1 和裁剪遵循 v8 已锁定实现；仅取 train raw partner，避免递归完整增强。transform 对象树和 worker 共享访问/触发/应用/拒绝计数归档。200 轮时零基195/第196轮同时关闭四种 mixer，重建 iterator/worker；几何、HSV、翻转继续，恢复时保持关闭状态。

训练内 val 的 batch 随该 run 的物理 batch 冻结，workers0，避免原生加倍；最佳轮和 patience 采用未舍入原生训练 val mAP50–95，相等时更新为较晚轮，另记原生 fitness。作者该提交的 fitness 权重是 P/R/AP50/AP75/mAP50–95 的 `[0,0,0,0,1]`，不可误标成 v8 的 `.1 AP50 + .9 mAP`。默认 start 只执行 preflight→train→best 独立 FP32 val→公共 CPU evaluate→summary，test=`not_requested`。finalize 仅在合法完成后复用同一个 best 与校验过的 val，再做独立 FP32 test。

公共 val/test 固定 imgsz640、batch16、workers0、seed42、conf.001、IoU.7、max_det300、max_nms30000、class-aware、rect=false、augment=false、native/FP32。原生 NMS 仅一次，关闭时间预算截断，保留完整批次；公共逆变换使用实际整数 resize/pad 和独立 x/y gain。NMS 后不裁剪/删除 padding 假阳性。记录稳定 image-id 顺序、每张图及空预测、原图浮点 xyxy。直接复用 `evaluation/evaluate.py` 的 `corrected_sorted_conf_mask_v1` 及协议哈希，原生训练指标与最终公共 P/R/AP50/AP75/mAP50–95 分别命名。

## 隔离、生命周期和结果归档

`.envs/yolov13l-configurable` 使用 venv `--copies`，可只读继承兼容母依赖；只在新 venv 装明确版本 overlay。可用 `bootstrap --fresh-torch` 新建无继承的 Torch2.2.2/TorchVision0.17.2 cu121 环境。探测兼容对后冻结实际 Python/Torch/TorchVision/NumPy/运行库、解释器、prefix、pip freeze。环境冻结后 bootstrap 不自动升级。专属 `.vendor`、`.runtime`、设置目录和每个 run 的 label/size cache；`--reuse-run` 仅复用经过检查的清单/GT，不复用标签缓存。当前母环境和其他实验不改动。

只锁本 run 的进程/会话，允许 GPU0 并行实验，不检查整卡空闲，不结束其他训练。start 自动创建 `comparison-yolov13l-<ID>`；普通 SSH 交互终端自动 attach，已在 tmux 则 switch-client，`--detach` 返回命令行。OOM/AMP/tee/信号/数据错误保留真实退出码、日志和阶段状态；不改变 batch/尺寸/AMP 后重试。status 在进程退出后识别遗留 running 状态，报告 preflight/train/public val/test 各阶段。失败的独立导出保留 partial 文件。

默认 `pack` 是只读审查包，排除 .pt，并明确不是权重备份；`--include-weights` 在合法完成后包含 best/last及哈希。收集配置、args、环境、初始化、优化器、增强、CSV/trace、状态/日志、GT/预测/公共指标、实现及公共协议和清单校验和；不装整个数据集、venv、vendor或重复缓存。训练中的审查包标记 `SNAPSHOT_INCOMPLETE_NOT_FORMAL_RESULTS`。

`archive-config` 将候选 NOT_RUN 或真实冻结配置/身份/实际 args/轻量结果导出到 `docs/comparison/archives/yolov13l`。生成文件不会自动提交推送；Git 操作步骤见交付文档。训练期间归档写入 Git 可跟踪路径会使 worktree 变脏，因此完成 finalize/pack 后再归档并提交；需要继续原 run 时回到其冻结实施 SHA。

`measure` 可计数实际 nc1 fused/unfused 的 Conv/Linear/mm/bmm，包含执行的 AAttn/HyperACE 矩阵乘法，MACs×2。报告明确不含 BN、激活、softmax/exp、池化、门控/逐元素、插值、decode/NMS 等，不将该部分统计称为完整 FLOPs。作者自动 summary 的 THOP 数字不是本工具的完整复杂度证据。未做独占硬件速度实验，并行 GPU 的时间不能作为论文速度。

## 检查与适用范围

交付证据见 `evidence/yolov13l_configurable_delivery_validation.json`。本机真实模型检查与 CUDA smoke 使用合成小预算，不能当正式200轮或真实精度。AutoDL 当前环境、实际 Linux tmux 与 POSIX 信号、正式数据 batch16/640 的训练、全量 val/test 及独占测速均须服务器执行，未执行项写 NOT_RUN。以下命令在独立环境可重验（CUDA smoke 只使用合成数据）：

```bash
PY="$PWD/.envs/yolov13l-configurable/bin/python"
bash -n scripts/autodl_yolov13l_configurable.sh
"$PY" benchmarks/comparison/yolov13l/test_configurable.py
"$PY" benchmarks/comparison/yolov13l/test_initialization_guards.py
"$PY" benchmarks/comparison/yolov13l/test_model.py
"$PY" benchmarks/comparison/yolov13l/test_b19.py
"$PY" benchmarks/comparison/yolov13l/test_lifecycle.py
"$PY" benchmarks/comparison/test_preparation.py
"$PY" benchmarks/comparison/yolov13l/smoke.py --output outputs/yolov13l-configurable-validation/cpu_recheck.json
"$PY" benchmarks/comparison/yolov13l/cuda_smoke.py --out outputs/yolov13l-configurable-validation/cuda_recheck.json
```

复用来源：v8 配置父提交 `a7b3d842df60adf28e5cf35c086c58ef752674f8`；v13 scratch 模型/后端代码 `38b49c547ad1fd93d7b85ef9917fa812b1568eeb`（未继承 random 限制、旧增强或自动 test）；同仓已验证 YOLO11m 配置生命周期辅助代码 `07bb47b`（只复用配置/生命周期/data/export helper，再按真实 v13 模型适配）；母版 `a0459d6a652cb702699087c88fa39a3e4c4087ec` 只核对数据/HSV/公共协议。详情和许可证见模块 `NOTICE.md`。
