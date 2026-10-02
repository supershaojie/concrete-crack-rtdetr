# YOLO11m scratch 接入说明

日期：2026-10-03（Asia/Shanghai）。本次只交付下一批实验代码，不代表正式结果。
参考父提交：61c386bc722b11208453cab0808d8f6edb15385d；分支：bench/yolo11m-scratch。
公共比较基点：529c456b9404f1d9ab66d82d2b2f9ec7e0c98545；母版锚点：a0459d6a652cb702699087c88fa39a3e4c4087ec。

## 模型与初始化

官方仓库：https://github.com/ultralytics/ultralytics ，v8.3.20，
commit f4d8f7765a490f3920e2d14c592a2967e347f185。
[官方说明](https://docs.ultralytics.com/models/yolo11/)；实现身份以固定提交为准。

配置直接读取 ultralytics/cfg/models/11/yolo11.yaml，字典明确设置 scale=m、nc=1，
scales.m=[0.50,1.00,512]。训练、恢复和导出均核验实际结构：
8 个 C3k2 的 M 分支使用 C3k，1 个 SPPF、1 个 C2PSA，P3/P4/P5 stride 8/16/32，
Detect.legacy=False，原生 depthwise 分类分支，reg_max=16。
nc80 的随机结构实测 20,114,688 参数，仅作官方结构交叉核验；没有加载 COCO 权重。

本次 YOLO11=**random**；既有 RT-DETR 母版=**ImageNet 骨干初始化**。
既有 scratch 和历史 COCO 结果保留各自真实条件，不把历史结果改标为随机初始化。

新 run 的 ComparisonTrainer.setup_model → get_model(weights=None) → DetectionModel 路径拒绝外部权重。
保留 BN 权重/偏置常量、eps=.001、momentum=.03、检测偏置先验和固定 DFL 投影。
初始参数 20,053,779，其中可训练参数 20,053,763；仅固定 DFL 的 16 个投影值。
initialization.json 记录零预训练迁移、官方 YAML 及哈希、尺度、参数数、seed42、
UUID/数据/源码/补丁/项目提交、完整展开训练参数及其哈希；原始记录不可覆盖。
预检独立进程，AMP 使用实际模型深拷贝，观测 FP32/FP16 输出 dtype 并恢复所有 RNG。

## 冻结配方与原生组件

完整配方在 benchmarks/comparison/yolo11m/recipe.yaml。
640、200 上限、patience50、batch16、train workers8、val workers0、GPU0、seed42、
nbs64、AMP、SGD(lr0=.01,lrf=.01,momentum=.937,weight_decay=.0005)、cosine、warmup5。
nbs64 对应正常阶段 accumulate4；warmup 累积值按官方插值实际记录。
不使用 AutoBatch、optimizer=auto、自动降 batch/尺寸/AMP 或自动恢复。

check 将官方 get_cfg 展开结果及组件来源写入 JSON；正式运行另存实际 expanded_train_args、
actual_training_setup 和逐轮 epoch_trace。损失为官方 v8DetectionLoss：
box7.5/cls0.5/dfl1.5，TaskAlignedAssigner(topk10,alpha.5,beta6)，原生 EMA(decay.9999,tau2000)
与 clip_grad_norm10。分类 auto_augment/erasing 在检测链不执行，cutmix 在此版本不适用。
增强概率、顺序及不可等价边界见 YOLO11M_AUGMENTATION.md。
零基190（第191轮）关闭真实 worker 中的 Mosaic/MixUp，重建 iterator/worker；其余增强保留。

best 和 patience 使用未舍入的训练 val mAP50–95，相等值更新到较新轮次；
另存原生 0.1 AP50 + 0.9 mAP fitness。test 只在训练正常结束后用于公共评测。

## 数据、环境与检查点隔离

源数据只读。轻量预检比对实际 YAML 三组路径、每个标签、图片/框数与既有冻结清单。
只读取 val/test 必要图像头，不做 43GB 全量图片哈希或解码审计，不转换全部 train。
首次 train 标签/尺寸缓存按需生成在本 run 的 cache 下；即使 cache=False，
仍显式隔离标签缓存及 .npy 路径。公共 GT 只有路径、尺寸和实际标签逐项一致才复用。

独立环境 .envs/yolo11m-scratch，源码 .vendor/yolo11m-scratch/ultralytics-v8.3.20，
配置 .runtime/yolo11m-scratch。首次 import 前设置 YOLO_CONFIG_DIR 并禁用集成回调和自动安装。
bootstrap 使用 venv --copies --system-site-packages；pip 前验证 sys.prefix/sys.executable，
始终保存 venv 解释器的词法路径，不对 python 路径调用 resolve。仅私有 venv 安装固定 overlay，
不安装最新版 ultralytics，不改 base Torch；不兼容时明确失败，另建环境再处理。
本地实际使用 Windows/Python3.9.25/Torch2.7.1+cu118，固定上游会给出 Python>=3.10 提示；
本次有限测试已通过，不把它声明为所有上游功能的兼容认证。

有限补丁 SHA256：461f2d4ce4ed99e691517ea316df4ebfb9a434d5f773240e6ffc9a51340ccbe1。
补丁仅为受控本 run 检查点显式 weights_only=False，以及保留带撇号的本地路径；
架构、损失、分配器未打补丁。完整补丁后源码规范化哈希：
7588e53839944a7cd34cde919744a7903343f06acb2f24655c94aa397b213dfb。

显式恢复只接受同一个 run 的 last.pt，并核验 UUID、random、模型、源码、数据、
完整配置、环境及 best/last 对应关系。保存 live model、EMA、optimizer 的 FP32 状态，
以及 scaler、scheduler、最佳轮次、stopper 和原初始化回执哈希。恢复记录单独追加。
这是原生训练算法外的序列化增强，避免丢失 scaler 或把 EMA 误当 live model；
checkpoint 占用会比原生 half/EMA-only 格式大。恢复按已完成 epoch 边界进行，
不声称重放中断时的预取 worker RNG、未完成 batch 或梯度累积。

## 导出、公共指标和资源统计

同一已选 best 分别导出 val/test：640、batch16、workers0、FP32、无 TTA、rect=False、
conf=.001、class-aware NMS iou=.7、max_det300、max_nms30000，取消 NMS 时间预算提前退出。
输出每图（包括空预测）全精度 jsonl.gz；Results.xyxy 已是原图坐标，只做一次逆变换；
类别0映射到公共category1，校验尺寸、路径顺序、完整图像集合。
调用既有 corrected_sorted_conf_mask_v1 评测器；P/R 为公共最大F1工作点。
保存 P/R/AP50/AP75/mAP50–95 的 0–1 原值与百分数。已有有效缓存优先 CPU 重算。

640、nc1、FP32 单张输入测量：

| 状态 | 参数 | THOP MACs | 补计注意力矩阵 MACs | MACs×2 GFLOPs |
|---|---:|---:|---:|---:|
| unfused | 20,053,779 | 34,094,310,400 | 61,440,000 | 68.3115008 |
| fused | 20,030,803 | 33,824,998,400 | 61,440,000 | 67.7728768 |

THOP 常规卷积/BN 等计数加上实际 Attention QK/AV 矩阵乘法。
softmax、SiLU/sigmoid、残差加法、attention 缩放、decode/NMS 未完整计入，
所以这是明确边界的运算量口径，不能称为每个标量操作的完整 FLOPs。
与 nc80 官方参考表、融合状态和只用 THOP 的数值差异均可解释。
speed.py 为以后独占硬件下的可选 FP32、batch1 前向计时入口，不是训练门槛，
也不把并行训练时的速度当论文速度。

## 验证证据与待验证项

机器回执位于 docs/comparison/evidence/yolo11m_scratch_validation.json，
展开参数位于 yolo11m_expanded_args.json。
实际数据轻量检查：train6048/45573、val1728/12840、test864/6663（图/框），冻结身份一致。
CPU 覆盖真实双 worker 增强切换、缓存隔离、stretch 框、HSV、
letterbox 逆变换、空预测公共评测及原始文件不变。
本机 RTX2060 上真实合成训练 batch2/64：第1轮检查点后中断，显式恢复到第2轮，
逐项对照 live model/EMA/scaler/stopper，原初始化记录不变；真实 best 执行 FP32/640 导出和公共评测。
另一次 batch1/640 AMP 前反向通过。合成模型、分数和权重仅为 smoke，均留在 ignored outputs。

Windows 已验证真实子进程/tee 退出码、Bash 会话保护及 tmux 调用顺序和实际 window ID。
真实 Linux tmux、POSIX 进程组信号和 Linux --copies 子进程执行仍待服务器验证；
本次没有登录服务器。正式 batch16/640、两模型 GPU0 并行容量、正式200轮和全量 val/test 推理
未执行。用户自行安排下一批启动时间；不会等待、停止或修改当前 v5/v8 实验。
