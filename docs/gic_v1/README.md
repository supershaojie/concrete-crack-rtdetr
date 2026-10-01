# GIC v1：几何区间校准

母版固定为 `a0459d6a652cb702699087c88fa39a3e4c4087ec`，实验分支为 `exp-rtdetr-r18-lite-gic-v1`。模型保留 RT-DETR-R18-Lite＋原 CBR＋原 LIF-Down v1，300 ordinary queries、3 decoder layers，单类未融合参数量 20,149,765。只增加损失计算，无新可训练参数。原 CBR、LIF、decoder、YAML、VFL 实现及原训练器源码逐字节（规范化 LF）核对母版。

## 损失与接线

`models/utils/loss.py` 的最终 ordinary 调用显式携带 `final_normal_context`，auxiliary 与 DN 调用不携带。`_get_loss` 保留原 Hungarian 匹配和原 q 计算，在原分类 loss 后传递同一次匹配的索引、GT、最终 CBR 框和 VFL q。不能用空 postfix、query 数或 DN 元数据猜测范围。

`models/rtdetr/gic_loss.py` 仅在最终普通匹配 query 的真实类别 logit 上加：

```
delta = eta_eff * q * ((q - t) * z + H(q) - H(t))
loss_class += sum(delta) / max(M, 1) * class_gain
t = stop_gradient(clip(stop_gradient(sigmoid(z)), lo, hi))
```

M 是母版 `_get_loss` 实際传入 `_get_loss_class` 的匹配 GT 数。原 VFL `mean(query).sum()` 再除 `M/nq`，因此差量只除 M，class gain 只乘一次（本母版为 1）。总损失字典键完全不增加；`sum(loss.values())` 正确计入一次差量，日志三项照旧。原 bbox/giou、负样本 `p^gamma` 调制梯度、其他类别、encoder/decoder auxiliary 和 DN 完整继承。原 VFL 构造 alpha=0.25、gamma=1.5；Hungarian 成本 class=2、bbox=5、giou=2，原 bbox/giou gains=5/2。

9 个中心平移取 dx,dy ∈ {-1,0,1} 像素，分别除实际输入 W/H；宽高不变，不裁剪边缘。区间同时显式包含原 q。沿用母版 `bbox_iou(xywh=True, eps=1e-7)`，不使用 GIoU。新增运算在 autocast 外 FP32；q/lo/hi/t 和假想几何全部 detach，z.float() 保留梯度。非法框、概率或非有限值报错，不用 abs/clamp 修补。

`shift_px=1.0`、`eta_max=0.5`；`eta_eff=0.5*clip((epoch-4)/15,0,1)`：e=0/4 为0，e=5为1/30，e=19/20为0.5。`GICTrainer._model_train` 和批次回调取真实 `trainer.epoch`，resume 不重置渐入。`eta=0` 或无 GT 直接调用母版；无额外 sigmoid/IoU/熵图，也不消耗 RNG。单点区间数值及 logit 梯度回到原正项。

GIC 配置保存在模型 `gic_config` 属性并由 `tasks.py:init_criterion` 显式传给 criterion；不向 Ultralytics train args 塞未知键。checkpoint 依赖的类位于可导入模块中。Native model、预测和推理路径保持原样。

## 配方与初始化

完整 **109 字段**见 [resolved_formal_config.yaml](resolved_formal_config.yaml)，新损失另存 [gic_config.json](gic_config.json)。真实母版归档 [mother_actual_args.yaml](mother_actual_args.yaml) 来自 `D:/rtdetr跑结果/C19＋LIF：原 CBR＋LIF-Down v1/training/args.yaml`，与母版提交中的完整 args 唯一差别是已经存在的 `-gatefix` 初始化路径别名，数值/类型完全一致。服务器 preflight 再次核对并输出完整解析配置及差异；不依赖当前库默认值补齐。

- 200 epochs、patience50、640/B16、nbs64、seed42、workers8、GPU0。
- AdamW，lr0=0.0005，lrf=0.01，weight_decay=0.0001，warmup_epochs=5，cos_lr/AMP/deterministic=True。momentum=0.937，warmup_momentum=0.8，warmup_bias_lr=0.1。
- 在线增强：hsv=(0.015,0.5,0.35)，degrees5，translate0.1，scale0.4，shear1.5，perspective0.0002，flipud0.2/fliplr0.5，mosaic0.8，mixup0.05，cutmix/copy_paste/erasing=0；其余字段完整保留。
- 公共源 `/root/autodl-tmp/projects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt`，SHA256 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
- 原 `init_c19_lif_v1.controlled_models/initialize/build_training_model` 执行相同构建顺序、源 nc80 严格加载和 native nc1 类别头适配。共同 533 个源 state、原 CBR/LIF 增量19个 state，nc1加载543/552，9个形状变化仅来自合法类别适配；完整 key 名及允许缺失说明在初始化/加载报告内。没有从 mother best 微调。
- 数据 YAML 取主仓库真实 `configs/crack_autodl.yaml`；6048/45573 train，1728/12840 val，864/6663 test。preflight 记录内容哈希、拆分清单和标签；后续复用检查元数据与 YAML，status/pack 不扫描数据。

正式输出固定为主仓库绝对路径 `runs/c_series/gic_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`，证据在实验 worktree `outputs/gic_v1`。不降低 batch、不换卡、不关闭 AMP、不要求 GPU 空闲。锁只保护本实验。原训练器 OOM 缩批重试被显式关闭，真实 OOM 保留后退出。

## 验证范围与诊断

见 [VALIDATION.md](VALIDATION.md) 和 reports 下真实输出。当前交付包含8项数学/接线测试、9项配置/评估/生命周期测试。逐项对比直接从指定母版 Git blob 加载的 criterion，检查输出 loss、独立输出张量梯度、零系数 RNG、矩形/边界/极端值、数值有限差分、总 loss 单次接线、序列化、同权重推理及结构。

在线每100个批次及每个 epoch 首批记录 detached 聚合 JSONL：匹配数，q/lo/hi/p，区间宽度分位数，区间内比例，GT短边组(<4/4–16/16–32/≥32输入像素)，delta，原/新正项梯度绝对值、实际减弱数量与比例。原梯度绝对值≤1e-8的样本不算梯度比并记录数量。没有跨批次图、匹配或特征缓存。

母版已有全 query val 导出的固定前64张，仅离线诊断，不重新推理、不读取 test 导出、不调 eta/shift。213个匹配中194个区间非零、168个差量非零且预计梯度减弱，区间宽度中位数0.011863；平均逐样本梯度减弱比例约12.45%。这是由导出几何与逆 sigmoid 重建的机制估计，不是训练批次或收益结论。样本短边覆盖有限（<4像素0个、4–16像素1个）；不据此推广小框收益。

固定几何与 q>0 下，新正项最优 p 仍为 q；梯度方向保持、幅度在原来的[1-eta,1]内。这不说明整网梯度不变、总体平衡不变或 AP 会提高。分类学习可能变慢；正项 loss 变小不能作为性能提升证据。该区间表示局部几何敏感性，不是标注噪声的统计置信区间。不对公式作论文归属或全球首创断言。

## 服务器流程与评估包

直接命令见根目录 [server_commands_gic_v1.md](../../server_commands_gic_v1.md)。`preflight` 一次总时限900秒，包含配置/初始化/单测及一个固定真实 B16 批次的最多两个临时优化步（FP32和AMP分别新建模型）。为隔离动态 GradScaler 初始溢出，AMP 数值冒烟用单位缩放；正式 GICTrainer 使用原生 scaler。预检不触发200轮训练或正式 val/test；成功记录按代码、配方、初始化、数据和环境身份复用。

`start` 在 `gic-v1-training` 启动；`finish` 必须训练与Python/tee成功后才在 `gic-v1-finish` 执行。pane 内在启动 Python 之前设置自身 remain-on-exit；不改 tmux 全局设置、不 kill session。后续明确请求的 finish/resume 在已退出会话新建 window，旧页面保留。`status` 区分进程运行、结束成功、失败、未启动及中断，读取退出码和日志尾部；`attach` 进入训练会话。

训练每轮原 val fitness/early stopping 不变。训练结束沿母版 strip_optimizer 保存固定 best；独立 FP32 评估移到 finish，避免训练结束额外重复一次无导出的正式 val。finish 首次 val/test前向同步导出300个普通query及所有GT、原始query索引、输入/原图尺寸和 stretch逆变换，以及原评估器TP矩阵/未舍入指标/曲线。协议为 `corrected_sorted_conf_mask_v1`，FP32/640/B16/workers0/conf0.001/iou0.7/max_det300/augment=False/rect=False/seed42。

复用要求 best哈希、完整配方/数据身份、评估参数、split和评估器源码版本一致，逐项校验产物哈希。已有成功评估缺导出时明确报缺项，禁止静默重跑整套推理。失败步骤可以补做；status/pack永不推理。P/R/F1以原报告最大F1点展示（test的该点只用于报告）；另外在val选阈值并固定到test的离线错误分析中绝不使用test选阈值。各短边组输出基于原检测贪心匹配的Recall/TP/FN与匹配分数；损失诊断的Hungarian不用于AP或检测TP。

`pack` 只消费已有证据，缺项标PARTIAL。完整包名 `GIC_v1_COMPLETE_<UTC时间>.tar.gz`，含best.pt、source snapshot/patch、完整args、初始化加载报告、测试、训练日志/曲线、正式val/test指标、所有普通query/GT导出、诊断和逐文件MANIFEST；全包SHA256并逐成员读回复核。同身份完整包可复用。包和权重留在服务器，不自动下载，不因指标低而删除。

## 文件与复用来源

- 核心：`ultralytics-main/ultralytics/models/rtdetr/gic_loss.py`，对 `models/utils/loss.py`、`nn/tasks.py` 的显式作用域/criterion接线。
- 工程：`tools/gic_v1{.sh,.py,_common.py,_training.py,_eval.py,_analysis.py,_pack.py,_lock.py,_report.py}`。
- 检查：`tools/check_gic_v1.py`、`tools/check_gic_v1_ops.py`。
- 完整配置、报告、来源核对在本目录；服务器命令在仓库根目录。

母版初始化与受核验的排序后置信度mask协议直接沿用母版；`gic_v1_common.py` 的数据/配方身份设施、`gic_v1_eval.py` 的query ID导出和离线阈值复核、`gic_v1_lock.py` 的进程/内核锁实现仅取自 GNR 分支提交 `649a711ce3604066ff772599c2353e47f88a2268` 的对应工具并作上述适配。没有复制该分支的loss、model、training或整支cherry-pick，也未引入GPC/CEA/ARG/GEO/QCC/GNR机制。复用的评估postprocess用母版函数独立对照测试。详细来源见 reports/mother_contract.json。

正式finish还生成纯离线 `outputs/gic_v1/metrics.html` 指标页面，显示val/test、all/crack、全IoU AP、best SHA与协议；不会触发推理或自动发布网站。
