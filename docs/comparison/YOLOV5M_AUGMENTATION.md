# YOLOv5m 冻结增强与训练配方

2026-10-02（北京时间）。本约定在观察本模型正式 val/test 之前制定。

依据是 `evidence/mother_args.yaml`（SHA256 `31029c8d3bcf0e2dc3cb7107fa7db316f37cd954eba21bf4fc854e2070fc476b`）、公共 `configs/mother_augmentation.yaml` 和母版固定源码 `a0459d6a652cb702699087c88fa39a3e4c4087ec`。YOLOv5 是官方 `915bbf294bb74c859f0b41f1c23bc395014ea679` 加本分支可重放 `upstream.patch`。表内上游位置以函数名定位，补丁后的行号会变化。

| 项目 | 母版依据/实际行为 | 本轮 YOLOv5 实现、值和调度 | 保留的例外 |
|---|---|---|---|
| 输入 resize | RTDETRDataset 先 stretch 到正方形 | 原生 LoadImagesAndLabels.load_image 保持比例，后续 letterbox；640 | 模型必要输入差异，不宣称逐像素相同 |
| Mosaic | augment.py Mosaic，四图 p=0.8、随机中心、buffer 伙伴 | 原生 load_mosaic 四图、p=0.8；index190 起关闭 | YOLOv5 从全 train 列表采伙伴，tile resize/裁剪和 buffer 抽样不同 |
| MixUp | 独立 p=0.05，Beta(32,32)，伙伴同一 pre_transform | 将原来仅 Mosaic 内的 MixUp 移至预变换后；整体 p=0.05，伙伴独立走同一 Mosaic/普通图几何链；Beta(32,32) | 不把原生 0.8×0.05=0.04 冒充总体 0.05；继承上述 resize/伙伴抽样差异 |
| 几何 | RandomPerspective：T×S×R×P×C | 原生 random_perspective；degrees=5、translate=.1、scale=.4、shear=1.5、perspective=.0002 | 相同分布/矩阵次序，输入 letterbox 和 Mosaic 几何仍不同 |
| HSV | RandomHSV 加法 hue、S/V 乘法 LUT | augment_hsv hue 改为 U[-.015,.015]×180 加法；S=.5、V=.35，uint8 LUT | 数值实现可能有浮点舍入差异；S=0 仍映射到0 |
| 翻转 | RandomFlip 上下 .2、左右 .5 | 原生 Bernoulli(.2)/Bernoulli(.5)，HSV 后先上下再左右 | 无新增翻转操作 |
| 框过滤 | 几何后裁剪；w/h>2、面积比>.10、长宽比<100 | 原生 random_perspective/box_candidates 同阈值；Mosaic 裁剪 | Mosaic 的中间过滤步骤不完全相同；YOLO 最后 xyxy→归一化 xywh 有 eps=1e-3 裁剪 |
| 第191轮 | trainer index=200-10=190；关闭 Mosaic/MixUp/CopyPaste/CutMix | bench.close_augmentation 在生成该轮 batch 前关闭，销毁旧 InfiniteDataLoader iterator/worker 并重建，沿用 dataset/cache/sampler/generator | 仅切换一次；不重建标签缓存；提前结束则不会强行补十轮 |
| 关闭后 | HSV、几何和翻转继续 | 保持这些 hyp 与 augment=True，只关闭组合增强 | 双 worker 实际 batch 已验证，不仅检查字典 |
| 隐式 Albumentations | 源码条件性 Blur/MedianBlur/ToGray/CLAHE；历史 freeze 无包，但历史激活未直接证实 | 本轮预先选择明确禁用，与是否安装该包无关 | 不用当前环境证明历史状态 |
| 其他字段 | CopyPaste/CutMix=0；erasing/auto_augment 属分类；bgr 未在母版检测链启用 | copy_paste=0；其余无额外实现；输出 RGB | 不因字段同名额外启用分类增强 |
| 额外机制 | 无 TTA/multiscale/quad/image weighting | 全部关闭，cache_images=false，rect_train=false，freeze=[0] 表示不冻结 | 保留 YOLOv5 原始模型/损失/分配/EMA/梯度裁剪/AutoAnchor |

完整展开 `hyp.yaml` 的 loss/anchor 参数源自固定版本 `data/hyps/hyp.scratch-med.yaml`，是官方 medium 配方；优化器/增强覆盖是本项目预先约定，不声称都是上游默认。box=.05、cls=.3、obj=.7、cls_pw=obj_pw=1、anchor_t=4、iou_t=.2、fl_gamma=0，label_smoothing=0。在 nc=1、nl=3、640 时上游训练器实际 cls gain=.00375，box/obj 不另变；单类官方损失不计算类别 BCE 分支。损失和标签分配算法未修改。

实际 batch=16；nbs=64；正常阶段 accumulate=4。warmup_steps=max(round(5×nb),100)，梯度累积由1向4线性插值并 round。weight_decay 按 16×4/64 缩放，仍为 .0005；bias/归一化组不加 weight decay。SGD momentum=.937、Nesterov 沿用官方；lr0=.01、lrf=.1、cos_lr=True。余弦乘子 f(e)=.1+.9×(1+cos(pi×e/200))/2，在 epoch=200 的调度边界为 .1；最后实际执行的零基199轮与该边界有差异。warmup bias lr=.1、其他组从0起；momentum 从.8向.937。逐 epoch 乘子与实际参数组写入 effective_training.json，原生 CSV 保存实际学习率。

seed=42，init_seeds(..., deterministic=True)。保留上游 loader generator seed=6148914691236517205+RANK（单进程 RANK=-1），worker_init_fn 从 torch.initial_seed 派生 NumPy/Python seed；不声称母版和 YOLOv5 具有完全相同随机流。AMP 在正式 CUDA 训练中明确开启，删除会下载额外模型的上游 check_amp 探测；不静默退为另一正式配方。CPU 合成检查 AMP=false，不能作为 CUDA AMP 已通过的证据。

best 和 patience 使用原生训练验证 mAP50–95，fitness 权重由上游 [0,0,.1,.9] 改为 [0,0,0,1]；相等时保存最近一轮并重置 patience，与上游 ties 规则一致。母版同样以 mAP50–95 选择，母版历史 validator 差异按公共报告保留，绝不重选母版 best。YOLO 原生训练验证仍按 v7.0 rect=True、batch=32、最大 workers=16、NMS IoU=.6、AMP 时 FP16 验证，原生日志与最终独立指标分开命名。max epochs=200，patience=50；每轮完成后记录 epoch_state.json，正常结束再写 training_complete.json。

AutoAnchor 仅读取 train loader 的标签/宽高，保留官方阈值及必要估计行为；运行时保存前后 anchors、stride、changed。当前没有针对正式数据运行 AutoAnchor，不能声称 anchors 已经/没有重新估计。
