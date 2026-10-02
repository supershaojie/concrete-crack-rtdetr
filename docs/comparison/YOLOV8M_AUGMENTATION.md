# YOLOv8m 在线增强冻结表

冻结日期：2026-10-02（北京时间）。母版依据是 `docs/comparison/evidence/mother_args.yaml`、
`benchmarks/comparison/configs/mother_augmentation.yaml` 和母版提交
`a0459d6a652cb702699087c88fa39a3e4c4087ec` 的实际源码；不从 test 的默认设置反推。
实际上游是官方 v8.3.20 / `f4d8f7765a490f3920e2d14c592a2967e347f185`。

## 活跃增强与调度

| 行为 | 母版依据 | YOLOv8m 本轮行为 | 数值/关闭时间 |
|---|---|---|---|
| 输入 resize | `models/rtdetr/val.py:RTDETRDataset.load_image`，训练square stretch | `IsolatedDataset.load_image`训练调用官方base的`rect_mode=False`，`v8_transforms(stretch=True)` | 640×640、OpenCV linear；训练不额外letterbox |
| Mosaic | 四图，使用dataset buffer取伙伴 | 官方`Mosaic`四图实现；cache=False时取buffer，与母版活跃路径一致 | p=.8；随机中心；零基190关闭 |
| MixUp | 相同预变换的伙伴图，Beta(32,32) | 官方`MixUp`及相同pre_transform | p=.05；零基190关闭 |
| 旋转 | `RandomPerspective.affine_transform` | 相同公式 | U[-5°,5°]；继续到结束 |
| 平移 | 同上 | T×S×R×P×C，相同中心平移 | 每轴U[.4,.6]×输出宽高；继续 |
| 几何scale | 同上 | 相同公式，非multi_scale输入抖动 | U[.6,1.4]；继续 |
| shear | 同上 | 每轴tan(均匀角度) | U[-1.5°,1.5°]；继续 |
| perspective | 同上 | 相同投影变换 | 每轴U[-.0002,.0002]；继续 |
| Hue | 母版新HSV是加法偏移 | `MotherHSV`移植母版加法LUT；官方v8.3.20原本为乘法Hue | U[-.015,.015]×180，mod180；继续 |
| Saturation | 母版乘法LUT，S=0保持0 | 相同LUT及uint8裁剪 | U[.5,1.5]；继续 |
| Value | 母版乘法LUT | 相同LUT及uint8裁剪 | U[.65,1.35]；继续 |
| 上下/左右翻转 | 母版`RandomFlip` | 相同box-only三通道活跃路径 | p=.2/.5；继续 |
| CopyPaste | p=0，检测无分割 | p=0，不加入分割增强 | close时也置0 |
| CutMix | p=0 | v8.3.20没有该字段/操作，不向API传入 | 不适用；未额外实现 |
| Albumentations | 历史freeze未出现该包；历史import状态未直接证明 | 显式no-op，即使环境装有该包也不激活 | 不启用Blur/MedianBlur/ToGray/CLAHE |
| bgr | 母版RTDETR的Format默认0 | Format固定bgr0，输出RGB | 0 |
| erasing / auto_augment | 分类字段，未在母版检测链执行 | 配置0/null，检测链不执行 | 不适用 |
| multi_scale | 母版未启用输入尺寸抖动 | false；不改变训练640 | 不启用 |

实际顺序：square stretch → Mosaic → CopyPaste(0) → RandomPerspective → MixUp →
Albumentations(no-op) → additive HSV → vertical flip → horizontal flip → Format。
未添加任何改变backbone、neck、检测头、DFL、损失或标签分配器的模块。

源码函数体比较（去除文档与类型注解）确认：`BaseMixTransform.__call__`、`Mosaic._mosaic4`、
`MixUp._mix_transform`、`RandomPerspective.apply_bboxes/__call__/box_candidates`活跃代码一致。
Mosaic空列表判断仅写法不同；母版几何路径补充灰度维度处理，翻转补充keypoint处理，均不影响本次OpenCV三通道水平框。
过滤规则相同：变换后clip，宽高>2px、面积比>.10（相对经scale后的原框）、长宽比<100；Mosaic还去零面积框。
这些是在线增强的过滤；独立GT转换不裁剪、不修复、不丢框。

## 第191轮的真实worker切换

固定200轮时，官方`BaseTrainer._do_train`在`epoch == epochs-close_mosaic == 190`进入关闭分支。
官方`YOLODataset.close_mosaic`清零Mosaic/MixUp/CopyPaste并重建transform；本版没有CutMix。
适配loader reset先关闭旧worker并丢弃prefetch，再建立包含新transform的worker副本。
resume起点已经超过190时同样刷新worker；不因提前停止把关闭时间挪到别处。

本地有限测试使用2个真实worker、8张合成训练图、64×64、batch2，并将测试专用Mosaic/MixUp概率设为1以确定观察应用行为。
执行的是固定官方源码中的实际关闭条件：零基189的batch仍实际调用两项组合增强；零基190后连续3个batch均无组合增强调用，
HSV仍调用，worker PID已换。测试没有执行191轮训练；测试概率和尺寸不写入正式recipe。

## 预先登记的边界

训练活动变换按上述有限适配对齐，但不声称相同seed产生逐图逐像素完全相同的随机轨迹。新旧pipeline中禁用节点的随机数消耗、
采样器及worker调度可不同。Albumentations的历史激活未直接证明，本轮预先冻结为关闭，不依据分数改变。

训练内验证和最终独立预测保留YOLO的等比resize/letterbox；母版RT-DETR验证是stretch。这一差异单列，不伪装成完全相同的前处理。
最终预测强制`auto=False`、640方形padding，原生`scale_boxes`已恢复并clip原图坐标，导出直接读取`Results.boxes.xyxy`，不再次逆变换。
已用非正方形合成图及已知框验证该映射；空预测图同样进入公共schema。

YOLO保留class-aware NMS，conf>.001、NMS IoU=.7、max_det300，未给RT-DETR加NMS。
训练val和最终预测仅取消官方NMS的时间预算提前退出，避免共享GPU时把后续图片变成空结果；框排序、IoU规则和max_det不变。
最终AP仍由公共评测器按0.50:0.05:0.95计算，.7不是AP单一阈值。
