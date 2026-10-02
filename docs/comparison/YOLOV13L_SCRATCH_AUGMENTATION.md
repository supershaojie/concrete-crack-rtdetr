# YOLOv13l scratch 增强映射

依据：母版a0459d6a652cb702699087c88fa39a3e4c4087ec，
YOLOv8m scratch参考61c386bc722b11208453cab0808d8f6edb15385d，
作者源码73289949533efac82bb5f72ec19b746618656bd2。

| 行为 | 实际路径/值 | 生命周期 |
|---|---|---|
| resize | 原生load_image(rect_mode=False)，真实框x/y缩放；640 square stretch | 全程训练 |
| Mosaic | 原生四图、dataset buffer取伙伴，p=.8 | 零基190前 |
| MixUp | 原生同pre_transform，Beta(32,32)，p=.05 | 零基190前 |
| 几何 | T×S×R×P×C；±5°、translate=.1、scale=.4、shear±1.5°、perspective±.0002 | 保留 |
| H | 母版加法LUT，H+U[-.015,.015]×180，再mod180 | 保留 |
| S/V | S×U[.5,1.5]、V×U[.65,1.35]，clip uint8，S=0保持0 | 保留 |
| 上下/左右翻转 | p=.2/.5，框同步 | 保留 |
| CopyPaste / Albumentations | p0 / 显式no-op，不随安装包改变 | 关闭 |
| bgr | Format固定0，输出RGB | 关闭随机通道交换 |
| erasing / auto_augment | 0/null，分类字段不在检测链执行 | 不适用 |
| CutMix | 固定版本无字段/操作，不向API传未知参数 | 不适用 |
| multi_scale / rect / cache | False/False/False，标签缓存仍单独隔离 | 固定 |

顺序：stretch→Mosaic→CopyPaste(0)→RandomPerspective→MixUp→
Albumentations(no-op)→加法HSV→上下翻转→左右翻转→Format。
验证/导出为640方形letterbox(auto=False)，官方坐标反变换一次。

与参考v8源码去文档/注解AST对比：
BaseMixTransform.__call__、Mosaic._mosaic4/get_indexes、MixUp._mix_transform、
RandomPerspective.affine_transform/apply_bboxes/__call__、RandomFlip.__call__相同；
box_candidates仅改成staticmethod，计算相同。几何过滤宽高>2px、面积比>.1
（以scale后原框为基准）、长宽比<100；Mosaic还去零面积框。公共GT不实施在线过滤。
跨模型随机消耗、采样和worker调度不保证逐像素一致；RT-DETR验证stretch与YOLO letterbox差异单列。

200轮时原生触发epoch==200-10==190，即第191轮关闭Mosaic/MixUp。
reset先结束旧worker并丢弃预取，再创建新transform worker；恢复超过边界也刷新。
早停不倒推关闭时间。真实2-worker测试执行官方分支：189仍组合增强，
190后三个batch无Mosaic/MixUp，HSV保留，worker PID变化；没有训练191轮。

正式recipe除model外与参考完全相等并有自动检查。其余默认值取固定官方源码。
本地默认展开见evidence/yolov13l_expanded_args.json。
实际run保存expanded_train_args.json、initialization.json、frozen_recipe.json、
actual_training_setup.json；epoch_trace.jsonl记录学习率、累积、Mosaic/MixUp阶段。
