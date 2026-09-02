# OBP-AIFI 来源与结构改造记录

## 实验边界

OBP-AIFI（Orthogonal-Branch Perception AIFI，正交分支感知 AIFI）是本项目针对混凝土裂缝检测重新设计的独立消融模块。它只替换 RT-DETR-R18-Lite Hybrid Encoder 中处理 S5/32 特征的 AIFI，不修改 Backbone、其余 Neck、CCFF、Decoder 或检测头，也不包含 CSCEF 或 SCPD。

## 原始 AIFI 来源

- 项目实现：`ultralytics-main/ultralytics/nn/modules/transformer.py` 中的 `AIFI` 与 `TransformerEncoderLayer`。
- 基线配置：`ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite.yaml` 第 9 层（从 0 计数）。第 7 层产生 512 通道 S5/32，经过第 8 层 1×1 投影后，AIFI 接收并输出 256 通道特征；其普通 FFN 隐藏维度为 1024，注意力头数为 8。
- OBP-AIFI 保留的内容：S5/32 插入位置、动态二维正弦余弦位置编码、全局多头自注意力、注意力残差与 `LayerNorm`、第二级残差与 `LayerNorm`，以及输入输出空间尺寸和通道数不变的接口。
- OBP-AIFI 替换的内容：原 AIFI 的两层逐 token 线性 FFN 被方向感知的卷积混合器替换；原线性 FFN 不会与新混合器重复保留。

## 条带卷积思想来源

- 思想参考：[Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection](https://arxiv.org/abs/2501.03775) 及其[公开实现](https://github.com/YXB-NKU/Strip-R-CNN)。该工作使用正交条带卷积建模细长目标的各向异性空间信息。
- 本次开始实现前，在固定基线 `cab1297` 的整个项目中检索了 `Strip_Block`、`StripMlp` 和 `C2f_Strip`，均未找到；任务提示所述 `RTDETR-main/ultralytics/nn/extra_modules/block.py` 不存在于当前项目。因此没有从用户模块包复制代码或结构。
- 被参考的仅是“窄而长的水平/垂直卷积可显式聚合方向上下文”这一思想。OBP-AIFI 不是 Strip R-CNN 模块的直接移植，也不能在论文或实验报告中描述成直接使用了 Strip R-CNN 模块。

## 本实验重新设计的结构

给定 `B×C×H×W` 的 S5/32 特征，OBP-AIFI 先执行与原 AIFI 等价的带二维位置编码的全局自注意力。随后将注意力输出恢复为空间特征，并送入 `AdaptiveOrthogonalMixer`：

1. 通过 1×1 卷积和 GELU 将 `C=256` 投影到基线 FFN 隐藏维度 `cm=1024`。
2. 让同一个隐藏特征同时进入三个并行深度卷积分支：1×7 水平分支、7×1 垂直分支和 3×3 局部分支。这里不是 Strip R-CNN 的串行正交结构。
3. 从隐藏特征经全局平均池化、1×1 降维、GELU 和 1×1 三通道投影，产生每个样本的三个标量 logits；在分支维度以 FP32 Softmax 得到可广播权重，三者之和为 1。
4. 对三个分支作加权求和，不使用通道拼接；经 GELU、Dropout 和 1×1 卷积恢复到 `C=256`。
5. 以逐通道可学习参数 `gamma` 缩放混合器输出，再进入第二残差与 `LayerNorm`。`gamma` 严格零初始化，使模块刚初始化时不会突然放大新方向分支，训练后再逐步引入其贡献。

门控最终层权重和偏置均初始化为零，因此三个分支的初始 Softmax 权重为 1/3。卷积核长度选择 7：R18-Lite 在 640 输入下的 S5/32 通常为 20×20，长度 7 能提供明显方向上下文，同时避免核尺寸接近整幅高层特征图造成额外开销或过度平滑。实现动态读取 H、W，也支持非方形特征图。

## 与混凝土裂缝形态的关系

- 原 AIFI 的全局自注意力继续负责远距离语义联系和裂缝连续性。
- 1×7 与 7×1 深度卷积显式聚合水平、垂直方向上下文。
- 3×3 深度卷积为弯曲、交叉及短局部裂缝提供局部二维感受野。
- 样本级三路 Softmax 门控允许不同图像根据方向构成调整三类特征的相对权重，又避免生成高成本的空间注意力图。
- 三路深度卷积只在共享的 `cm` 隐藏特征上运行，融合采用求和而非 Concat；因此该模块仍以轻量开销增强 S5/32 高层语义，而不承担浅层边缘提取职责。

以上是结构动机与设计假设，不代表已经获得训练精度收益；是否提升裂缝检测指标必须由后续独立训练与消融实验验证。
