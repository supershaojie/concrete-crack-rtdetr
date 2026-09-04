# GSDR-AIFI 来源追踪

## 参考思想

### DAT

- 论文：[Vision Transformer with Deformable Attention](https://arxiv.org/abs/2201.00520)。
- 本项目借鉴的思想：由内容预测有限范围的采样偏移，通过双线性采样取得稀疏 K/V，再让 Query 与这些采样特征建立关系。
- 未复制的内容：没有移植 DAT 的 stage/block、完整 DAttention 类、位置编码实现、分层 backbone 或任何第三方训练配置。

### CrossFormer Dynamic Position Bias

- 论文：[CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention](https://arxiv.org/abs/2108.00154)。
- 本项目借鉴的思想：用小型 MLP 从相对坐标连续生成位置偏置，避免固定尺寸的位置 bias table。
- 本项目差异：输入是 AIFI 完整 Query 坐标与四组实际变形采样坐标之间的连续位移；每组使用独立 `2→32→1` MLP，最后层零初始化，并服务于单尺度稀疏 K/V 关系。

### RT-DETR Hybrid Encoder

- 论文：[DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)。
- 项目基线：`ultralytics/nn/modules/transformer.py` 中的 `AIFI`，以及 `rtdetr-resnet18-lite.yaml` 第 9 层。
- 本项目遵循的边界：AIFI 仍在 S5/32 投影后的 256 通道特征上执行全局 self-attention；GSDR 只作为同一注意力残差阶段的补充关系旁路，不替换原 MHA 或 FFN，也不改变后续 CCFF 与 Decoder。

## 本仓库机制追踪

实现前检查了准确 C2 提交 `67c3078e54a657fd96d65fee657a75fbb1dae0d6` 的：

- `ultralytics/nn/modules/transformer.py`：确认 `AIFI` 继承 `TransformerEncoderLayer`，以及 post-norm/pre-norm 的真实残差次序；
- `ultralytics/nn/modules/__init__.py` 和 `ultralytics/nn/tasks.py`：确认模块公开导出和 parser 自动注入 `ch[f]` 的方式；
- `rtdetr-resnet18-lite.yaml`：确认 AIFI 实际输入为 256 通道，FFN 为 1024，MHA 为 8 heads，Decoder 输入为 `[19, 22, 25]`；
- baseline ImageNet backbone 初始化脚本，以及 CSCEF-v3/v4、OBP-AIFI 分支中的受控映射、审计和 unittest 风格。

在准确基线工作树中检索了 `DAttention`、`DynamicPositionBias`、`DPB`、`offset_range_factor` 等标识，没有找到用户旧模块包或可直接复用的 DAttention/DPB 实现。仓库已有的 `MSDeformAttn` 属于 RT-DETR Decoder 的多尺度可变形注意力，接口和用途不同，未被拼接到 GSDR-AIFI。

## 代码来源声明

`ultralytics/nn/modules/gsdr_aifi.py` 是依据本项目当前 AIFI 接口重新实现的代码。没有复制第三方 `extra_modules/transformer.py`，没有移植旧 Ultralytics 8.0.201 类，也没有引入 DAT、CrossFormer 或其他仓库的源码文件。引用论文只用于说明机制思想来源，不能将“可变形采样”或“动态位置偏置”表述为本项目完全原创。

## 与参考工作的具体区别

- 不是新的分层视觉 Transformer backbone，而是 RT-DETR AIFI 内的一条 128 通道辅助关系旁路；
- 原 256 通道、8-head 全局 MHA 继续执行，稀疏关系不会取代全局关系；
- Query 保留全部 `H×W` token，只有 K/V 使用 stride-2 的动态参考网格；
- offset 分为 4 组，连续位移 bias 也按组计算并映射到对应 heads；
- 原 `fc1→GELU→fc2` FFN 以及两个 LayerNorm/残差/dropout 语义完整保留；
- 最终稀疏输出投影为零，使加载 baseline 后严格等价，而不是让新旁路在实验起点扰动基线；
- 不依赖自定义 CUDA、DCNv3、Mamba、timm、einops 或额外 pip 包。

## 实验表述约束

GSDR-AIFI 的研究假设是：完整全局关系负责远距离语义联系，较低成本的内容自适应稀疏 K/V 为裂缝形态提供位置可变的补充关系，连续位移 bias 描述 Query 与实际采样点的几何关系。这只是结构动机。未经正式训练和独立 test 评估，不声称已经提升 Precision、Recall 或 mAP；后续实验只比较 C2 与单一 GSDR-AIFI 变量。
