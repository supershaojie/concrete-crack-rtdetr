# GSDR-AIFI 设计记录

## 实验边界

GSDR-AIFI（Global–Sparse Deformable Relation AIFI，全局—稀疏可变形关系 AIFI）只替换
RT-DETR-R18-Lite Hybrid Encoder 中第 9 层的 `AIFI` 类名。Backbone、输入投影、CCFF/FPN/PAN、Decoder、类别数和训练超参数均不改变。本轮只实现、初始化和验证代码，不执行正式训练、验证集或测试集评估；是否提升精度必须由后续单变量实验回答。

## 为什么停止 OBP-AIFI 路线

C12 OBP-AIFI 用 1×7 水平、7×1 垂直和 3×3 局部分支替换了原 AIFI 的逐 token FFN，并以样本级 Softmax 选择三路方向。其 test mAP50-95 约为 0.443641，低于 C2 baseline 约 2.60 个百分点；已有结果还显示方向权重偏向垂直分支，Precision、大目标召回和总体定位质量下降，假阳性增加。更关键的是，它删除了 `fc1→GELU→fc2`，且零初始化不等价于 baseline。GSDR-AIFI 因此不修补条带卷积路线，也不包含门控、方向分支或 learnable alpha。

## 实际数据流

输入为动态尺寸 `x: B×256×H×W`。原 AIFI 主路径完全保留：

1. 生成动态 `H×W` 二维正弦—余弦位置编码；
2. 用原 256 通道、8 heads 的 `nn.MultiheadAttention` 计算全局稠密关系；
3. 在同一注意力残差阶段计算独立的稀疏旁路；
4. 保留原 `norm1`；
5. 保留原 `fc1(256→1024)→GELU→dropout→fc2(1024→256)`；
6. 保留原第二残差、`dropout2` 和 `norm2`。

稀疏旁路为：

```text
x (B×256×H×W)
  → 1×1 input projection (128 channels)
  ├─→ dense Q projection at H×W
  ├─→ dense K/V projections, then group-wise grid_sample at ceil(H/2)×ceil(W/2)
  └─→ depthwise 3×3 stride-2 offset predictor (4 groups)
         → GroupNorm → GELU → grouped 1×1 (dx, dy per group)

QKᵀ / sqrt(d) + MLP(query_position - deformed_sample_position)
  → FP32 softmax → weighted sparse V
  → 1×1 output projection (128→256, zero initialized)
  → add beside dense attention residual
```

在 20×20 输入下，Query 数为 400，稀疏 K/V 位置数为 10×10=100。Query 不降采样。K/V 的 128 个通道被分成 4 组，每组按自身变形网格进行双线性采样。默认 4 个稀疏 attention heads 与 4 个 offset groups 一一对应。

## 坐标、偏移与连续位移

参考点采用 `grid_sample(..., align_corners=False)` 的像素中心约定，坐标顺序严格为 `(x, y)`。尺寸为 `Hs×Ws` 的参考网格中心位于
`2*(index+0.5)/size-1`。偏移 logits 经 `tanh`，再乘以对应稀疏网格轴的归一化间距和 `offset_range_factor=2.0`。单点轴把位移尺度置零，避免除零；最终采样坐标裁剪到 `[-1, 1]`。

每个 offset group 有一个独立 `2→32→1` MLP。输入不是写死索引，而是每个 Query 坐标到实际变形采样坐标的连续二维位移。各组 bias 分配给该组对应的 attention heads，并加入缩放后的 QK 分数。Softmax 和 QK/AV 累加采用 FP32，输出再恢复输入 dtype，以降低 AMP/FP16 非有限风险。

## 初始化等价性与梯度

`GSDRAIFI` 继承 `AIFI`，原 `ma/fc1/fc2/norm1/norm2/dropout/dropout1/dropout2` 名称和形状不变。受控初始化将 baseline checkpoint 中每一个原参数按同名、同形状逐键复制。

以下边界严格零初始化：

- 稀疏旁路最终 `128→256` 输出投影的 weight 和 bias；
- offset 最后一个 grouped 1×1 的 weight 和 bias；
- 四个连续位移 MLP 最后线性层的 weight 和 bias。

因此初始化时稀疏残差逐元素为零，CPU FP32、eval、dropout=0 下 post-norm 与 pre-norm 都与原 AIFI 逐元素完全相等。第一次 backward 会直接更新最终输出投影；该投影非零后，Q/K/V、offset 输出层和位移 bias 输出层获得梯度；后两项发生更新后，其上游 offset 卷积和 bias MLP 第一层也获得梯度，所以零边界不会永久切断训练。

## 复杂度预算

默认旁路新增 117,644 个可训练参数，低于 0.20M 上限。640 输入时 AIFI 特征通常为 20×20；包含卷积/线性、QK/AV、连续 bias、双线性采样和 Softmax 主要操作的单图估算约为 0.144 GFLOPs，低于 0.30 GFLOPs 上限。

Ultralytics `model.info()`/THOP 主要按模块 hook 统计。它可以看到卷积和线性层，但不会完整统计函数式 `torch.matmul`、`softmax` 和 `grid_sample`；因此 `model.info()` 的差值只能作为可复现的工具统计，不能冒充完整理论 FLOPs。

## 与 CSCEF-v4 的隔离

CSCEF-v4 位于跨尺度融合处，使用 Scharr、结构张量、结构置信度、语义一致性和残差调制。GSDR-AIFI 位于最高层 AIFI 的注意力阶段，只研究原全局注意力之外的内容自适应稀疏 K/V 采样与连续相对位移。它不使用边缘算子、结构张量、coherence/reliability、跨尺度一致性、门控、频域、wavelet 或条带卷积，因而技术变量与创新点一分离。
