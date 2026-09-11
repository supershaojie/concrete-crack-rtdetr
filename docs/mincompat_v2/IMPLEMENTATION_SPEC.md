# RT-DETR C17 / C19 / C24 最小干预兼容重构 v2 —— Codex 完整实施规范

> **任务名称**：RTDETR C17/C19/C24 Minimal-Intervention Compatibility Redesign v2
> **项目**：混凝土裂缝检测 / RT-DETR-R18-Lite
> **日期**：2026-09-11
> **核心原则**：**先保护已经证明有效的单模块结构，再只对“模块之间真正发生冲突的连接”做最小兼容处理。**
> **最高目标**：尽可能保留 C17、C19、C24 的单模块能力，同时让组合实验不再出现旧 C20/C25/C26 那种负交互。
> **本轮不是重新设计三个模块。** C17、C19、C24 原成功版本必须保留，SCCA 不再做 v2 梯度隔离，CBR 不再先改 stable-reference/rho，CSCEF 不再移到 PAN 后。

---

# 0. 给 Codex 的总命令

请直接执行本文档，不要只输出建议。

本任务需要：

1. 读取并审计：
   - C17 / CSCEF-v5.1 成功版本；
   - C19 / CBR 成功版本；
   - C24 / SCCA-AIFI 成功版本；
   - 旧组合 C20/C25/C26；
   - 第一版 Compatibility Redesign 的源码与失败结果；
   - 用户提供的两份新失败实验包；
2. 确认第一版重构为什么损失单模块能力；
3. **保留原 C17/C19/C24 成功 forward、原插入点、原主要梯度结构；**
4. 新建一个“只处理 C17↔C24 冲突边”的最小兼容版本：
   - 原 C24 `SCCAAIFI` 完全恢复，不改；
   - 原 C17 `CSCEFv51` 的位置、内容、置信度、PAN 传播完全恢复；
   - 仅在 **CSCEF 读取 Y4 semantic 的这一条附加支路** 上做固定梯度衰减；
5. 同时准备：
   - `CSCEF-v5.2-Compat standalone`；
   - `C17-v5.2-Compat + 原 C24 SCCA`；
   - `上述 pair + 原 C19 CBR`；
   - 一个旧 C25 replay 配置，只用于本地对照，不正式训练；
6. 完成 forward 等价、梯度比例、公共权重、AMP/half/DN/save-load/optimizer coverage 等验证；
7. commit 并普通 push；
8. **不要启动正式 200 epoch 训练，不要执行完整 val/test。**

本轮最重要的限制：

> **不得再次大规模重路由 RT-DETR。不得再次移动 CSCEF。不得再次 detach SCCA 的 src/s。不得再次改 CBR 的 P3 来源或 rho。**

---

# 1. 当前研究事实：先承认 v1 Compatibility Redesign 失败

## 1.1 成功的原单模块

统一独立 test 已知：

| 模型 | Precision | Recall | mAP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| C2 baseline | 0.833960 | 0.808044 | 0.858163 | 0.458360 | 0.469636 |
| C17 / CSCEF-v5.1 | 0.844895 | 0.825004 | 0.879297 | 0.532830 | 0.512045 |
| C19 / CBR | 0.849034 | 0.837312 | 0.885186 | 0.519995 | 0.503892 |
| C24 / SCCA-AIFI | 0.855563 | 0.819901 | 0.878244 | 0.506461 | 0.499872 |

已知源码 commits：

- C2：`67c3078e54a657fd96d65fee657a75fbb1dae0d6`
- C17：`0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139`
- C19：`025997e3c51eaf6933534308a95da6ebf97bff53`
- C24：`f6e9dfda765046ae7691302cf5ec89d3f76cec5d`

这些结果说明：

- C17 本身有效；
- C19 本身有效；
- C24 本身有效；
- 下一版必须优先保护它们，而不是为了“理论解耦”重新改写它们。

---

# 2. 旧组合问题

历史两两组合：

| 实验 | P | R | mAP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| C20 = C17+C19 | 0.837321 | 0.814948 | 0.866303 | 0.494811 | 0.487043 |
| C25 = C17+C24 | 0.793838 | 0.777127 | 0.818478 | 0.404978 | 0.425180 |
| C26 = C19+C24 | 0.842168 | 0.814948 | 0.860372 | 0.496019 | 0.490730 |

对应 commits：

- C20：`81d5f0175e0168e022f29feb1ceb54de8a52deb5`
- C25：`ac32e223a509981ee9e61be66a352f2b80c5bfd1`
- C26：`beedcfa307e250fb2de47587097c51f9c141123b`

最严重的是：

`C17 + C24`

mAP50-95 从单模块：

- C17 = 0.512045
- C24 = 0.499872

组合后：

- C25 = 0.425180

因此本轮首先只解决：

> **C17 ↔ C24 的冲突。**

C19 暂时完整保留，等 C17+C24 的兼容 pair 先成立后再加入。

---

# 3. 第一版 Compatibility Redesign 的失败证据

第一版 compatibility commit：

`05e6de8b582f4a5f6c11cd50f8bc0406dacd7d0b`

第一版做了：

- DR-CSCEF-v6：stable P3/P4 ref、PAN 后置、只改 Decoder P3、side refs detach；
- GI-SCCA-v2：`scca_channel(x.detach(), s.detach())`；
- SR-CBR-v2：stable P3 ref、query/geometry detach、rho 0.075。

该版本工程验证通过，但正式训练性能明显下降。

## 3.1 GI-SCCA-v2 正式独立 test

用户失败包中 `variant=scca_v2`：

- P = `0.8031705127956372`
- R = `0.7771274200810446`
- mAP50 = `0.8296869092291662`
- AP75 = `0.4281460030593854`
- mAP50-95 = `0.4426717781864705`

原 C24：

- mAP50-95 = `0.4998720584810606`

说明：

> 仅仅把 SCCA 通道支路的 x/s detach，就已经明显破坏原 C24 成功机制。

因此本轮：

**恢复原 `SCCAAIFI`，不再使用 `GISCCAAIFI`。**

## 3.2 DR-CSCEF-v6 正式独立 test

用户失败包中 `variant=cscef_v6`：

- P = `0.7538670628356574`
- R = `0.7510130571814498`
- mAP50 = `0.7886893478058470`
- AP75 = `0.3957623500147743`
- mAP50-95 = `0.4165237430966872`

原 C17：

- mAP50-95 = `0.5120450850441516`

说明：

DR-CSCEF-v6 一次改变了太多有效因素：

- semantic 来源从 Y4 改成 projected P4；
- CSCEF 从原 FPN P3 融合前移到 PAN 后；
- CSCEF residual 不再传播进 PAN 的 P4/P5；
- side refs detach。

因此本轮：

**完全恢复 C17-v5.1 的原位置、原 Y4 semantic、原 P3 lateral、原 PAN 传播。**

---

# 4. 更新后的设计原则

第一版原则：

> 先彻底解耦，再组合。

正式结果证明这个原则过度。

第二版原则改为：

> **Preserve Successful Path First —— 先保护成功路径，再只削弱冲突边。**

具体：

1. 原 C17 standalone 不修改；
2. 原 C24 standalone 不修改；
3. 原 C19 standalone 不修改；
4. Pair 模型尽可能保持旧 C25 的 forward **数值完全一致**；
5. 只改变旧 C25 中：
   `CSCEF semantic residual branch → Y4/SCCA/AIFI`
   这条额外 backward 依赖的强度；
6. 不改变主 Y4→Concat→FPN/PAN 的梯度；
7. 不改变 SCCA 自身内部梯度；
8. 不改变 C17 lateral branch；
9. 不改变 C17 residual 强度；
10. 不改变 CBR。

---

# 5. 新版本核心：CSCEF-v5.2-Compat

建议代码类名：

`CSCEFv52Compat`

工作名称：

**Semantic-Gradient Attenuated CSCEF-v5.2**

论文最终名字暂不确定。

文件建议：

`ultralytics-main/ultralytics/nn/modules/cscef_v52_compat.py`

必须继承：

`CSCEFv51`

不要复制一整份 v5/v5.1 源码。

---

# 6. CSCEF-v5.2-Compat 的唯一行为变化

原 C17：

```text
lateral = projected backbone P3
semantic = upsampled Y4

lateral, semantic
       ↓
CSCEF-v5.1
       ↓
enhanced lateral
       ↓
Concat(upsampled Y4, enhanced lateral)
       ↓
RepC3
       ↓
P3
       ↓
PAN P4/P5
```

新 v5.2：

**forward 完全保持这条路径。**

唯一变化：

CSCEF 内部用于 semantic content projection 的 `semantic` 使用一个**前向恒等、反向缩放**的 view。

定义：

```python
def grad_scale_identity(x, gamma: float):
    return x.detach() + gamma * (x - x.detach())
```

正式固定：

```python
gamma = 0.25
```

因为：

- 旧 C25 冲突极重；
- 但 full detach 已经证明“过度隔离”风险很大；
- gamma=0.25 保留 25% 的 CSCEF→semantic 协同梯度；
- 同时削弱 75% 的额外 cross-module backward coupling；
- 原 Y4 主路径仍然 100% 正常训练。

必须验证：

```text
grad_scale_identity(x, 0.25)
```

前向数值与 x 完全相同。

---

# 7. 为什么只缩放 semantic，不动 lateral

C17 的结构置信度来自 lateral P3：

```text
P3 lateral
 ↓
Scharr
 ↓
coherence/reliability
 ↓
confidence
```

C17 的强 AP75 很可能与这条结构路径有关。

因此：

- `lateral` 不 detach；
- lateral 梯度不缩放；
- Scharr/structure confidence 保持 C17 原样；
- output projection 保持原样；
- residual addition 保持原样。

只处理：

`semantic → semantic_projection → content branch`

向 Y4/AIFI/SCCA 返回的额外梯度。

---

# 8. 为什么不 full detach semantic

不要使用：

```python
semantic.detach()
```

正式 v2 不允许。

原因：

- 第一版“完全隔离”的总体思路已经显示会损害成功模块；
- 原 C17 很可能利用 semantic 与 CSCEF 的联合适应；
- 我们要降低冲突，不是完全切断协同。

因此正式只用：

`gamma=0.25`

不要同时实现 0 / 0.5 / 0.75 的正式模型来做 test 扫描。

如果未来 v2 结果表明 gamma=0.25 过强，再基于 **val** 设计 v3；本任务不提前扫。

---

# 9. `CSCEFv52Compat` 推荐实现

逻辑应非常小。

示意：

```python
class CSCEFv52Compat(CSCEFv51):
    def __init__(
        self,
        c_lateral,
        c_semantic,
        hidden_channels=32,
        num_groups=8,
        eps=1e-6,
        semantic_grad_scale=0.25,
    ):
        super().__init__(
            c_lateral,
            c_semantic,
            hidden_channels,
            num_groups,
            eps,
        )
        self.semantic_grad_scale = float(semantic_grad_scale)

    def _semantic_grad_view(self, semantic):
        g = self.semantic_grad_scale
        return semantic.detach() + g * (semantic - semantic.detach())

    def forward(self, inputs):
        lateral, semantic = inputs
        semantic_proxy = self._semantic_grad_view(semantic)
        return super().forward((lateral, semantic_proxy))
```

但必须按当前 `CSCEFv51/CSCEFv5` 的真实签名调整。

要求：

- 不改任何原参数 key；
- 不增加 trainable parameter；
- 不改变 state_dict 数量；
- 不改变 parameter count；
- `semantic_grad_scale` 不是 Parameter；
- 作为普通 Python float / buffer-like configuration 记录；
- 不能让 scale 参与 optimizer。

---

# 10. 强制 forward 等价测试

这是本版最关键的工程性质之一。

构建：

- 原 `CSCEFv51`
- 新 `CSCEFv52Compat`

复制完全相同的**非零**参数。

输入：

- 随机 lateral；
- 随机 semantic；
- FP32；
- AMP；
- half（可行环境）。

要求：

```text
old_output == new_output
```

记录：

- max_abs_error
- max_rel_error

目标：

FP32 下优先 `max_abs=0`。

如果由于表达式实现导致极小非零，但逻辑上可避免，则改实现，尽量做到 exact identity。

不能只在 zero-init output projection 时测。

必须人为把 output projection 改成非零后再测，否则测试没有意义。

---

# 11. 强制 backward 比例测试

同样复制完全相同的非零参数。

构造：

```python
loss = output.square().mean()
```

分别 backward。

对比：

## 11.1 semantic 输入梯度

新旧应满足：

```text
grad_semantic_new ≈ 0.25 * grad_semantic_old
```

记录：

- L1 norm ratio
- L2 norm ratio
- max_abs difference vs 0.25×old

## 11.2 lateral 输入梯度

因为 lateral 没有缩放：

```text
grad_lateral_new ≈ grad_lateral_old
```

## 11.3 CSCEF 参数梯度

因为仅在输入边界缩放梯度，CSCEF 内部 forward 完全相同：

- lateral_projection grad：应一致；
- semantic_projection grad：应一致；
- mix/depthwise/output_projection grad：应一致。

注意：

输入 semantic 的梯度缩放，不应把模块自身 parameter gradient 乘 0.25。

必须做非零 output projection 的测试。

---

# 12. 本轮 SCCA：完全恢复原 C24

正式模型使用：

`SCCAAIFI`

来源：

C24 commit：

`f6e9dfda765046ae7691302cf5ec89d3f76cec5d`

不得使用：

`GISCCAAIFI`

不得：

- detach x；
- detach s；
- 改 Q/K/V；
- 改 temperature；
- 改注入位置；
- 改 pre/post norm；
- 改 channel width；
- 改 heads；
- 加 residual scale。

必须保留原：

- width=64；
- heads=4；
- head dim=16；
- LayerNorm no affine；
- Q from spatial MHA s；
- K/V from x；
- token-centered Q/K；
- L2 normalization；
- bounded temperature；
- FP32 core；
- zero-init `scca_o`；
- 原 AIFI forward_pre/forward_post 注入位置。

---

# 13. 本轮 CBR：完全恢复原 C19

正式三模块模型使用原：

- `CrackBoundaryRefinement`
- `RTDETRDecoderCBR`

来源：

C19 commit：

`025997e3c51eaf6933534308a95da6ebf97bff53`

必须恢复：

```text
CBR reads decoder x[0] = final neck P3
rho = 0.10
normal_fraction = 0.10
query 不 detach
P3 不 detach
原 displacement width/height scale 行为
```

本轮不要使用：

- `StableReferenceCBR`
- `RTDETRDecoderCBRv2`
- stable P3 fourth input
- rho=0.075
- query detach
- geometry gradient firewall

原因：

C19 原版本已经单独有效；
C19+C24 旧组合只是轻度下降，不像 C17+C24 那样灾难；
本轮先解决最严重冲突，不同时改变第三个模块。

---

# 14. 新版模型一：CSCEF-v5.2-Compat standalone

建议 YAML：

`rtdetr-resnet18-lite-cscef-v52-compat.yaml`

拓扑必须与 C17-v5.1 原 YAML一致。

当前已知原 C17：

```yaml
- [-1, 1, AIFI, [1024, 8]]                  # AIFI 原版
...
- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # upsampled Y4
- [5, 1, Conv, [256, 1, 1, None, 1, 1, False]] # projected P3
- [[17, 16], 1, CSCEFv51, []]
- [[16, 18], 1, Concat, [1]]
- [-1, 3, RepC3, [256, 0.5]]
...
- [[20,23,26], 1, RTDETRDecoder, ...]
```

新 standalone 只改：

```text
CSCEFv51
→
CSCEFv52Compat
```

其余一律不动。

目的：

> 检查只衰减 semantic side gradient 后，是否仍尽量保持 C17 原来的单模块性能。

这项正式实验未来建议与 pair 同时跑。

---

# 15. 新版模型二：C17-v5.2-Compat + 原 C24 SCCA

建议 YAML：

`rtdetr-resnet18-lite-cscef-v52-scca-compat.yaml`

这是本轮最关键模型。

拓扑必须等于：

**旧 C25 forward topology**

只有一处 backward 行为变化。

即：

```text
P5 projection
    ↓
原 SCCAAIFI
    ↓
Y5
    ↓
原 FPN → Y4
    ↓
Upsample Y4
    ├──────────────→ 原 Concat 主路径（100% 正常梯度）
    │
    └──→ CSCEFv52Compat semantic input
            backward × 0.25
            forward value unchanged
                  ↓
            enhanced P3 lateral
                  ↓
Concat(upY4, enhanced lateral)
                  ↓
RepC3
                  ↓
P3
                  ↓
原 PAN → P4/P5
                  ↓
Decoder
```

不得：

- 把 CSCEF 移到 PAN 后；
- 把 semantic 换成 P4_ref；
- 把 lateral 换成 final P3；
- detach SCCA；
- detach整个 Y4；
- 改 PAN；
- 改 Decoder；
- 加新 gate。

---

# 16. 新版模型三：Triad-MinCompat-v2

建议 YAML：

`rtdetr-resnet18-lite-triad-mincompat-v2.yaml`

结构：

```text
原 SCCAAIFI
    ↓
原 C17 位置的 CSCEFv52Compat
    ↓
原 FPN/PAN
    ↓
原 RTDETRDecoderCBR
```

也就是：

```text
C24 original
+
C17 forward-preserving minimal compat
+
C19 original
```

注意：

- CBR 继续读取最终 Neck P3；
- 不使用第四个 reference 输入；
- rho=0.10；
- 不使用第一版 Triad 的 `DRCSCEFv6/GISCCAAIFI/StableReferenceCBR`。

这才是 v2 三模块模型。

---

# 17. 旧 C25 Replay YAML

建议：

`rtdetr-resnet18-lite-c25-replay.yaml`

它应精确表示：

```text
原 SCCAAIFI
+
原 CSCEFv51
```

不做兼容处理。

用途：

- 本地 forward/backward regression；
- 与新 pair 做 exact forward 对照；
- 不用于新的正式 200e 训练，除非用户明确要求重跑历史 C25。

---

# 18. Pair 级 exact-forward 证明

这是新方案的核心证据。

建立：

- old C25 replay；
- new `cscef-v52-scca-compat`。

将所有公共参数和创新参数设置完全一致，并人为使 SCCA/CSCEF output projection 非零。

同一输入：

```text
640×640
160×192
```

要求：

```text
old_pair_forward == new_pair_forward
```

至少对：

- Y5；
- Y4；
- CSCEF output；
- P3；
- P4；
- P5；
- decoder bbox；
- score

记录：

- max_abs
- max_rel

理想：

`max_abs = 0`

这项证明意味着：

> 新 v2 不靠改变前向特征来“兼容”，只改变冲突边的 backward 强度。

---

# 19. Pair 级 gradient attribution

需要新增诊断工具：

`tools/check_mincompat_v2_gradients.py`

至少做三类检查。

## 19.1 CSCEF semantic-only objective

使用：

`CSCEF residual output`

构造非零 objective。

比较 old C25 / new pair。

在 Y4 semantic tensor 上：

```text
new semantic grad ≈ 0.25 * old semantic grad
```

## 19.2 Main concat path

构造只经过：

```text
Y4 → Concat → RepC3 → ...
```

而不经过 CSCEF semantic branch 的 objective。

old/new：

```text
Y4 main-path grad 应一致
```

证明我们没有把 Y4/SCCA 主路径切断。

## 19.3 SCCA parameter gradient

使用完整 pair objective。

要求：

- SCCA 参数仍有有限非零梯度；
- 不允许像 `GISCCAAIFI` 那样把 SCCA 自己的 x/s side path detach；
- old/new forward 相同；
- new 的来自 CSCEF semantic side 的额外影响被减弱。

不要声称 total SCCA grad 必须精确是某个比例，因为其同时收到正常主路径梯度。

---

# 20. 为什么 gamma 不设为 0

第一版失败已经说明：

“所有创新支路尽量 detach”不是正确方向。

本轮保留：

```text
25% cross-module adaptation
```

目的是：

- 仍允许 CSCEF semantic branch 与上游表示联合适应；
- 但不允许它像旧 C25 那样完整地把额外梯度回灌到 C24/SCCA；
- 主 Y4/FPN/PAN 的正常梯度完全不变。

---

# 21. 为什么本轮不加 residual scale

暂时不要：

```python
output = lateral + beta * delta
```

正式 v2 固定：

`beta = 1`

原因：

- C17 单独已经证明 residual 强度有效；
- 本轮要一次只解决一个可归因问题；
- 如果同时改 gradient 和 residual 强度，正式结果无法判断哪个有效。

如果 v2 standalone 保持 C17，但 pair 仍不好：

未来 v3 才考虑：

**Compatibility Residual Budget**

例如固定 beta<1。

本任务不要提前实现正式 v3。

---

# 22. 新分支策略

建议新分支：

`codex/rtdetr-mincompat-v2`

建议从第一版 compatibility commit：

`05e6de8b582f4a5f6c11cd50f8bc0406dacd7d0b`

建立。

原因：

- 它已经包含原 `CSCEFv51`；
- 已包含原 `SCCAAIFI`；
- 已包含原 CBR；
- 有成熟的统一初始化/服务器/打包工具；
- C2 regression 已验证。

但是：

第一版失败模块：

- `DRCSCEFv6`
- `GISCCAAIFI`
- `StableReferenceCBR`

只能保留做历史/回归，**新正式 v2 YAML 不得引用它们。**

---

# 23. 不能从失败 Triad YAML 直接改几行就算完成

必须逐项证明新 YAML 使用的是：

- C17 原位置；
- C17 原 semantic；
- C17 原 PAN 传播；
- C24 原 SCCAAIFI；
- C19 原 RTDETRDecoderCBR。

新三模块模型不能残留：

```text
DRCSCEFv6([P3_base,P3_ref,P4_ref])
GISCCAAIFI
RTDETRDecoderCBRv2([P3,P4,P5,P3_ref])
```

任何一个。

---

# 24. 参数量预期

`CSCEFv52Compat` 不新增 trainable parameter。

因此：

## standalone

应与原 C17-v5.1 参数量完全相同。

已知 C17 相对 C2：

`+26,912`

## pair

应等于：

`C2 + C17 + C24`

不因为 compatibility 增加参数。

理论：

`20,082,772 + 26,912 + 65,540 = 20,175,224`

## triad

应等于：

`C2 + C17 + C24 + C19`

理论：

`20,221,113`

若与理论不同：

必须解释公共模块是否重用或 state key 是否发生变化。

---

# 25. 受控初始化

所有正式新实验仍从 C2 统一初始化源：

`weights/rtdetr_r18_lite_imagenet_backbone_init.pt`

SHA256：

`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`

禁止使用：

- C17 best.pt；
- C19 best.pt；
- C24 best.pt；
- C25 best.pt；
- CSCEF-v6 best.pt；
- SCCA-v2 best.pt。

训练权重只能用于历史分析，不用于初始化。

---

# 26. State dict key 兼容

`CSCEFv52Compat` 应尽量继承 `CSCEFv51` 原 state key。

要求：

- same parameter names；
- same tensor shapes；
- same buffers；
- `semantic_grad_scale` 不增加 trainable state；
- 能把 C17-v5.1 module state 精确加载到 v5.2。

做 test：

```text
old.state_dict()
→
new.load_state_dict(..., strict=True)
```

优先要求成功。

---

# 27. 原单模块 regression

新分支必须继续支持：

- C17 原 YAML；
- C19 原 YAML；
- C24 原 YAML。

构建和 forward 不得变化。

尤其：

原 C17：

`CSCEFv51`

原 C24：

`SCCAAIFI`

原 C19：

`RTDETRDecoderCBR`

都不能被 parser 默认替换成新类。

---

# 28. 新 variant IDs

建议新增：

```text
cscef_v52_compat
cscef_v52_scca_compat
triad_mincompat_v2
```

另有内部回归：

```text
c25_replay
```

第一版：

```text
cscef_v6
scca_v2
cbr_v2
triad_v1
```

保留历史，但在文档标注：

`deprecated_for_performance_experiments`

不要删除历史代码与报告。

---

# 29. AutoDL 工具

建议新脚本：

`tools/autodl_mincompat_v2.sh`

接口：

```bash
bash tools/autodl_mincompat_v2.sh cscef_v52_compat start-direct
bash tools/autodl_mincompat_v2.sh cscef_v52_scca_compat start-direct
bash tools/autodl_mincompat_v2.sh triad_mincompat_v2 start-direct
```

以及：

```bash
status
val
test
pack-complete
```

不要让新脚本默认跑第一版失败 variants。

---

# 30. 第一轮正式实验计划

Codex 本任务不启动正式训练。

代码完成以后，用户服务器 Round A 建议两个 GPU 并行：

## GPU 1

`cscef_v52_compat`

目的：

> 判断只减少 semantic side gradient 后，C17 的强单模块性能还能保留多少。

## GPU 2

`cscef_v52_scca_compat`

目的：

> 直接验证最严重的 C17+C24 冲突能否在 forward 完全不变的情况下，仅靠最小 backward compatibility 修复。

SCCA 不需要重新做 standalone：

原 C24 已经是正式已知成功版本。

---

# 31. Round A 结果判定

基准：

```text
C2 mAP50-95    = 46.964%
C17             = 51.205%
C24             = 49.987%
旧 C25 pair     = 42.518%
```

## 理想

```text
CSCEF-v5.2 standalone ≈ 50~51+
pair > max(standalone, C24)
```

## 可以接受并继续

例如：

```text
CSCEF-v5.2 = 49.5~50.5
pair       = 50.5~51+
```

说明：

- 单模块牺牲有限；
- 兼容后出现累加。

## 需要 v3

如果：

```text
CSCEF-v5.2 standalone 仍接近原 C17
但 pair 仍低于约 49~50
```

说明主要冲突更偏 forward/residual over-enhancement。

此时才做：

`CSCEF-v5.3 Compatibility Residual Budget`

不要回到第一版大解耦。

## 若 standalone 自身大掉

说明 gamma=0.25 太强。

下一版可基于 val 考虑：

`gamma=0.5`

但不使用 test 做多 gamma 扫描。

---

# 32. Round B

只有 Round A 支持兼容方向后，才跑：

`triad_mincompat_v2`

它使用：

- 原 C24；
- CSCEF-v5.2 minimal compat；
- 原 C19。

如果三模块加入 C19 后下降，再针对 CBR 做**最小修改**。

不要提前套用第一版 SR-CBR-v2。

---

# 33. 若 CBR 后续冲突，下一步规则

未来若 pair 成功但 triad 加 CBR 下跌：

第一优先不是 stable reference/rho/多项同时改。

优先只改一条冲突边，例如：

```text
CBR branch → final P3/query
```

做 gradient attenuation。

仍然保持：

- CBR 原 final P3；
- rho=0.10；
- 36 sampling；
- box residual；
- 原 forward。

即继续遵循：

**forward first, gradient-edge minimal intervention。**

本任务现在不实现。

---

# 34. 训练配方完全保持 C2

从真实 C2 args 全字段 diff。

关键：

```text
epochs=200
patience=50
batch=16
imgsz=640
device=0
workers=8
optimizer=AdamW
seed=42
deterministic=True
cos_lr=True
close_mosaic=10
amp=True

lr0=0.0005
lrf=0.01
weight_decay=0.0001
warmup_epochs=5
```

在线增强全部与 C2 相同。

禁止：

- 150e；
- batch32；
- MuSGD；
- 关闭增强；
- 改 imgsz；
- 特殊 lr；
- freeze；
- 新 loss。

---

# 35. 正式评估

继续统一：

```text
split=test
imgsz=640
batch=16
conf=0.001
iou=0.7
max_det=300
half=False
augment=False
seed=42
```

保留：

- Precision
- Recall
- mAP50
- AP75
- mAP50-95
- PR/F1/P/R curve
- confusion matrix
- checkpoint SHA256
- effective args

---

# 36. 本地验证：必须完成

Codex 不正式训练，但至少完成：

1. 原 C17/C19/C24 构建回归；
2. 新 standalone 构建；
3. 新 pair 构建；
4. 新 triad 构建；
5. nc=1；
6. 参数量；
7. same-weight forward exact equivalence；
8. semantic gradient ×0.25；
9. lateral gradient unchanged；
10. CSCEF parameter gradients unchanged；
11. pair forward == old C25 replay；
12. AMP forward/backward；
13. true half inference；
14. DN dynamic query；
15. save/load；
16. optimizer coverage；
17. original C2 regression；
18. state strict load from v5.1→v5.2。

---

# 37. 第一版失败包必须纳入审计

用户会同时提供两份第一版失败包。

当前外层文件名：

- `69fac72a-217b-4625-ae8e-470d620d7407.gz`
- `e20e19a8-2175-41d3-9f0e-5bed389e8f06.gz`

不要根据外层名字判断。

必须读取：

`test/metrics.json`

根据 `variant` 识别：

- `scca_v2`
- `cscef_v6`

并把真实指标写入：

`docs/mincompat_v2/failure_evidence.md`

---

# 38. 文档

新增：

```text
docs/mincompat_v2/FAILURE_ANALYSIS.md
docs/mincompat_v2/ARCHITECTURE.md
docs/mincompat_v2/GRADIENT_EDGE_DESIGN.md
docs/mincompat_v2/VALIDATION.md
docs/mincompat_v2/AUTODL.md
```

`FAILURE_ANALYSIS.md` 要明确承认：

第一版 compatibility 的 engineering correctness ≠ detection effectiveness。

不能把失败归因写成已经证明的因果结论。

正确表述：

> v1 results falsified the assumption that aggressive architectural/gradient decoupling would preserve standalone capability; therefore v2 preserves successful forward paths and changes only one cross-module backward edge.

---

# 39. Git

开始前：

```bash
git status
git worktree list
git remote -v
git branch -a
git log --all --decorate --oneline
```

建议新分支：

`codex/rtdetr-mincompat-v2`

建议 worktree：

`D:\MyProjects\Crack_RTDETR\outputs\worktrees\mincompat-v2`

服务器：

`/root/autodl-tmp/projects/Crack_RTDETR-mincompat-v2`

禁止：

- reset --hard
- clean
- stash
- force push
- 删除其他 worktree
- 覆盖用户结果

---

# 40. Commit

建议：

`feat(rtdetr): minimally attenuate CSCEF-SCCA conflict edge`

普通 push：

`origin codex/rtdetr-mincompat-v2`

---

# 41. 服务器 sync

实现：

```bash
bash tools/sync_mincompat_v2.sh <FULL_SHA>
```

要求：

- fetch 新分支；
- 验证 FULL SHA；
- detached worktree；
- 不修改主树；
- 不安装环境；
- 不启动训练。

---

# 42. start-direct

预检必须检查：

- SHA；
- rtdetr env；
- model variant；
- C2 init source SHA；
- 参数量；
- train config diff；
- `semantic_grad_scale=0.25`；
- variant 不引用 v6/GI-SCCA/SR-CBR-v2；
- optimizer coverage；
- loss/DN smoke。

---

# 43. status / val / test / pack

接口：

```bash
bash tools/autodl_mincompat_v2.sh <variant> status
bash tools/autodl_mincompat_v2.sh <variant> val
bash tools/autodl_mincompat_v2.sh <variant> test
bash tools/autodl_mincompat_v2.sh <variant> pack-complete
```

强制：

`train SUCCESS → val → test → pack`

---

# 44. 打包要求

每个包收集：

- training；
- val；
- test；
- console；
- model YAML；
- source snapshot；
- git SHA；
- initialization；
- weight mapping；
- gradient-edge validation；
- parameter report；
- resolved config；
- environment；
- best/last policy按现项目；
- curves；
- confusion matrix；
- metrics；
- manifest；
- SHA256；
- inventory；
- verification。

---

# 45. Codex 最终回复必须包含

1. v1 failure evidence；
2. 为什么不再使用 CSCEF-v6 / GI-SCCA-v2 / SR-CBR-v2；
3. 原 C17/C19/C24 是否保持；
4. `CSCEFv52Compat` 精确实现；
5. gamma=0.25；
6. forward exact equivalence 数值；
7. semantic grad ratio；
8. lateral grad ratio；
9. CSCEF parameter grad equivalence；
10. old C25 replay vs new pair forward equivalence；
11. 新 YAML；
12. 参数量；
13. public weight mapping；
14. AMP/half/DN/save-load/optimizer coverage；
15. C2/C17/C19/C24 regression；
16. 新 FULL SHA；
17. push 状态；
18. AutoDL 固定 SHA sync 命令；
19. Round A 两个启动命令；
20. Round B triad 启动命令；
21. val/test/pack 命令；
22. 未验证内容。

---

# 46. 最终研究逻辑

本轮核心不是“再做一个模块”。

而是：

```text
C17 成功 → 保留
C24 成功 → 保留
C19 成功 → 保留

旧 C25 冲突
   ↓
只处理：
CSCEF semantic side branch
      →
Y4 / SCCA / AIFI
的额外 backward 强度

forward 不改
主路径不改
SCCA 不改
PAN 不改
CBR 不改
```

这使得新的实验具有很强的可解释性：

如果 pair 提升：

> 说明减弱特定 cross-module gradient edge 有助于兼容。

如果 pair 仍失败：

> 说明主要冲突更可能来自 forward residual / feature over-enhancement，再进入 v3 的 residual-budget 设计。

无论结果怎样，都比一次同时改位置、输入、梯度、rho 更容易定位原因。

---

# 47. 最后约束

请 Codex：

- 先读取失败包再改代码；
- 不再次重构整个 RT-DETR；
- 不移动 CSCEF；
- 不 detach SCCA x/s；
- 不使用 stable-ref CBR；
- 不改 rho；
- 不加新模块；
- 不改训练超参数；
- 不启动正式训练；
- 只做最小 conflict-edge intervention；
- 完成验证后 commit、普通 push。

本轮目标：

> **尽可能保持单模块的原成功性能，同时让最严重的 C17+C24 组合从负交互转向至少不降，最好出现正向累加。**
