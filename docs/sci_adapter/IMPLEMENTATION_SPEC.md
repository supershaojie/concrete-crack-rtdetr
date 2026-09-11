# RT-DETR SCCA ↔ CSCEF Semantic Interface Adapter —— Codex 完整实施规范

> **任务名称**：RTDETR SCCA-CSCEF Semantic Compatibility Interface Adapter  
> **项目**：混凝土裂缝检测 / RT-DETR-R18-Lite  
> **日期**：2026-09-11  
> **核心目标**：**不再修改已经证明有效的 C17 / CSCEF-v5.1 和 C24 / SCCA-AIFI 本体，只在两者之间新增一个极轻量、可退化、可审计的 semantic interface adapter，用于校准 SCCA 改写后的 Y4 分布，使其更适合原 CSCEF 消费。**  
> **最高原则**：保护成功模块、保护原 forward、保护原梯度协同；只修“模块接口”，不再修模块本体。

---

# 0. 给 Codex 的总命令

请直接执行本文档，不要只给建议。

本任务必须完成：

1. 审计原 C17、原 C24、旧 C25、上一版 mincompat-v2 的源码和实验记录；
2. 明确确认：
   - 原 C17 / CSCEF-v5.1 单独有效；
   - 原 C24 / SCCA-AIFI 单独有效；
   - 旧 C25 组合失败；
   - mincompat-v2 通过 semantic gradient attenuation 改善了一部分 pair 的严格定位，但严重伤害 C17 standalone；
3. **恢复并保护原 C17、原 C24 源码与训练路径，不再修改它们；**
4. 新增一个只存在于组合模型中的：
   - `SemanticCompatibilityAdapter`
   - 简称可用 `SCIAdapter`
5. Adapter 只处理：
   - `SCCA-enhanced Y4 → CSCEF semantic input`
   - 不进入主 FPN/PAN 路径
   - 不替代 Y4
   - 不修改 SCCA
   - 不修改 CSCEF
6. 准备并验证两个正式 Round A variant：
   - `C17 + Adapter control`
   - `C24 + Adapter + C17`
7. 预留后续：
   - `C24 + Adapter + C17 + 原 C19 CBR`
8. 完成：
   - zero-init 等价
   - old C17 / old C25 forward regression
   - gradient path audit
   - parameter audit
   - AMP / half / DN / save-load / optimizer coverage
   - AutoDL sync/train/status/val/test/pack
9. commit 并普通 push；
10. **不要启动正式 200 epoch 训练，不要执行完整 val/test。**

本轮禁止再次做：
- CSCEF 位置迁移；
- semantic gradient ×0.25；
- SCCA x/s detach；
- stable-ref CBR；
- rho 改动；
- Dual Semantic Stream；
- 新 attention；
- 新 edge branch；
- 新 gate；
- 新 loss；
- 新 matcher；
- 新 optimizer；
- 新超参数搜索。

---

# 1. 当前已确认的研究事实

## 1.1 C2 baseline

C2 FULL SHA：

`67c3078e54a657fd96d65fee657a75fbb1dae0d6`

正式独立 test：

- P = `0.8339600078513765`
- R = `0.8080444244334384`
- mAP50 = `0.858162985651444`
- AP75 = `0.4583604005521489`
- mAP50-95 = `0.46963623190802783`

---

# 2. 三个成功单模块

## 2.1 C17 / CSCEF-v5.1

source commit：

`0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139`

正式 test：

- P = `0.8448953234408928`
- R = `0.825003752063635`
- mAP50 = `0.8792974548298313`
- AP75 = `0.5328297758576255`
- mAP50-95 = `0.5120450850441516`

这是必须保护的成功模块。

关键成功路径：

```text
Projected P3 lateral
          │
Upsampled Y4 semantic
          │
          ├──→ CSCEF-v5.1
          │        ↓
          │  enhanced lateral
          │        ↓
          └────→ Concat
                   ↓
                 RepC3
                   ↓
                  P3
                   ↓
                  PAN
               ↓       ↓
              P4       P5
                   ↓
                Decoder
```

必须保留：
- 原 Y4 semantic；
- 原 P3 lateral；
- 原 CSCEF 插入位置；
- 原 Scharr；
- 原 confidence；
- 原 content branch；
- 原 residual；
- 原 PAN 传播；
- 原完整 backward。

---

## 2.2 C24 / SCCA-AIFI

source commit：

`f6e9dfda765046ae7691302cf5ec89d3f76cec5d`

正式 test：

- P = `0.8555632739746718`
- R = `0.819900945520036`
- mAP50 = `0.8782438758197731`
- AP75 = `0.5064610022981055`
- mAP50-95 = `0.4998720584810606`

必须使用原 `SCCAAIFI`：

- 原 Q/K/V；
- 原 temperature；
- 原 LayerNorm；
- 原 token centering；
- 原 L2 normalize；
- 原注入位置；
- 原 zero-init；
- 原 forward/backward。

禁止使用第一版 compatibility 中的 `GISCCAAIFI`。

---

## 2.3 C19 / CBR

source commit：

`025997e3c51eaf6933534308a95da6ebf97bff53`

正式 test：

- P = `0.8490338445575341`
- R = `0.837311905431872`
- mAP50 = `0.8851863565670782`
- AP75 = `0.5199952552147155`
- mAP50-95 = `0.5038924194554971`

后续 triad 必须使用原 C19：

- final neck P3；
- 原 query；
- rho = 0.10；
- normal_fraction = 0.10；
- 原 36 点采样；
- 原 box residual。

本轮 Round A 暂时不跑 C19。

---

# 3. 旧 C25 的失败事实

旧 C25 = 原 C17 + 原 C24。

source commit：

`ac32e223a509981ee9e61be66a352f2b80c5bfd1`

正式 test：

- P = `0.793838`
- R = `0.777127`
- mAP50 = `0.818478`
- AP75 = `0.404978`
- mAP50-95 = `0.425180`

与单模块相比严重下降。

这说明：

> **两个单模块都有效，但直接串联的接口存在不兼容。**

---

# 4. 第一版 aggressive compatibility 失败

第一版 commit：

`05e6de8b582f4a5f6c11cd50f8bc0406dacd7d0b`

失败点：

- `DR-CSCEF-v6`
- `GI-SCCA-v2`
- `SR-CBR-v2`

正式 test：

## CSCEF-v6
- mAP50-95 ≈ `0.4165237430966872`

## SCCA-v2
- mAP50-95 ≈ `0.4426717781864705`

结论：

> 大幅改位置、输入、PAN 传播和梯度，会破坏原单模块能力。

因此本轮绝对不能再次大改。

---

# 5. 第二版 mincompat-v2 失败，但提供了重要信息

第二版核心：

```text
CSCEF semantic branch backward:
1.0 → 0.25
```

forward 保持原 C17/C25。

结果：

## CSCEF-v5.2 standalone

正式 test：

- P ≈ `0.77343`
- R ≈ `0.77107`
- mAP50 ≈ `0.80636`
- AP75 ≈ `0.40407`
- mAP50-95 ≈ `0.42300`

## CSCEF-v5.2 + SCCA

正式 test：

- P ≈ `0.78028`
- R ≈ `0.77728`
- mAP50 ≈ `0.81519`
- AP75 ≈ `0.43554`
- mAP50-95 ≈ `0.44025`

两个结论：

### A
把 semantic gradient 降到 25% 会严重破坏 C17 standalone。

说明：
> C17 的成功依赖 CSCEF ↔ Y4/FPN/AIFI 的完整联合优化。

### B
Pair 相对 standalone 又有所提升，而且相对旧 C25：
- mAP50-95 约 +1.5pp
- AP75 约 +3pp

说明：
> C17+C24 并非“天生互斥”，但旧 C25 除 backward conflict 外，很可能还有 forward feature/interface mismatch。

因此本轮不再调 gamma。

---

# 6. 本轮核心假设

新的研究假设：

> **SCCA 本身有效，CSCEF 本身有效；问题主要出在 SCCA 改写后的 Y4 分布并不是原 CSCEF 最容易消费的 semantic distribution。**

因此不再修改模块内部。

新目标：

```text
SCCA-enhanced Y4
       ↓
轻量 semantic interface calibration
       ↓
CSCEF-friendly Y4
       ↓
原 CSCEF-v5.1
```

---

# 7. 新整体设计

主结构：

```text
P5
 ↓
原 SCCAAIFI
 ↓
Y5
 ↓
原 FPN
 ↓
Y4
 ├──────────────────────────────→ 原主路径
 │                                  ↓
 │                                Concat
 │                                  ↓
 │                                RepC3
 │                                  ↓
 │                                 P3
 │                                  ↓
 │                                 PAN
 │
 └────→ Semantic Compatibility Adapter
                  ↓
             Y4_compat
                  ↓
            原 CSCEF-v5.1
                  ↓
           enhanced P3 lateral
                  ↓
              原 Concat
```

注意：

- `Y4` 主路径完全不经过 Adapter；
- Adapter 只生成给 CSCEF 的 semantic proxy；
- CSCEF 位置不变；
- SCCA 位置不变；
- PAN 不变；
- Decoder 不变；
- loss/matcher/DN 不变。

---

# 8. Adapter 的正式定位

建议类名：

`SemanticCompatibilityAdapter`

短名：

`SCIAdapter`

文件：

`ultralytics-main/ultralytics/nn/modules/semantic_compat_adapter.py`

它不是第四个“增强模块”。

定位是：

> **接口校准器 / compatibility adapter**

只负责：
- channel distribution calibration；
- small channel mixing；
- residual correction。

不负责：
- spatial enhancement；
- edge enhancement；
- attention；
- large receptive field；
- deformable sampling。

---

# 9. Adapter 结构

输入：

`Y4: [B,256,H,W]`

正式隐藏宽度：

`hidden = 32`

结构：

```text
Y4_detached
    ↓
GroupNorm(8 groups, affine=True)
    ↓
Conv1x1 256 → 32
    ↓
SiLU
    ↓
Conv1x1 32 → 256
    ↓
zero-init
    ↓
Δcompat

Y4_compat = Y4 + Δcompat
```

公式：

\[
R = A(\operatorname{stopgrad}(Y4))
\]

\[
Y4_{compat}=Y4+R
\]

最后：

\[
CSCEF(lateral, Y4_{compat})
\]

---

# 10. 为什么 Adapter 输入使用 detach

这里的 detach 和 mincompat-v2 完全不同。

错误做法：

```python
CSCEF(lateral, Y4.detach())
```

这会切断 CSCEF 原本需要的语义联合优化。

本轮：

```python
delta = Adapter(Y4.detach())
Y4_compat = Y4 + delta
CSCEF(lateral, Y4_compat)
```

因此：

## 原 C17 主 semantic gradient

```text
CSCEF
  ↓
Y4_compat
  ↓
Y4
```

仍然是：

`100%`

## Adapter 自己

```text
Adapter(Y4.detach())
```

不会额外把自己的梯度传回 SCCA/FPN。

也就是说：

> 保留原 C17 有益梯度，同时让 Adapter 自己只学习“如何修正接口”。

---

# 11. zero-init

最后一个：

`Conv1x1(32→256)`

必须：

```text
weight = 0
bias = 0
```

第一层 Conv：

- Kaiming/Xavier 正常初始化；
- GN affine 正常初始化。

初始化时：

\[
R=0
\]

因此：

\[
Y4_{compat}=Y4
\]

所以 pair 初始化：

> **严格退化为旧 C25 forward。**

Control 初始化：

> **严格退化为原 C17 forward。**

禁止：

- zero-init 第一层；
- 再加 zero alpha；
- Python if-zero shortcut。

---

# 12. GroupNorm

固定：

```text
num_groups = 8
num_channels = 256
affine = True
eps = 1e-5
```

理由：

- batch=16，但裂缝任务不依赖 batch statistics；
- GN 对 train/eval 更稳定；
- 只做分布校准；
- affine 参数允许 Adapter 学习轻微 scale/shift。

不要使用 BatchNorm。

不要使用 spatial attention。

---

# 13. 参数量

Adapter 理论 trainable 参数：

## GroupNorm
- weight = 256
- bias = 256
- 合计 512

## Conv1x1 256→32
若 bias=False：
- `256×32 = 8192`

## Conv1x1 32→256
建议 bias=True：
- `32×256 = 8192`
- bias 256
- 合计 8448

Adapter 总计：

`512 + 8192 + 8448 = 17,152`

若当前代码风格需要第一层 bias=True，则参数量变化必须明确记录，但优先按本文：

- reduce bias=False
- restore bias=True

正式目标：

`+17,152 params`

不要为了凑参数破坏仓库习惯；实际值必须审计。

---

# 14. 为什么不加 3×3

本轮不允许：

- 3×3 conv
- depthwise 3×3
- Scharr
- strip conv
- dilation

因为：

> Adapter 只处理 channel/statistical mismatch，不再引入新的 spatial inductive bias。

否则它会变成第四个独立创新点，难以归因。

---

# 15. 为什么不加 gate / attention

不允许：

- SE
- CBAM
- channel attention
- spatial attention
- sigmoid gate
- softmax gate
- dynamic routing

因为：

SCCA 本身已经是 attention；
CSCEF 本身已有结构/内容融合。

Adapter 再做 attention 很容易继续 over-enhancement。

---

# 16. Round A 正式 variant 1：C17 + Adapter control

建议 YAML：

`rtdetr-resnet18-lite-cscef-v51-sci-control.yaml`

逻辑：

```text
原 AIFI
 ↓
原 FPN
 ↓
Y4
 ├→ 原主路径
 └→ SCIAdapter
       ↓
    Y4_compat
       ↓
   原 CSCEF-v5.1
```

注意：

**没有 SCCA。**

目的：

> 验证 Adapter 本身不会破坏原 C17。

理想结果：
- 接近 C17；
- Adapter 若无必要，应学接近 identity。

这是非常重要的 control。

---

# 17. Round A 正式 variant 2：原 C24 + Adapter + 原 C17

建议 YAML：

`rtdetr-resnet18-lite-scca-sci-cscef-v51.yaml`

逻辑：

```text
原 SCCAAIFI
 ↓
原 FPN
 ↓
Y4
 ├→ 原主路径
 └→ SCIAdapter
       ↓
   Y4_compat
       ↓
原 CSCEF-v5.1
       ↓
原 PAN
       ↓
原 Decoder
```

这是本轮最核心实验。

---

# 18. Round B：后续 triad

仅在 Round A 支持 Adapter 方向后准备：

`rtdetr-resnet18-lite-scca-sci-cscef-v51-cbr.yaml`

结构：

```text
原 C24 SCCA
+
SCIAdapter
+
原 C17 CSCEF-v5.1
+
原 C19 CBR
```

CBR 完整原样。

本轮 Codex 可以准备 YAML 和工具，但不要启动正式训练。

---

# 19. 原模块代码保护

必须保护：

## 原 C17
- `CSCEFv51`
- 不改源码；
- 不改 state key；
- 不改 forward；
- 不改 gradient。

## 原 C24
- `SCCAAIFI`
- 不改源码；
- 不改 x/s；
- 不改 Q/K/V；
- 不改注入。

## 原 C19
- `RTDETRDecoderCBR`
- 不改源码；
- rho=0.10；
- 不改 P3 来源。

如果为 parser/export 必须改 `__init__.py/tasks.py`，只能新增 `SCIAdapter` 注册。

---

# 20. 旧 C17 / C25 regression

必须构建原：

- C17-v5.1
- C24
- C25

确认新分支没有改变它们。

建议使用历史 YAML/source commit 做独立子进程 regression。

---

# 21. Adapter zero-init forward 等价

## Control

将原 C17 和：

`C17 + SCIAdapter`

公共参数完全对齐。

SCI final projection zero-init。

同输入：

- 640×640
- 160×192

比较：
- Y4
- Adapter output
- CSCEF output
- P3/P4/P5
- bbox
- score

要求：
- `Y4_compat == Y4`
- 最终输出与原 C17一致
- max_abs/rel 记录

## Pair

将旧 C25 与：

`SCCA + SCIAdapter + CSCEF`

公共参数完全对齐。

Adapter zero-init。

要求：
- pair forward == 旧 C25
- 同样记录全部关键节点。

---

# 22. 非零 Adapter 测试

不能只测试 zero-init。

手工让 restore projection 变成小非零值。

验证：

- Adapter 只改变 `Y4_compat`
- 原 Y4 主路径不被 in-place 修改；
- P3/PAN 通过 CSCEF 路径产生合理变化；
- finite；
- shape/dtype 不变。

禁止任何 in-place 修改原 Y4。

---

# 23. 梯度测试

新增：

`tools/check_sci_adapter_gradients.py`

必须验证：

## 23.1 原 CSCEF→Y4 gradient 保留

Control 模型中：

- 与原 C17 对比；
- 在 Adapter zero-init 时：
  - CSCEF→Y4 主梯度应保持；
  - 不再像 v5.2 那样 ×0.25。

## 23.2 Adapter 对输入无额外 gradient

单独 isolating Adapter branch：

```python
delta = adapter(y4.detach())
```

要求：
- y4 不收到 Adapter input branch gradient；
- Adapter 参数有 gradient。

## 23.3 总模型

在非零 Adapter 情况：

- Y4 仍通过 `Y4 + delta` 的 identity 路径收到 CSCEF gradient；
- Adapter 参数可训练；
- SCCA 参数可训练；
- CSCEF 参数可训练；
- 无 NaN/Inf。

---

# 24. Adapter 学习能力验证

构造 synthetic tensor：

```text
Y4_source
Y4_target = fixed affine/channel-mix transform(Y4_source)
```

冻结 source，只训练 SCIAdapter 几步。

证明：
- loss 能下降；
- Adapter 能学习基础 channel shift/mixing；
- 不需要 3×3。

此测试只证明工程学习能力，不是检测性能证据。

---

# 25. Adapter 参数统计诊断

正式训练工具应可额外导出：

- GN scale mean/std
- GN bias mean/std
- reduce weight RMS
- restore weight RMS
- residual RMS / Y4 RMS
- residual mean abs
- channel mean shift
- channel std shift

用于以后判断：

> Adapter 是否真正学到接口校准，还是基本保持 identity。

---

# 26. Adapter residual 安全监控

不加硬 gate，但训练时记录：

\[
r = RMS(\Delta) / (RMS(Y4)+1e-6)
\]

只记录，不限制。

如果正式实验后发现 r 极大，再考虑 v2 residual budget。

本轮不提前加 clamp。

---

# 27. 不做 Dual Semantic Stream

本轮明确禁止实现正式 Dual Semantic Stream。

原因：
- 侵入性更强；
- 双路语义梯度更复杂；
- 当前 Adapter 更符合最小修改原则。

可在文档中保留 future work note。

---

# 28. Git 策略

建议新分支：

`codex/rtdetr-sci-adapter`

建议基点：

优先从已包含：
- 原 C17
- 原 C24
- 原 C19
- 最新服务器工具

的兼容分支建立，但必须审计，确保新正式 YAML 使用原成功模块。

可以从：

`33535ab2a5d9acee4e62d34ae382df8b98b9bbde`

建立新 worktree，因为其中已包含：
- 原 CSCEFv51
- 原 SCCAAIFI
- 原 CBR
- 成熟 sync/eval/pack 工具

但新正式模型禁止引用：
- `CSCEFv52Compat`
- `DRCSCEFv6`
- `GISCCAAIFI`
- `StableReferenceCBR`

建议本地 worktree：

`D:\MyProjects\Crack_RTDETR\outputs\worktrees\sci-adapter`

服务器：

`/root/autodl-tmp/projects/Crack_RTDETR-sci-adapter`

禁止：
- reset --hard
- clean
- stash
- force push
- 覆盖用户文件
- 删除历史结果。

---

# 29. 新 variant IDs

建议：

```text
cscef_v51_sci_control
scca_sci_cscef_v51
scca_sci_cscef_v51_cbr
```

不要叫：
- cscef_v7
- scca_v3

因为模块本体没有变。

---

# 30. 参数量预期

已知：

C2：
`20,082,772`

C17：
约：
`20,109,684`

C24：
约：
`20,148,312`

旧 C25：
约：
`20,175,224`

SCI Adapter：
约：
`17,152`

因此预期：

## Control
`20,126,836`

## Pair
`20,192,376`

## Triad
旧理论 triad：
`20,221,113`

加 Adapter：
约：
`20,238,265`

实际必须审计。

---

# 31. 初始化

所有正式实验仍从 C2 公共初始化：

`weights/rtdetr_r18_lite_imagenet_backbone_init.pt`

SHA256：

`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`

禁止从：
- C17 best
- C24 best
- C25 best
- mincompat best
- compatibility best

继续训练。

新模块：
- GN 默认；
- reduce 正常初始化；
- restore zero-init。

公共权重映射必须逐项审计。

---

# 32. 训练配方

严格与 C2 一致：

```text
epochs=200
patience=50
batch=16
imgsz=640
workers=8
device=0
optimizer=AdamW
lr0=0.0005
lrf=0.01
weight_decay=0.0001
warmup_epochs=5
cos_lr=True
amp=True
seed=42
deterministic=True
close_mosaic=10
```

其他所有 augmentation 从真实 C2 args 继承。

---

# 33. 评估协议

统一：

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
- P
- R
- mAP50
- AP75
- mAP50-95
- PR/F1/P/R curves
- confusion matrix
- args
- checkpoint SHA256。

---

# 34. Round A 正式训练顺序

Codex 不启动。

服务器以后两个 GPU 并行：

## GPU 1

`cscef_v51_sci_control`

意义：
> Adapter control，确认不会破坏 C17。

## GPU 2

`scca_sci_cscef_v51`

意义：
> 真正验证 semantic interface calibration 能否解决 SCCA↔CSCEF incompatibility。

---

# 35. Round A 结果判断

参考：

- C2 = 46.964
- C17 = 51.205
- C24 = 49.987
- C25 = 42.518
- v5.2 standalone ≈ 42.300
- v5.2 pair ≈ 44.025

## Control 理想

`≈50~51+`

如果 control 自身显著掉到 <49：
- Adapter 即使在无 SCCA 时也干扰 C17；
- 需要检查 Adapter residual / training dynamics。

## Pair 理想

`> max(control, C24)`

尤其：
`50.5~52+`

## Pair 只恢复但不累加

例如：
`49~50`

也有价值，说明 interface mismatch 被缓解。

之后再判断是否做 Adapter v2。

---

# 36. 如果 Adapter 仍失败

不要立刻改 SCCA/CSCEF。

下一优先级：

**Dual Semantic Stream**

概念：
- SCCA-enhanced semantic 走主检测流；
- base semantic 走 CSCEF guidance 流；
- 仍保留 CSCEF 原位置。

但本任务不实现。

---

# 37. 服务器工具

建议：

`tools/sync_sci_adapter.sh`

以及：

`tools/autodl_sci_adapter.sh`

接口：

```bash
bash tools/autodl_sci_adapter.sh cscef_v51_sci_control start-direct
bash tools/autodl_sci_adapter.sh scca_sci_cscef_v51 start-direct
bash tools/autodl_sci_adapter.sh scca_sci_cscef_v51_cbr start-direct
```

支持：
- status
- val
- test
- pack-complete。

---

# 38. start-direct 预检

必须检查：

1. FULL SHA
2. rtdetr conda env
3. ultralytics path
4. variant YAML
5. 参数量
6. C2 init source SHA
7. public mapping
8. Adapter final projection zero
9. variant 使用原：
   - SCCAAIFI
   - CSCEFv51
   - RTDETRDecoder / CBR
10. 不引用：
   - CSCEFv52Compat
   - DRCSCEFv6
   - GISCCAAIFI
   - StableReferenceCBR
11. train recipe diff
12. loss/DN smoke
13. optimizer coverage。

---

# 39. 打包

每个包至少：

- training
- val
- test
- console
- YAML
- source snapshot
- git SHA
- initialization
- mapping
- SCI adapter stats
- resolved config
- environment
- curves
- confusion matrix
- manifest
- SHA256
- inventory
- verification。

---

# 40. 文档

新增：

```text
docs/sci_adapter/FAILURE_HISTORY.md
docs/sci_adapter/ARCHITECTURE.md
docs/sci_adapter/GRADIENT_PATHS.md
docs/sci_adapter/VALIDATION.md
docs/sci_adapter/AUTODL.md
```

必须记录：
- aggressive decoupling 失败；
- gradient attenuation 失败；
- 为什么转向 interface adaptation；
- 不能把因果写成“已证明”，应写“实验支持/更一致”。

---

# 41. 本地必须完成的验证

1. C2 regression
2. C17 regression
3. C24 regression
4. C25 replay regression
5. control build
6. pair build
7. triad build
8. zero-init control == C17
9. zero-init pair == C25
10. nonzero adapter affects only semantic proxy
11. no in-place Y4
12. CSCEF→Y4 gradient full strength
13. Adapter input no extra grad
14. Adapter params nonzero grad
15. SCCA params grad
16. CSCEF params grad
17. AMP
18. true half
19. DN dynamic query
20. save/load
21. optimizer coverage
22. nc=1
23. parameter counts
24. rectangular input。

---

# 42. Codex 最终回复

必须包含：

1. 旧两轮失败结论；
2. 为什么本轮不再改 SCCA/CSCEF；
3. SCIAdapter 精确结构；
4. 参数量；
5. zero-init control 与 C17 等价误差；
6. zero-init pair 与 C25 等价误差；
7. nonzero Adapter 验证；
8. gradient path 结果；
9. C2/C17/C24/C25 regression；
10. AMP/half/DN/save-load/optimizer；
11. YAML 名；
12. variant IDs；
13. C2 SHA；
14. base SHA；
15. 新 FULL SHA；
16. push 状态；
17. AutoDL 固定 SHA sync 命令；
18. Round A 两组启动命令；
19. Round B triad 启动命令；
20. val/test/pack 命令；
21. 未验证范围。

---

# 43. 最终研究逻辑

本轮必须始终保持：

```text
原 C24 SCCA
    ↓
    Y4
    ├──────────────→ 原主路径
    │
    └→ SCIAdapter
          ↓
      Y4_compat
          ↓
    原 C17 CSCEF-v5.1
```

Adapter 只解决接口。

**不要再让成功模块本体互相迁就。**

如果成功：

> 下一步再加入原 C19。

如果失败：

> 才进入 Dual Semantic Stream。

---

# 44. 最后约束

请 Codex：

- 不修改原 SCCA；
- 不修改原 CSCEF；
- 不修改原 CBR；
- 不移动模块；
- 不改梯度缩放；
- 不加 attention/gate/spatial conv；
- 不扫 hidden/gamma/beta；
- 不启动正式训练；
- 不完整 val/test；
- 只实现最小 semantic interface calibration；
- 完成后 commit、普通 push。

本轮目标：

> **通过修改 RT-DETR 模块接口，而不是修改成功模块本体，让 SCCA 与 CSCEF 能同时工作。**
