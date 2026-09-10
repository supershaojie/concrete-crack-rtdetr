# RT-DETR C17 / C19 / C24 兼容性排查与三模块重构：Codex 完整实施规范

> **任务类型**：兼容性诊断 + 全局架构重构 + 三个有效创新点的低干扰版本实现 + 全组合配置准备  
> **项目**：混凝土裂缝检测 / RT-DETR-R18-Lite  
> **日期**：2026-09-10  
> **核心目标**：不再把 C17、C19、C24 当成三个孤立“外挂模块”简单串联，而是从整个 RT-DETR 信息流与梯度流出发，把三者重构为一个**可组合、低干扰、主路径保持、稳定参考驱动**的整体架构。  
> **最高优先级**：组合后涨点。允许单模块兼容版比原单模块少涨一点，但希望满足 `C2 < 单模块 < 双模块 < 三模块`，至少避免旧组合中的明显负交互。

---

# 0. 给 Codex 的总命令

请直接执行本文档，不要只输出建议。

本任务必须完成：

1. 审计用户提供的 C17、C19、C24 三个实验包和 RT-DETR 模块包；
2. 审计本地仓库、分支、worktree 和已有 C20/C25/C26 组合实验（若本机存在）；
3. 画清楚 C2、C17、C19、C24 与旧组合的 **forward dependency graph** 和 **gradient dependency graph**；
4. 形成可审计的 `compatibility_audit.md/json`；
5. 在一个全新的兼容性重构分支中实现：
   - **CSCEF-v6 / DR-CSCEF**；
   - **SCCA-AIFI-v2 / GI-SCCA**；
   - **CBR-v2 / SR-CBR**；
6. 同时准备三个单模块、三个两两组合和一个三模块整体模型的独立 YAML；
7. 实现受控初始化、公共权重映射、结构验证、梯度隔离验证、AMP/half、save/load、RT-DETR 原生 loss/DN smoke、参数量审计、服务器同步/训练/val/test/pack 工具；
8. commit 并普通 push；
9. **不要启动正式 200 epoch 训练，不要执行完整数据集 val/test**；
10. 最终返回固定 FULL SHA 和后续 AutoDL 可直接复制的命令。

本任务不是“复制三个旧模块然后写三个 YAML”。  
如果最终代码仍然只是：

`SCCA -> 原 C17 CSCEF -> 原 CBR`

的简单串联，则视为任务没有完成。

---

# 1. 总体研究目标与判断标准

目前已有大量新创新实验，但真正明确单独涨点的核心仍是 C17、C19、C24。

本轮研究策略改为：

> **优先修复已经被真实实验验证有效的机制之间的兼容性，而不是继续盲目增加第四、第五个新模块。**

最终论文的整体网络图也应基于“融合后真正有效的整体 RT-DETR”来画，而不是画成原始 RT-DETR 外面机械挂三个方框。

因此本轮允许：

- 改三模块自身；
- 改模块插入位置；
- 改 Neck/Decoder 输入组织方式；
- 给模块建立稳定参考支路；
- 调整梯度路径；
- 为组合模型重新组织 feature flow；
- 新建整体模型 YAML；
- 修改 parser / decoder 容器以支持新的 feature routing。

但禁止：

- 改数据集；
- 改 train/val/test split；
- 改正式训练超参数；
- 改 loss；
- 改 matcher；
- 改 DN 规则；
- 改 query 数；
- 为了涨点偷偷改 conf/iou/max_det；
- 用测试集扫描结构超参数；
- 从 C17/C19/C24 的 best.pt 继续训练新模型；
- 让新模型依赖用户提供的外部模块包才能运行。

---

# 2. 必须读取的用户附件

用户会把本文档与以下四份附件一起提供给 Codex。文件名可能保留为 UUID 名称，也可能被用户重命名；应优先根据**内部目录结构和内容**识别，不要只依赖外层文件名。

当前附件：

## 2.1 C17：CSCEF-v5.1 完整包

当前外层文件名：

`7c246325-7893-4485-b95d-9eea47a58c0c.gz`

内部根目录应类似：

`c17_cscef_v51_complete_20260906_153449_116232399/`

重点读取：

- `test/metrics_summary.json`
- `training/args.yaml`
- `source/source_commit.txt`
- `source/working_tree.patch`
- `source/files/docs/C17_CSCEF_V51.md`
- `source/files/ultralytics-main/ultralytics/nn/modules/cscef_v5.py`
- `source/files/ultralytics-main/ultralytics/nn/modules/cscef_v51.py`
- `source/files/ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cscef-v51.yaml`
- `metadata/cscef_v51/initialization.json`
- `metadata/cscef_v51/audit.json`

## 2.2 C19：CBR 完整/小包

当前外层文件名：

`3f4ee9f5-eb80-42a1-8893-eedc9e347a08.gz`

内部根目录应类似：

`c19_cbr_test_small_20260907_132852_339107473/`

重点读取：

- `metrics_summary.json`
- `cbr_parameter_stats.json`
- `source/files/docs/C19_CBR.md`
- `source/files/ultralytics-main/ultralytics/nn/modules/cbr.py`
- `source/files/ultralytics-main/ultralytics/nn/modules/transformer.py`
- `source/files/ultralytics-main/ultralytics/nn/modules/head.py`
- `source/files/ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr.yaml`
- `launch_reports/launch_c19/initialization.json`

## 2.3 C24：SCCA-AIFI 完整包

当前外层文件名：

`0038367d-c23a-40b7-977d-f4d2cc1f6d9a.gz`

内部根目录应类似：

`c24_scca_complete_20260908_212957_GmHKX5/`

重点读取：

- `launch_and_evaluation/evaluation_test/metrics.json`
- `launch_and_evaluation/evaluation_val/metrics.json`
- `launch_and_evaluation/initialization.json`
- `launch_and_evaluation/parameter_diff.json`
- `metadata/source_at_commit.tar.gz`
- `metadata/working_tree.patch`

继续打开 `metadata/source_at_commit.tar.gz`，重点读取：

- `docs/scca/README.md`
- `docs/scca/VALIDATION.md`
- `docs/scca/evidence/*`
- `experiment_records/scca_aifi.md`
- `ultralytics-main/ultralytics/nn/modules/scca_aifi.py`
- `ultralytics-main/ultralytics/nn/modules/transformer.py`
- `ultralytics-main/ultralytics/nn/modules/head.py`
- `ultralytics-main/ultralytics/nn/tasks.py`
- `ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-scca.yaml`
- `ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cscef-scca.yaml`

## 2.4 用户 RT-DETR 模块包

当前外层文件名：

`8a64ab86-40ff-4565-8f5e-fc7c5c96f6dd.zip`

内部根目录：

`RTDETR-main/`

仅作为实现和机制参考。

尤其可查看：

- `ultralytics/nn/extra_modules/CAFM.py`
- `ultralytics/nn/extra_modules/FreqFusion.py`
- `ultralytics/nn/extra_modules/FAAFusion.py`
- `ultralytics/nn/extra_modules/MAFusion.py`
- `ultralytics/nn/extra_modules/SFSFusion.py`
- `ultralytics/nn/extra_modules/block.py`
- `ultralytics/nn/extra_modules/attention.py`
- `ultralytics/nn/extra_modules/wtconv2d.py`
- `ultralytics/nn/extra_modules/SFS_MSDeformAttn/`
- `ultralytics/nn/modules/transformer.py`

模块包只用于理解成熟的：

- feature routing；
- 多尺度融合；
- residual side branch；
- sampling / dtype / padding；
- attention 的实现细节。

**禁止**直接复制一个现成模块、改名后宣称为新创新。  
**禁止**把整个 `extra_modules` 移植到当前仓库。  
**禁止**增加当前服务器没有的 CUDA/C++ 依赖。

---

# 3. 已知基准数据：必须自行从附件复核

以下数据是本任务当前已知参考值。Codex 必须从附件/本地原始记录重新读取核对，不能只相信本表。

## 3.1 C2 baseline

C2 源提交：

`67c3078e54a657fd96d65fee657a75fbb1dae0d6`

C2 nc=1 unfused 参数：

`20,082,772`

统一独立 test：

- Precision = `0.8339600078513765`
- Recall = `0.8080444244334384`
- mAP50 = `0.858162985651444`
- AP75 = `0.4583604005521489`
- mAP50-95 = `0.46963623190802783`

## 3.2 C17 / CSCEF-v5.1

独立 test：

- Precision = `0.8448953234408928`
- Recall = `0.825003752063635`
- mAP50 = `0.8792974548298313`
- AP75 = `0.5328297758576255`
- mAP50-95 = `0.5120450850441516`

相对 C2，mAP50-95 约：

`+4.2409 pp`

这是目前三个模块中 **严格定位/AP75 最强** 的候选之一。

C17 新增参数：

`26,912`

## 3.3 C19 / CBR

独立 test：

- Precision = `0.8490338445575341`
- Recall = `0.837311905431872`
- mAP50 = `0.8851863565670782`
- AP75 = `0.5199952552147155`
- mAP50-95 = `0.5038924194554971`

相对 C2，mAP50-95 约：

`+3.4256 pp`

C19 在三个模块中：

- Recall 很强；
- mAP50 很强；
- box refinement 明确有效。

新增参数：

`45,889`

## 3.4 C24 / SCCA-AIFI

独立 test：

- Precision = `0.8555632739746718`
- Recall = `0.819900945520036`
- mAP50 = `0.8782438758197731`
- AP75 = `0.5064610022981055`
- mAP50-95 = `0.4998720584810606`

相对 C2，mAP50-95 约：

`+3.0236 pp`

三个模块中 Precision 最强。

新增参数：

`65,540`

---

# 4. 旧两两组合的已知问题

本地已有历史组合目录，用户当前记录中至少包括：

- `c20 cs+cbr`
- `c25 CSCEF + SCCA`
- `c26 CBR + SCCA`

Codex 必须在用户本地：

- `D:\rtdetr跑结果\`
- 当前项目已有 `runs/`
- 当前各 worktree / experiment_records / docs

中主动寻找这些组合的原始结果包、源码、YAML、训练 args 和 test 指标。

不要要求用户重新提供，除非本机确实找不到任何可用记录。

当前已知历史参考值（**必须复核**）：

| 模型 | P | R | mAP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| C2 | 83.396 | 80.804 | 85.816 | 45.836 | 46.964 |
| C17 | 84.490 | 82.500 | 87.930 | 53.283 | 51.205 |
| C19 | 84.903 | 83.731 | 88.519 | 52.000 | 50.389 |
| C24 | 85.556 | 81.990 | 87.824 | 50.646 | 49.987 |
| C20 = C17+C19 | 83.732 | 81.495 | 86.630 | 49.481 | 48.704 |
| C25 = C17+C24 | 79.384 | 77.713 | 81.848 | 40.498 | 42.518 |
| C26 = C19+C24 | 84.217 | 81.495 | 86.037 | 49.602 | 49.073 |

必须重点解释：

1. 为什么三个单模块都涨，但两两组合没有形成加法收益；
2. 为什么 C17+C24 冲突最严重；
3. 为什么 C17+C19 次之；
4. 为什么 C19+C24 的损失相对较轻；
5. 这些现象究竟来自：
   - forward feature distribution shift；
   - 同一特征被创新模块重复改写；
   - residual stacking；
   - gradient coupling；
   - box over-refinement；
   - decoder feature source 改变；
   - 初始化映射；
   - 还是其他真实代码原因。

不要先入为主把下文假设写成“已证明”。  
必须先做代码与实验记录审计。

---

# 5. 第一阶段：必须先做 Compatibility Audit

在修改任何生产代码前，先创建：

`docs/triad_compat/compatibility_audit.md`

以及：

`docs/triad_compat/compatibility_audit.json`

至少包含以下内容。

## 5.1 C2 拓扑

从当前真实 C2 YAML 和 parser 建立语义拓扑，不只写层号。

当前预期 640 输入：

- Backbone P3/8：约 80×80；
- Backbone P4/16：约 40×40；
- Backbone P5/32：约 20×20。

当前 C2 head 逻辑预期：

```text
Backbone P5
    ↓
input_proj.2
    ↓
AIFI
    ↓
Y5
    ↓
top-down
    ↓
Y4
    ↓
top-down
    ↓
P3_base
    ↓
bottom-up PAN
    ├── P4_base
    └── P5_base
    ↓
RTDETRDecoder(P3, P4, P5)
```

当前 YAML 常见语义点：

- layer 8：projected P5
- layer 9：AIFI
- layer 10：Y5
- layer 12：projected backbone P4
- layer 15：Y4
- layer 16：upsampled Y4
- layer 17：projected backbone P3
- layer 19：final neck P3
- layer 22：final neck P4
- layer 25：final neck P5
- layer 26：RTDETRDecoder

**这些层号只作为当前 C2 预期。生产代码和工具不得把语义写死成这些整数。**

## 5.2 C17 真实 dependency

当前 C17-v5.1 预期：

```text
projected P3 (lateral)
      ┐
      ├── CSCEFv51 ── enhanced P3 lateral
upsampled Y4 (semantic)
      ┘
                     ↓
concat(upsampled Y4, enhanced P3)
                     ↓
RepC3
                     ↓
P3
                     ↓
PAN → P4/P5
                     ↓
Decoder
```

必须验证当前 `CSCEFv51`：

- lateral 输入；
- semantic 输入；
- Scharr/structure confidence 来自哪里；
- semantic content 是否参与 residual；
- residual 是否再次进入 PAN；
- C24 改 AIFI 后 semantic 输入是否随之改变；
- semantic 是否既被 CSCEF 读取、又在随后 concat 中再次使用。

## 5.3 C24 真实 dependency

当前 SCCA 预期替换原 AIFI：

```text
P5 projection
   ↓
SCCA-AIFI
   ↓
Y5
   ↓
Y4
   ↓
upsampled Y4
   ↓
P3 fusion
```

必须验证：

- `scca_channel(src, s)` 的 Q/K/V 来源；
- `delta` 注入点；
- SCCA branch 的 loss gradient 是否会回到 `src` 和 MHA 的 `s`；
- 在 C17+C24 中，SCCA 是否通过 Y5→Y4→layer16 改变 CSCEF semantic 输入。

## 5.4 C19 真实 dependency

当前 CBR 预期：

```text
Decoder input x[0] = final neck P3
       ↓
p3_proj / boundary sampling
final_query
       ↓
condition
final box
       ↓
geometry
       ↓
CBR residual
       ↓
refined final box
```

必须验证：

- CBR 是否读取最终 Neck P3；
- C17/C24 是否会改变该 P3；
- CBR branch 是否通过 p3/query 对主模型产生额外反向梯度；
- box geometry 哪些地方 detach；
- 原始 box 主梯度是否保留；
- `rho=0.10` 对最大边界修正的含义；
- 组合中是否可能出现对已被 C17 改善的 box 再次过修。

## 5.5 画两张图

审计文档必须包含 Mermaid 或 ASCII：

1. **forward dependency graph**
2. **gradient dependency graph**

分别针对：

- C17；
- C19；
- C24；
- C17+C19；
- C17+C24；
- C19+C24。

## 5.6 审计结论门槛

如果真实代码证据支持以下假设，则按本文后续架构实施：

### H1
C17+C24 的主要风险之一是：

`SCCA → Y5/Y4 → CSCEF semantic → CSCEF residual`

同时同一个 `upsampled Y4` 又进入原 concat，形成语义的重复使用和串联改写。

### H2
C17+C19 中：

CSCEF 改写 decoder P3，而 CBR 又把这个已改写 P3 当作边界 reference，且 CBR branch 对 P3/query 产生额外梯度。

### H3
C19+C24 中：

SCCA 改变最终 Neck P3/query，CBR 再以其作为 boundary evidence/condition，因此存在分布耦合，但没有 C17+C24 那么直接。

### H4
三个模块本身不是无效，因为各自单独训练均明显超过 C2；问题更像是**模块之间缺乏稳定参考与梯度隔离**。

如果审计明显推翻其中某项：

- 不得机械照抄本文；
- 必须保留“稳定参考 + 主路径保持 + 低干扰组合”总目标；
- 在 `compatibility_audit.md` 中说明哪项假设被推翻；
- 对后续局部实现做最小必要调整；
- 最终报告明确列出调整理由。

---

# 6. 新整体架构：Decoupled Reference Triad（工作名）

本轮不是简单的三模块串联。

整体重构原则：

## 6.1 Main Path First

原 C2 的主路径尽可能保持：

- Backbone；
- AIFI 主空间 MHA/FFN；
- FPN；
- PAN；
- Decoder；
- loss / matcher / DN。

创新分支优先作为 sidecar residual，而不是取代主路径。

## 6.2 Stable Reference

创新分支尽量从**不被其他创新模块直接改写的中间特征**读取 reference。

本轮指定两个稳定 reference：

- `P4_ref`：Backbone P4 经原 `input_proj.1` 后的 256-channel feature；
- `P3_ref`：Backbone P3 经原 `input_proj.0` 后的 256-channel feature。

在当前 C2 里通常对应 layer 12 和 layer 17，但实现和审计必须按语义识别。

## 6.3 Gradient Firewall

所有创新 side branch 读取主模型特征时，优先使用 read-only feature：

```python
ref = feature.detach()
```

目的：

- 主模型仍通过原主路径正常训练；
- 创新分支自己正常学习；
- 创新分支不再通过额外梯度去拖动它所读取的上游表征；
- 避免三个创新分支形成互相拉扯的梯度回路。

注意：

**不是 detach 整个主模型。**

主路径本身的梯度必须完整保留。

## 6.4 Orthogonal Responsibility

三部分职责尽量分离：

- **SCCA-v2**：P5/AIFI 高层判别关系；
- **CSCEF-v6**：只增强 Decoder 的高分辨率 P3；
- **CBR-v2**：只做最终框边界 refinement。

尤其：

> CSCEF-v6 不再把 residual 继续送进 PAN 去重新构造 P4/P5。

这样高层语义、P3 结构强化、最终 box refinement 不再彼此多次循环改写。

---

# 7. 最终推荐整体拓扑

当前预期整体模型：

```text
                           ┌─────────────────────────────┐
Backbone P5 → input_proj → │ GI-SCCA-AIFI-v2            │
                           │ 原 AIFI 主路 + 隔离通道支路 │
                           └─────────────┬───────────────┘
                                         ↓
                                        Y5
                                         ↓
                               原 C2 top-down FPN
                                         ↓
                 ┌────────────────────── Y4
                 │                       ↓
                 │              原 C2 P3 fusion
                 │                       ↓
                 │                    P3_base
                 │                       ↓
                 │              原 C2 bottom-up PAN
                 │                 ┌─────┴─────┐
                 │                 ↓           ↓
                 │               P4_base     P5_base
                 │
                 │
Backbone P4 → input_proj.1 ──→ P4_ref ──────────────┐
                                                     │
Backbone P3 → input_proj.0 ──→ P3_ref ──────┐       │
                                            │       │
P3_base ────────────────────────────────────┼───────┤
                                            ↓
                                    DR-CSCEF-v6
                               stable refs / read-only
                                            ↓
                                         P3_enh
                                            │
                ┌───────────────────────────┼───────────────┐
                ↓                           ↓               ↓
              P3_enh                     P4_base          P5_base
                └───────────────────────────┬───────────────┘
                                            ↓
                                      RT-DETR Decoder
                                            ↓
                                      final query/box
                                            │
P3_ref (read-only) ─────────────────────────┤
                                            ↓
                                         SR-CBR-v2
                                            ↓
                                       refined box
```

关键点：

1. SCCA 仍然在 AIFI 位置；
2. 原 FPN/PAN 仍完整构造 `P3_base/P4_base/P5_base`；
3. CSCEF-v6 **在 PAN 三尺度已经构造完成之后**才生成 `P3_enh`；
4. `P3_enh` 只作为 Decoder P3 输入；
5. 原 PAN 的 P4/P5 不再由 CSCEF residual 二次影响；
6. CSCEF 的 side branch 使用稳定 `P3_ref + P4_ref`；
7. CBR 使用稳定 `P3_ref`，而不是最终 `P3_enh`；
8. CBR 只对最终 box 做一次有界修正。

这才是本轮的“整体 RT-DETR 重构”。

---

# 8. CSCEF-v6：DR-CSCEF

建议正式代码类名：

`DRCSCEFv6`

工作解释：

**Decoupled-Reference CSCEF v6**

不要覆盖原：

- `CSCEFv5`
- `CSCEFv51`

---

# 9. CSCEF-v6 输入与位置

v6 不再放在旧 C17 的：

`P3_ref + upsampled Y4 -> enhanced lateral -> concat`

位置。

改为：

```text
先完整运行原 C2 Neck：

P3_ref + upsampled Y4
        ↓
      Concat
        ↓
      RepC3
        ↓
     P3_base
        ↓
     PAN → P4/P5

待 P3_base/P4/P5 都构造完：

[P3_base, P3_ref, P4_ref]
        ↓
   DRCSCEFv6
        ↓
     P3_enh

Decoder:
[P3_enh, P4_base, P5_base]
```

因此：

- `base` = 最终原 Neck P3；
- `lateral_ref` = 原 projected backbone P3；
- `semantic_ref` = 原 projected backbone P4。

`semantic_ref` 不再读取：

- Y4；
- upsampled Y4；
- SCCA 后 top-down semantic。

这直接切断：

`SCCA -> CSCEF semantic reference`

的强依赖。

---

# 10. CSCEF-v6 计算：尽量继承 C17 已验证内容

不要重新发明 CSCEF 内核。

尽量复用 C17-v5/v5.1 中已经涨点并验证过的：

- `hidden_channels=32`
- `num_groups=8`
- `eps=1e-6`
- lateral 1×1 projection
- semantic 1×1 projection
- GroupNorm
- concat
- mix 1×1
- depthwise 3×3
- SiLU
- Scharr `/32`
- reflect/replicate boundary
- FP32 structure calculation
- coherence
- reliability
- `sqrt(coherence * reliability)`
- v5.1 的 **per-image H/W mean structure confidence**
- output projection zero-init。

但 forward 改成三输入：

```python
base, lateral_ref, semantic_ref = inputs
```

核心原则：

```python
lateral_side = lateral_ref.detach()
semantic_side = semantic_ref.detach()
```

然后：

```text
l = proj(lateral_side)
s = proj(semantic_side)
semantic spatial size → align to lateral
h = SiLU(GN(DW3(mix(cat(l,s)))))
c = CSCEF-v5.1 mean structure confidence(l)
delta = output_projection(c * h)

output = base + delta
```

注意：

- `base` 不 detach；
- `delta` 可正常对 DR-CSCEF 自身参数反向；
- `lateral_ref/semantic_ref` 仅在 DR-CSCEF side branch 中 detach；
- 两个 ref 在原 C2 主路径中的正常梯度完全保留；
- 不给 side branch 增加额外 attention；
- 不引入新 loss；
- 不加 SE/CBAM/gate；
- 不从模块包复制融合模块。

---

# 11. CSCEF-v6 为什么放到 PAN 之后

这是一项必须贯彻的全局设计，不是实现细节。

旧 C17：

```text
CSCEF residual
   ↓
P3 FPN
   ↓
P3
   ↓
PAN downsample
   ↓
P4/P5
```

因此 C17 会同时改变：

- Decoder P3；
- Decoder P4；
- Decoder P5。

当 SCCA 已经修改高层语义后，再让 CSCEF 从 P3 一路向上改变 P4/P5，容易形成跨尺度互相追逐。

新 v6：

```text
原 C2 PAN 先独立完成 P3/P4/P5
                    ↓
                CSCEF sidecar
                    ↓
              只改 Decoder P3
```

因此职责更正交：

- SCCA 负责高层语义；
- CSCEF 负责 Decoder 高分辨率分支；
- CBR 负责最终 box。

---

# 12. CSCEF-v6 参数与初始化

目标：

尽量保持与 C17-v5.1 相同的新增参数量：

`26,912`

如果因类组织略有差异，必须逐项解释。

要求：

- output projection 全零初始化；
- 其他新 projection 使用和 C17-v5.1 一致的初始化；
- 用 `torch.random.fork_rng(devices=[])` 或现有等价机制隔离新增模块 RNG；
- 初始化时：
  `DRCSCEFv6(base, refs) == base`
- 最终整模型初始化输出应与对应不含 CSCEF 的父模型一致。

例如：

- `C2 + DRCSCEFv6` 初始化时 == C2；
- `SCCA-v2 + DRCSCEFv6` 初始化时 == `SCCA-v2` 父模型（SCCA 的 own zero-init 也应使其最初等价 C2）。

---

# 13. SCCA-AIFI-v2：GI-SCCA

建议类名：

`GISCCAAIFI`

或：

`SCCAAIFIv2`

工作解释：

**Gradient-Isolated SCCA**

不要覆盖原 `SCCAAIFI`。

---

# 14. SCCA-v2 原则：forward 尽量不变，gradient graph 改变

C24 本身单模块已经明显有效。

所以 v2 **不重新设计 SCCA attention**，不贸然把它移到其他层。

保留：

- 256 输入通道；
- SCCA width=64；
- 4 heads；
- 每头16；
- 两个无 affine LayerNorm；
- Q from spatial MHA output；
- K/V from input；
- Q/K 沿 N 去均值；
- L2 normalize；
- 有界 temperature；
- `[B,4,16,16]` channel relation；
- FP32 numerical core；
- `scca_o` zero-init；
- 原 post/pre-norm 行为；
- delta 仍在原 C24 的注入位置。

唯一核心变化：

SCCA side branch 的 source 改为 read-only。

例如：

```python
def scca_channel(self, x, s):
    x_ref = x.detach()
    s_ref = s.detach()

    xn = self.scca_x_norm(x_ref)
    sn = self.scca_s_norm(s_ref)
    ...
```

这样：

- SCCA Q/K/V/O/temperature 自己仍然获得梯度；
- 原 AIFI 主 spatial MHA / FFN 仍从主路径获得正常梯度；
- 但 SCCA branch 不再通过 Q/K/V 额外拉动 `src` 和 `s`。

不要写：

```python
delta = delta.detach()
```

因为那会让 SCCA 自身不能训练。

---

# 15. SCCA-v2 必须做的验证

建立同参数 v1/v2 小模型：

1. 将 v1 SCCA 的所有参数复制到 v2；
2. 相同输入；
3. eval 模式；
4. v1 和 v2 forward 数值应一致（detach 只改 backward graph，不应改 forward value）；
5. 记录 max_abs/max_rel；
6. backward 时：
   - v2 SCCA 参数有梯度；
   - SCCA side path 对 x/s 的额外梯度被切断；
   - AIFI 原主路径梯度仍存在；
7. 初始 `scca_o=0` 时整模型等价 C2；
8. params delta 应保持约：
   `+65,540`。

本轮 v2 **不要同时移动 delta 到 FFN 之后**。

如果后续正式组合仍冲突，未来才考虑：

`SCCA-v3 = post-AIFI residual sidecar`

本任务不提前实现 v3，避免一次改动太多导致无法归因。

---

# 16. CBR-v2：SR-CBR

建议：

`StableReferenceCBR`

与：

`RTDETRDecoderCBRv2`

工作解释：

**Stable-Reference CBR v2**

不要覆盖原 `CrackBoundaryRefinement` / `RTDETRDecoderCBR`。

---

# 17. CBR-v2 的关键修改一：boundary reference 改为稳定 P3_ref

原 C19：

```text
Decoder input x[0] = final Neck P3
       ↓
CBR boundary sampling
```

这意味着：

- C17 改 P3 → CBR reference 跟着变；
- C24 改高层语义/FPN → CBR reference 也跟着变。

v2：

Decoder 接收：

```text
[
  core_P3,
  core_P4,
  core_P5,
  stable_P3_ref
]
```

其中：

- 前 3 个 feature 只用于原 RT-DETR decoder；
- 第 4 个 feature 只用于 CBR。

CBR evidence：

```python
boundary_ref = stable_p3_ref.detach()
```

不要用：

- `P3_enh`
- final Neck P3 作为 CBR reference。

---

# 18. CBR-v2 的关键修改二：Gradient Firewall

CBR side branch 中：

```python
boundary_ref = p3_ref.detach()
query_condition = final_query.detach()
geometry_condition = final_box.detach()
```

CBR 根据它们预测 residual。

但是最终：

```python
original = final_box.float()  # 不 detach
refined = original + residual
```

必须保留。

因此：

- CBR 自身参数能获得 loss gradient；
- CBR branch 不通过 p3/query/box-condition 额外改变核心网络；
- 原 decoder box 仍通过 `refined = original + residual` 保留直接梯度。

必须写专门梯度测试证明：

- `d refined / d original` 主路径仍存在；
- CBR side branch 对 ref/query 的额外梯度为 0；
- CBR 参数梯度非零/有限（考虑 zero-init staged activation）。

---

# 19. CBR-v2 的关键修改三：更保守的 box residual

原 C19：

```python
rho = 0.10
```

v2 正式预设：

```python
rho = 0.075
```

理由：

- C17 已经显著改善严格定位；
- 组合模型不需要 CBR 再进行同等强度的大幅二次修正；
- 从最大几何约束看，单边最大位移从 box width/height 的 10% 降到 7.5%；
- 宽高最极端改变量由约 ±20% 收敛到约 ±15%；
- 这是为“兼容性”先验设定，不允许利用 test 扫描得到。

保留：

```python
normal_fraction = 0.10
```

保留原：

- 36 fixed samples/query；
- 4 sides；
- 3 along positions；
- inside/edge/outside；
- grid_sample；
- `align_corners=False`；
- `padding_mode=border`；
- signed inside-outside evidence；
- side embedding；
- geometry condition；
- offset_out zero-init；
- 最终 xywh residual 公式。

暂时不要加：

- confidence gate；
- IoU predictor；
- quality branch；
- 新 loss；
- learnable rho。

如果 v2 组合仍只有轻微过修，未来再考虑 CBR-v3。

---

# 20. CBR-v2 Decoder 容器

建议：

```python
class RTDETRDecoderCBRv2(RTDETRDecoder):
```

构造器接收 4 个通道：

```text
ch = (P3, P4, P5, P3_ref)
```

内部：

```python
core_ch = ch[:3]
ref_ch = ch[3]
super().__init__(..., ch=core_ch, ...)
self.cbr = StableReferenceCBR(ref_ch, hd, rho=0.075)
```

forward：

```python
core = x[:3]
p3_ref = x[3]
```

必须确认：

- `len(x)==4`
- core P3/P4/P5 空间顺序正确；
- ref P3 与 core P3 spatial size 匹配或按设计验证；
- `_get_encoder_input()` 只收到 core 三尺度；
- DN/query/matcher 逻辑不变；
- 只最终 layer box 被 CBR refine；
- decoder 前几层 reference 不受 CBR feedback；
- eval_idx 规则保持。

参数新增应仍约：

`45,889`

---

# 21. 三个单模块 YAML

必须全部新建，不修改原 YAML。

## 21.1 `rtdetr-resnet18-lite-cscef-v6.yaml`

逻辑：

```text
C2 原 Neck 0...25 全部正常完成
↓
DRCSCEFv6([P3_base, P3_ref, P4_ref])
↓
RTDETRDecoder([P3_enh, P4_base, P5_base])
```

当前 C2 层号预期可表现为：

```text
19 = P3_base
22 = P4_base
25 = P5_base
17 = P3_ref
12 = P4_ref
26 = DRCSCEFv6([19,17,12])
27 = RTDETRDecoder([26,22,25])
```

但代码、审计和初始化映射不能只靠“19/22/25”。

## 21.2 `rtdetr-resnet18-lite-scca-v2.yaml`

只把原 AIFI 替换为 `SCCAAIFIv2/GISCCAAIFI`。

Decoder 保持：

```text
[P3_base,P4_base,P5_base]
```

## 21.3 `rtdetr-resnet18-lite-cbr-v2.yaml`

原 C2 Neck 不变。

Decoder：

```text
RTDETRDecoderCBRv2([
    P3_base,
    P4_base,
    P5_base,
    P3_ref
])
```

---

# 22. 三个两两组合 YAML

全部准备，但本任务不正式训练。

## 22.1 CSCEF-v6 + SCCA-v2

建议：

`rtdetr-resnet18-lite-cscef-v6-scca-v2.yaml`

逻辑：

- layer9 用 SCCA-v2；
- 原 FPN/PAN 完整运行；
- P3/P4/P5 完成后才追加 CSCEF-v6；
- CSCEF stable refs 来自原 projected P3/P4；
- Decoder `[P3_enh,P4,P5]`。

这是最关键的兼容性组合，因为旧 C25 冲突最大。

## 22.2 CSCEF-v6 + CBR-v2

建议：

`rtdetr-resnet18-lite-cscef-v6-cbr-v2.yaml`

Decoder 输入：

```text
[
 P3_enh,
 P4_base,
 P5_base,
 P3_ref
]
```

CBR 的 ref 必须还是 `P3_ref`，不能因为已有 `P3_enh` 就改回 x[0]。

## 22.3 SCCA-v2 + CBR-v2

建议：

`rtdetr-resnet18-lite-scca-v2-cbr-v2.yaml`

Decoder：

```text
[
 P3_base,
 P4_base,
 P5_base,
 P3_ref
]
```

---

# 23. 三模块整体 YAML

建议：

`rtdetr-resnet18-lite-triad-compat-v1.yaml`

工作名：

`Triad-Compat-v1`

不要急着定论文最终模型名。

逻辑：

```text
SCCA-v2
  ↓
原完整 FPN/PAN → P3_base/P4/P5
  ↓
DR-CSCEF-v6(stable P3/P4 refs) → P3_enh only
  ↓
RTDETR Decoder(P3_enh,P4,P5)
  ↓
SR-CBR-v2(stable P3 ref)
```

当前预期：

```text
layer 9  = SCCA-v2
layer 19 = P3_base
layer 22 = P4_base
layer 25 = P5_base
layer 17 = P3_ref
layer 12 = P4_ref
layer 26 = DR-CSCEF-v6([19,17,12])
layer 27 = RTDETRDecoderCBRv2([26,22,25,17])
```

该整体模型不是：

`原 RT-DETR + 三个串联外挂`

而是：

> **高层语义增强 + 原主 Neck + 稳定参考 P3 sidecar + 稳定边界 refinement 的重新路由 RT-DETR。**

---

# 24. 参数量预期

以 C2：

`20,082,772`

为基准。

预期新增：

- DR-CSCEF-v6：约 `26,912`
- SCCA-v2：约 `65,540`
- CBR-v2：约 `45,889`

若三者参数没有额外新增：

三模块理论参数约：

`20,221,113`

Codex 必须实际构建 nc=1 unfused 模型逐项统计：

- C2
- CSCEF-v6
- SCCA-v2
- CBR-v2
- CSCEF-v6+SCCA-v2
- CSCEF-v6+CBR-v2
- SCCA-v2+CBR-v2
- Triad-v1

任何差异必须解释。

不要为了“凑理论值”修改合理实现。

---

# 25. 公共权重与初始化：绝对公平

新模型唯一初始化来源仍应和 C2 相同：

`weights/rtdetr_r18_lite_imagenet_backbone_init.pt`

已知 SHA256：

`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`

不得从：

- C17 best.pt
- C19 best.pt
- C24 best.pt
- C20/C25/C26 best.pt
- 其他创新模型 best.pt

继续训练。

这些 trained checkpoints 只能用于：

- 机制审计；
- 诊断；
- 读取历史指标；
- 非训练分析。

不能成为新模型正式初始化源。

---

# 26. 初始化映射工具

建议实现统一工具：

`tools/init_triad_compat.py`

接口示例：

```bash
python tools/init_triad_compat.py \
  --variant cscef_v6 \
  --source ... \
  --output ... \
  --report ...
```

支持：

- `cscef_v6`
- `scca_v2`
- `cbr_v2`
- `cscef_v6_scca_v2`
- `cscef_v6_cbr_v2`
- `scca_v2_cbr_v2`
- `triad_v1`

逐项分类：

- COMMON
- NEW
- CLASS_ADAPTATION
- MISSING
- UNEXPECTED
- SHAPE_MISMATCH

要求：

1. 公共 state 精确映射；
2. 新参数只来自本轮 side branches；
3. nc80→nc1 的分类相关适配沿用真实 C2 原生规则；
4. 不以 `strict=False` 未报错作为成功依据；
5. 所有公共 tensor 报告 max_abs；
6. 建模 RNG 不得因新模块改变后续公共层随机初始化；
7. 每个 variant 输出独立 initialization report。

---

# 27. Zero-init 等价要求

每个兼容模型在新 residual 输出层 zero-init 时，都必须尽可能退化为父模型。

要求至少验证：

## CSCEF-v6

`C2 + CSCEF-v6` 初始输出 ≈ C2。

## SCCA-v2

`C2 + SCCA-v2` 初始输出 ≈ C2。

## CBR-v2

`C2 + CBR-v2` 初始 final boxes ≈ C2。

## Pair

例如：

`SCCA-v2 + CSCEF-v6`

两边都 zero-init 时 ≈ C2。

## Triad

三者所有 output head zero-init 时：

`Triad-v1 ≈ C2`

比较：

- P3/P4/P5；
- decoder outputs；
- bbox；
- score；
- nested tensors。

记录：

- max_abs_error
- max_rel_error

不得通过 Python：

```python
if all_zero:
    return baseline
```

伪造等价。

必须是结构数学上自然等价。

---

# 28. 梯度隔离测试：本任务最重要的新增测试

新增专门脚本：

`tools/check_triad_compat_gradients.py`

必须验证。

## 28.1 CSCEF-v6

side branch backward：

- DR-CSCEF 参数能训练；
- `P3_ref` / `P4_ref` 作为 side inputs 不接受来自 CSCEF side branch 的梯度；
- 但这些 feature 通过原 C2 main path 仍可获得梯度；
- `P3_base` 主输出保留正常梯度。

建议通过：

- 分离小模块测试；
- 整网 hook/retain_grad 对照；

分别证明。

## 28.2 SCCA-v2

- SCCA 参数获得梯度；
- side branch 对 `src/s` 不产生额外梯度；
- 原 MHA/FFN 主路仍获得正常梯度。

需要有 v1/v2 对照证明 detach 真正生效。

## 28.3 CBR-v2

- CBR 参数能训练；
- `P3_ref` side input 无 CBR branch gradient；
- `final_query` 条件 branch 无额外梯度；
- geometry 条件使用 detached box；
- `original final_box` 通过 `refined=original+residual` 主梯度仍存在。

## 28.4 Triad

整网一次 synthetic/真实小 batch loss backward：

- 三个模块参数都被 optimizer 覆盖；
- 无 NaN/Inf；
- 共享主路梯度正常；
- 不存在某个 innovation branch 意外把主路整个 detach。

---

# 29. Forward 兼容性诊断

实现：

`tools/diagnose_triad_compat.py`

仅轻量验证，不正式训练。

至少记录：

- feature shape；
- feature dtype；
- P3_ref/P4_ref/P3_base/P3_enh 的 RMS；
- CSCEF delta RMS / P3_base RMS；
- SCCA delta RMS / AIFI src/S RMS；
- CBR 平均 abs box correction；
- CBR `abs(tanh)>=0.95` 比例；
- 每个 side branch 是否 finite；
- 同图 batch-invariance（适用时）；
- AMP 与 FP32 对比；
- forward time/peak memory 粗测。

这些是诊断，不是性能证据。

不要把 synthetic benchmark 宣称为 4090 正式训练速度。

---

# 30. CBR rho 的研究纪律

正式 v2 配置固定：

`rho=0.075`

不要用 test 扫：

- 0.05
- 0.075
- 0.10

然后选最好。

若本机存在 C19 trained val checkpoint，可以做**标记为 diagnostic-only** 的 inference intervention，帮助理解修正强度，但不能因此使用 test 选择 rho。

本任务正式实现仍固定 0.075。

---

# 31. Parser / module registry 原则

允许修改：

- `ultralytics/nn/modules/__init__.py`
- `ultralytics/nn/tasks.py`
- 必要 head/transformer helper

但必须：

1. 不改变原模块默认行为；
2. 原 C2 YAML 构建结果不变；
3. 原 C17/C19/C24 YAML 仍能构建；
4. 不 monkey patch；
5. 不让普通 `AIFI` 自动变 SCCA；
6. 不让普通 `RTDETRDecoder` 自动变 CBR；
7. 不让所有多输入模块都走特殊 parser；
8. 新 parser branch 仅针对明确的新类。

---

# 32. 推荐代码文件

按当前仓库风格调整，但优先：

```text
ultralytics-main/ultralytics/nn/modules/cscef_v6.py
ultralytics-main/ultralytics/nn/modules/scca_aifi_v2.py
ultralytics-main/ultralytics/nn/modules/cbr_v2.py
```

测试：

```text
ultralytics-main/tests/test_cscef_v6.py
ultralytics-main/tests/test_scca_aifi_v2.py
ultralytics-main/tests/test_cbr_v2.py
ultralytics-main/tests/test_triad_compat.py
```

工具：

```text
tools/audit_triad_compat.py
tools/init_triad_compat.py
tools/check_triad_compat.py
tools/check_triad_compat_gradients.py
tools/train_triad_compat.py
tools/sync_triad_compat.sh
tools/autodl_triad_compat.sh
```

文档：

```text
docs/triad_compat/compatibility_audit.md
docs/triad_compat/compatibility_audit.json
docs/triad_compat/ARCHITECTURE.md
docs/triad_compat/VALIDATION.md
docs/triad_compat/AUTODL.md
```

---

# 33. Git 方案

本地主项目：

`D:\MyProjects\Crack_RTDETR`

目标远程：

`supershaojie/concrete-crack-rtdetr`

建议新分支：

`codex/rtdetr-triad-compat`

建议独立 worktree：

`D:\MyProjects\Crack_RTDETR\outputs\worktrees\triad-compat`

基点必须追溯到真实 C2：

`67c3078e54a657fd96d65fee657a75fbb1dae0d6`

但在建分支前必须检查：

```bash
git status
git branch --show-current
git worktree list
git remote -v
git log --all --decorate --oneline
```

同时定位：

- C17 source commit；
- C19 source commit；
- C24 source commit；
- C20/C25/C26 组合 commit（若存在）。

原则：

> 从 C2 建干净兼容重构分支，再把 C17/C19/C24 的必要机制按审计结果**选择性重实现/迁入**。

不要直接从某个组合失败分支继续堆改动，因为那可能携带无关实验历史。

禁止自动：

- `git reset --hard`
- `git clean`
- `git stash`
- 删除其他 worktree
- 覆盖用户结果
- force push

---

# 34. 为什么优先一个统一 compatibility 分支

本轮所有 variant 需要共享：

- 同一套新模块；
- 同一套 parser；
- 同一套初始化工具；
- 同一套服务器脚本。

因此优先在一个：

`codex/rtdetr-triad-compat`

实现并审计所有 variant。

每个正式训练仍通过：

- 独立 YAML；
- 独立 run name；
- 独立 output；
- 独立 tmux；
- 固定同一 FULL SHA；

实现严格隔离。

这样可以避免：

“CSCEF-v6 分支和 SCCA-v2 分支又因为 parser / base commit 不同产生新的非实验变量”。

---

# 35. 正式训练配方：必须完整继承 C2

不要只硬编码下面十几个字段。

必须读取 C2 原始 `args.yaml`，对全部字段和类型做 diff。

当前关键值已知：

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
momentum=0.937
weight_decay=0.0001
warmup_epochs=5

hsv_h=0.015
hsv_s=0.5
hsv_v=0.35
translate=0.1
scale=0.4
shear=1.5
perspective=0.0002
flipud=0.2
fliplr=0.5
mosaic=0.8
mixup=0.05
copy_paste=0
auto_augment=None
erasing=0
```

训练 data：

`/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml`

project：

`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series`

允许变化：

- model YAML；
- run name；
- save_dir；
- 为不同 variant 必需的模型结构。

禁止变化：

- dataset；
- split；
- epochs；
- batch；
- imgsz；
- optimizer；
- lr；
- augmentation；
- seed；
- patience；
- AMP；
- loss/matcher；
- query；
- DN。

---

# 36. 评估协议

独立 val/test 必须沿用统一入口。

当前已知 test 关键设置：

```text
split=test
imgsz=640
batch=16
device=0
seed=42
conf=0.001
iou=0.7
max_det=300
half=False
augment=False
```

workers 按当前统一评估脚本实际规则。

必须保留：

- Precision
- Recall
- mAP50
- AP75
- mAP50-95
- PR curve
- F1 curve
- P curve
- R curve
- confusion matrix
- results.csv
- args.yaml
- checkpoint SHA256
- effective eval args

不能调 threshold 制造提升。

---

# 37. 不用 test 做架构搜索

本轮兼容版本的选择优先依据：

- val；
- 训练曲线；
- 机制诊断；
- P/R/AP75/mAP50-95 的一致性。

不要在 test 上：

- 扫 rho；
- 扫 residual scale；
- 扫插入层；
- 扫多个 v6 子版本后挑最好。

test 用于最终确定的候选进行统一报告。

已有历史 test 已经被使用过，但新一轮代码工具仍应尽量避免进一步 test-driven tuning。

---

# 38. 必须创建 7 个 variant

注册统一 variant ID：

```text
cscef_v6
scca_v2
cbr_v2
cscef_v6_scca_v2
cscef_v6_cbr_v2
scca_v2_cbr_v2
triad_v1
```

每个 variant 要有：

- YAML
- expected topology
- parameter count
- initialization mapping
- run name template
- log dir
- evaluation dir
- package dir

推荐统一 registry，避免七套脚本复制粘贴漂移。

---

# 39. 服务器工具

建议：

```bash
bash tools/autodl_triad_compat.sh <variant> start-direct
bash tools/autodl_triad_compat.sh <variant> status
bash tools/autodl_triad_compat.sh <variant> val
bash tools/autodl_triad_compat.sh <variant> test
bash tools/autodl_triad_compat.sh <variant> pack-complete
```

例如：

```bash
bash tools/autodl_triad_compat.sh cscef_v6 start-direct
bash tools/autodl_triad_compat.sh scca_v2 start-direct
bash tools/autodl_triad_compat.sh cscef_v6_scca_v2 start-direct
bash tools/autodl_triad_compat.sh triad_v1 start-direct
```

每个 variant：

- 独立 tmux 名；
- 独立 log；
- 独立 run dir；
- 独立 lock/state；
- 可以并行；
- 不 kill 其他实验；
- 不改 GPU 参数；
- OOM 就失败并保留日志，不自动降 batch。

---

# 40. 服务器环境

服务器主仓库：

`/root/autodl-tmp/projects/Crack_RTDETR`

建议兼容性 worktree：

`/root/autodl-tmp/projects/Crack_RTDETR-triad-compat`

服务器必须使用已有：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
```

运行时应检查：

```text
python executable
torch version
cuda version
GPU
ultralytics.__file__
new module __file__
git FULL SHA
```

不得：

- pip install；
- conda install；
- 升级 torch；
- 新建环境；
- 切到错误 ultralytics。

---

# 41. `sync_triad_compat.sh`

要求：

```bash
bash tools/sync_triad_compat.sh <FULL_SHA>
```

必须：

1. fetch `codex/rtdetr-triad-compat`；
2. 验证 FULL SHA 属于该分支历史；
3. 建立/复用独立 worktree；
4. checkout 固定 SHA；
5. 支持 detached HEAD；
6. 已有 worktree 有修改时停止；
7. SHA 不一致时停止并报告；
8. 不 reset/clean；
9. 不删除其他 worktree；
10. 不覆盖用户文件；
11. 不启动训练。

---

# 42. `start-direct` 预检

每个 variant 正式启动前必须验证：

1. 当前 FULL SHA；
2. rtdetr 环境；
3. YAML 可构建；
4. nc=1；
5. 参数量；
6. C2 初始化源 SHA；
7. controlled init report；
8. public mapping 无异常；
9. 训练 args 全字段与 C2 对齐；
10. data YAML；
11. run dir 不存在或符合不可覆盖规则；
12. tmux 不冲突；
13. 没有同 variant 活跃 process；
14. AMP smoke；
15. RT-DETR 原生 loss/DN smoke。

通过后才启动。

---

# 43. status

至少区分：

```text
NOT_STARTED
DISPATCHED
RUNNING
SUCCESS
FAILED
```

不能：

- tmux 存在 = SUCCESS；
- best.pt 存在 = SUCCESS；
- results.csv 存在 = SUCCESS。

必须结合：

- process token；
- exit code；
- worker completion；
- train artifact completeness。

---

# 44. val / test / pack 顺序

建议强制：

```text
train SUCCESS
    ↓
val COMPLETE
    ↓
test ALLOWED
    ↓
pack-complete
```

test 前必须核对：

- 选择的 checkpoint；
- checkpoint SHA256；
- val 使用的是同一个 checkpoint。

pack-complete 不应自动替用户补跑 val/test。

---

# 45. 完整打包

输出建议：

`/root/autodl-tmp/projects/Crack_RTDETR/downloads/triad_compat/<variant>/`

每个包至少包含：

```text
training/
val/
test/
console/
model YAML
source snapshot / patch
git FULL SHA
C2 SHA
initialization report
weight mapping report
compatibility audit
topology report
gradient firewall report
resolved train config
effective eval config
environment
parameter report
best.pt / last.pt（若当前项目打包政策允许）
results.csv
args.yaml
results.png
PR/F1/P/R curves
confusion matrices
manifest
SHA256
inventory
verification
```

若现有统一政策不把大权重放下载包，则保持当前项目一致做法，不要为了本文强行改变。

---

# 46. 本地必须完成的验证

本任务不启动正式训练，但至少完成：

## 46.1 构建

7 个 YAML 全部能构建 nc=1 RTDETRDetectionModel。

## 46.2 原 C2 regression

原：

`rtdetr-resnet18-lite.yaml`

构建和 forward 未被新 parser 修改。

## 46.3 老模块 regression

尽可能确认原：

- C17 YAML
- C19 YAML
- C24 YAML

仍可构建/运行，不被新类污染。

## 46.4 Shape

640 输入预期：

- P3 80×80
- P4 40×40
- P5 20×20

同时测试矩形输入对应 shape。

## 46.5 zero-init

7 variant 的 initial equivalence。

## 46.6 gradient firewall

按第 28 节执行。

## 46.7 DN

真实或合理 synthetic batch，确认训练时 DN query 数变化不会让 CBRv2 写死 Q=300。

## 46.8 AMP

CUDA 可用时：

- FP32
- AMP forward/backward
- true half inference

无 CUDA 时明确 NOT_RUN。

## 46.9 Save/load

保存已人为改成非零的新模块参数，再 reload，参数和输出保持。

不能 reload 时再次 zero-init 已学习参数。

## 46.10 Optimizer coverage

真实 `Trainer.build_optimizer`：

- 每个新可训练参数恰好覆盖一次；
- bias/norm/weight decay 分组符合原项目规则；
- 无漏参；
- 无重复参数。

---

# 47. Compatibility-specific 单元测试

至少加入以下测试。

## 47.1 SCCA 不再额外改动 AIFI source gradient

在小图/小 token 中：

- v1 side branch 对 source 梯度非零；
- v2 side branch detach 后该额外梯度为 0；
- v2 SCCA 参数仍有梯度。

## 47.2 CSCEF stable reference

改变 SCCA/Y4 feature，但保持：

`P3_ref/P4_ref/P3_base`

不变时，CSCEF-v6 side residual 不应因 Y4 改变。

证明 v6 不再依赖旧 semantic layer16。

## 47.3 CSCEF 不污染 PAN

hook：

- 构造 P4/P5 时没有经过 DR-CSCEF-v6 output；
- Decoder P3 使用 P3_enh；
- Decoder P4/P5 使用 original PAN outputs。

## 47.4 CBR stable ref

人为改变 core P3，而固定 P3_ref/query/box：

CBR boundary sampling evidence 应保持。

人为改变 P3_ref：

CBR evidence 应改变。

## 47.5 CBR gradient

如第18节。

## 47.6 Triad topology

证明三者同时启用时：

- SCCA only AIFI；
- CSCEF only decoder P3 sidecar；
- CBR only final box；
- stable P3/P4 ref 正确。

---

# 48. 不要过度改动的地方

本轮特意避免“一次把一切都重写”。

因此不要顺手加入：

- LIF-Down
- FSA-Deform
- LCR-AIFI
- RSC-Head
- EVC
- DRA
- OBP-AIFI
- 其他 attention
- 新 backbone
- 新 loss
- 新 matcher
- 新 query selector
- Focal/IoU loss 改动
- 训练参数调优

FSA 和 LIF 当前可以在服务器独立运行，但与本兼容性重构实验完全隔离。

本轮只研究：

> 如何让已经证明有效的 C17/C19/C24 互相不打架。

---

# 49. 模块包使用规则

可以从用户模块包学习：

- stable feature routing；
- sidecar residual 写法；
- dtype；
- interpolation；
- sampling；
- feature fusion 代码规范。

但此次核心创新不应变成：

“从 CAFM/FAAFusion/FreqFusion 拿一个 fusion 放进去”。

如果模块包中存在更好的工程实现方式，可以借鉴代码组织，但必须：

1. 在文档中记录参考文件；
2. 说明借鉴了什么实现思想；
3. 说明没有复制什么；
4. 当前项目独立运行；
5. 不引入重依赖。

---

# 50. 诊断后允许的小范围修正

本文给出的 DR-CSCEF / GI-SCCA / SR-CBR 是目标方案。

Codex 可以在审计后做**小范围必要修正**，例如：

- 当前 parser 的真实参数顺序不同；
- 当前 P3/P4 semantic index 与预期不同；
- 当前 CBR 已经另有 decoder wrapper；
- 当前 common state mapping 需要层号偏移；
- 当前 C2 是 pre-norm 而不是 post-norm；
- 某个 old package 与本地 source commit 有一处差异。

但不能擅自：

- 换掉核心机制；
- 加第四个模块；
- 加新 attention；
- 大幅搜索超参数；
- 直接把 CSCEF-v6 又放回 old C17 位置；
- 让 CBR 又读取 final P3；
- 取消 gradient firewall；
- 让 CSCEF residual 再进入 PAN。

若认为必须偏离这些核心点，先在最终回复中列出证据，不要悄悄修改。

---

# 51. 实验执行顺序：代码现在全部准备，正式训练按阶段进行

Codex 本任务只准备，不启动。

后续用户正式实验建议顺序：

## Round A：两个最关键单模块兼容版

并行：

```text
CSCEF-v6 standalone
SCCA-v2 standalone
```

目的：

确认兼容化之后没有把原有有效机制完全破坏。

期望：

- 不要求达到原 C17/C24 的全部涨幅；
- 但优先希望仍明显 > C2。

## Round B：最关键冲突组合 + CBR-v2 standalone

并行：

```text
CSCEF-v6 + SCCA-v2
CBR-v2 standalone
```

因为旧 C17+C24 崩得最严重。

如果新的 CSCEF+SCCA 不再崩，而且：

`mAP50-95(pair) > max(mAP50-95(single A), single B)`

则说明兼容化方向真正成功。

## Round C：三模块整体

优先：

```text
Triad-v1
```

若 Triad-v1 明显 > 最强 pair，再补全论文所需的另外两个 pair：

```text
CSCEF-v6 + CBR-v2
SCCA-v2 + CBR-v2
```

如果时间允许，也可以先补 pair 再 triad。

---

# 52. 结果判定核心

本研究不再只追求“某一个模块单独最高”。

最理想的消融趋势：

```text
C2            < A
A             < A+B
A+B           < A+B+C
```

或者：

```text
C2            < A
C2            < B
C2            < C
max(A,B)      < A+B
max(A+B,...)  < A+B+C
```

我们接受：

- C17 原来 +4.24pp；
- 新 CSCEF-v6 可能只 +2～3pp；

只要最终组合形成更高的整体结果。

例如以下趋势比旧实验更有价值：

```text
C2       46.96
A        49.3
A+B      51.0
A+B+C    52.2
```

而不是：

```text
A        51.2
B        50.0
A+B      42.5
```

---

# 53. 哪些结果说明值得继续做 v2/v3

正式训练以后，不是失败就立刻换新创新。

## CSCEF-v6

若：

- AP75 仍明显涨；
- Recall 或 mAP50-95 轻微掉；

可考虑 future v6.1：

- 调 residual location 的轻量细化；
- 不回到 old semantic dependency。

## SCCA-v2

若：

- Precision 继续涨；
- Recall 小幅降低；
- pair 已变得可组合；

优先保留，不为了单模块再追原 C24 最大值破坏兼容性。

## CBR-v2

若：

- Recall/strict IoU 仍涨；
- 单模块涨幅比 C19 小；
- triad 进一步增加；

就是理想结果。

若仍发生过修，再考虑 future CBR-v3：

- rho 更保守；
- reliability-aware correction。

## Pair/Triad

如果组合已经消除大跌但还没超过强单模块：

先分析：

- residual RMS；
- CBR box correction；
- gradient；
- AP75 vs mAP50；
- P/R tradeoff；

有明确机制再做下一版，不盲换新模块。

---

# 54. 当前阶段不要画最终论文图

可以生成：

- Mermaid architecture；
- debug topology；

但不要花时间做最终论文级模块图。

只有当：

`Triad-v1` 或后续兼容版组合真实涨点

之后，才基于最终模型画完整网络图。

最终图应表现：

- RT-DETR 主干；
- SCCA 的高层语义作用；
- Stable P3/P4 reference；
- late P3 sidecar；
- decoder；
- stable boundary refinement；

而不是“原模型旁边放三个外挂框”。

---

# 55. 审计与源码可追溯要求

最终提交中保存：

- C2 full SHA；
- C17 source SHA；
- C19 source SHA；
- C24 source SHA；
- C20/C25/C26 source SHA（若找到）；
- 用户附件 SHA256；
- 模块包 SHA256；
- 新 branch full SHA；
- 每个新文件 SHA256；
- 每个 YAML topology；
- 每个 variant 参数量；
- initialization mapping；
- gradient firewall report。

不要提交用户的巨大结果包和训练权重到 Git。

---

# 56. Codex 本地工作流程

开始：

```text
1. 读 AGENTS.md（若存在）
2. git status
3. git worktree list
4. git remote -v
5. git branch -a
6. git log --all
7. 识别 C2/C17/C19/C24/C20/C25/C26
8. 读取四个用户附件
9. 写 compatibility audit
10. 新建独立 compatibility worktree
11. 实现三个新模块
12. 实现 7 YAML
13. 实现 init/audit/train/server tools
14. 跑本地轻量验证
15. git diff/status
16. commit
17. 普通 push
18. 核对远端 FULL SHA
```

---

# 57. 不能做的 Git 操作

禁止：

```text
git reset --hard
git clean
git stash
git checkout -- .
git restore .   （针对用户已有工作）
force push
删除未知 worktree
删除用户结果目录
覆盖现有下载包
```

若当前目录不干净：

- 保留；
- 新建独立 worktree；
- 不处理用户已有修改。

---

# 58. 最终 commit

建议 commit message：

```text
feat(rtdetr): decouple CSCEF SCCA CBR for compatible triad
```

普通 push：

```text
origin codex/rtdetr-triad-compat
```

不 force。

---

# 59. Codex 最终回复必须包含

完成后，最终只需清晰汇报：

1. compatibility audit 的核心结论；
2. 旧 C17/C19/C24 冲突的真实证据；
3. 是否找到 C20/C25/C26，实际指标是什么；
4. DR-CSCEF-v6 实现；
5. GI-SCCA-v2 实现；
6. SR-CBR-v2 实现；
7. Triad-v1 实际 topology；
8. 7 个 YAML 文件名；
9. baseline/各 variant 实测参数量；
10. zero-init 等价误差；
11. 公共权重映射结果；
12. gradient firewall 测试；
13. AMP/half/DN/save-load/optimizer coverage；
14. 修改/新增文件清单；
15. C2 FULL SHA；
16. 新 branch FULL SHA；
17. push 状态与远端 SHA；
18. 未验证内容；
19. AutoDL 首次同步固定 SHA 命令；
20. Round A 的两条启动命令；
21. 后续 Round B / Round C 启动命令；
22. val/test/pack 命令。

---

# 60. AutoDL 最终接口期望

Codex 最终应实际给出类似以下命令，但必须替换为**真实 FULL SHA**，不要使用占位符。

同步：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR
git fetch origin codex/rtdetr-triad-compat
git show <FULL_SHA>:tools/sync_triad_compat.sh | bash -s -- <FULL_SHA>
```

Round A：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-triad-compat

bash tools/autodl_triad_compat.sh cscef_v6 start-direct
bash tools/autodl_triad_compat.sh scca_v2 start-direct
```

状态：

```bash
bash tools/autodl_triad_compat.sh cscef_v6 status
bash tools/autodl_triad_compat.sh scca_v2 status
```

Round B：

```bash
bash tools/autodl_triad_compat.sh cscef_v6_scca_v2 start-direct
bash tools/autodl_triad_compat.sh cbr_v2 start-direct
```

Round C：

```bash
bash tools/autodl_triad_compat.sh triad_v1 start-direct
```

训练成功后：

```bash
bash tools/autodl_triad_compat.sh <variant> val
bash tools/autodl_triad_compat.sh <variant> test
bash tools/autodl_triad_compat.sh <variant> pack-complete
```

Codex 最终必须给出无占位符的首次同步命令；variant 名可以保留为参数。

---

# 61. 研究边界与最终目标

本轮最重要的思想不是：

> “把三个涨点模块全加上，看看会不会更高。”

而是：

> **三个单模块都已经证明某种能力有效；现在要重新设计 RT-DETR 内部的信息路由，让它们分别在最适合的位置工作，并通过稳定 reference 和 gradient firewall 避免彼此破坏。**

三个能力应该最终对应：

```text
SCCA-v2
= 高层判别语义

DR-CSCEF-v6
= 高分辨率结构/严格定位增强

SR-CBR-v2
= 最终边界局部修正
```

整体模型应体现：

```text
Semantic discrimination
        +
Stable structural reference
        +
Boundary refinement
```

而不是三次无约束的 feature rewriting。

如果后续正式结果证明：

- 单模块兼容版略低于旧 C17/C19/C24；
- 但双模块和三模块出现稳定累加；

则本轮设计成功。

最终论文更值得强调的是：

> **从三个独立有效但互相冲突的增强机制，重构为一套低干扰、职责解耦的 RT-DETR 多阶段协同检测架构。**

这比继续增加大量互不相关的新模块更有研究价值，也更适合作为最终整体网络图与消融实验主线。

---

# 62. 最后约束

请 Codex：

- 先审计，后改代码；
- 不跳过附件；
- 不把假设当事实；
- 不启动正式训练；
- 不自动完整 val/test；
- 不修改正式配方；
- 不破坏其他实验；
- 不把代码做成不可审计的大改；
- 以“最终三模块组合涨点”而不是“单模块最高”作为架构设计优先级。

完成后 commit、普通 push，并按第59节格式汇报。
