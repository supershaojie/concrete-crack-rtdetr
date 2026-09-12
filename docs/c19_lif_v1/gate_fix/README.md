# FP16 截止位门禁工程修复

在 `codex/rtdetr-c19-lif-v1` 的 `216bba006a61382ebecef4bc5eb44374e67a2c80` 上追加工程修复。生产模型、YAML、CBR/LIF、top-k、融合数学、独立评估及正式配方均未修改。没有启动正式训练或完整 val/test。服务器操作以本目录 [AUTODL.md](AUTODL.md) 为准，不重复执行历史恢复命令或两遍完整预检。

## 原 LIGHT 证据复核

原包 120965 字节，SHA256 `b1485f1349fac86bac33ba07409840364076a33974814835ac0de2ef121d0aa8`。四份完整逐模式报告逐字节保存于 [light_fixture](light_fixture)，哈希见 [manifest.json](light_fixture/manifest.json)。超长总报告仅有 excerpt，不作为完整 JSON 解析。原报告的 BLOCKED 状态未改写；独立审查见 cutoff_tests.json 的 LIGHT_original_review。

| 原服务器模式 | operator | natural | acceptance |
|---|---|---|---|
| CPU FP32 | OPERATOR_CHECK_PASSED | PERMUTATION，2 个位置，300 个共同候选 | PASSED |
| CUDA FP32 | OPERATOR_CHECK_PASSED | IDENTICAL | PASSED |
| CUDA AMP | OPERATOR_CHECK_PASSED | PERMUTATION，214 个位置，300 个共同候选 | PASSED |
| CUDA true-half | OPERATOR_CHECK_PASSED（旧版已有检查范围） | NATURAL_SELECTION_DRIFT，228 个位置，299 个共同候选 | BLOCKED，缺少新增动态证据 |

这次 CPU 自然 raw_boxes 最大差 `0.20000004768371582`，按 candidate ID 对齐后为 0，最终 boxes 差 `2.9802322387695312e-08`，排序前 logits 差 `5.960464477539062e-07`。它解释了本次记录的同类、同幅度 0.2；第一次遗失张量的旧 fixture 没有被逐比特重放。

原 true-half 仅替换一个候选：A_only=[405]、B_only=[194]。ma=`0.00048828125`，mb=0，cutoff_gap A=`0.00048828125`、B=0，全局 logits 扰动 `0.00146484375`。同次 C2 的两侧完整 300-ID 序列分别与 Pair 完全一致。原 A 集合双方重放 raw_boxes 差 0、最终 boxes 差 `1.1883676052093506e-06`；旧包没有 Pair B 重放、C2 A/B 全链重放和新的 FP64 局部证据，不能直接升级旧报告为成功。

## 新判据

`c19_lif_v1_cutoff.py::review_evidence` 同时供融合诊断和真实 `require_preflight` 调用。原 PASS 集合没有加入 REQUIRES_REVIEW。CPU/CUDA FP32 和 AMP 仍要求原严格连续比较及相同集合的 ID 对齐。

只有 CUDA true-half 可以返回 `PASS_WITH_BASELINE_CUTOFF_TIE / OPERATOR_CHECK_PASSED / NATURAL_SELECTION_DRIFT / ACCEPTED_WITH_WARNING`，且必须同时满足：

1. 完整 schema、实际 gather/top-k 计数与 IDs、合法唯一 300 候选、shape、dtype、finite/mask、LIF BN 和全部 LIF 状态不变；普通 BN 融合实际发生。
2. Pair 排序前连续张量全部通过。分别将自然 A 和自然 B 的完整集合在双方模型重放，selected tensors、reference、三层 query、raw 输出、CBR 前后、最终输出均通过。只替换诊断所用 IDs，不替换双方各自 feature。
3. 统计只对已经生成的 logits 做 FP64 运算，保存全部 630 分数、ID、k/k+1、并列数、ULP。每一对交换计算 `ma=sA[i]-sA[j]`、`mb=sB[j]-sB[i]`、`e_local=abs(sA[i]-sB[i])+abs(sA[j]-sB[j])`。slack=`8*eps(float64)*max(abs(produced logits), tiny(float64))`，不再用放大的 FP16 epsilon。
4. 除局部扰动解释外，交换成员距截止位不得超过原 dtype 的一个局部 ULP，两边 k/k+1 gap 也不得超过各自 ULP。此处是窄化边界规则，不是放宽 allclose，不允许稳定高分候选丢失。
5. C2 来自同次 FP32 公共源状态，533 项 shape/value 精确映射，Pair cast 逐项相等，记录源状态 SHA256；相同输入/RNG/精度，原生 FP32 fuse→half。C2 排序前严格比较，A/B 两组全链重放；C2 两侧完整自然 ID 分别与 Pair 相等，截止 gap 和并列数一致。C2 不使用人工自然候选。

缺字段、越界 ID、dtype/mask 错误、Pair/C2 任意连续超差、只 A 通过、C2 列表不同、稳定候选变化或其他精度集合漂移，仍 BLOCKED，并记录具体原因。自然输出差值保留。这一告警不证明检测指标相等，也不是训练完成的 C2 性能消融。

容差保持：融合 FP32 `atol=2e-5, rtol=2e-4`；half/AMP `atol=3e-3, rtol=3e-2`；原其他比较不变。没有 model.double() 或生产候选固定。

## 本地验证及服务器边界

[local_checks.json.gz](local_checks.json.gz) 是完整可解析 JSON 的 gzip，包含原始比较和 loss/DN/梯度证据；不是权重/张量包。环境为 Python 3.9.25、Torch 2.7.1+cu118、RTX 2060。报告中的 runtime.commit 为测试时未提交补丁所基于的 216bba...，没有伪造为后续提交 SHA；实际源码哈希另见 audit.json。最终只增加共享 fixture 的重放元信息和收紧缺失 dtype/finite 字段拒绝规则；共享 fixture 重放和真实证据门禁另行验证。

| 本地普通有限检查 | operator | natural | acceptance |
|---|---|---|---|
| CPU FP32 | OPERATOR_CHECK_PASSED | IDENTICAL_CANDIDATES | PASSED |
| CUDA FP32 | OPERATOR_CHECK_PASSED | IDENTICAL_CANDIDATES | PASSED |
| CUDA AMP | OPERATOR_CHECK_PASSED | CANDIDATE_PERMUTATION | PASSED |
| CUDA true-half | OPERATOR_CHECK_PASSED | CANDIDATE_PERMUTATION | PASSED |

本地普通输入没有产生原服务器那次截止替换。另以**仅测试用 logits 注入**制造一个相邻 FP16 ULP 的截止并列；仍调用原生 top-k/gather 和原模型连续算子，完成 Pair/C2 双集合重放，得到带告警接受。完整证据见 [synthetic_cutoff_evidence.json.gz](synthetic_cutoff_evidence.json.gz)，不能冒充原 AutoDL fixture 的新动态结果。该测试 ma=`0.00048828125`、mb=0、e_local=`0.00048828125`、slack=`1.3322676295501878e-15`；533 项公共状态精确映射，Pair A/B、C2 A/B 全链均无超差。候选 ID 由原生排序选择，代码没有 405/194 许可名单。

CPU loss、CUDA loss 和 CUDA AMP loss 均实际各执行三次有限 optimizer smoke，原生 loss/backward、两模块梯度、动态 DN、保存/重载均通过。复用一次性 AMP smoke 副本，六个原零初始化 bbox 投影张量变为非零；其更新后 FP32 融合检查通过。

原生 EMA 验证入口使用两张训练图完成半精度 forward、原生 loss、RTDETR postprocess、finite/shape 检查及 FP32 恢复；不计算数据集指标。AutoBackend 原生 fuse→half 入口通过，输出 `[1,300,5]`，CBR 输出 FP32 为原设计，LIF BN 保留。

**本地 B16/640 容量为 NOT_RUN。旧服务器的 CUDA loss、AMP loss、后续 DN、B16 也仍为 NOT_RUN。** 新服务器必须在一次 start-direct 内实际执行四模式诊断、后续 CUDA loss/DN/精度入口及原 B16/640 AMP loss/backward；全部通过才派发。未降低 batch、未关闭 AMP、未预先保证服务器通过。

测试证据：27 项 cutoff 正反例（cutoff_tests.json）；30 项原故障/真实算子注入回归（original_fault_tests.json）；11 项真实启动谓词测试（gate_flow_tests.json）；单次 checker→单次 dispatch、失败保留与状态生命周期（lifecycle_tests.json）；6 项小包测试（light_tests.json）。启动谓词测试的 B16 元数据和生命周期的 subprocess/dispatch **明确使用 mock**，不算实际容量或训练成功。诊断后 hooks、top-k 包装、原生 forward、CPU/CUDA/Python/NumPy RNG 均恢复。

## 实际源码精度路径

| 入口 | 本仓库实现及精度 | fuse |
|---|---|---|
| 训练 forward/loss | trainer.py:325/334/429，FP32 参数 + AMP autocast + 原生 GradScaler | 否 |
| EMA 每 epoch 验证 | trainer.py:747 → validator.py:147/151/222，CUDA 且 trainer.amp 时 model.half()，输入 half、原生 loss，结束 model.float() | 否 |
| trainer.final_eval | trainer.py:832 复用 validator；沿用上一轮 args.half，加载保存的 half EMA，load_checkpoint 先 float，AutoBackend 再 half | FP32 fuse→half |
| 独立 val/test | c19_lif_v1_results.py:26/98 固定 half=False，原独立评估政策 | FP32 fuse，保持 FP32 |

AutoBackend 默认 fuse=True（autobackend.py:163/223/228/235），checkpoint load 的 float/fuse 在 tasks.py:1522/1531。以上来自本项目源码读取，没有用最新版网页替代。原生半精度入口测试的源码 SHA 保存在报告内。

## 不变量与现场保留

nc=1 未融合参数量 **20,149,765**。109 字段原配方保持 200e、640、batch16、AdamW、seed42、amp=True。CBR 仍为最终 Neck P3、rho=0.10、36 点采样及原几何/梯度规则；LIF 仍为 BN 前残差。相对 216bba...，整个 ultralytics-main、configs、原初始化和独立评估工具均无差异；recipe 函数与原版本 AST 相同。

LF 规范化 SHA256：LIF `26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7`；CBR `d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787`。Windows CRLF 的原始文件字节哈希单独记录，不能混淆为模型修改。

每个 device 的多精度诊断共享一个 FP32 snapshot；成功只保存 JSON，不再每个成功模式写两套模型和大 records。失败仍保留服务器本地完整 records，必要时引用共享 snapshot。`diagnose_c19_lif_v1_fusion.py --fixture ... --precision half` 可读取新共享现场；这是可选排错入口，不是推荐启动前额外必跑步骤。

`pack_c19_lif_v1_light.py` 硬上限 **8,000,000 字节**，包括压缩归档全部内容；超限不输出半成品。默认只含必要 JSON/源码/元数据及每个日志 32 KiB tail，完整保留逐模式 630 logits 与 A/B 证据；超长总报告使用带 omission 字段的合法结构化 JSON。排除 .pt、权重、大 records、图像、数据集及旧归档。此次现场打包测试为 66929 字节。原包、旧大包、日志、张量、旧 worktree 和锁归档全部保留；没有操作 C17。
