# R1 实施与证据边界

本次依据用户在本任务中的明确批准实施 R1。历史提案 `acceptance_review_bb768787.md` 保持原文，包括其当时的“未批准/未实施”状态；历史状态不回填。获批文档的 LF 规范化 SHA256 为 `d8c0b79c54654e0bf263fb21a11bd22ac5d9fde5446f2e1a6fcd6758b8d9ea0e`。

新合同为 `dpr_acceptance_v2`，完整预检为 `dpr_full_preflight_v3`，定向补证为 `dpr_r1_supplement_v1`。支持范围固定为 `native_amp_native_half_epoch_final_independent_fp32_v1`。未知、旧版、缺项、源码或输入身份不符均拒绝。补证始终 `admission_eligible=false`，只能解除“不得开始完整预检”的前置阻断，不能生成正式训练许可。

## B1–B8

`supplement_dpr.collect_mode` 与完整预检共享采集器；`dpr_acceptance.validate_b` 从事实重新计算结论，不以 `assessment.status` 放行。

| 条款 | 本次实现 |
| --- | --- |
| B1 | 每模式在本次会话执行原 A；复用实际 `optimizer_fp32_v1` saver，绑定原始 checkpoint SHA、代码、模式、输入和会话。A 原有完整恢复、同梯度更新、CPU continuation、AMP 有效步/overflow skip 门槛保持。 |
| B2 | 父版/候选各两个独立 live FP32 副本；初态 exact、无共享存储；逐次 Python/NumPy/CPU/CUDA RNG、完整 batch 张量、真实 scaler、autocast dtype、TF32/cuDNN/deterministic 设置；前向与 loss exact。 |
| B3 | 完整参数、buffer、optimizer 分组/超参/两阶矩/step、EMA、scaler、epoch/update 清单；缺梯度、非有限、buffer/整数/元数据差异拒绝。新增四组参数的 model、EMA、两阶矩、step 共 20 个持久状态原容差硬检查。 |
| B4 | 原始失败名、全部超差坐标、max_abs、relative_L2、超差比例及父版对照均保留。逐个持久浮点失败必须对应同阶段有限且不相等的梯度。失败名称子集事实仍报告，不再单独决定放行。 |
| B5 | 同一 backward 的有效核梯度、W、独立四映射转置、成对差值映射、固定输入/upstream/norm 状态局部重放；按真实 scale 仅做一次算术除法；独立 FP64 参考。 |
| B6 | backward 之后、原生 optimizer_step 之前采集全部状态和原始 scaled 梯度。独立副本完整恢复 GradScaler 内部张量，直接调用未修改的原生 unscale/全局 clip/AdamW/scaler/EMA 更新。每次完整 post-state exact，成对 replay delta 与实际 delta exact；完整原始张量留取证主机。 |
| B7 | 两个更新后的候选，以相同 batch、eval FP32、暂时关闭且恢复 TF32，检查核/target/P3–P5/encoder/scores。连续量原 FP32 容差必须通过；output 须 raw 通过，或实际 query indices 改变且固定候选重放通过。 |
| B8 | 独立验证器核对 B1–B7、文件哈希与完整清单。只有 GPU 可得到 `EXPLAINED_BACKWARD_VARIATION`；raw B/failed_tensors 不改写；不宣称长期训练轨迹等价。 |

原 FP32 `2e-5/2e-4`、half `2e-3/2e-2`、FP64 `1e-10/1e-9`、三项 FLOPs `1e-9` 门槛均保留。

## H0–H4

H0 在真实更新过的 AMP trainer 的独立副本上，调用继承的原生 validator `__call__`、预处理、推理、GT loss、postprocess，恰好一个已有真实 batch。只替换全 split 指标累加/绘图出口，不修改生产类。检查 U16、输入/参数实际 half、非零四组参数、原 norm、有限 loss/postprocess，以及原生 `.float()` 返回的 half→float 舍入状态。原 trainer 完全不变。随后对同一 disposable EMA 调用实际 saver。

H1 检查 FP32 构核折叠、父版融合再 half 的部署对象保存/重载：参数与 buffer 字节、文件哈希、dtype、表示、norm、输出及连续量原 half 容差。

H2 覆盖真实 saver 的 half EMA 文件、只在副本上 `strip_optimizer` 的原生 final_eval 文件表示，以及独立评估 `RTDETR(path).model` 的内存 FP32 backend 入口。各自实际 AutoBackend 与同文件独立加载参考比较；沿用所有身份、真实 selector、连续量和 raw output 硬门槛，无 precision-note 例外。

H3（U32→U16、D32→D16、N32→N16）和 H4（U16→D16、D16→N16、U16→N16）保留全部原始误差，能力明确为 `NOT_SUPPORTED_UNDER_THIS_PROFILE`。这不是 PASSED 或 PRECISION_NOTE；非有限仍是硬失败。完整预检还保留原四项 half 检查的 raw 树，对同路径 reload/backend 仍执行原门禁。

每条 H 路径设置独立 120 秒硬预算，超时即停止自身 worker 并保留 BLOCKED。整个定向 worker 默认 900 秒、最大 1800 秒，只终止自己创建的进程树。每个模式只执行一组成对更新，不重抽样寻找通过。`--modes` 的开发子集缺另一模式时必然 BLOCKED。

## 执行顺序

1. 固定版本同步；新目录重新执行初始化与数学审计，已有 controlled init 只 verify，不覆盖。
2. `dpr_server.sh supplement`：仅 CUDA FP32/原生 AMP 的新 B 证据、鲜活 A、真实 EMA H；不跑 CPU/容量矩阵，不运行旧完整诊断。
3. 返回 `r1_light.tar.gz`、manifest、退出记录与报告；完整 tensors 留服务器。
4. 只有同版本、同输入、同服务器的补证被独立验证通过，才允许设置 `DPR_R1_SUPPLEMENT` 并执行一次 `init-preflight`。完整预检包含 CPU/CUDA 生命周期、全部原始 FP32 检查及原 B16/640/AMP 有限容量。
5. 即使 verify 通过也不串接 start；正式训练仍未启动，最终 test 仍未执行。

`eval_dpr.py` 固定独立 FP32 路径并记录相同 scope；打包校验同合同/scope，不能把训练许可当成任意 half 部署许可。原生 half epoch/final 验证、best 选择、训练配方、DPR/CBR/LIF 数学与初始化均不修改。

本地验证结果和固定 SHA 的可复制服务器命令在交付记录中单列。本地 PyTorch 2.7.1 / RTX2060 结果不替代服务器 PyTorch 2.1.2 / RTX4090 补证。任何未执行或新检查失败继续 BLOCKED。

本地结果：门禁反例、CPU/AMP 原生更新副本重放、原有观测回归通过；主配置两种 CUDA 模式的 B6 均 exact。B7 仍失败：FP32 encoder_features 最大差值 `4.887580871582031e-05`；AMP 多处连续量超差，P5 最大差值 `0.003418445587158203`。保持原容差并阻断，不据此调整模型或 AMP。真实 EMA 的 H0–H2 本地实现检查通过，H3/H4 仍不支持。单独 `dpr_v1` 只完成候选采集器/初态/B6/FP64 实现验证，不冒称已完成其父版对照、A/H 或完整准入。

开发过程的 scaler 副本、SimpleNamespace 夹具及临时恢复夹具失败一并保存于 `r1_validation_20260920`，原始 JSON/日志采用可逐字节还原的 gzip；manifest 记录原始与压缩哈希。测试中的合成正例只用于隔离反例门禁，明确标记 `UNIT_ONLY_NOT_OBSERVED`，没有写回原始报告。开发记录保留采集时的真实 HEAD/源码哈希，不改写成最终提交 SHA。
