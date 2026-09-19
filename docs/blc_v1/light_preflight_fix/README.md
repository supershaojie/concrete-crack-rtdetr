# AutoBackend 比较修复与轻量预检

起点：`343df78240277301a03fa520a532312008bf6850`，分支 `exp-rtdetr-r18-lite-blc-v1`。

用户提供的旧服务器事实保留：`preflight-20260919T183334684697Z/report.json` 的B16/640/AMP容量通过且2次有效更新；配套 `lifecycle-20260919T183404923829Z.json` 的保存恢复、optimizer/scaler/EMA及双恢复同梯度更新通过，非零增量范数为3.971256228396669e-05。整体因第一条FP32 AutoBackend比较失败；后续AMP/half、真实单批half EMA验证和真实原生续训未完成。本次未登录服务器或改写这些原报告，也不把旧结果计作新版本全部通过。

## 源码事实与修复

旧手工参考在目标 GPU 上执行 `deepcopy(quantized).fuse()`；文件 AutoBackend 则经
`load_checkpoint` 在 CPU 读取、转 FP32、融合，随后 eval/迁移设备/设置请求 dtype。
这证明旧“Same-path”两端路径不同，不能据此宣称已证明服务器失败的唯一原因。

现在同一个临时文件分别交给原生 `load_checkpoint(fuse=True)` 和 AutoBackend 独立加载。
对齐融合设备/顺序、dtype、eval、输入和 autocast；两端仅在测试副本上清空 decoder 的
shapes 缓存，让原生逻辑根据当前输入重新生成 anchors。逐项检查状态一致且不共享存储。
每端使用 hook 记录 BLC 调用及非零增量；正常或异常均移除 hook、恢复 TF32 开关。

未融合/融合仍另行比较，使用未融合文件加载与原生 CPU 融合文件加载两条自然路径。
保持原 `difference` 的 atol=2e-5、rtol=2e-4 和既有 PRECISION_NOTE 判据。
固定索引重放仅保留为已有诊断，不替代自然输出；未达原判据仍为 PENDING。
每种模式先登记，再记录实际统计、路径、dtype/device、阶段和异常；断言失败不会丢行。
未执行的模式和外层阶段明确记录 NOT_RUN。

生产模型、CBR/LIF、配置、正式训练配方、公共预训练源、精度阈值没有改动。
`blc_common.py` 的严格 JSON 数据身份修复和证据身份比较保持原样。

## 本机结果

环境：RTX2060 / torch2.7.1+cu118 / Python3.9.25，不能代替指定4090服务器验证。

一次定向检查，单个固定 1×3×160×160 输入、明确非零 Wo 测试副本，合计 **15 次前向**：
六次同文件直接/AutoBackend 前向、六次自然融合比较、三次已有固定索引诊断。

| 检查 | FP32 | AMP | half |
|---|---|---|---|
| 同文件独立加载 / AutoBackend | PASSED，max_abs=0 | PASSED，max_abs=0 | PASSED，max_abs=0 |
| 每端 BLC 调用次数 | 1 | 1 | 1 |
| BLC 非零增量范数 | 0.0371916741 | 0.0365773737 | 0.0365773737 |
| 状态/存储 | 364项完全一致，无共享 | 同左 | 同左 |
| 自然未融合/融合比较 | PRECISION_NOTE | PENDING | PENDING |

FP32 自然 max_abs=0.2500000596，有2个候选位置交换；连续特征/encoder分数满足原容差，
诊断重放 max_abs=1.1920929e-7，按未改动的既有判据记录 PRECISION_NOTE。
AMP/half 自然 max_abs 分别为0.7499509696、0.7002563477；encoder分数及重放仍未满足
原判据，因此**整体定向检查 PENDING，退出码1**。未放宽阈值或追加重跑。

`backend.json` 为完整小型原始报告（12,199字节），约40.6MB的临时 checkpoint 已删除。
`faults.json` 验证失败行/异常阶段保留，未执行项正确，hook/TF32恢复，16 batch守卫、
单批验证包装，以及正常/失败/KeyboardInterrupt临时清理；借用AMP资源的成功/失败清理也通过。
故障夹具执行两次：第一次基本清理/预算，第二次新增外层资源清理覆盖；均无实际训练。

两份新的轻量 init-preflight 在本机通过：当前受控源参考逐张量比较、结构/零初值和真实
`RTDETR.train` 的 Trainer 重建，均 **0个训练 batch**；临时参考权重及重建输出均已清理。
COMMON 列表只留计数、清单哈希及失败样本，完整比较仍先在内存严格执行。
数学长矩阵明确 NOT_RUN；本次没有改变模型数学，不重复旧的整套诊断。

本次上述运行共新增留存 **250,574字节（约0.239MiB）**，包括两次故障夹具报告、两份init审计、定向报告和init日志；留存测试checkpoint为0。清单见 `final_audit.json`。提交内归档为小型文本副本，不含权重。

报告在提交前产生，runtime.commit 是起点 HEAD。定向报告产生后只补充了外层 AMP 资源
清理失败的元数据记录/留存打印；被测生命周期/对照函数没有再改动。最终源码及权重核对
见交接附带审计；不能将这些本机小输入结果视为新 SHA 的服务器容量准入。

## 默认预检行为

- `init-preflight --both`：审计现有两份 controlled_init，临时参考只用于比较，不重建或覆盖正式初值。
- `preflight`：真实 B16/640/native AMP，start与resume合计最多16个训练batch；start达到原2次有效更新和上游梯度要求即停，resume至少1次有效更新。预算不足保持PENDING，不加重试。
- 一次原生保存供生命周期与真实resume共用，仍按原策略保存half EMA与optimizer状态。实际resume继续核查epoch、moments/steps、scaler、EMA、updates并执行真实有效更新。
- 移除重复的双模拟恢复/外部梯度更新诊断；真实resume承担恢复状态检查，旧通过事实保留在历史报告中。
- 实际原生half EMA验证只消费一个真实val batch。它及真实resume没有被移出整体判定。
- 外层bounded_start及其last/best、生命周期state_dict/full_model均位于本次独占TemporaryDirectory；成功/异常/可捕获中断后清理，只保存用途/大小/SHA256/清理状态。不会清理旧结果或正式权重。
- 不下载模型包；原生AMP所需资源缺失时只临时借用现有资源，检查结束移除本次副本，已有资源保留。
- 预检自身关闭绘图输出；正式配方不变。默认留存小型JSON和必要日志，目标每次≤10MiB，不靠删除失败报告达标。真实服务器留存量待该次执行观察。

## 下一步

服务器安全更新至交付的完整固定SHA，再执行一次 `init-preflight --both` 和一次主组合 `preflight`。
不运行 `init` 重建权重。每次入口自行激活现有rtdetr环境；沿用已有tmux会话 `blc-check`。
新提交服务器init审计、B16/640容量、half EMA验证、真实续训均 **PENDING**。
成功标志为两份init报告PASSED、主preflight报告PASSED且命令退出码0；PENDING/FAILED不放行。
本次未登录服务器，未启动正式训练、独立完整val或最终test。
