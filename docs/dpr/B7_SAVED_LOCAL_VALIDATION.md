# B7 已保存状态对照：本地交付核验

原基线 `f0eace10a882049e7053fbe9f69cad0c38149554`，分支 `exp-rtdetr-r18-lite-dpr-v1`，实施前 DPR 工作树干净。主实验工作树保持 `f34502c43a88edc9cb1076bd996b9c3fc2d23e31` 及其既有未跟踪文件，不修改其他实验。仅新增独立诊断、故障测试、服务器入口和审查记录；原模型、CBR/LIF、配方、AMP、原生 half EMA、best 选择、验收器及原报告没有改动。

## 已完成（本地）

- 本地 Python 3.9.25 / PyTorch 2.7.1+cu118；不是服务器 Python 3.10.13 / PyTorch 2.1.2+cu121 环境，没有改依赖。
- `python tools/test_dpr_b7_saved.py`：12 项通过。覆盖 actual_post 唯一选择、原始字节/名称/dtype/buffer 核对、两种模型构造、前向预算/超时前置检查、原指标方向与索引、3/4 次调用分支、连续失败保留、观察异常清理检查、标志恢复、旧 B7 不一致优先停止、缺失/损坏输入、非有限失败数值可留档与禁止训练调用的 AST 审查。
- 两个 Python 文件 `py_compile` 通过；服务器 shell `bash -n` 通过。
- 原包与补传清单身份匹配，20/20 文件通过，16 个包内源码与报告身份一致，原报告 323 个源码身份均匹配当前未改的实现。
- 原验收函数 `validate_b(..., files=False)` 离线复核：cuda_fp32 和 cuda_native_amp 均仍在 B7 target 拒绝。这里只核验原记录/判定逻辑，不是重新测量。
- 按服务器报告的原清单额外核对 nc=1 构造器：母版 345 个参数、207 个 buffer；DPR 349 个参数、207 个 buffer。两种模式的原清单均相符。该核验没有调用检测器前向。
- 元数据核验记录：`outputs/dpr/cbr_lif_dpr_v1/b7_saved_20260919T174218_076569_29128/`；collection_status=LOCAL_METADATA_ONLY，forward_counts={}，9 份服务器 tensor 均标 PENDING_SERVER，整体约 6.23 秒。随后审查增加结束时源码复核、禁写 bytecode 和非有限失败留档；最终再次做静态、故障测试及元数据核验，最终目录在交付消息中列出。

## 已发生的本地失败，保留

第一次元数据核验目录 `outputs/dpr/cbr_lif_dpr_v1/b7_saved_20260919T173752_085309_3896/` 保留 STOPPED 原记录。原因是新脚本初稿的 250 MB 成员上限小于原始 supplement.json 的 421,296,366 字节；并非原包大小或 SHA 不符。已移除不适用的大小上限，仍严格匹配固定清单的原大小和 SHA。随后在新目录完成核验，没有覆盖首次失败或原 R1 失败。

## 待服务器，不得写成通过

八份 replay 和 batch.pt 不在本地。其实际路径/大小/SHA、actual_post 恢复字节、按原路径推导的缓存、同状态重复、四组 B7 观察、DPR 与原记录逐项一致性、实际硬超时执行均待服务器。单元测试里的计数/超时和清理模拟不冒充真实 CUDA 运行结果。工具没有原缓存独立序列化字节；缓存恢复依据及边界详见 `B7_SAVED_STATE_REVIEW.md`，缺失前提或新结果不一致时停止。

正式训练 NOT_STARTED；最终 test NOT_RUN；完整预检 NOT_RUN；DPR 保持 BLOCKED。本轮不重新运行 R1 更新、容量或完整诊断来获取新样本。服务器只需固定版本同步后调用 `tools/run_saved_dpr_b7.sh`，使用原会话，退出 3/124 后保留报告并停止。
