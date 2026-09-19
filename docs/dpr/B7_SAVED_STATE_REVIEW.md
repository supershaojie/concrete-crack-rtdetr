# 已保存状态的 B7 母版对照

本工具只增加诊断，不修改 `dpr_acceptance_v2` 或任何准入入口。原 R1 两种模式的 B7 失败、原始报告、原容差及 BLOCKED 均保留。没有服务器新测量时不能称 B7 通过；收集完毕也不是准入。正式训练、最终 test、完整预检均不执行。

## 输入与源码边界

原源码固定为 `f0eace10a882049e7053fbe9f69cad0c38149554`，原会话固定为 `r1_20260919T165019_256581`。入口 `tools/compare_dpr_b7.py` 核对补传清单 SHA `ff76e9d8645864a0fe555af9c3ea441dc414bab26de3ff88283f13f801f5b711`、原包 SHA `c34d69c0565f0e02930156961053e42c2d5a04b503bf3fab81e17c47a1b9aa9b`、20/20 成员、包内 16 个源码以及原报告全部 323 个源码身份。Windows 源码比较只作 Git CRLF/LF 规范化。任何原源码变化均停止。

八份 `native_replay_{0,1}.pt` 和 `batch.pt` 必须位于原绝对路径，文件大小、SHA256 必须与原 `supplement.json` 一致；全部核验后才反序列化。只选择 `actual_post.model` 作为权重，`actual_post.buffers` 仅用于交叉核查。`pre`、EMA、优化器从不应用到模型；不读 `last.pt`。原 payload 的容器须整体反序列化，不声称 PyTorch 可以只从此格式读取单个键。

## 非 state_dict 属性审查及恢复依据

这是**固定源码路径推导**，不是原缓存字节的直接对照。原 B6 payload 没有序列化缓存；报告明确记录 `old_cache_bytes_independently_compared=false`。以下依据须同时成立，否则停止，不新增训练取样：

1. `supplement_dpr.worker` 用原生 nc=1 构造器重建两个模型；`init_dpr.native_rebuild` 调用 `RTDETRTrainer.get_model`，只按相同 YAML 构造及加载 state_dict。`DetectionModel.__init__` 的虚拟前向仅适用于 Detect，RTDETR 走固定 stride 分支；verbose=False 无 info/profile 前向。新脚本也不以推理生成初始化或缓存。
2. `make_trainer` 仅给模型设置 nc、names、args，持久状态由原保存值覆盖。固定 YAML 恢复 routing/from/save、卷积属性、BN eps/momentum、激活、attention/decoder 属性、CBR rho/normal_fraction=0.1 和原 LIF 结构。训练更新不改变这些构造属性；DPR 的 deployed=False，四组差分参数从 actual_post 恢复。`criterion` 是原 loss 惰性安装的无持久状态子树，仅为匹配完整 named_modules/modes 清单构造；新前向使用 image，绝不执行 criterion。criterion.device 等仅供 loss 使用，不进入 eval 数据流。
3. `RTDETRDecoder` 的可变 eval 缓存为 shapes、anchors、valid_mask，不在 state_dict；`_get_decoder_input` 仅在 dynamic 或 shape 变化时生成缓存。两个模型由空缓存开始，原 `collect_mode` 都先在同一真实 batch=2/160 上至少完成两次更新，随后 B 副本复用缓存。此处三层网格由固定 YAML 得到 20×20、10×10、5×5；原真实 top-k 分数形状也记录为 [2,525]。
4. 生成时严格复用原 `_generate_anchors`：FP32 模式传 FP32；AMP 模式在原 CUDA autocast 下传 FP16，保留 log 等操作的原 autocast 行为，**不能直接生成 FP32 缓存，也不能清空 shapes 让 eval 重新生成**。`preflight_dpr.move` 和 `RTDETRDetectionModel._apply` 只移动缓存；`checkpoint_a` 在另外的恢复副本上做继续更新，原 candidate 不被这些更新替换。B7 的 `.float().eval()` 最后才把缓存升到 FP32。新脚本复现这条生成/移动/升精度路径，不执行训练前向。
5. CBR/LIF 前向不持久保存中间特征；AIFI 位置编码按当前形状在前向内生成，无历史缓存。Torch 标准模块的 eval 属性来自相同版本构造器；原服务器 Python、Torch、CUDA、GPU 型号必须一致。原记录的 deterministic/warn_only/cuDNN 标志被恢复用于观察，TF32 临时关闭，退出恢复全部这些进程内标志。原 seed42 设置的 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 只在独立 worker 环境中恢复，且在 CUDA 初始化之前设置。模型 FP32、autocast 关闭，和 B7 一致。

该推导只适用于这一份经过哈希验证的源码与会话，不能用于任意 checkpoint、未知模型修改、不同输入形状或未知 autocast 历史。若运行时参数、buffer、模块模式、requires_grad、decoder 属性/空缓存前提不符，或任一观察前后 state/cache/mode/hook 改变，则停止。参数及 buffer 核对名称、shape、dtype、原始字节，加载不得隐式 dtype 转换。原缓存没有独立保存的事实始终留在结果中；DPR 与原 B7 的逐项记录对照是额外检查，不能把推导说成独立缓存测量。

## 有限前向与统计

四组按 `cuda_fp32/parent`、`cuda_fp32/candidate`、`cuda_native_amp/parent`、`cuda_native_amp/candidate` 顺序执行。每组依次：

1. actual_post 0 一次前向。
2. 同一模型/状态重复一次，原容差内所有观察点通过且真实 top-k 完全相同才继续；同时记录是否逐字节相同。失败立即停止，不重试。
3. actual_post 1 一次前向；以 0 为左、1 为右比较。
4. 仅原始顺序 output 不通过时，由原 `compare_traces` 用 0 的实际 top-k 索引固定重放 1 一次。未失败时只复用原 metric/selection_difference 聚合逻辑，不调用会无条件重放的 compare_traces。

所有检测器调用都通过同一计数器，每组≤4、全局≤16，无额外 warmup/profile。独立父进程从启动、输入核验、导入、重建到观察共同计时 **300 秒**，到限杀掉 worker，退出 124，保留已经写出的进度，不重试。强制终止时 worker 进程内标志随进程消失，不影响其他实验进程。报告收尾写入单独 exit_status；超时/中断的 partial JSON 不得作为完整结果。

原 `capture_trace` 也适用于母版 ConvNormLayer，其 effective_kernel 为真实 conv.weight。两者均观察 dpr_input（母版中表示同位置输入）、effective_kernel、conv_output、norm_output、target、P3/P4/P5、encoder_features、candidate_scores、真实 query indices 和 output。原 `comparison` 保持 `atol=2e-5, rtol=2e-4`、右侧相对容差参考、原 relative_L2 分母和统计版本，绝不改归一化方式。DPR 构核仍是模型原执行路径，不涉及 FLOPs 或部署修改。

JSON 的 parallel_table 并列两种模型各阶段 raw_allclose、max_abs、relative_L2、超差比例；索引集合、换序数、固定重放、重复前向另列。first_exceeded 仅表示首次**观察**到超差的位置，不是首次错误操作。

DPR 新记录的指标、索引、固定输出与原 B7 做精确记录对照，任何不一致优先报告 RECONSTRUCTION_OR_OBSERVATION_MISMATCH 并停止。这是重建检查，不是改变验收容差。母版也失败只能支持共同背景；仅 DPR 失败只能缩小定位范围。不能用排序对齐替换连续失败，不以失败行数或误差大小直接归因。没有具体实现错误就停止扩大诊断，保留挂起 DPR 或另行审议合同的选项。

## 入口与产物

服务器已有环境中运行 `python tools/compare_dpr_b7.py`；无需任何新权重、初始化审计、训练或完整 R1。每次只写新的 `outputs/dpr/cbr_lif_dpr_v1/b7_saved_<UTC>_<pid>/`，不接受覆盖旧输出的路径参数。JSON、exit_status、worker.log、source_manifest.json 和 source/ 可直接作为轻量交付，不包含 tensor。

退出 3 表示保持 BLOCKED（包括完整收集），124 表示超时。应同时查看 collection_status、每组 status 和 original_B7_consistency，不能仅按退出码判断是否完成收集。只在本地用 `--metadata-only --archive ... --manifest ...` 时，完成输入清单/源码/原拒绝逻辑核验可退出 0，明确标注 LOCAL_METADATA_ONLY，未读取服务器 tensor、未测量 B7。
