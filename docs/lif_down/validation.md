# LIF-Down v1 本地验证

环境：Windows，Python3.9.25，PyTorch2.7.1+cu118，RTX2060 6GiB。
仅进行轻量检查，未执行正式 epoch 循环或完整数据集 val/test。

| 检查 | 结果 |
|---|---|
| 独立 YAML / parser / nc1 | PASS，仍是 RTDETRDetectionModel + RTDETRDetectionLoss |
| 精确拓扑 | 仅 P3 RepC3→P4 Concat 的下采样为 LIFDown，另一处原 Conv |
| 参数 | C2=20,082,772，LIF=20,103,876，delta=21,104 |
| shape | 80×80、79×80、80×79、79×81、1×1 均与原 Conv 一致 |
| Haar | /2、逐潜在通道 Dh/Dv/Dd 相邻排列、奇数 replicate 手算通过 |
| Align | 3×3 小 tensor 手算、横纵分离、左/上 replicate 边界通过 |
| 初始化等价 | CPU/CUDA，640×640：downsample/final/bbox/score max_abs=max_rel=0（实测） |
| 容差 | 等价检查 atol=2e-6，rtol=2e-5；不要求跨设备/版本固定零误差 |
| 梯度 | O 首步非零；B/P/U 首步为0，实际一次 O 更新后均有限非零 |
| CUDA FP32 / AMP / half | 推理 finite，真正 model.half() 输出 float16 |
| 非零 O save/load | 全部 state 值相同，输出误差0 |
| 非零 O 融合 | 保留 LIF BN，整网输出 max_abs 约 1.8e-7 以内 |
| 原生 DN/loss | 两张真实 train 图，batch2/imgsz320；CPU/CUDA forward/backward 有限，DN=[198,300] |
| AMP DN | 同一小真实 batch，有限 backward 和一步 AdamW 更新；仅 smoke scaler=128 |
| 公共映射 | nc80 533 COMMON+5 NEW；异常 missing/unexpected/shape mismatch=0 |
| nc1 原生重建 | 529/538 同形状态精确加载，9个预期类别适配；533个公共状态与C2相同 |

数值、梯度范数、样本文件散列与源码散列见 checks.json。
deterministic 使用 CUBLAS_WORKSPACE_CONFIG=:4096:8、禁用 TF32、cudnn benchmark=false、deterministic warn_only。
原 RT-DETR Decoder 的 CUDA grid_sample backward 没有确定性实现，其警告如实保留；不宣称训练反向位级可复现。

ops_checks.json 是显式模拟的生命周期保护测试：tmux 调用未真实派发，OOM 为注入异常；
状态、重复启动、batch保持、缺失证据打包拒绝均检查。大于20MiB 的打包写入/读取/散列/拒绝覆盖为真实文件 IO，
该包使用测试 fixture 和模拟 metadata 校验，不是正式训练包。

evaluation_checks.json 只对复制到隔离目录的两张 train 图片走评估入口，验证曲线生成、600个预测与GT导出、
val/test同checkpoint和错误SHA拒绝；不能解释为独立验证集成绩或完整 val/test。

sync_checks.json 记录本地真实 Git bare/linked-worktree 集成测试，只有 fetch 目标重定向到隔离本地仓库。
验证远端前进后仍固定原 SHA、detached HEAD、重入、脏目标/不同SHA/错误仓库拒绝、主工作树修改保留。

尚未验证：AutoDL Linux/PyTorch2.1.2、正式 batch16 显存/吞吐、服务器真实 tmux、200e 收敛、完整 val/test、
真实训练完成证据及其完整下载包、任何涨点或创新性结论。最终回复提供已推送的 SHA，不以本地 smoke 代替服务器结果。
