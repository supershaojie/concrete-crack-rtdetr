# LSRT v1 本地验收（2026-09-16）

两配置均完成本地必要检查；总状态为 PENDING，等待服务器原环境与真实 B16/640 容量补检。正式训练 NOT_STARTED；最终 test NOT_RUN；正式优化器更新为 0；未登录或同步服务器。

开发机：Windows、Python 3.9.25、torch 2.7.1+cu118、RTX 2060 6GB。不能将这些结果替代既有服务器 Python 3.10.13 / torch 2.1.2+cu121 / RTX 4090 验收。

每个 variant 的15项检查通过：受控初始化、原生nc1 Trainer、拓扑、真实train API在训练前停止的构建审计、AdamW覆盖、复杂度、CPU/CUDA数学与初始等价、CPU/CUDA FP32 loss反向、CUDA AMP loss反向、零/非零Wo的重载/EMA/融合及half推理。公共状态包括参数和BN buffers，逐值一致。原始CBR/LIF模块回归亦通过。

原CBR+LIF公共状态552项；原基线公共状态533项。分别仅增加5个LSRT参数张量，共32,777参数。nc80→nc1精确9键白名单见 classification_adaptation.json。所有训练参数恰好进入AdamW一次，包括5个LSRT参数。

| nc=1 / 640 / FP32 eval | 未融合参数 | 融合参数 | 未融合 THOP 2×MAC GFLOPs | 融合 THOP 2×MAC GFLOPs |
|---|---:|---:|---:|---:|
| cbr_lif_lsrt_v1 / parent | 20149765 | 19944965 | 58.6725120 | 57.5649536 |
| cbr_lif_lsrt_v1 / LSRT | 20182542 | 19977742 | 58.9420288 | 57.8344704 |
| lsrt_v1 / parent | 20082772 | 19877716 | 58.2766080 | 57.1657728 |
| lsrt_v1 / LSRT | 20115549 | 19910493 | 58.5461248 | 57.4352896 |

使用同一THOP及显式LSRT functional投影hook，两种状态的增量均为134,758,400 MAC（0.2695168 GFLOPs按2×MAC）。这是工具覆盖的算子估计，不是完整实际算量或延迟；LSRT归一化、softmax、mask、差分和访存未全部计入。

全109字段配方差异见 recipe_diff.json，仅模型/名称/输出及本地环境路径变化。数据train/val/test=6048/1728/864图；框数45573/12840/6663；原路径与标签指纹一致。本次记录全图像字节哈希，历史没有该项，不能声称历史图像字节已回溯证明。

服务器待补：原软件/GPU环境的同套预检；真实train在线增强B16/640、AMP及原DN完整前后向；固定顺序和高GT批次的GT/DN与峰值显存；原生AMP参考资源与实际Trainer启动门禁。容量预检不执行optimizer.step，OOM不改配方。

实际记录CUDA grid_sampler_2d_backward_cuda的deterministic warn_only警告；不宣称整网CUDA逐bit确定。融合诊断沿原容差，TF32关闭并恢复。start/resume/val/test本轮均未执行；这些入口完成语法、帮助、源码复核与门禁/打包故障测试，正式运行留待后续授权。

完整本地报告在 outputs/lsrt_v1/final_preflight_combo/report.json 与 final_preflight_single/report.json；摘要及报告SHA见 VALIDATION.json。交付版本绑定见提交外 DELIVERY_ATTESTATION.json。服务器命令见 SERVER_HANDOFF.md 的交付SHA展开副本。
