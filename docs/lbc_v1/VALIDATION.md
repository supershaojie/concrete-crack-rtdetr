# 验证边界

本地已通过：独立 Python 朴素区域/采样/公式参考；重叠框与所有 GT 缓冲区排除；极细/边界/非法/无 GT/负点不足；
loss 方向、稳定 softplus、零特征 eps、合法空 pair、非法输入报错、不消耗 RNG；e=5/6/20 调度数值；
同初值/同 RNG 的单线程 CPU L0、公共梯度及 AdamW 一步更新逐位一致；关闭时 head.grad=None 且不更新；
同一次 S3 traversal 的原预测/L0 不变；LBC 单项无第 6 层以后直接梯度，总 loss 保留 LIF/CBR；
有效→空 micro-batch 的累积梯度保留；全空窗口不会触发 head 的 AdamW decay/momentum；
真实 Trainer.get_model nc 适配、精确 552 个公共状态和 9 个类别适配 keys；两个新增参数恰好一次入普通 decay 组；
EMA、epoch 边界 scheduler/scaler/optimizer/未清窗口恢复，恢复 e=20；继续一步公共参数逐位一致；
独立标准部署文件无 head、无 GT，推理输出逐位一致。

真实 `LBCTrainer.__init__/_setup_train` 使用原增强和数据 loader 的本地有界检查：CPU FP32、B4/160、nbs64、累积 16，
最多两个真实 train batch（后续复用），16 micro-batch 中 1 次有效更新，head 和 S3 均更新，耗时约 29.4 秒。
69 个 GT 中 62 个有效 pair，覆盖率 89.86%；5 个无正网格点、2 个负区不足。
保存并检查 8 张图。灰色 padding 中确实可能出现选中负点，是方法的明确限制。
这不代表服务器 B16/640 容量通过，也不代表收敛/性能通过。

严格 CPU FP32 融合逐行通过：最大绝对误差 1.6391e-7，原容差 atol=2e-5、rtol=2e-4，不需置换豁免。
精度设置正常退出已恢复。验证使用隔离 eval 副本，原 LIF BN 保留。

Git sync 离线临时仓库验证：首次创建、同 SHA 重入、干净前进更新通过；tracked 修改、冲突 untracked、冲突 ignored、
过期 SHA 均拒绝且文件/HEAD 保留；旧报告备份和主工作区保持已检查。shell 入口通过 bash -n。
LIGHT 打包与逐项 SHA256 回读校验已通过，体积小于 8MB；补丁按原始字节保存以兼容 Windows 中文内容。

测试过程发现并修复：CPU 多线程 embedding 归约不保证逐位一致，确定性对照改用单线程；部署原 decoder 缓存必须保留；
部署两侧 requires_grad 状态统一到推理状态，避免 attention 执行路径差别。未复制梯度/参数伪造原路径通过。
本地 B2/nbs64 在 16 micro-batch 内不足一个更新，明确失败后改用允许的本地 B4/160；服务器配置仍为 B16/640/nbs64。

服务器待运行：指定 Python3.10.13/torch2.1.2+cu121/RTX4090 上的原生 AMP、B16/640 显存/实际有效更新/启用 scaler 状态，
该环境下的融合检查、Linux tmux 正式派发/状态/恢复。未启动正式训练，未访问 test 推理，未产生本实验 mAP。
CPU scaler 测试是 disabled scaler 的状态恢复；不能把它写成已验证服务器 enabled scaler。
preflight 会单独验证服务器 scaler round-trip，报告 allocated/reserved 峰值、真实更新与 overflow 跳步。

训练结果未来仍须按原 200e/patience50 判断，不能以短检 loss 降低代替 mAP 提升，不进行额外超参搜索。
