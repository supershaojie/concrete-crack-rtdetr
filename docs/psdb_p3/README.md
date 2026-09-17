# PSDB-P3 v1

本实验直接从成功母版 `a0459d6a652cb702699087c88fa39a3e4c4087ec` 派生，分支为 `exp-rtdetr-r18-lite-psdb-p3-v1`。原 SDB worktree 不变；PSDB 是第三模块的替代候选。完整授权与数学合同保存在 [IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md)。

## 实现与来源

`PSDBRepC3` 继承原 RepC3，只替换 model.19 包装与输入 `[[18,4],3,PSDBRepC3,[256,0.5,32,8]]`。原 RepC3 路径先生成完整 T3，再接收 model.4 的 P2。所有新增状态仅在 `model.19.psdb.*`。27 个节点、P3→model.20 和 Decoder `[19,22,25]` 接线保持。主组合保留 CBR、LIF；单模块保留 C2 下采样和原 Decoder。

四相位按 00、10、01、11 排列；奇数尺寸仅右/下 replicate 补齐，P3 尺寸错误直接失败。未修改 T3 生成 Q，Q 和共享键投影产生四相位余弦分数，`a=softmax(2*scores, dim=1)`，`w=0.75+a`。每相位一个空间标量广播到全部 64 通道，加权后继续按相位拼成 256 通道。权重和为 4 不等于特征能量不变。

共享 D/Q/g/W_o 部分源自固定 SDB 提交 `f976779bcdb114178f83a3d2fa959baf6d9655ec`，保留其精度边界与 seed42 初始化顺序；新增 q/k 使用隔离 seed43 流。构造恢复调用方 CPU RNG。仅 W_o 全零，其余卷积 Xavier uniform，GN affine 为一/零。q/k 投影、归一化、点积、softmax 和权重形成局部 FP32；函数式卷积的可微参数转换兼容显式 half，不修改 Parameter 或模块 dtype。初始输出恒等不意味着内部 attention 均匀或 D 与 SDB 相同。

指定 ZIP 在有界本机搜索中不可见，未读取其 SPDConv/CARAFE，详见 [source_audit.json](source_audit.json)。实现依据完整数学合同。已阅读 [SPD-Conv](https://arxiv.org/abs/2208.03641) 与 [CARAFE++](https://arxiv.org/abs/2012.04733) 摘要，仅作空间重排、内容相关重组的理论背景；不据此宣称本实验精度收益或论文新颖性。

| nc=1 模型 | 未融合参数 | 融合后参数 |
|---|---:|---:|
| CBR + LIF 父模型 | 20,149,765 | 19,944,965 |
| CBR + LIF + PSDB | 20,178,629 | 19,973,829 |
| C2 父模型 | 20,082,772 | 19,877,716 |
| C2 + PSDB | 20,111,636 | 19,906,580 |

新增 **28,864** 参数。640 输入卷积分析值为 193,536,000 MAC / **0.387072 GFLOPs**（2 FLOPs/MAC），其中 q/k 为 14,745,600 MAC。GN、L2 归一化、点积、softmax、激活和逐元素算子另计。THOP 报告标记 PARTIAL，不等同完整整网 GFLOPs；功能式 q/k 必须显式计入且避免重复。

## 初始化、配方与数据

`init_psdb_p3.py` 校验统一未训练源的 SHA256、epoch=-1 和无 optimizer/EMA/scaler 已训练状态，重建受控 C2/CBR+LIF 父模型并逐 key、形状、数值复制审计公共参数及 buffers。两个变体新增状态一致，保存后立即重载。真实 `RTDETRTrainer.get_model` 的 nc80→1 只允许原 9 项分类形状适配，并与匹配 RNG 的父模型 nc1 张量逐值比较。正式 start 只接受受控初值，resume 保留已学习状态。

[parent_args.yaml](parent_args.yaml) 来自实际成功训练记录，与附件附录 109 项完全一致。两个 `recipe_diff_*.json` 给出逐字段差异，仅替换实验模型/输出身份；200 轮、patience50、B16/640、AdamW lr0=0.0005、原在线增强、native AMP 全部保留。`exist_ok=False`，不自动换名或删除已有 run。

`parent_dataset_inventory.json` 固定划分路径和标签内容指纹。当前本机 train/val/test 为 6048/1728/864，完整路径与标签指纹匹配成功父记录；读取 test 清单不执行 test 推理。服务器须对自身数据再次验证。

## 验证与生命周期

核心检查覆盖编号相位/奇数补齐、边界和权重和、共享 k/单次 q、非零 W_o 下均匀权重退化、输入敏感性、首步 W_o 与后续上游梯度、独立 P2 导数、真实 hook 接线和精度路径。模型预检使用一次性副本检查真实检测 loss/GT/DN、同步 RNG/BN 的父子等价、非零状态保存重载/EMA/原生恢复/融合/half。低精度 top-k 漂移与连续特征检查分开记录，固定候选诊断不冒充原生等价。

发现 AutoBackend.warmup 使用未初始化的 `torch.empty`；本机 CUDA allocator 复用 NaN 存储时可复现非有限输入。只将 warmup 输入改为 `torch.zeros`，保留所有数值异常检测。原 CBR、LIF、Decoder、loss 和训练默认设置不修改。

服务器预检要求真实数据、原在线增强、B16/640/native AMP 至少两次有效 optimizer 更新，校验全部可训练参数恰好被优化器覆盖一次。资源缺失或不足须 PENDING，不能降 batch、关 AMP 或将 GradScaler 跳步计成有效更新。严格 FP32 融合诊断使用局部 TF32 关闭并恢复标志，不改变正式训练策略。

工具分离 init/check、plan、start、resume、val/test、pack。start 绑定完整提交、源码配置哈希、源/初值/数据哈希、variant 和 PASSED 服务器预检。on_train_start 使用 start_epoch；轮数完成与 final_eval 状态分离，200 轮后的验证失败不会触发重训。

独立评估保留 `corrected_sorted_conf_mask_v1`：先独立 val，再对同一 SHA256 best.pt 显式 test；FP32、640、B16、workers0、conf0.001、iou0.7、max_det300，不加 NMS 或 test 调阈值。轻量包只收集现有证据、有限日志尾和必要源码，排除数据/权重，不触发训练或 test。

## 交付

[SERVER_COMMANDS.md](SERVER_COMMANDS.md) 是命令模板；`outputs/psdb_p3_delivery/SERVER_COMMANDS.md` 是最终提交后填入完整 40 位 SHA 的执行副本。实际检查报告、初始化审计、CLI help、固定代码身份、推送核验与轻量包均在 `outputs/psdb_p3_delivery/` 或 `outputs/psdb_p3/<variant>/`，不作为运行结果提交 Git。每段命令独立 source 固定环境并检查 SHA。保留正在运行的 SDB；GPU 空间不足时等待资源，不杀进程。

交付时：**正式训练 NOT_STARTED，最终 test NOT_RUN，服务器未登录。** 服务器容量与目标服务器端验证为 PENDING，须用户执行预检后再显式 start。检查通过不构成涨点证据。
