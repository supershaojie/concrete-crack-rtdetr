# TRC v1 交付验证

两候选本地必要检查均 **PASSED**；完整预检状态保持 **PENDING**，因为服务器真实在线增强 B16/640/AMP 容量门槛尚未执行。正式训练 **NOT_STARTED**，最终 test **NOT_RUN**，两份正式初值的优化器更新次数均为 **0**。没有登录或同步服务器。

机器可读汇总见 [local_checks.json](local_checks.json)。原始报告在本工作树 `outputs/trc_v1/final_combo_01/preflight.json` 和 `outputs/trc_v1/final_single_01/preflight.json`；汇总保留其 SHA256、被测源码清单摘要和实际环境。两次最终检查均通过首末源码/配置/权重指纹一致性核对。报告中的 runtime.commit 是测试时的 base；最终提交 SHA 在提交外 `outputs/trc_v1_delivery.json`，避免提交自引用。

## 已验证

- 15 项模块测试全部通过、无 skip：小张量逐项数学参考、self 项、零描述、相同描述、N=1、重复样本统计、正负方向、batch/head 排列、pre/post norm、位置编码来源、bool/float 2D/3D mask、padding、全遮蔽行原语义、梯度、CPU/CUDA FP32、native FP16 AMP、half 参数及可用的 BF16。
- 真实父模型状态迁移：组合父模型 552 个状态、单模块父模型 533 个状态逐值一致，包括 BN buffers。新增仅 `model.9.trc.descriptor.weight`、`model.9.trc.coefficient.weight`、`model.9.trc.coefficient.bias`，共 10,248 参数，两候选 TRC 初值逐值相同。
- 实际 `RTDETRTrainer.get_model` 和 `RTDETR(...).train` 的有界重建路径，完成 nc=80→1 适配后停止于训练之前。9 项 shape 白名单为 layer26 denoising embedding、encoder score weight/bias、3 层 decoder score weight/bias；其他公共状态及 TRC 完整加载。原 QKV/FFN/LN 未改名或漏载。
- 原 AdamW 构建中每个参数恰好一次；新参数在构建前已注册。零和非零 checkpoint、EMA 及原生 resume 的模型/优化器状态恢复通过。
- 640 整网前向测得 AIFI 输入 `[1,256,20,20]`；两候选对各自父模型在 CPU/CUDA FP32 的初始输出 max_abs 均为 0。本次测量不构成对所有 kernel 逐 bit 一致的承诺，声明容差仍为 atol=2e-5、rtol=2e-4。
- 非零 TRC 以及组合中非零 CBR/LIF 副本的严格 FP32 整网融合通过；TF32 设置在检查结束/异常时恢复，原 LIF BN 被保留。单模块 CUDA 融合出现原查询排序的排列变化，使用完全一致的候选集合和一一 ID 对齐核验全部输出，并做固定查询重放；记录 `PASS_WITH_CANDIDATE_PERMUTATION`。没有放宽数值容差、改变生产查询选择或用部分候选掩盖差异。
- CUDA 整网 native AMP 和 FP32 fuse→half 推理有限，TRC 学习状态保留。可选诊断仅保存 r/lambda/bias 和梯度汇总。
- 真实训练集 2 张图、160 输入上的原 loss 与反向：CPU FP32、CUDA FP32、CUDA AMP 均通过。这是隔离副本的工程检查，不是完整训练、验证集评估或检测精度证据。两候选 AMP 均由原生 65536 按 32768→16384→8192→4096 回退，随后完成两个有限梯度副本更新。没有固定 scale=1、关闭 AMP 或对非有限值做替换。
- 7 项生命周期检查、14 项门槛/指纹/TF32 恢复检查、9 项真实隔离 Git 同步检查通过。plan 不创建正式 run；PENDING、旧指纹、改动数据或已更新初值不能授权 start；resume、旧 run 和轻量包覆盖保护均有检查。同步测试只用本地 fixture，没有访问服务器。

## 算量口径

同次 nc=1、B1、640，THOP 使用副本及 TRC 自定义 tensor/buffer hook，显式计算函数式投影和相似度，未重复计算子层。完整父/候选、融合/未融合数值在 local_checks.json。主组合未融合参数为 20,160,013，父模型 20,149,765；差为 10,248。TRC 矩阵增量恰为 9,216,000 MAC（0.018432 GFLOPs），不包含 functional LN/L2/exp/log/tanh、逐元素 bias/mask；原 MHA 和 deformable/CBR/LIF 部分函数运算也不是 THOP 完整覆盖，因此表中 THOP 总量不是完整实测算量，也不是实测延迟。B16 一个 FP32 完整 bias 张量为 81.92 MB，服务器实际峰值待测。

## 待服务器补检

按 [AUTODL.md](AUTODL.md) 激活服务器已有 rtdetr 环境并执行 `preflight_trc_v1.py ... --server`。服务器 PyTorch 2.1.2 历史环境尚未在本次连接实测；工具保留旧 AMP API 兼容分支，不升级依赖。必须通过实际 assets/bus.jpg/参考权重的原 check_amp、原 train 在线增强、B16/640/native AMP、原 loss/DN/反向、GT/DN 和峰值显存记录。start 严格要求整体 PASSED 及该容量项 PASSED；本地 PENDING 报告不能启动训练。

## 开发中修正与保留证据

首次 loss 诊断发现缺少 scheduler 建立的 `initial_lr`，已按原生命周期补上；首次单模块融合的直接逐行比较发现查询排列差异，已增加候选身份审计；最初门槛测试使用本机 Python 不支持的 unittest 辅助 API，已替换为兼容写法。失败记录仍在 outputs 中，最终上述检查均重新执行通过。未改变实验配方或网络主路来规避这些诊断问题。
