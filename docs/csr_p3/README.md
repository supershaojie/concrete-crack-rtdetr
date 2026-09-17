# CSR-P3 v1

独立实验分支 `exp-rtdetr-r18-lite-csr-p3-v1` 从成功组合
`a0459d6a652cb702699087c88fa39a3e4c4087ec` 派生。原始执行合同见
[IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md)，实际附件、模块 ZIP、源权重与成功参数记录身份见
[source_audit.json](source_audit.json)。原 CBR/LIF 源码未修改。正式训练与最终 test 不由交付过程启动。

## 实现

两种模型仅把原 model.17 的侧向 `Conv` 换成继承它的 `CSRConv`，保留 conv/bn/act 状态键、from=5、act=False，新增七个参数张量均位于 `model.17.csr.*`。
Concat18、RepC3 19、下采样20、Decoder26 和 from=[19,22,25] 的编号不变。主组合保留原 CBR/LIF，单模块消融来自 C2。

水平/垂直两条链、负/正两侧分别执行可微分 cumsum；负侧按距离3、2、1排列，增量不额外取负。
按像素中心 `(2*(x+0.5)/W-1, 2*(y+0.5)/H-1)` 构造坐标，使用同一特征、同一 bilinear/border/align_corners=False 采样与 softmax 权重做曲线减直线。
中心差分恒零，计算省略这两次差分，但保留七点 softmax 的完整分母。forward 流式处理12个非中心点，共24次 grid_sample，不复制完整七点输入。

只有 offset_pw 的 weight/bias 置零，theta=0 对应非零均匀1/7权重，W_i/offset_dw/W_o 用显式 Xavier 初始化。
新增层构造隔离 CPU RNG，不改变后续 head 初始化；两个变体 CSR 状态相同。输出投影非零使 offset_pw 首步可获得任务梯度，其余层在支路启动后逐步获得梯度。
偏移末投影、tanh/cumsum、网格、softmax、采样与累加局部 FP32；float 参数视图保留可微连接，显式 half 不替换 Parameter。
SiLU 通过 `F.silu(..., inplace=False)` 固定，防止全网 `initialize_weights` 把普通 SiLU 改为 inplace。

`CSRConv.forward_fuse` 明确执行原融合投影后再执行 CSR；使用学习后的非零偏移测试支路保留。
AutoBackend 的 warmup 输入单独修复为有限 zeros，原 deterministic 和正式 CUDA AMP 策略不变。

## 工具入口

- `tools/init_csr_p3.py VARIANT --source ... --output ... --report ...`：验证统一未训练源；逐键加载父模型完整状态；保存/重载；原生 Trainer nc80→1 的九项分类适配与匹配 RNG 父模型逐值相等。
- `tools/check_csr_p3_math.py --device cpu|cuda:0 --output ...`：独立标量双线性 oracle、矩形/斜坡/脉冲/边界、初始化和分阶段梯度、非零融合、half。
- `tools/check_csr_p3_runtime.py --complexity --output ...`：确定性 warmup 和 THOP int/Tensor 计数回归；同一640/nc1工具口径的父/目标融合前后统计。
- `tools/preflight_csr_p3.py VARIANT --mode local|server ...`：实际 loss/DN、实际 model.train 加载入口、公共状态/输入梯度、学习后 checkpoint/EMA/恢复/融合。服务器模式另外执行原增强、原生动态 GradScaler、B16/640有效更新。
- `tools/train_csr_p3.py plan|start|resume|status ...`：完整109字段配方与差异记录；start 必须匹配 PASSED 服务器预检身份；run 已存在不会另起 name2；resume 只载实际 last.pt。
- `tools/csr_p3_results.py val|test ...`：独立 corrected_sorted_conf_mask_v1 评估，先 val 后同一 best.pt/hash 的 test，训练中拒绝 test。
- `tools/pack_csr_p3.py --output ... --evidence ...`：证据清单、实际源码、完整指标、最多64KiB日志尾；排除权重/数据/巨大预测/参考包；超20MiB保留临时包并报告大文件。
- `tools/sync_csr_p3.sh FULL_SHA [BUNDLE]`：有界获取固定提交、创建/核验独立 worktree、生成环境文件。已有工作目录不 reset、不删除。
- `tools/autodl_csr_p3.sh --help`：上述 Python 工具的既有 Conda 环境封装。

每个入口的真实参数以 `--help` 为准。交付目录另有填入最终完整 SHA 的 `SERVER_COMMANDS.md`，包括独立的同步、初始化/预检、显式 start、resume、val/test、打包段。

## 验证口径

| nc=1模型 | 未融合参数 | 融合参数 | CSR增量 |
|---|---:|---:|---:|
| 原 CBR+LIF | 20,149,765 | 19,944,965 | — |
| CBR+LIF+CSR | 20,167,313 | 19,962,513 | 17,548 |
| 原 C2 | 20,082,772 | 19,877,716 | — |
| C2+CSR | 20,100,320 | 19,895,264 | 17,548 |

THOP 只报告所覆盖算子的 MACs/2×MAC 部分操作数；函数式采样、注意力及部分元素算术未完整覆盖，不能视为完整 GFLOPs。
CSR四层卷积在B1/P3=80×80时为109,158,400 MAC；额外24次采样约4,915,200个通道值。
插值与差分加权另附明确分析估算，不能从参数量推断速度。B16/640训练显存必须单独实测。

预检报告不可改写旧 commit/status。最终报告保存在交付输出目录并绑定最终代码 SHA、源码/配置、初值/源权重、配方及数据身份；代码变更后重跑受影响检查。
服务器容量和 native AMP 仅由服务器实际检查给出 PASSED；本机小输入通过不会授权正式 start。
CUDA grid_sample 的确定性警告按实际运行记录，不改变父策略。

本机 PyTorch 2.7.1 的整网 CPU half 曾出现非有限输出，插入观测 hook 后行为也会变化；部分父模型对照同样失败，单模块对照曾出现父模型有限而目标非有限。因此不能笼统归因于父模型，也没有确认具体算子根因。保留原失败报告和 CPU-half 诊断，绝不记为 PASSED。规定的 CPU FP32、CSR 模块 half 与整网 CUDA half 分别独立检查；服务器的 CUDA half 仍是必须通过的门槛，不通过即停止。未以修改 Decoder、关闭 AMP 或关闭 deterministic 来绕过问题。

成功父数据身份归档为 [parent_dataset_inventory.json](parent_dataset_inventory.json)，校验训练/验证/测试路径与标签内容指纹；读取 test 清单不等于运行 test。
配方归档 `configs/csr_p3_parent_args.yaml` 来自成功实验实际 args.yaml，与合同附录109字段完全一致。

曲线采样仅借鉴细长曲折结构采样动机（[原论文](https://arxiv.org/abs/2307.08388)）；不引入分割损失、拓扑约束或新标注，不保证拓扑连续性或精度提升。
