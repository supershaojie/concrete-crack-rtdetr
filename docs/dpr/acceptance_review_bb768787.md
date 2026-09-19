# DPR 验收规则定向审查与变更提案

审查日期：2026-09-19。被审查代码：`bb7687874a13782a768335adc60a38d00f150706`。母版：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。

**结论：未发现可证实的模型实现错误。现行准入继续 BLOCKED；本文件是待审查提案，未应用任何门禁变更。**

本轮只读取源码、核验附件、离线复核已有 JSON 的数值和分类条件。没有加载服务器 `.pt`，没有执行模型、反向、完整诊断、容量、start、resume、val 或 test，也没有登录服务器或切换服务器代码。全部既有模型、门禁、初始化、数据和 109 字段配方保持不变。

建议审查的路线是：保留保存恢复 A、FP32 折叠/文件推理硬门槛；为 B 定义有额外证据约束的条件接受；分别定义 half 文件一致性、原生训练验证可运行性与跨精度/跨转换路径等价性。**不能把这条路线写成“研究阶段完全不使用 half”：母版的 CUDA AMP epoch 验证实际使用 half EMA，且影响 best 选择。**

## 1. 附件、工作树与证据边界

开始审查时 DPR worktree 为 `D:/MyProjects/Crack_RTDETR/outputs/worktrees/Crack_RTDETR-dpr-v1`，分支 `exp-rtdetr-r18-lite-dpr-v1`，HEAD 等于上述诊断 SHA，工作区干净，没有需要回退的后续代码。独立主工作树及其他实验没有修改。

| 对象 | 本轮实际核验 |
| --- | --- |
| `diagnostic_light.tar.gz` | 777,989 bytes；SHA256 `52e7e2e38b07dd19a253c5ac219dd0e1afe03626f67dd60c4c402d7203cc3ccb` |
| `package_manifest.json` 原始字节 | SHA256 `63a8c7f54a8f63c099c7367bb61f461f98fb8ae8e6ce3c17b98d721afbc2c6bd` |
| 包内条目 | 恰好 9 个普通文件；无重复、缺失或额外条目；大小与 SHA 全部匹配 |
| `diagnostic.json` | 10,590,744 bytes；SHA256 `5f69fcc7be8029f44fd10413dda921a2c7fb670ecf0f5a305425dc694ee2bfb6` |
| 随包 6 个源码 | 与报告哈希、bb768787 Git blob、当前工作树归一化字节一致 |
| 开始/结束身份 | 完全相同；`source_stable=true` |
| 服务器记录 | Python 3.10.13、torch 2.1.2+cu121、RTX 4090、THOP 2.1.6；79.1690908894 秒；无超时 |
| 本次样本 | B2/160，6 个 GT；两张 `crack_000001` 派生增强图；不能代表完整数据或性能收益 |
| 总状态 | `COMPLETED / BLOCKED`，worker exit 3，`admission_eligible=false` |
| 没有执行的项目 | A=`NOT_RERUN_DIAGNOSTIC`；容量=`NOT_RUN_DIAGNOSTIC`；正式训练=`NOT_STARTED`；最终 test=`NOT_RUN` |

原包、原清单和离线审查结果保存在同名子目录 [acceptance_review_bb768787](acceptance_review_bb768787/)。`reviewed_metrics.json.gz` 保留两个 CUDA 模式的完整失败名称、四组梯度指标和解释性证据；原始 JSON 在无损保留的原包内。旧的 `failure_fix_20260919`、THOP 报告等没有覆盖或改写。

报告中声明的服务器张量路径仍指向 `outputs/dpr/cbr_lif_dpr_v1/diagnose_20260919T102122_041914Z`。`batch.pt`、`gradient_replay_*.pt`、`learned_diagnostic.pt` **不在附件内，本轮没有读取**。以下“实测”均指经过字节核验的服务器记录，不是本机重新计算张量所得。

## 2. 原始失败与解释性证据分开列示

### 2.1 已有检查修复得到本次支持

实际文件 AutoBackend 的 CPU FP32、CUDA FP32、CUDA half 三项均 `PASSED`，输出最大差为 0；独立 storage、源状态/文件绑定、前后文件 SHA、观察前后输出、部署与 norm 保留均成立。half 部署重载全部捕获路径最大差为 0。

CPU 的 `legacy_comparison` 仍是 `raw_allclose=false`，max_abs=`0.34705445170402527`，relative_L2=`0.025110136717557907`，超限比例=`0.0013333333190530539`。新 `legacy_diagnostic` 显示连续路径通过；两张图集合都相同，仅第一张两个索引位置交换；固定候选输出最大差=`1.7881393432617188e-7`。这次已经有旧比较的候选敏感性证据，但没有把其 raw=false 改成 true。27 个模式记录和 219 个 requires_grad 记录不同；state/buffer/cache 差异数为 0，不能把某一状态标志单独认定为唯一原因。

已证实的是此前的检查修复有效：对齐文件加载条件、真实 backend 观察、统计口径统一。本轮没有新增模型修复。`backend_audit` 的硬 raw 检查不需要另开候选排序豁免。

**文件测试范围仍有限：** 本次三个 backend 的 `provenance.selected` 是 `model`；审计临时文件为 `model=已学 FP32 模型, ema=None`。它并不是正式保存入口产生的 half EMA 文件。该结果不能代替新版本 checkpoint A、真实 EMA 文件路径及原生 epoch validator 的证据。

### 2.2 B 的 raw 失败仍然成立

固定 `atol=2e-5, rtol=2e-4`。表中顺序均为“母版 / 候选”。

| 模式 | 失败持久状态张量 | 失败梯度张量 | 候选独有失败状态名 | 当前 B |
| --- | ---: | ---: | ---: | --- |
| CPU FP32 | 0 / 0 | 0 / 0 | 0 | PASSED |
| CUDA FP32 | 22 / 20 | 108 / 114 | 8 | FAILED |
| CUDA AMP | 50 / 51 | 159 / 155 | 17 | FAILED |

| DPR 独立梯度 | CUDA FP32 max_abs | CUDA AMP max_abs | 两种模式 raw_allclose |
| --- | ---: | ---: | --- |
| CD | 4.1078776121139526e-5 | 1.33514404296875e-4 | false |
| HD | 4.291161894798279e-5 | 1.049041748046875e-4 | false |
| VD | 4.9384310841560364e-5 | 7.05718994140625e-5 | false |
| AD | 3.679841756820679e-5 | 1.1444091796875e-4 | false |

每种模式中，四组 DPR 的 20 条 model/EMA/optimizer 状态记录全部满足原容差。原始初态、storage 独立、前向、loss、buffers/元数据前提均通过；两种 CUDA 模式的每条失败持久状态，都符合现有 `gradient_support()` 的“对应梯度有限且不完全相等”条件。

两次运行各自的四组解析梯度、有效核梯度=W 梯度、固定输入/上游梯度/norm 初态的局部输出与梯度重放均最大差 0。梯度差传播残差 CPU=0、CUDA FP32=`3.725290298461914e-9`、AMP=0。AMP 实际 scale=4096，两边同阶段、除以同一真实 scale 一次；没有把 scale 改为 1。

这些证据支持局部线性映射和差异传播，**不证明整网独立梯度 allclose、不证明某个 CUDA 算子是唯一根因、不证明长期轨迹等价**。警告只是线索。

本轮从已核对源码中只提取无 PyTorch 依赖的 `gradient_support/classify_b`，对原 JSON 离线重算：得到同样的 CPU PASSED、CUDA FAILED。另做明确标记的反事实逻辑检查：即使只把分类器的 A 输入假设为 PASSED，两个 CUDA 模式仍 FAILED，因为名称子集和新增梯度条件依然不成立。该假设不是 A 证据，不写回报告。

### 2.3 half 失败的对象必须准确命名

原容差 `atol=0.002, rtol=0.02`，未改。U32 为未折叠 FP32；D32 为仅 DPR 折叠且保留 norm；N32 为再按母版原生规则融合；U16/D16/N16 是对应模型转 half。

| 比较对象 | 首个观察超限点 | 输出 max_abs | 当前 raw 结果 |
| --- | --- | ---: | --- |
| U32→D32 | 无 | 0 | 通过 |
| D32→N32 | 无 | 8.940696716308594e-8 | 通过 |
| U32→同有效核普通卷积 FP32 | 无 | 0 | 通过 |
| U32→U16 | norm_output | 0.8998547196388245 | 失败 |
| D32→D16 | norm_output | 0.8998544216156006 | 失败 |
| N32→N16 | norm_output | 0.8998544216156006 | 失败 |
| U16→D16 | 骨干 P4 | 0.8993836641311646 | 失败 |
| D16→N16 | encoder_features | 0.8993837237358093 | 失败 |

`audit_cuda_half()` 的旧键 `fp32_fold_then_half` **实际比较 U16 与 N16**（`half_capture` 对 `deployed_capture`，check_dpr.py:322），不是单独 D32→D16，更不是 FP32→FP16 的唯一比较。它的输出 max_abs=`0.8993837237358093`，raw=false。名称保留，但新合同应明确两端表示。

U32→U16 的 norm/P5 max_abs 为 `0.006237030029296875 / 0.03380107879638672`；同有效核普通卷积转 half 的对应值为 `0.006237030029296875 / 0.033059537410736084`。局部固定输入的两种核舍入顺序，核差=`6.103515625e-5`、输出差=`0.001953125`，均 allclose；整网 U16→D16 的 P4 差=`0.00390625`、P5 差=`0.015625` 仍超限。

U32→U16 有 392 个索引位置变化，两张图集合都改变。固定候选回放通过不能覆盖连续失败。P3/P4/P5 是骨干 `model[5]/[6]/[7]` 的观察点；“首个观察超限点”不等于第一个错误算子，不能据此断言 LIF、BN 有 bug。约 0.9 的逐位置输出误差不是 mAP 下降 0.9。

## 3. 原始 A/B 原则与当前实现的关系

已完整读取原始 `DPR_Codex_Prompt.md`，重点为专属验收第 10 条、E2 第 286–298 行。其要求 A 是保存正确性硬门槛，B 是另行报告的独立 CUDA 反向重复性；有充分的梯度/状态/父候选证据且 A 通过，才可解释 B。没有明确要求“候选失败名必须是母版失败名子集”，也没有明确要求“新增独立梯度全部 allclose”。

同时只读核对其指定的 DCC `784fb5b...:docs/dcc/acceptance_v2/README.md`：它保留新增参数更新门槛，但记载新增 W_o 部分梯度 raw 失败。该参考支持 A/B 的概念分层，不能把 DCC 的具体判定、报告或阈值直接当成本 DPR 证据。

当前 `preflight_dpr.py:443–464` 与 `train_dpr.py:303–319` 对精度说明增加了两个保守条件，且二者一致：

| 当前条件 | 能提供的约束 | 不能证明的事项 |
| --- | --- | --- |
| 候选失败状态名称 ⊆ 母版失败状态名称 | 本次有限样本没有新增“跨容差的公共持久状态名称” | 同名意味着同原因、误差幅度受限、未来样本/多步等价；阈值附近名称变化也可能来自微小误差 |
| 新增 DPR 状态及独立梯度均 allclose | 新增状态/梯度在原容差内，保守防止新分支异常 | 某次梯度超限必然是差分公式错误；父模型分支不同，失败数量更少也不能证明等价 |
| `gradient_support()` | 每条失败状态能映射到有限且非完全相等的梯度记录 | 实际 AdamW/裁剪/EMA 状态差完全由该梯度造成，或影响足够小；它没有重建更新，也没有检查误差上界 |

因此可以提出修订以更准确对应 A/B 原则，但不能把当前保守门槛当作已经授权删除的 bug。当前分类 FAILED 与实现相符；本轮维持它。A 本次没有执行，不能写 A 失败，也不能把历史 PASSED 标签填进来。

## 4. 实际精度路径审查

这里区分“由固定源码确定的执行路径”和“本次附件实际测得”。正式训练尚未发生，本轮没有伪造其运行 trace。父/当前 `engine/trainer.py`、`engine/validator.py`、`engine/model.py`、`nn/autobackend.py`、RTDETR trainer/validator、DetectionValidator 的 Git blob 相同。`torch_utils.py` 与父版的差异仅在 DPR FLOPs 接入；autocast、ModelEMA、strip_optimizer 函数未变。详见 `parent_precision_sources.json`。

| 阶段 | 真实调用及 dtype/转换顺序 | 本次证据和限制 |
| --- | --- | --- |
| start | `strict_gate` 后，受控初始化由 RTDETR/load_checkpoint 读入，`.float()`，按原 trainer 重建；训练 forward/loss 在原 `autocast(self.amp)` 内，GradScaler backward，原 unscale→clip(10)→step→update→EMA | 原配方 `amp=true`；实际 `check_amp` 仍会决定运行时开关，不能只凭 args 断言。附件 AMP 模式已实测 scale=4096；正式 start 未执行。 |
| live EMA | `ModelEMA` 初始为独立 FP32 eval 副本；按原 EMA 规则更新 | 不与未量化活动模型声称逐元素相同。 |
| epoch 验证 / best 选择 | `trainer.validate()`→原 `RTDETRValidator`；CUDA 且 `trainer.amp` 时覆盖 `args.half=True`，直接将 **EMA 对象** `.half()`，不做 fuse；输入也转 half；原 forward、loss、postprocess/metrics；结束 `model.float()` | `.float()` 只还原 dtype，不能还原 half 舍入损失；该过程可能改写 EMA 浮点值。这是母版原路径，保持不动。val fitness 决定 best/early stopping，不能仅因独立 val/test FP32 就删除此路径检查。新 DPR 诊断没有实际执行完整 validator 调用。 |
| checkpoint 保存 | `DPRCheckpointTrainer.save_model()` 保留 optimizer **FP32** 独立副本；`model=None`，EMA 独立副本 `.half()`；保留 epoch/scaler/updates/绑定 | 保存精度策略 `optimizer_fp32_v1` 已获原任务授权，未改；不能把 EMA half 存储与整网 half 算术等同。新 A 尚缺。 |
| resume | 在 cast 前检查原始 checkpoint 策略、名称顺序/组、当前 HEAD 和实验身份；load_checkpoint 选择 EMA→FP32；trainer 原生重建并恢复 FP32 moments/scaler、EMA `.float()` state 和 epoch+1 | FP32 EMA 是从已保存 half 数值提升，不是丢失精度的恢复；A 的参照是文件实际字节。没有调用 resume。 |
| 训练末尾 final_eval | 原生先 strip_optimizer：EMA 写入 model，half 保存，optimizer/scaler/EMA 等清空，epoch=-1；同一 validator 接收 best 文件。它**没有将之前的 args.half 清回 false**，按本配方 epoch 验证后的路径仍为 half；文件 CPU load→float→CPU fuse→eval→device→half | 原生最终验证通常是 half，不是独立 FP32 val。已 strip 的终态文件不再可续训；正式续训仍须未完成 last.pt。此完整路径本次未执行。 |
| 独立 val/test | `eval_dpr.py` 的 EVAL 固定 half=False；`RTDETR(weights)` 先文件 CPU load、选择 EMA 或 model、float（不融合）；`Model.val` 将内存模型交给新 validator→AutoBackend，在模型当前 CPU 上 fuse（本项目非 Jetson路径），再搬设备并 float；输入 float | 不只 args：`StreamingValidator.init_metrics` 还断言实际 `self.args.half=False`。该路径与直接传文件 AutoBackend 的入口形式不同，仍应有受控非指标审计。全量独立 val/test 没有执行。 |

精确源码锚点：`validator.py:143–151,164–172,244–245`；`trainer.py:429–445,514–533,721–729,735–753,819–833,887–896`；`models/rtdetr/train.py:86–89`；`models/yolo/detect/val.py:75`；`nn/tasks.py:1527–1545`；`nn/autobackend.py:215–238`；`engine/model.py:610–614`；`tools/dpr_checkpoint.py:167–220`；`tools/train_dpr.py:397–438`；`tools/eval_dpr.py:26,93–104,140,173–205`。

母版 `c19_lif_v1_results.py` 同样显式独立 FP32；`c19_lif_v1_precision.py:28–66` 已专门检查原生 half epoch-validator 和 fuse→half backend。它们仅作父实现路径依据，不把历史测试标签拿来放行 DPR。CBR 本身还有局部 FP32 运算；模型参数/输入为 half 不代表每个内部算子或最终输出都是 FP16。

## 5. 待审查规则：现规则 → 候选规则 → 必需证据 → 未解决风险

以下仅是候选方案 **R1**。保守方案 **R0** 是不变更合同，继续 BLOCKED；只在发现具体实现错误时修复。建议审查 R1，不建议现在直接把本报告变成许可。

| 条款 | 现规则 | 候选规则 R1（尚未生效） | 必需证据 | 未解决风险 |
| --- | --- | --- | --- | --- |
| A 保存恢复 | 三模式 A 硬门槛 | **不变**。先于 B 条件接受，缺失直接阻塞 | 新固定版本原保存字节、FP32 optimizer 与 live 独立副本、参数/组/超参数绑定、model/EMA/buffers/moments/step/scaler/epoch/updates 完整；独立 storage；相同梯度完整原生更新 exact；CPU native continuation；AMP 有效更新和 overflow skip | half EMA 相对未量化 live 状态的变化仍存在；不承诺中断/不断训练长期轨迹相同 |
| B 原始记录 | raw 比较；附加子集与新增 raw 梯度条件 | raw 全部保留。仅 GPU 可申请 `EXPLAINED_BACKWARD_VARIATION`，必须通过下节 B1–B8；不再把失败名称子集作为独立硬条件，改为逐条数值重放及一步功能边界。**新增持久状态仍需原 allclose**，只允许解释新增独立梯度 raw 失败 | 同条件父/候选、RNG/精度指纹、完整有限状态、对应梯度支持、独立解析/冻结局部/差分映射、原生全更新重放和一步后功能检查；新 A | 这是明确放宽两个旧条件并增加替代证据；局部正确和单步边界不保证长期轨迹或精度收益 |
| FP32 折叠及文件推理 | 原层级精度检查；实际 backend raw 硬门槛 | **保持**，包括独立参考、字节/源状态身份。已解释的原 native_fuse 候选规则不扩大到 backend | 非零 learned 状态、独立数学参考、未折叠/仅 DPR 折叠/母版对称融合/幂等/重载；文件真实调用和每条连续路径；EMA 实际保存文件也覆盖 | 两个同样错误的加载器一致不够；不能省掉独立映射、未折叠对照和保存字节审计 |
| half 保存重载 | `half_deploy_reload` 必须通过 | **不变，独立能力 H1**；同表示同 dtype 重载仍需原容差及状态字节检查 | 已学非零 DPR，表示/键/部署/norm/dtype 一致，原始文件哈希，真实重载输出 | 不证明 FP32→FP16 或不同转换路径等价 |
| half 实际 backend | `autobackend_half` raw 必须通过 | **不变，独立能力 H2**；同精度、同文件、同加载顺序严格比较 | 现有所有 backend 事实 + 正式 half EMA 保存对象，覆盖 final_eval 的文件表示 | 当前 `ema=None` 的 FP32 fixture 不是 EMA 保存链完整证据 |
| FP32→FP16 | 跨精度中间层及输出必须通过才能研究启动 | 候选支持限制 **H3：跨精度等价性不承诺**；原 raw FAIL 永久保留，评估值仍照录；不得标 PASSED 或普通 PRECISION_NOTE | 非有限值仍硬失败；原生训练验证可运行且状态正确的 H0 必须另行通过；独立 FP32 评估路径身份必须通过 | 改变支持范围；half epoch 选出的 best 与 FP32 选出的 best 可能不同，绝不据此改 best 规则 |
| U16→D16 / D16→N16 / U16→N16 | 旧 half 聚合门禁包含转换顺序等价性 | 候选支持限制 **H4：不同转换顺序不能互换**；固定每个正式阶段的既有顺序，保留每条失败；不允许通称“half 已通过” | U16 原生 epoch 验证与 N16 文件验证各自的表示/有限性/不丢 DPR；FP32→DPR fold→母版 fuse 为唯一声明的部署构造顺序；不用 half().fuse() 替代 | 本次 U16→D16 连续特征失败是真失败；即使局部核 allclose，也不保证整网等价 |
| 原生 half 训练验证 H0 | 原实现路径存在，但当前 DPR 有限诊断没有实际调用它 | **增加明确硬门槛，不能被 H3/H4 范围限制豁免** | 一次受控真 EMA 原生 validator 工程调用：确认 unfused U16、真实 GT loss/postprocess 有限、输入/参数 dtype、未丢差分、结束 FP32 dtype 恢复；再验证真实 EMA 文件加载路径；无全量指标 | 仅两张图仍是工程证据；不得因此宣称整网 FP32/half 精度等价 |
| 新鲜证据与许可 | `dpr_acceptance_v1` + `dpr_full_preflight_v2`，固定身份严格检查 | 获明确批准后才实现新合同版本；报告分 raw evidence、能力状态、admission 三层。所有旧/局部报告不能直接准入 | 新固定 HEAD、内容哈希、服务器环境、variant、源/初值/数据、109字段、数学/初始化/A/B/推理/H0–H2/容量完整；validator 独立重验事实 | 文档提交本身使 HEAD 改变，不产生新许可；不得修改旧 JSON 的 SHA/版本来冒充新证据 |

### B1–B8：替代条件必须同时成立

1. **B1 新 A**：同一新固定版本对应模式的 A 全部硬事实通过；不能从本次 `NOT_RERUN_DIAGNOSTIC` 或旧版标签推导。
2. **B2 控制一致**：父、候选各自两个独立 live FP32 副本；完整初态 exact/storage 不共享；同 batch/标签/预处理身份；开始前 CPU/CUDA/Python/NumPy RNG 指纹相同，记录 autocast dtype、实际 scaler、TF32/cuDNN/deterministic flags；前向连续路径与 loss exact。父与候选不要求学后参数互相相同，要求各自成对控制成立。
3. **B3 状态完整**：全参数、buffers、optimizer 组/绑定/超参数/moments/step、EMA、scaler、epoch、updates 无缺项且有限；buffers、整数/步数、scaler 和元数据 exact。两边有效更新。新增四组的参数/EMA/moments/step **继续使用原 raw 门槛**；模型梯度注册覆盖完整，结构性零空间不作非零错误。
4. **B4 梯度对应**：保留每条公共/新增 raw 失败名称、坐标/幅度/relative_L2/超限比例及母版对照；持久浮点失败必须有对应有限、非完全相等的同阶段梯度记录。只有“母版也失败”或总数更少不能通过。
5. **B5 新分支因果分解**：每次有效核梯度与 W 梯度、四种独立转置映射、两次差分映射及同输入/同 upstream/同 norm 初态局部重放，均在原 FP32 阈值内；FP64 参考仍为 `1e-10/1e-9`。同 scaler、同 backward、pre-clip；不重复 unscale，不更改 optimizer 顺序。
6. **B6 全更新解释**：对每个父/候选控制，保留 backward 完成、`optimizer_step` 之前的完整可变状态及全量原始 scaled 梯度（不只 DPR），包括前向已更新的 buffers。从该状态和原 scaler 用捕获梯度走一次原 native unscale/全局 clip/step/scaler/EMA 重放，与该次真实更新后的完整状态 **exact** 对齐；再核对两个重放结果的逐状态差等于已报告差异。unscaled 分析副本不能冒充原始 scaled 梯度；不要重复 unscale，也不能只对四组梯度手算 AdamW。A 的同梯度恢复重放不能自动替代此处 live B 因果链。
7. **B7 一步功能边界**：对 B6 中两份已更新候选，原固定输入、eval FP32、预先规定且恢复的 TF32 设置，比较有效核、target、骨干 P3/P4/P5、encoder/候选分数；仍用 `2e-5/2e-4`。输出 raw 必须通过；若仅候选离散变化，则需严格连续+实际候选+固定回放全部成立。这是本提案为新增比较显式规定的诊断标准，不代表现门禁已有此 B 豁免。不允许只因为 B6 能重现一个大差异就放行；未通过仍 BLOCKED，禁止再调门槛。
8. **B8 独立决策**：raw 仍为 FAILED/false；条件接受单独记录具体 B1–B7 证据引用、哈希、模式、未解决风险和批准的合同。任何缺项/非有限/共享状态/异常更新/前向不一致/重放失败均不能得到条件接受。`long_term_trajectory_equivalence=NOT_CLAIMED`。

**本次并不满足这套候选规则。** B1 没有执行；B2 有相同 seed 调用和前向证据，但缺少逐次起始 RNG/精度完整指纹；B5 有已核验记录；B6 缺少整个 live 更新前状态/全梯度/实际重放；B7 未执行。绝不能仅凭 B5 自动生成条件接受。额外的 B6/B7 是本提案为替代两个保守条件提出的约束，不伪称原提示词已经要求或本次已经通过。

## 6. 新合同与状态迁移设计（仅提案）

若用户批准 R1，再单独实现例如 `dpr_acceptance_v2` / `dpr_full_preflight_v3`，并在报告中绑定实际批准的 policy 文档哈希；本轮不改 `CONTRACT` 或任何验证器。拒绝 draft/未知/缺版本，不从可选字段推测适用范围。

建议最终报告至少明确：

```text
raw.B.cuda_fp32.status                 = 原始 FAILED（不可重写）
raw.B.cuda_native_amp.status           = 原始 FAILED（不可重写）
assessment.B.*                        = PENDING / REJECTED / EXPLAINED_BACKWARD_VARIATION
capability.checkpoint_A                = PASS 才能准入
capability.fp32_fold_and_file_inference= PASS 才能准入
capability.native_epoch_validation_H0 = PASS 才能准入
capability.half_same_path_H1_H2        = PASS 才能准入
capability.fp16_cross_precision_H3    = NOT_SUPPORTED_UNDER_THIS_PROFILE
capability.fp16_cross_conversion_H4   = NOT_SUPPORTED_UNDER_THIS_PROFILE
admission.profile                    = 原生 AMP + 原生 half epoch/final val + 独立 FP32 评估
admission.status                     = BLOCKED，直到获批合同及全部必需事实满足
```

H3/H4 的候选能力标签不是“测试通过”，不能被放进 `{PASSED, PRECISION_NOTE}` 集合。评估、部署和打包也必须读取同一支持范围，防止训练许可被误作任意 FP16 部署许可。原始叶子、阈值和误差不变。

之后只在获准的新固定版本上收齐必要证据：初始化/数学、A、B补足事实、FP32与真实EMA推理、H0–H2、原 B16/640/AMP 有限容量等；完整汇总必须逐项重新校验。旧报告可用于审查背景或明确的内容身份追溯，但不能复制状态标签、换 HEAD 后成为新准入。当前 `strict_gate` 仍严格拒绝旧 HEAD、局部诊断和缺项。

开始补检时依原流程归档撤销旧许可；失败不留许可。即使将来 verify 通过，也不能自动串接 start；正式训练仍需后续显式启动。另一配置消融不自动训练。

## 7. 具体缺口、最小补证与停止条件

**本轮没有必要重跑同版完整诊断。** 以下缺口不是改数学结构的理由，也不是要求用户上传完整权重。实施仅在下一步规则/取证范围被明确接受后进行。

| 缺口 | 最小方法 | 停止条件 / 最小交付 |
| --- | --- | --- |
| 真 EMA 文件与原生 half epoch/final 路径未覆盖 | 先只核对现存文件角色。已有 learned_diagnostic 只能作已学模型对象，不冒称真 EMA。下一次必要 A 保存时复用其真实 EMA 文件，独立 disposable 副本、同一已有 batch，原生 validator 只 1 batch（不计全 split 指标、不改变生产类），再做同文件精度路径身份检查；不得写入正式 run/best | 每路径最多一次，超时 120 秒即停止；出现 dtype/表示/非有限/异常清理问题保持 FAILED。只返回 JSON/日志和源码哈希；完整张量留服务器。**本轮不执行 validator/val。** |
| B 的梯度支持仅为相关性 | 审查获批后，在受影响 CUDA FP32/AMP 的同一次 B 更新内采集 B2/B6 所需完整状态与梯度，并只做原生更新重放和 B7；不再运行 CPU/half/容量矩阵，不变 seed/input/阈值 | 每模式父/候选各一组成对更新，预先固定预算；只补新字段，失败停止，不重新抽样找通过。完整 state/grad 留服务器，交付逐条 residual/失败名与哈希。现存局部 `gradient_replay_*.pt` 缺全局 optimizer/全梯度，不能完成 B6。 |
| 真正新 A/容量缺失 | 在合同获准且上述定向问题有结论后，纳入一次新的完整准入汇总；沿原有限预算 | 本轮不跑；不搬用旧容量或 A 标签，不把元数据检查当 A |

下面是可选的**只读文件角色核验**短命令，明确仅查看现存 CUDA FP32 `learned_diagnostic.pt`，无 forward/backward/val、无 GPU 分配、无写文件。其目的只是核对“当前 retained 文件不是原生 half EMA checkpoint”，不能补足 A、H0 或 H2。服务器张量本轮未读取、此命令本轮未执行；不需要上传该权重。

```bash
bash <<'DPR_METADATA_ONLY'
set -eo pipefail
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
set -u
cd /root/autodl-tmp/projects/Crack_RTDETR-dpr-v1
test "$(git rev-parse HEAD)" = bb7687874a13782a768335adc60a38d00f150706
export PYTHONPATH="$PWD/ultralytics-main${PYTHONPATH:+:$PYTHONPATH}"
timeout 60s python - <<'PY'
import hashlib, json
from pathlib import Path
import torch
base = Path('outputs/dpr/cbr_lif_dpr_v1/diagnose_20260919T102122_041914Z')
r = json.loads((base/'diagnostic.json').read_text())
ref = r['modes']['cuda_fp32']['learned_artifact']
p = base/'cuda_fp32/learned_diagnostic.pt'
assert p.resolve() == Path(ref['path']).resolve()
assert hashlib.sha256(p.read_bytes()).hexdigest() == ref['sha256']
c = torch.load(p, map_location='cpu', weights_only=False)
selected = 'ema' if c.get('ema') is not None else 'model'
m = c[selected]
print(json.dumps(dict(selected=selected, purpose=c.get('purpose'),
    ema_present=c.get('ema') is not None, optimizer_present=c.get('optimizer') is not None,
    parameter_dtypes=sorted({str(v.dtype) for v in m.parameters()}),
    devices=sorted({str(v.device) for v in m.parameters()}),
    admission_eligible=False, scope='metadata only; no model execution'), indent=2))
PY
DPR_METADATA_ONLY
```

预计结构来自已核对源码；上面未产生本轮实测输出。任一身份不匹配应停止，不用搜索其他权重代替。将来若需要新张量证据，只请求 B2/B6/B7 与 H0–H2 的小型 JSON/日志；不请求全部权重或 GB 包。

## 8. 本轮交付与需要审查的决策

本轮仅新增本文和附件/审查证据目录；没有候选代码 diff，因为改变生效门禁尚未获准，也没有可信模型缺陷需要代码修复。原 DPR、CBR/LIF、3,072 参数、THOP、精度阈值、200e/B16/640/seed42/AdamW/lr0=0.0005/warmup5/原在线增强/AMP 均不改。

需要用户分别审查的具体条款：

1. 是否接受 B1–B8 对 B 精度说明的定义，明确以替代证据取代“失败名称必须子集”和“四组独立梯度必须 raw allclose”，同时保留新增持久状态 allclose？本次证据尚不足以获得该条件接受。
2. 是否接受 H3/H4 作为明确的能力限制，保留所有原始失败，同时将 **H0 原生 half epoch 验证、H1/H2 同路径文件一致性**作为硬门槛？这改变验收范围，不是修好 half，更不改变原 best 选择路径。
3. 是否接受新合同/报告版本、证据与能力/准入分层、原始记录只读和新鲜证据规则？文档提交不代表批准，局部诊断永不生成许可。

本地验证范围：附件与清单、六份源身份、原始 JSON 数值/失败集合、实际纯分类函数离线结果、父版精度相关源码比较、文档命令语法、Git 差异范围与原门禁文件哈希。没有重新执行 PyTorch/GPU 诊断；未执行的 A、容量、真实 EMA 验证、B6/B7 均明确保留 PENDING。验证明细见 `review_validation.json`。

**现行合同：不变。当前准入：BLOCKED。规则变更：PROPOSED_NOT_APPROVED / NOT_IMPLEMENTED。正式训练：NOT_STARTED。最终 test：NOT_RUN。**
