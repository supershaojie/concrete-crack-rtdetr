# TCR v1 服务器融合精度诊断修复

修复起点：`6a179e6b2577ffe3a9579074d3589691b719c9bb`，分支 `exp-rtdetr-r18-lite-tcr-v1`。母版仍为 `a0459d6a652cb702699087c88fa39a3e4c4087ec`。

## 服务器原始证据与已定位缺口

用户提供的失败目录为 `/root/autodl-tmp/projects/Crack_RTDETR-tcr_v1/outputs/tcr_v1/preflights/20260926T103719Z_33b1932d`。本次没有登录服务器读取或修改该目录。用户提供的 `fusion/fuse_diagnostic.json` 片段：

```json
{
  "runtime": {
    "python": "3.10.13", "torch": "2.1.2+cu121", "cuda": "12.1",
    "threads": 4, "tf32": false, "cudnn_tf32": true, "deterministic": true
  },
  "precision_path": {
    "model_a_dtype": "torch.float32", "model_b_dtype": "torch.float32",
    "input_dtype": "torch.float32", "autocast": false
  }
}
```

旧报错为 `FAILED_REAL_NUMERICAL_MISMATCH`，`cuda.fp32.fuse.pre_selection/lif_input`，`[1,256,80,80]`，最大绝对误差约 `0.0004224`，超差元素 `384609`，非有限值 `0`。原阈值是 `atol=2e-5, rtol=2e-4`，本修复保持不变。

已确认的实现缺口：母版独立 checker 关闭两种 TF32，而 TCR 服务器预检继承 Trainer 后的运行环境。旧 `precision_path.autocast` 来自 `precision == 'amp'`，并非实时状态读取；FP32 路径使用空上下文，不能阻止外层 autocast。原比较顺序先查 `lif_input`，所以旧报错不能说明它是最早发生差异的节点。

PyTorch 2.1 的 [官方 CUDA 精度说明](https://github.com/pytorch/pytorch/blob/v2.1.2/docs/source/notes/cuda.rst#tensorfloat-32tf32-on-ampere-devices) 明确区分 matmul 和 cuDNN TF32；张量 dtype 为 float32 并不排除 TF32 卷积。服务器记录中开启的 cuDNN TF32 是优先核查的因素。BN 折叠使用原生 `torch.mm`，因此严格控制覆盖融合模型的构造与前向两部分，即使本次服务器 matmul TF32 已关闭也保留这一边界。

**尚未确认：RTX4090 上同一旧 fixture 关闭 cuDNN TF32 后是否通过。** 本机 RTX2060（计算能力 7.5）不能执行 Ampere 及后续架构的 TF32 运算；本地两个精度标志设置均通过不能代替这项因果证据。

## 实现与证据结构

新增 `tools/tcr_v1_fusion.py`，只由诊断入口使用。TCR 公式、`ConvTCR.forward_fuse`、原 LIF/CBR 和母版诊断/候选判据源码均未修改。

1. 诊断副本先核对与 Trainer 全部权重/BN 完全相同，再 eval/float。只在这个副本上注入非零 O；记录唯一变更键为 `model.17.tcr.O.weight`。两个融合副本均在此后创建。Trainer 的模式、权重、BN 以及 P 不变。
2. `precision_scope(strict=True)` 在局部设置 matmul precision=highest、matmul TF32=false、cuDNN TF32=false，并关闭 CPU/CUDA 两种实际 autocast。`finally` 恢复进入时的设置和外层 autocast，保留 `medium` 与 `high` 的区别。确定性、benchmark、AMP、GradScaler 和正式超参数不受此作用域修改。
3. 记录 Trainer 构造前、构造后、`_setup_train` 后、诊断前后的实际精度状态。每一对照保存完整模型状态/BN/P/O 哈希、eval/dtype、输入哈希、CPU/CUDA/Python/NumPy RNG 身份。
4. 顺序钩子追踪骨干内部 `ConvNormLayer/BasicBlock`、节点 0–25、节点 17 原 Conv 块输出、TCR 输入/P/O/输出/残差、节点 19、`lif_input`。原 Conv 块输出指 conv/bn/act 后的语义边界；BN 折叠后不将裸 conv 与未折叠裸 conv 当作同一语义比较。
5. 分开报告第一个非逐位相同节点与第一个超过原容差的节点。所有节点继续收集；早期节点失败时仍运行母版完整诊断，保留原 `lif_input` 失败、`records.pt`、`fixture.pt` 和候选证据，不改错误状态。
6. 审计 `ConvTCR.forward_fuse` 绑定、TCR/P/O 各一次真实执行、非零 O 和可见残差、P/O 精确保留、LIF BN/状态/forward 保留，确认其他普通 BN 确实融合。

一次诊断目录含：

| 路径 | 内容 |
|---|---|
| `precision_diagnostic.json` | 入口/恢复状态、各对照结果、权重/输入/RNG 控制及设置差异 |
| `strict_fp32/fuse/` | 在严格精度下创建并运行的原生 fuse |
| `strict_fp32/backend/` | 严格精度下的真实 AutoBackend fusion 与 wrapper 精确比较 |
| `runtime/same_strict_weights/` | 使用完全相同的严格融合权重，只恢复原运行精度做前向 |
| `runtime/native_fuse/` | 从相同未融合状态，在原运行精度下构造并运行融合模型 |
| 每个对照的 `early/early_nodes.json` | 顺序节点统计，首次超差另保存原始张量 |
| 每个对照的 `full/fuse_diagnostic.json` | 原母版完整协议报告，失败证据保留 |
| `historical_weights_strict/`、`historical_weights_runtime/` | 仅重放入口：严格加载旧 fixture 已保存的融合权重，分别检查 |

每个 `comparison.json` 的 `settings.autocast_cuda/autocast_cpu` 是实际查询结果。母版 `full/fuse_diagnostic.json` 原有 autocast 字段仍表示其请求的诊断模式，应结合新的实际查询字段读取。

`same_weight_precision_control` 必须证明输入、两份模型及 RNG 全部相同；`setting_differences` 明确列出严格/原精度差异。若服务器新证据显示唯一差异是 cuDNN TF32 且同权重对照由失败变为通过，才能将该次差异归因于这一精度设置。模型构造时的精度影响另由 `matches_strict_fused_weights` 和历史权重对照判断。

## 结果状态与启动门槛

原 `2e-5/2e-4` 阈值、`lif_input`、完整融合和母版候选判据全部保留。严格 FP32 失败时预检失败并阻止启动。两个精度都通过时预检为 `PASS`；只有严格 FP32 通过时明确为 `PASS_STRICT_FP32_ONLY`，要求有界 B16/640 容量、实际更新/P 梯度等其余门槛也全部完成。其 `fusion.runtime_accepted=false`，原精度失败状态与张量继续保留，不宣称原精度通过。该作用域定义不改变正式 AMP 训练配方。

重放入口只检查保存的输入和权重，不运行 Trainer、数据扫描、优化、val 或 test。它恢复旧报告中实际记录的两类 TF32 标志和保存的 RNG；其他生效设置逐项记录，不假定旧报告未记录的 autocast/cuDNN benchmark。原失败文件仅被读取，新目录拒绝覆盖。

## 针对性验证与限制

本地测试入口：`tools/check_tcr_v1_fusion.py`。使用实际受控初始化并完成原生 nc80→nc1 加载；两次无梯度小输入前向产生非平凡 BN 状态，然后以非零 O 做融合测试。

- CPU B1/160：原容差下严格融合、AutoBackend、原精度、同权重对照通过。
- CUDA B1/640：本机 RTX2060 上相同项目通过；这不是服务器 B16/640 容量测试。
- 成功/异常、CPU/CUDA 外层 autocast、matmul highest/high/medium 恢复均实测。
- 修改 P/O、替换掉 TCR 的融合 forward、遗漏 TCR 执行、移除 LIF BN 均被拒绝。
- 注入节点 0 真实误差后，顺序追踪先定位节点 0，母版仍失败于 `lif_input`；失败不转 PASS。严格门槛失败和不完整容量不能进入训练。
- 原故障注入 fixture 的只读重放另执行，并检查文件哈希保持不变；此夹具用于验证重放入口，不作为 RTX4090 TF32 根因证据。

精确运行信息、最大误差、源文件哈希和证据路径见随本修复提交的 `fusion_precision_validation.json`。原交付的本地报告仍保留其原时间与源码身份。新服务器 B16/640 预检、RTX4090 原 fixture 因果对照均为 **PENDING**；正式训练及完整 val/test 未执行。

服务器使用更新后的 `outputs/tcr_v1/server_commands.md`，先按完整新 SHA 同步。可先执行 `diagnose-fusion --fixture .../20260926T103719Z_33b1932d/fusion/fixture.pt --device 0` 检查旧失败输入；随后 `prepare`、`preflight` 会创建新预检目录，不删除旧证据。
