# PBI 数学融合审计精度修复

基线提交：`3daded7f3ec7412f800b687111f930f374bf536f`。本修复只改变数学融合审计、报告准入验证和对应回归测试。PBI/PBIConv 数学结构、forward/forward_fuse、初始化、CBR/LIF、两个模型配置、依赖以及全部 109 字段训练配方保持不变。

## 精度合同

数学证据版本为 `pbi_math_precision_v2`，融合证据版本为 `pbi_fusion_precision_v1`，策略为 `native_diagnostic_strict_fp32_reference_v1`。全工程合同仍为 `pbi_acceptance_v1`。

同一输入、同一原始 wrapper 和同一非零 PBI 权重用于两组检查：

1. `native`：保持进入审计时的设置，完整保留 `before_pbi` / `after_pbi` 的 `raw_allclose` 和误差。
2. `strict_fp32`：仅在该项参考检查内临时设置 matmul/cuDNN `allow_tf32=False`、float32 matmul precision=`highest`，容差仍为 `atol=2e-5, rtol=2e-4`。

每组均从同一原始 wrapper 独立 deepcopy，执行相同 `fuse_conv_and_bn` 和 `forward_fuse` 流程。融合包含矩阵乘法，因此严格范围同时包含融合构造与 forward；原生与严格融合后的 Conv 哈希允许因舍入不同，原始 wrapper、输入和 PBI 权重哈希必须一致。两组均记录融合前后 PBI 调用次数、PBI 参数哈希一致性、非零残差和有限性。

上下文使用 `try/finally`，先恢复两个 TF32 布尔设置，再恢复原始 matmul precision 字符串，最后逐项核对。这个顺序避免将原来的 `medium` 意外恢复成 `high`。作用范围不扩展到其他数学检查、完整工程预检或正式训练。

`native.raw_allclose=false` 保持为 false，原始误差保留，并标记 `PRECISION_NOTE`。只有严格参考数值通过、两组调用/有限性/参数/残差检查通过才会得到本数学策略的 `PASSED`，这不表示原生 CUDA 数值通过。其他原生 CUDA、AMP、half、保存恢复和容量验收仍独立遵循原标准。

错误分别标记 `PBI_CALL_COUNT`、`NONFINITE`、`PBI_STATE`、`ZERO_RESIDUAL`、`TOLERANCE` 或 `EXECUTION_ERROR`，精度恢复失败为 `PRECISION_RESTORE`。失败设备记录在抛出异常前绑定到最终报告；非有限误差用 null 和非有限元素计数记录，避免 NaN/Infinity JSON。

`train_pbi._math_pass()` 校验证据版本、精度策略、两个模式分段数值/调用/状态/哈希、上下文恢复和当前源码内容身份，拒绝旧格式、缺失精度或来源不符报告。`check_pbi.py` 在运行工程预检前执行同一验证。数学通过不能代替其他准入条件。

## 证据与验证状态

用户提供的服务器诊断来自上述基线 SHA，未在本次本机运行中重现或重新认证：

| 模式 | before PBI raw / max_abs | after PBI raw / max_abs | 其他证据 |
|---|---|---|---|
| 原生 CUDA | false / 约 3.7832558e-4 | false / 约 3.7671626e-4 | finite=true, pbi_calls=1, state_equal=true, residual≈0.122282 |
| 诊断中临时关闭两类 TF32 | true / 约 1.6689301e-6 | true / 约 1.6689301e-6 | finite=true, state_equal=true, residual 非零 |

这些值保留为用户提供的近似诊断，原生具体 flags 未提供，不能由本机设置推定。本次未访问、移动或覆盖服务器旧失败报告。现有 `init-preflight` 每次生成新时间戳目录/文件，旧失败报告继续保留；旧训练许可会按原机制归档撤销。

本机实测环境：Python 3.9.25、torch 2.7.1+cu118、CUDA 11.8、RTX2060。新证据在本目录 `evidence/`，gzip 文件解压即为原始 JSON，清单记录原始内容和压缩文件 SHA256。

| 检查 | 状态 | 证据说明 |
|---|---|---|
| 完整数学审计 CPU + CUDA | PASSED | `math_all_final.json.gz`，两设备原生/严格 pre/post 最大误差均 1.5497207641601562e-6，原容差通过；包括原生 AMP/half 数学检查与两变体结构检查 |
| 6 种有效设置 × 正常/异常退出 | PASSED | 12 项，三个精度设置全部恢复 |
| 非零正确融合及故障注入 | PASSED | CPU/CUDA 各验证正确融合，漏调用/重复调用/非有限输出/严格数值超差均拒绝 |
| 原生失败保留逻辑 | PASSED（合成夹具） | 明确注入原生 0.1 偏移，保留 false/误差；不构成 RTX4090 TF32 实测 |
| 报告准入回归 | PASSED（合成夹具） | 11 个损坏报告被拒绝，包含旧版本、缺精度、来源不符、虚标原生成功和原生 AMP/half 失败 |
| 真实数学 CLI 失败落盘 | PASSED（预期失败） | `omitted_pbi_math_FAILED.json.gz` 自身状态为 FAILED；漏调用被正确拒绝，保留具体数值与精度，后续结构检查未执行 |
| 原有准入拒绝回归 | PASSED（合成夹具） | 19 项；不生成训练许可 |
| 完整准入接口兼容性 | PASSED（合成夹具） | 6 项；使用历史工程数值及明确合成的环境/容量身份，仅验证接口和拒绝逻辑 |
| 源码保护核对 | PASSED | `scope_preservation.json.gz` 核对 882 文件及函数/AST，`supplementary_scope.json.gz` 补充 4 文件；配方及其他验收逻辑保持不变 |
| RTX4090 / torch2.1.2+cu121 完整数学、原生 CUDA/AMP/half、保存恢复 | PENDING | 必须在新 SHA 重新执行服务器完整预检 |
| 服务器 B16/640/AMP 容量 | PENDING | 继续使用原配方，最多 16 批、至少 2 次有效更新，失败不能放行 |

新报告记录测试时的基线 HEAD 和实际源码内容哈希；提交后不重写历史证据。准入要求在实际服务器当前源码/环境重新生成报告。旧 `docs/pbi/local_evidence` 未修改，新回归夹具及旧本机工程结果均不能充当服务器通过证明。

正式训练 **NOT_STARTED**；最终 test **NOT_RUN**；DPR 和其他实验未修改。

复现新增本机测试（使用新的输出文件名，已有文件受保护）：

```bash
python tools/check_pbi_math.py --device all --output outputs/pbi/math_precision_new.json
python tools/check_pbi_math_precision.py --device all --math-report outputs/pbi/math_precision_new.json --output outputs/pbi/precision_regression_new.json
python tools/check_pbi_ops.py --output outputs/pbi/admission_faults_new.json
```

固定提交的服务器同步与完整预检命令在推送并独立核实远端 SHA 后交付；入口仍为 `pbi_server.sh environment` 和 `pbi_server.sh init-preflight`，不执行 `start` 或最终 test。
