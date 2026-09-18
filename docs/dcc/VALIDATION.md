# Actual local validation

Formal training **NOT_STARTED**. Final test **NOT_RUN**. Start gate **BLOCKED**.
Public initialization, DCC math, graph wiring, parameter counts, gradient startup, learned reload/EMA and fusion have passing evidence.
The complete engineering result is **PRECISION_NOTE**, not an unconditional pass: native CUDA next-update equivalence did not meet its original tolerance.

Environment: Python3.9.25, PyTorch2.7.1+cu118, RTX2060 6GiB, current worktree ultralytics. This is not a server2.1.2 test.
The final small smoke uses original deterministic=True/warn-only behavior, no benchmark, preserved default TF32 flags; strict fusion temporarily disables TF32 and restores it.

| Path | Result | Effective updates | Native next-weight max abs | Gradient max abs | Strict FP32 fusion |
|---|---|---:|---:|---:|---|
| cpu_fp32 | PASSED | 3 | 0 | 0 | PASSED |
| cuda_fp32 | PRECISION_NOTE | 3 | 5.57133865 | 7.16745853e-06 | PASSED |
| cuda_native_amp | PRECISION_NOTE | 3 | 59.0951314 | 0.00122070312 | PASSED |

Each path used two actual train images at160 with valid GT/native detection loss. These are disposable fixed-LR smoke models,
not the formal warmup/capacity protocol. CPU next update is exact in the final run. CUDA restores all optimizer/scaler/epoch/EMA
state exactly and has identical captured forward tensors and loss, but native backward differences produce unequal next weights.
Measured native serialization turns about19.37million nonzero second-moment entries into half zeros. Amplification by these
quantized moments is a numerical inference, not a claim that the observed weight differences are small or harmless.
Identical-gradient native optimizer replay is exact; it proves restoration consistency only. The start gate requires native
next-update allclose and rejects this unresolved result. It cannot be bypassed by a precision note or local2.7 evidence.

Original v1/v2 failed diagnostic controls retained half-quantized ephemeral anchor caches in only one control; v3+ rebuild the
same cache state. v4 retained an overly broad FP32 gradient assertion on AMP; v5 recorded the diagnostic note but used overly
broad top-level PASSED wording. Final v6 preserves declared tolerances and propagates PRECISION_NOTE. All original JSONs/logs
remain in outputs; attempt_history.json records their identity. Old reports are not relabeled.

Local native B16/640 capacity result: **FAILED**. Exact report/error, memory and timing are in capacity_local.json.
The first local attempt hit Windows dataset cache permissions; the bounded authorized retry used the native cache path.
No batch reduction, AMP disable, stopped foreign process, full epoch, validation or test was used.

Initialization full per-key reports contain552/533 public states and9 exact nc adaptations. Both DCC initial states match.
Module reports capture B×4×8×8 attention,18,432 trainable parameters and nonzero-path math/casts/fusion.
Parameter_counts.json records all8 unfused/fused totals. Actual parent recipe has109 fields, all matching the appendix.
Actual data paths/labels match historical inventories exactly (6048/1728/864 images).

Operations fixtures verify sorted confidence masking, attainable tied-score fixed-P logic, full CLI, exclusive outputs,
light archive readback and conservative start gates. Operations_gate_final.json supersedes the earlier permissive gate fixture.
The final source identity report preserves measured HEAD and explains the sole post-network-test shell gate addition;
all Python, model and configuration hashes are identical to the final network test.

Remaining PENDING: server2.1.2 checks/capacity, missing reference ZIP correspondence, actual training/accuracy/fixed-P comparison.
No FLOP completeness or capacity/speed result is inferred from parameter count. Native THOP summary is PARTIAL.
