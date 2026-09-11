# Local validation

Environment: Windows, Python 3.9.25, torch 2.7.1+cu118, CUDA 11.8, RTX 2060 (6 GB).
Evidence records the pre-commit base HEAD and dirty implementation under test; release source hashes
are in source_files_sha256.json. This is an implementation verification, not a 200e result.

| Check | Result |
|---|---|
| C2 / C17 / C24 / C25 / C19 historical subprocess regression | Exact state load; every nested forward tensor max abs/rel = 0 |
| Protected originals / C2 execution AST / parent transformer | Unchanged |
| Control vs original C17, activated CSCEF | 640x640 and 160x192: all named features/bbox/score abs/rel = 0 |
| Pair vs old C25, activated CSCEF/SCCA | 640x640 and 160x192: all named features/bbox/score abs/rel = 0 |
| All three zero-initialized variants vs C2 | Exact output, lateral, P3/P4/P5 at both sizes |
| Nonzero SCI | Original Y4/upY4/lateral and actual Concat tensor exact; proxy/CSCEF/P3/P4/P5 change, finite and same shape/dtype |
| Original semantic gradient | Control and pair: Y4/upY4 L2 ratio = 1; abs/rel = 0 |
| Nonzero SCI input Jacobian | Identity gradient exact; isolated correction input gradient None |
| Active SCI/SCCA/CSCEF parameters | Every parameter receives finite nonzero gradient |
| FP32 and CUDA AMP loss/DN | Three updates per variant; native loss, backward, scaler, AdamW all pass |
| Dynamic DN | GT groups [2,1], [3], [1,4]; total queries 500, 498, 500 |
| True FP16 and CUDA AMP inference | Activated original modules and SCI; finite outputs and rectangular input |
| Optimizer | Every parameter exactly once; native bias/norm decay groups; learned optimizer state reload exact |
| Save/load | Full learned model, optimizer tensors/groups, output and native trainer rebuild exact |
| nc80 -> nc1 / public C2 mapping | Each public tensor exact; 9 allowed class adaptations; no unexplained keys |
| Standalone adapter | Zero identity in train/eval; constructor RNG unchanged; no first-layer zero |
| Train statistics | First training batch per epoch, validation ignored; diagnostics detach and do not gate |
| Ops | 13 focused unit tests; actual Bash sync with Git fixture; syntax checks |
| Val/test pipeline | Exactly 2 copied images per split with untrained triad; no full dataset evaluation |

Synthetic affine/channel mixing: loss 0.00343778403476 -> 9.26407810766e-05 in 30 steps. This proves engineering learnability only.

Parameter counts (unfused nc=1): control 20,126,836; pair 20,192,376; triad 20,238,265.
Each adds exactly 17,152 parameters. Diagnostic GradScaler initial scale=128 is bounded-smoke-only;
formal training uses the unchanged native AMP scaler/settings. GN/reduce learning begins on the third
native optimizer step due to the two serial zero projections, as explicitly tested.

Full values: validation_summary.json, gradient_paths.json, historical_regression.json, source_audit.json,
ops_validation.json, evaluation_pipeline.json, pack_validation.json, evidence/*_initialization.json.

Reproduction from repository root (use an unused output path):

```bash
python tools/audit_sci_adapter.py --report outputs/sci_source_audit.json
python tools/check_sci_adapter_regression.py --report outputs/sci_regression.json
python tools/check_sci_adapter.py --source weights/rtdetr_r18_lite_imagenet_backbone_init.pt --output outputs/sci_validation
python tools/check_sci_adapter_ops.py --bash bash --report outputs/sci_ops.json
```

Unverified scope: real AutoDL 2.1.2/cu121 environment (mandatory preflight will run there),
actual remote tmux dispatch, long-run training statistics/convergence, complete-dataset val/test,
and detector improvement. No formal 200e training was launched. A packaging fixture uses synthetic
training lifecycle records plus actual two-image evaluation files and labels itself accordingly.
No claim is made that the archive is a real completed experiment.
