# Local validation

Environment: Windows, Python 3.9.25, PyTorch 2.7.1+cu118, RTX 2060 6GB. No dependency installation/upgrade. No formal epoch loop and no full val/test. Validation runtime records show parent 05e6de8 and dirty=true because the new code was under development; these are not claims that the parent contained v2. Final committed files are fingerprinted in source_files_sha256.json.

| Check | Observed result |
|---|---|
| Nonzero CSCEF FP32/AMP/true-half forward | max_abs=0, max_rel=0 |
| Identity proxy | exact for the tested finite inputs |
| Semantic input backward (FP32, AMP) | L1=L2 ratio 0.25; max_abs versus 0.25×old=0 |
| Lateral input backward | L1=L2 ratio 1.0; max_abs=0 |
| All five CSCEF parameter gradients | max_abs=0, max_rel=0 |
| v5.1→v5.2 strict load | identical 7 state tensors and 26,912 parameters |
| Nonzero old C25 vs new pair, 640×640 and 160×192 | Y5/Y4/upY4/lateral/CSCEF/P3/P4/P5/bbox/score all max_abs=max_rel=0 |
| Pair residual-only gradient at upY4 and Y4 | L1=L2 ratio 0.25; max_abs=0 |
| Pair main-Concat gradient | L1=L2 ratio 1.0; roundoff max_abs≤3.56e-15 |
| SCCA full objective and internal x/s | finite nonzero gradients; no fixed total-gradient ratio asserted |
| C2 public state mapping | all 533 tensors exact, no missing/unexpected/shape errors |
| Parameters standalone/pair/triad | 20,109,684 / 20,175,224 / 20,221,113 |
| Same-zero-initialized model vs C2 | full outputs and neck features exact at 640×640 and 160×192 |
| Original CSCEF residual propagation | changing its nonzero output changes P3 and both PAN P4/P5; upstream projections unchanged |
| Native RT-DETR loss/DN/AdamW | 2 synthetic updates per model, CPU FP32 and CUDA AMP; finite loss/gradients |
| Dynamic DN | batch2→batch1; total queries500→498, regular queries fixed300 |
| Optimizer coverage | every trainable tensor exactly once, original bias/norm/weight-decay groups; no scale parameter |
| Learned model save/load | exact state and output after updates; native reconstruction preserves learned innovation tensors |
| Complete-model/YAML API init reload | exact compatible states; only original nine nc80→nc1 classification states adapt |
| True half whole-model inference | finite nonzero-model outputs for all three variants |
| C2/C17/C19/C24 historical regression | separate historical-source processes, same nonzero state, strict load and exact outputs at160×192 |

The bounded full-model loss checks use batch1/2 at160 for memory. They do not change formal batch16/imgsz640. A debug AMP scaler initial value128 is local-only; it does not alter the formal Trainer configuration. Same-precision equivalence is exact; AMP versus FP32 may select/reorder different native encoder top-k queries and is not claimed to be elementwise equivalent.

Ops tests exercise nine failure-sensitive lifecycle cases, actual Bash syntax, and sync with explicit Git stubs: detached creation/reuse, historical pin, different SHA, dirty or attached worktree, wrong origin, outside-history pin and existing pin preservation. Start-direct preflight/dispatch ordering is mocked; no tmux training session is started. C25 replay and v1 variants are absent from the formal launcher allowlist.

The real evaluation pipeline was run only on two copied images in each original val/test split, with untrained controlled triad weights, 640/batch16/conf0.001/iou0.7/max_det300/FP32/seed42. It verifies AP75, all-query export, curves, confusion matrices and identical checkpoint gates. These zero-quality untrained metrics are tool tests, not research results.

Complete packing is tested with explicitly synthetic train-completion evidence plus those real tiny evaluation outputs. The manifest is read back, every archived file hash checked, and overwrite/broken-test/implicit-evaluation paths rejected. Best/last follow the existing complete-package policy. No training weights, datasets, images or binary archives are committed.

Primary evidence: validation_summary.json, gradient_edge.json, historical_regression.json, source_audit.json, ops_validation.json, evaluation_pipeline.json, pack_validation.json and per-variant initialization reports in evidence/. Input-package evidence is separate from synthetic validation evidence.

Reproduce with new output paths:

```bash
python tools/audit_mincompat_v2.py --report outputs/v2_audit.json
python tools/check_mincompat_v2.py --source weights/rtdetr_r18_lite_imagenet_backbone_init.pt --output outputs/v2_check
python tools/check_mincompat_v2_gradients.py --report outputs/v2_gradients.json
python tools/check_mincompat_v2_regression.py --report outputs/v2_regression.json
python tools/check_mincompat_v2_ops.py --bash bash --report outputs/v2_ops.json
```

Not verified: actual AutoDL Python3.10/PyTorch2.1.2+cu121/RTX4090 execution, formal batch16 memory capacity, 200 epochs, full val/test, trained-gradient conflict angles, repeated seeds, retained detection effectiveness or positive pair/triad interaction. Server start-direct repeats selected-model AMP/half, loss/DN and gradient checks in the existing environment and stops on failure without changing batch or installing packages.
