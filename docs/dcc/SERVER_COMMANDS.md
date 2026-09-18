# Server entry points (explicit dispatch)

The final chat handoff supplies the fixed 40-character delivery SHA and the bounded fetch bootstrap.
After that bootstrap, `tools/sync_dcc.sh SHA` verifies the origin, commit, base ancestor, module hashes and
independent worktree. It uses existing origin and never rewrites Git configuration. No server login was performed.

Every block is independent: the wrapper sources `docs/dcc/environment.sh`, activates the existing rtdetr
environment and verifies the actual import path. No package upgrade is needed. The paths below refer to
the main combination. Preparing the single-module configuration does not start an ablation experiment.

Initialization and preflight (does not start formal training):

```bash
bash -Eeuo pipefail <<'BASH'
bash /root/autodl-tmp/projects/Crack_RTDETR-dcc-v1/tools/autodl_dcc.sh init-preflight
BASH
```

This verifies/reuses the same controlled initial checkpoint, runs module checks, full engineering checks on
real train samples, and the native B16/640 online-augmentation capacity check (≤16 batches, ≥2 effective
updates). Contract `dcc_acceptance_v2` binds initialization and mathematical reports
to full engineering checks and capacity, re-evaluates complete restore/trajectory
evidence and verifies all current code/runtime/data/init identities. Only after
all required checks pass is `outputs/dcc/cbr_lif_dcc_v1/passed_gates.sh` written. Logs have real exit-code
sidecars. A resource failure stops the preflight and does not change B16, disable AMP, start training or stop
another experiment. Reports require the exact runtime, source/config hashes, init hash and dataset identity
at later start/resume. Source public hash is checked by initialization and engineering checks.

Read-only plan:

```bash
bash -Eeuo pipefail <<'BASH'
bash /root/autodl-tmp/projects/Crack_RTDETR-dcc-v1/tools/autodl_dcc.sh plan
BASH
```

Explicit formal start, only after the NEW complete `init-preflight` genuinely passes
and the user chooses to run it (not executed in this delivery). `resume-verify` is
a partial diagnostic and can never generate `passed_gates.sh`. CUDA raw differences
remain false/recorded when independently classified as `PRECISION_NOTE`:

```bash
bash -Eeuo pipefail <<'BASH'
bash /root/autodl-tmp/projects/Crack_RTDETR-dcc-v1/tools/autodl_dcc.sh start
BASH
```

Resume the same unfinished run (last.pt must retain native optimizer/scaler/EMA/epoch):

```bash
bash -Eeuo pipefail <<'BASH'
bash /root/autodl-tmp/projects/Crack_RTDETR-dcc-v1/tools/autodl_dcc.sh resume
BASH
```

Independent val of selected best.pt:

```bash
bash -Eeuo pipefail <<'BASH'
bash /root/autodl-tmp/projects/Crack_RTDETR-dcc-v1/tools/autodl_dcc.sh val
BASH
```

Independent final test, later, only after same-weight independent val (NOT_RUN in this delivery):

```bash
bash -Eeuo pipefail <<'BASH'
bash /root/autodl-tmp/projects/Crack_RTDETR-dcc-v1/tools/autodl_dcc.sh test
BASH
```

Light package after the formal run exists (no training or evaluation side effects):

```bash
bash -Eeuo pipefail <<'BASH'
bash /root/autodl-tmp/projects/Crack_RTDETR-dcc-v1/tools/autodl_dcc.sh pack
BASH
```

Optional val-only fixed-P diagnostic uses a new output directory; keep its raw score/TP file on server:

```bash
bash -Eeuo pipefail <<'BASH'
source /root/autodl-tmp/projects/Crack_RTDETR-dcc-v1/docs/dcc/environment.sh
python -u tools/eval_dcc.py val --variant "$DCC_VARIANT" --fixed-precision \
  --weights "$DCC_MAIN/runs/c_series/${DCC_VARIANT}_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt" \
  --data "$DCC_DATA" --output "$DCC_META/evaluation_val_fixedP"
BASH
```

Original P/R remain each model's own maximum-F1 point. Fixed-P is a separate attainable-threshold diagnostic,
not a replacement metric or a test-set threshold search. Recompute the parent with `--parent-baseline` and
the pinned historical parent best SHA; aggregate historical metrics cannot supply raw prediction evidence.
