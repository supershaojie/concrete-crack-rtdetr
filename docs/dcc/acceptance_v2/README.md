# DCC acceptance contract v2

Contract: `dcc_acceptance_v2`. Previous source: `3edabd6ae46a3f05701b7fe04abcf2c25b452c42`.
This change implements the explicitly requested acceptance semantics, without changing
DCC mathematics, node 17 wiring, 18,432 added parameters, CBR/LIF, controlled initialization,
loss, AMP, accumulation, or the archived 109-field successful-parent recipe.

## Evidence actually inspected

The user-provided `DCC_RESUME_REVIEW_20260918T170728_700241Z.zip` (called
`95b8c495-6f9e-44e4-8ef0-b1305cc9419a.zip` in the request) has SHA256
`ae1def3a0d2236c576fe0a206ec621a5a66676cc99d28f8c367f8eefcd67be41`.
All 37 manifested files were checked by size and SHA256; packaged source matched
3edabd6 and the recorded server checkout was clean. The >8 MiB aggregate report
was omitted by the packer. The nine complete individual reports, verification
summary, capacity report and source provide the relevant evidence.

`evidence/` preserves original JSON bytes (nine large cases gzip-compressed without
changing their content), manifest, summary, capacity, log and provenance. No old
failure report has been edited or removed. `offline_reassessment_final.json` is a NEW
classification of historical evidence, not a fresh server certification.

| Original next-step model failures | CPU FP32 | CUDA FP32 | CUDA AMP |
|---|---:|---:|---:|
| Parent CBR+LIF live FP32 twins | 0 | 3 | 6 |
| DCC live FP32 twins | 0 | 3 | 6 |
| DCC checkpoint restored twins | 0 | 5 | 7 |

All nine forward captures and losses were equal. The three checkpoint modes
strictly passed saved-live optimizer equality, saved-byte reference restoration,
name/group/hyperparameter/state/step/epoch/scaler/EMA checks, storage isolation and
identical-gradient complete-state replay. Native AMP scaler replay and overflow
skip checks passed. The four added DCC parameter updates passed original tolerance.
AMP **W_o gradients did not all pass**: live and checkpoint each have 1/8192 elements
outside tolerance; max_abs is respectively 2.1391315385699272e-05 and
2.093985676765442e-05, relative_L2 0.0013345103057246389 and 0.001297938012443927.
The original rows and failure lists remain available.

Earlier FP16 optimizer checkpoint storage amplified the observed error: the
FP32-optimizer control reduced maximum state errors from 90.545572/66.921374 to
0.000591057/0.000575373 (FP32/AMP). That repair remains intact: only saved optimizer
precision changed; active optimizer, AMP and native half EMA semantics did not.
Upcasting old FP16 checkpoints cannot recover information already lost, and their
historical continuation trajectory is not claimed equivalent.

## Two properties, two reported outcomes

A — checkpoint saving/restoration correctness — remains a hard gate. It checks
live FP32 optimizer bytes, independent saved-byte oracles, complete names/groups/
hyperparameters/model/optimizer/step/scaler/epoch/EMA, disjoint storage, exact CPU
native continuation, complete exact identical-gradient CUDA replay, and native
AMP effective/overflow paths. It reads detailed evidence, not just `status`.

B — repeatability after independent native CUDA backward — runs and records the
unchanged `atol=2e-5`, `rtol=2e-4`, raw flags, failed tensors, gradients, max_abs,
relative_L2 and exceeded_fraction. Matched parent and DCC live controls start
from independently cloned real-updated FP32 state, on the same real batch, RNG
and precision. They never pass through half checkpoint conversion.

The old gate conflated A with B by requiring CUDA independent-next-update
allclose. V2 permits `PRECISION_NOTE` only when A strictly passes and the full
same-mode control evidence supports finite backward-only variation: initial states,
forward captures and losses agree; permitted variations are parameter/EMA-parameter
and matching Adam moment fields; buffers, steps, scaler and metadata stay exact;
DCC parameter updates meet original tolerance. It requires supporting gradient
variation for parameter differences and validates both live controls. Missing,
nonfinite, inconsistent, aliased, unsupported or unexplained evidence blocks.
This is neither a blanket waiver because the parent also varies nor a model-only
fixed-gradient test. Raw false values are never rewritten to true.

A single B2/160 bounded comparison does not establish long-run or statistical
training equivalence, and does not prove a particular CUDA kernel root cause.
Independent CUDA trajectories remain non-identical; no accuracy gain is claimed.

## Complete entry and regression checks

`check_dcc_resume.py` is explicitly `partial_resume_diagnostic`, reports A/B
separately and never authorizes start. Its successful diagnostic exit code may be
0 with `PRECISION_NOTE`; failed or missing evidence returns 3.

`check_dcc.py` is `full_preflight_engineering`. It retains initialization equivalence,
mathematics prerequisites, wiring, gradient startup, nonzero state/full-model
reload, native reconstruction, EMA, class adaptation, fusion and CUDA half checks;
it adds the matched live controls needed to interpret checkpoint continuation.
It binds initialization and mathematical report paths and SHA256 values.

`train_dcc.strict_gate` independently re-evaluates A/B and required leaf evidence,
verifies report kinds/version, current source/runtime/init/dataset identity,
all archived recipe fields except explicit run identities, mathematical CPU+CUDA
coverage and native B16/640 AMP capacity. Stale or partial reports cannot satisfy it.
`autodl_dcc.sh init-preflight` archives/revokes an existing permit before checks,
then atomically writes a new `passed_gates.sh` only after all gates pass. Start
rechecks evidence, never follows automatically from verification.

Offline tests: `check_dcc_acceptance.py` uses the original nine reports and mutates
mapping, optimizer, scaler/EMA, storage, forward, gradients and required evidence.
`check_dcc_gate.py` exercises the actual formal admission function with explicitly
synthetic composite fixtures, including stale/partial reports, missing math,
wrong wiring/data/recipe, and corrupted restore state. `check_dcc_shell_flow.py`
uses inert stubs to test exit codes and permit lifecycle; no detector is executed.
Final regression evidence: `offline_reassessment_final.json` (nine cases, 23 rejection
fixtures), `full_gate_regression_final.json` (30 actual-gate fixtures, including a
positive precision-note case), and `shell_flow_final.json` (11 inert shell flows).
`source_audit.json` records syntax/import-help, unchanged experiment sources and
109-field recipe comparisons. These are regression evidence only; the full-gate
positive fixture deliberately combines historical records with a synthetic current
schema/runtime/identity, and can never serve as actual server acceptance.

## Delivery status

New server full initialization/math/wiring/startup/restore/fusion/CUDA-half/capacity/
identity acceptance: **PENDING**. Historical B16/640 5-batch/2-update pass belongs
to 3edabd6, not this changed source. Run one complete `init-preflight` after syncing
the fixed delivery SHA. Do not repeat the partial `resume-verify` as a substitute.
Formal training: **NOT_STARTED**. Final test: **NOT_RUN**.
