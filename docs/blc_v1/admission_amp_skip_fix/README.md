# BLC admission: native AMP backoff repair

Contract: blc_training_admission_v3_amp_backoff.

This repairs the metadata evaluator deployed at
3906d10280a2f1236ea54a6cef9fb8853ea1f85c. The original preflight producer remains
0d2335122a09203dfa2ebf4b469a68eede27cd61. Neither the PENDING original preflight
nor the FAILED v2 admission is edited, replaced, deleted or relabeled.
The two controlled_init files keep their original bytes.

## Source-derived scaler contract

The protected producer's tools/blc_preflight.py records full-model scaled-gradient
finiteness before calling the native optimizer step. Its four BLC norm entries
measure float gradients divided by the current scale. These are distinct
observations: full-model overflow can coexist with four finite BLC norms.
The evaluator keeps the BLC norm/delta finiteness checks and the global JSON
NaN/Inf scan intact.

At the producer revision, engine/trainer.py constructs the PyTorch CUDA
GradScaler with only enabled=self.amp, then uses unscale, clip, step and update
without overriding its scale. The server version is torch 2.1.2+cu121.
The checked defaults are initial scale 65536, backoff factor 0.5, growth factor 2
and growth interval 2000, with growth tracker initially zero. See the exact
[PyTorch v2.1.2 scaler implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/torch/cuda/amp/grad_scaler.py)
and [CUDA scale update implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/aten/src/ATen/native/cuda/AmpKernels.cu).
These constants are tied to that source/runtime contract, not guessed from the
server's error list or taken from the local machine's different PyTorch version.

## Attempt classification

Every attempt requires a corresponding completed batch with finite loss and
positive integer GT count, increasing valid batch indices, positive finite
scale values, a nonnegative integer optimizer step, explicit Boolean flags and
all four finite nonnegative BLC norm/delta entries.

- AMP_BACKOFF requires full-model gradients nonfinite, scaler_skipped=true,
  effective_update=false, the verified native scale reduction, unchanged
  optimizer step, and zero delta on all four BLC parameters.
- EFFECTIVE_UPDATE requires finite gradients, scaler_skipped=false, one native
  optimizer step increment, consistent scale, and positive Wo delta with the
  effective flag set.
- APPLIED_NO_WO_CHANGE records a native step that advanced but had zero Wo
  delta, with effective_update=false. It is neither a backoff nor an effective
  update.
- Missing or contradictory observations yield INVALID with specific reasons.
  After an invalid attempt, the next state is unproven rather than invented.

Capacity starts from an empty fresh AdamW state, so its producer's recorded
maximum step begins at zero. Scale and growth state are derived from the
verified defaults and the full sequence of recorded attempts. Native checkpoint
serialization leaves optimizer step unchanged; the producer's resume audit
asserts exact optimizer moments/steps, scaler, EMA and epoch restoration.
Only when those restore checks and the capacity sequence pass does resume
inherit capacity's final optimizer/scaler state. Its first attempt never assumes
a zero optimizer origin or a reset scale.

The report labels derived states/classifications separately from the immutable
original observations. CLI output prints every attempt's classification, batch,
scale before/after and optimizer state step, followed by backoff/non-effective/
effective counts and both final admission/diagnostic statuses.

Backoffs still consume actual batch budget. Three consecutive backoffs alone do
not fail. The existing two effective start updates, one effective resumed update,
finite nonzero upstream-gradient rule and combined 16-training/one-validation
batch limits remain. Model/data/init/recipe identity, native half EMA validation,
same-file AutoBackend comparisons and other original gates remain intact.
Fusion diagnostic PENDING is preserved; admission does not prove numerical
equivalence, convergence or an accuracy gain.

## Restricted migration

The migration verifies both exact parent edges:

    0d2335122a09203dfa2ebf4b469a68eede27cd61
      -> 3906d10280a2f1236ea54a6cef9fb8853ea1f85c
      -> this repair commit

Each segment records its actual changed paths and normalized before/after
SHA256 values. The first retains the original seven-file allowlist and exact
entry-point AST proof. The repair permits only blc_admission.py,
blc_admission_identity.py, check_blc_admission.py and this directory's README.md
and validation.json. All other runtime content, including blc_server.py/.sh,
must equal the deployed revision; the cumulative original entry-point AST
proof is also rerun. Unknown intervening commits, merges and production-code
changes are rejected. This is not a generic ancestor exemption.

Original reports are still verified by their full LF-normalized code.files
manifest against their actual supported producer tree. The fixed production
SOURCE is unchanged. Both older init audits retain their own source verification
and initialization-semantic proof. New reports bind the complete current
contexts and immutable raw JSON hashes. admission/start/resume recompute the
same repaired decision and reject stale v2 contracts or edited summaries.

## Validation and server use

18 quick metadata tests, with explicitly synthetic FIXTURE input, cover three
initial backoffs followed by two effective updates and one restored update,
backoff-only insufficiency, parameter/step/scale contradictions, missing fields,
finite versus nonfinite loss, nonfinite-gradient updates, inconsistent resume
origins, legitimate resumed backoff and applied-but-ineffective steps.
Existing source/context/hash/precision/fusion tests remain.

The fixed original server report is not available locally:

    /root/autodl-tmp/projects/Crack_RTDETR-blc-v1/outputs/blc_v1/cbr_lif_blc_v1/preflight-20260919T193011968491Z/report.json

After the separately delivered fixed-SHA safe update, run one reassess action
against that file with BLC_MAIN, BLC_DATA and BLC_VARIANT set. It reads old JSON
and existing context only, creates one new exclusive admission report, and exits
nonzero with specific reasons when evidence is insufficient. Do not rerun init,
init-preflight or preflight to repair this evaluator issue. No checkpoint is
deserialized and no model forward, backward, training or validation is run.

Only after training_admission=PASSED with exit code zero may the user run the
separate admission plus normal start commands. Start rechecks all bindings and
uses original controlled_init and the unchanged 200e recipe. Preserve an existing
formal run; unfinished training must use the existing checked native resume
entry. Updating the code never starts training.

Local fixture success is not server reassessment. Server reassessment remains
PENDING until those commands run. Formal training is NOT_STARTED by this task;
final test is NOT_RUN. Added training/validation batches and checkpoints are all
zero. New source/tests/JSON/logs target less than 1 MiB; no large tensor dumps.
RDM and all other experiment worktrees are outside this task.
