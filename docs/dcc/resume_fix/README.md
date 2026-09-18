# DCC resume engineering repair (2026-09-19)

This repair descends from delivered `5049bb0010f91aa89cd0294024ea73f6b0396164`
on `exp-rtdetr-r18-lite-dcc-v1`, whose fixed parent is
`a0459d6a652cb702699087c88fa39a3e4c4087ec`. Formal training is **NOT_STARTED**;
final test is **NOT_RUN**. No experiment accuracy claim is made.

## Evidence actually read before editing

Both supplied ZIP files were opened, their JSON reports inspected, and all included
DCC Python/shell sources compared against the delivered tree. Archive/file SHA256
and original entry names are in [provenance.json](evidence_20260918/provenance.json).
The copied JSON evidence is byte-for-byte original, including failed comparisons.
It describes the historical server run, **not acceptance of this changed tree**.

The only source difference in the first ZIP is the four-line
`DCC_PREFLIGHT_DETACH_BEFORE_SAVE_V1` block in `preflight_dcc.py`. It removes temporary
forward hooks and clears their handles before deepcopy/pickle. Previously the local
`loss_finite` hook remained attached and could not be pickled. Cleanup in `finally`
was too late for serialization.

The server capacity report records PyTorch 2.1.2+cu121 / RTX4090, B16/640 native AMP,
five observed batches and two effective updates. CPU checkpoint continuation is
exact. CUDA checkpoint controls have equal initial state, captured forward tensors
and loss, but different backward gradients.

The second ZIP changes optimizer checkpoint precision only; both policies retain
the same native half-precision EMA model:

| Mode | Half optimizer max state error | FP32 optimizer max state error | Half second moments becoming zero | FP32 remaining failed state tensors |
|---|---:|---:|---:|---:|
| CUDA FP32 | 90.54557204246521 | 0.0005910570034757257 | 19,379,804 | 3 |
| CUDA native AMP | 66.92137360572815 | 0.00057537283282727 | 19,318,652 | 7 |

FP32 optimizer saving produces zero nonzero-to-zero second-moment conversions;
all four DCC tensors pass the original tolerance in these two controls. FP32 still
fails `model.9.norm2.bias`, `model.12.bn.bias`, and `model.17.bn.bias`. AMP also fails
three `model.7` convolution weights and `model.9.ma.in_proj_bias`; all names and
per-tensor errors remain in the original JSON.

This is strong evidence that FP16 optimizer serialization amplifies backward
differences. It does not explain every remaining CUDA difference, and does not
establish elementwise-equivalent continuation or full DCC acceptance.

## Narrow saving policy

`tools/dcc_checkpoint.py::DCCCheckpointTrainer.save_model` is shared by formal DCC
training and engineering checkpoint checks. It preserves the reviewed native
fields and last/best/periodic file selection. The optimizer state is deep-copied
without `convert_optimizer_state_dict_to_fp16`; live optimizer calculations,
GradScaler, clipping, AMP and EMA updates remain native. Checkpoint `model` remains
`None`, and checkpoint EMA remains half. Shared Ultralytics source is unchanged.
The override checks the identity of the native serializer to detect upstream drift.

Additional `dcc_checkpoint` metadata records `optimizer_fp32_v1`, parameter names
in optimizer group order, and serializer identity. Resume validates the original
checkpoint precision and this mapping. **This is a checkpoint storage precision
change. Its restoration trajectory is not claimed to match historical FP16
checkpoints. Casting an already rounded checkpoint back to float cannot restore
lost information.**

Diagnostic checkpoints use temporary directories; only JSON, hashes and needed
logs are retained. Existing output directories and historical weights/reports are
never deleted. The original preflight hook block is retained verbatim.

## Separate properties and unchanged admission rule

The old check combined several properties in one lifecycle result and, on CUDA
failure, replayed unscaled gradients with AMP disabled while comparing model state
only. That replay did not establish scaler, optimizer or EMA update correctness.
The new audit reports these properties separately:

1. **Saving/restoring state:** exact parameter-name/group/order/hyperparameter
   binding, all model and optimizer state (including step), scaler, epoch,
   EMA weights/updates, and disjoint mutable tensor storage between controls.
2. **Identical-gradient update:** identical gradients pass through the native
   enabled/disabled scaler as appropriate, unscale, clipping, optimizer step,
   scaler update and EMA update. The updated model, optimizer, scaler and EMA
   must all be exact. Both effective and overflow-skip behavior are audited.
3. **Independent native next step:** same real batch and restored RNG, original
   precision settings, independently computed forward/backward/update. Original
   `atol=2e-5`, `rtol=2e-4` and every failed tensor/error are retained.
4. **Live FP32 control:** original CBR+LIF and CBR+LIF+DCC each perform real bounded
   updates, then independently duplicate their own live FP32 state without any
   half checkpoint conversion. Exactly one paired next-step comparison per model
   and precision mode is run. No alternate seed, batch or relaxed tolerance is
   tried to obtain a pass.

Property 1 passing certifies restoration under the declared new storage policy.
Property 2 isolates update mechanics. Neither proves property 3. Parent variation
in property 4 is contextual evidence and **cannot authorize training**.
`strict_gate` still requires `raw_next_update_allclose=true`, as well as the new
complete restoration and replay evidence and all existing capacity/fusion gates.
An unresolved CUDA failure remains **BLOCKED**, even when restoration passes.
The targeted `resume-verify` command never writes a formal start authorization.

## Unchanged experiment and server verification

DCC mathematics, graph node 17, the four new tensors / 18,432 parameters, original
CBR/LIF modules and controlled public initialization are unchanged. The recipe is
still copied from all 109 actual parent arguments, allowing only model/data/output
identity fields to differ: 200 epochs, B16/640, seed42, native AMP, AdamW lr0=.0005,
and the same loss, warmup, accumulation and online augmentation.

The new server verification is **PENDING** until the fixed revision is actually
run on PyTorch 2.1.2+cu121 / RTX4090. Older capacity or engineering reports cannot
be used with the new source identity. Local evidence and its runtime are reported
separately. `resume-verify` reruns the changed checkpoint checks, the bounded live
control and the necessary native capacity check; it does not rerun unrelated
mathematical or inference/fusion checks and does not promote their old reports.

The new local run completed once in 181.308 seconds on the already installed
PyTorch 2.7.1+cu118 / RTX2060. All three modes passed complete checkpoint restoration
against independently mapped saved bytes, disjoint-storage checks, and exact
same-gradient model/optimizer/scaler/EMA updates. The AMP overflow fixture also
passed; it records the inherited EMA update on a scaler-skipped optimizer step.

| Local mode | Parent live FP32 next model state | DCC live FP32 next model state | DCC checkpoint next model state |
|---|---|---|---|
| CPU FP32 | allclose | allclose | allclose |
| CUDA FP32 | fails 1 tensor | fails 1 tensor | allclose in this one local run |
| CUDA native AMP | fails 6 tensors | fails 6 tensors | fails 7 tensors |

All nine paired forward captures and losses were exact. CUDA FP32 live controls
both fail `model.17.bn.bias` (parent max error 2.1485182514879853e-5;
DCC 2.2129571220830258e-5). CUDA AMP DCC checkpoint max model error is
0.0005528894253075123. All four DCC parameter tensors pass the original tolerance
in this local run. These observations do not overwrite the different historical
4090 results or turn a server failure into a pass.

The log contains a `grid_sampler_2d_backward_cuda` nondeterminism warning. Together
with the non-quantized parent control this supports an inherited CUDA backward
contribution, but no individual kernel's contribution to each residual has been
isolated. Residual trajectory mismatch remains **UNRESOLVED / BLOCKED**.

[local_validation_summary.json](local_validation_summary.json) retains every failed
model/gradient tensor with the original tolerance and errors. The byte-exact full
33,245,039-byte JSON is retained locally and losslessly compressed in
[local_reports/resume_checks.json.gz](local_reports/resume_checks.json.gz) (776,246
bytes); its SHA256, source identity, complete state comparisons and log are provided alongside
the summary. No `.pt` remains under this new diagnostic directory. Separate reports
record the serializer differential test, strict-gate negative tests, hook pickle
regression and seven Git synchronization fixtures. Protected source hashes and
all 109 recipe fields were checked again without rerunning unrelated model tests.

Server synchronization accepts a clean existing DCC worktree or precisely the
four-line verified hook fix at old HEAD5049. That recognized patch is backed up
before being absorbed into the new pinned revision. Other tracked changes are
refused and preserved; untracked/ignored output collisions prevent checkout.
Neither `reset` nor `clean` is used, and GRA is not accessed.
