# DPR-1 experiment

Base: `a0459d6a652cb702699087c88fa39a3e4c4087ec`; branch: `exp-rtdetr-r18-lite-dpr-v1`.

Only `model.5.blocks.1.branch2b` changes. `BlocksDPR` preserves the original
BasicBlocks and replaces that wrapper with `DPRConvNormLayer`. The original
`conv`, `norm`, and `act` keep their state keys. The target is dense
128→128, 3×3, stride/padding/dilation 1, groups 1, no bias, original BN, Identity.
At 640 its input/output are B×128×80×80. No top-level node is inserted.

Four zero-initialized tensors (`dpr_cd` 128×9, `dpr_hd` 128×3, `dpr_vd`
128×3, `dpr_ad` 128×9) add **3,072 parameter elements**. Each forward constructs
the four documented difference kernels in FP32 (retaining FP64 for mathematical
checks), adds their sum only to the channel diagonal of the original dense W,
and executes one dense convolution followed by the original BN and Identity.
The original W remains trainable. CD/AD centers have structural zero gradients;
the parameterization is redundant and does not add 3,072 independent degrees of
freedom or a new deployed convolution function family. No accuracy gain is claimed.

| Variant | Parent | Candidate configuration |
|---|---|---|
| `cbr_lif_dpr_v1` | original CBR + LIF | `rtdetr-resnet18-lite-cbr-lif-dpr-v1.yaml` |
| `dpr_v1` | original C2 | `rtdetr-resnet18-lite-dpr-v1.yaml` |

Both retain all 27 top-level nodes and from indices. The main model keeps CBR
rho/normal_fraction 0.10, LIF node 20, hidden_dim 256, 300 queries, three Decoder
layers, eval_idx 2. The ablation keeps the plain RTDETRDecoder and Conv at node 20.
The original `cbr.py` and `lif_down.py` normalized SHA256 values are enforced.

## Initialization and recipe

`tools/init_dpr.py` accepts only the public ImageNet initialization with SHA256
`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`.
It checks epoch −1/no trained state, reuses the original controlled CBR/LIF
initialization, strictly copies every parent key, and adds exactly four zero
parameters with no new buffers. Initialization is saved in FP32, immediately
reloaded, and audited through native `RTDETRTrainer.get_model` and the actual
`RTDETR.train` reconstruction. The nine nc80→nc1 classification adaptations must
match the corresponding parent tensors. These reconstruction checks do not claim
any real loss or optimizer update. Existing initialization files are preserved.

`parent_args.yaml` is the actual successful parent's 109-field record, identical
to the supplied appendix. `source_audit.json` records its source/hash and the
fixed Git blob checks. Formal recipe differences are restricted to experiment
identity and verified equivalent data location. Epochs/patience 200/50,
B16/640, AdamW, lr0 .0005, seed42, native AMP, all augmentation/loss/EMA settings
remain the parent recipe. Native automatic OOM batch reduction is disabled so
an OOM cannot silently turn this experiment into B8. Public parameters are not frozen.

`parent_dataset_inventory.json` is historical identity evidence, not a new model
result. `tools/dpr_data.py` verifies the actual configured train/val/test paths
and label-byte hashes against it. Counts are 6048/45573, 1728/12840, 864/6663
images/GT. Reading test identity does not evaluate test.

## Deployment and checkpoint policy

Standard deployment uses an independent **eval copy**: fold DPR into the original
conv while retaining its original BN, remove the four difference parameters,
then apply the unchanged parent `BaseModel.fuse()` policy to all other layers.
Native fuse/AutoBackend invoke this conversion. Conversion is idempotent and
full-model serialization retains deployment state. Deploy objects are inference
only and must not be passed to resume. Live model/optimizer/EMA retain the four
parameters. FP32 folding precedes optional half conversion.

THOP receives a DPR-specific counting hook because functional conv2d would
otherwise be missed. Kernel construction cost is separate from convolution
GFLOPs. The acceptance report records actual counts in all three representations;
expected main counts are 20,152,837 / 19,948,037 / 19,944,965 and ablation counts
20,085,844 / 19,880,788 / 19,877,716 (training / native fusion without folding DPR /
DPR fold plus native fusion). No extra backbone BN fusion is included.

The explicit `optimizer_fp32_v1` checkpoint policy preserves independent FP32
AdamW moments when saving. EMA still uses the native half checkpoint, and native
live optimizer, AMP scaler, clipping, EMA updates, epoch and best/last selection
remain unchanged. The override is scoped to DPRTrainer and guards the parent
serializer's source/AST. This is a storage precision change; it is not claimed
to reproduce historical FP16-moment resume trajectories. Casting old FP16 moments
to float is rejected as a substitute for original precision.

Acceptance A compares the actual saved quantized EMA and original FP32 optimizer
bytes, name/group bindings, scaler/epoch/EMA updates, no shared storage, and
native identical-gradient replay. B separately records independent live
backward repeatability at atol=2e-5, rtol=2e-4 for both parent and candidate.
Raw failures remain visible. Missing or unexplained evidence blocks start.

## Operations

`tools/dpr_server.sh` separates environment, init-preflight, plan, start, resume,
val, test, and pack. Every command loads the existing rtdetr Conda environment
and verifies the worktree import. There is no GPU-idle guard and no package
upgrade. Preflight does not invoke start. Formal start revalidates full HEAD,
code/config bytes, source/init hashes, variant, dataset identity, complete recipe,
server environment and all evidence before atomically recording admission.

Capacity is a native real-train online-augmentation B16/640/native-AMP run,
bounded by 16 batches and a target of at least two actual optimizer updates.
Scaler calls are not counted as updates. Smaller real-loss checks are explicitly
separate from capacity. Reports use new timestamp directories. Old permits are
revoked before new checks; local reports cannot authorize a different server.

`tools/eval_dpr.py` implements independent val then test of the same frozen
`best.pt` under `corrected_sorted_conf_mask_v1`: 640, B16, workers0, FP32,
conf .001, iou .7, max_det300, no augment/rect, seed42. It retains native one-to-one
matching, corrects the post-sort confidence mask, and adds no NMS. JSON contains
full precision P/R at each model's own max-F1 point, AP50/AP75/mAP, ten AP values,
GT/image counts and speed; predictions remain on the server. Failed/existing
evaluation outputs are preserved. The optional matched-precision diagnostic is
not part of required acceptance and is not reported as measured.

Historical parent independent val: P .8655727028797452, R .835202492211838,
AP50 .8956724997170779, AP75 .5453045665315482, mAP .5245427221919051.
Historical parent independent test: P .8602393656297294, R .8353594476962329,
AP50 .8919968337794693, AP75 .5400236205750717, mAP .5220090191444802.
These are historical records, not DPR results. The correct parent test mAP is
52.2009%, not the obsolete 56.51185% claim.
Historical parent best SHA256:
`24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa`;
original test metrics JSON member SHA256:
`51f9e6c0d225a545c6913783e872fc906580287bea9d01994ce6913c9ad8edaa`.
These supplied historical identifiers are not claimed as new DPR evaluations.

`tools/pack_dpr_light.py` creates a read-back verified archive strictly below
20 MiB. It includes source, small reports, args/results, metrics/curves and log
tails, with a member size/SHA manifest and explicit missing/omitted lists. It
excludes weights, data, ZIP references and full predictions, and never triggers
training or evaluation. Archive integrity does not imply complete experiment
acceptance. The CLI prints an absolute path and size.

## Provenance and current results

The requested reference ZIP was not found in the bounded attachment/project/
Downloads searches and was **not read**. No archive code or dependencies were
imported. The mathematical mappings were implemented from the supplied contract.
The prompt is not a third-party code license; repository source retains AGPL-3.0.
PiDiNet/DEA-Net are background references only; no novelty or crack-detection
benefit is inferred from their results. No DCC model, results, or initialization
are included. Its separate A/B checkpoint methodology was only a design reference.

See `VALIDATION.md` and the accompanying JSON for this DPR implementation's
actual checks, including original failures and environment-dependent pending work.
Fixed-SHA server handoff is generated **after commit** under `outputs/dpr/`, so
editing a handoff's SHA does not cause a chain of documentation commits.

**Formal training NOT_STARTED; final test NOT_RUN; no server login in this task.**
