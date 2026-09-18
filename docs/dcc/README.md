# DCC v1 implementation and evidence

Branch: `exp-rtdetr-r18-lite-dcc-v1`. Fixed parent: `a0459d6a652cb702699087c88fa39a3e4c4087ec`.
The implementation starts from this commit, never from another candidate's HEAD.
Formal training is **NOT_STARTED**; final test is **NOT_RUN**. Engineering evidence is not evidence of accuracy gains.

The current checkpoint repair and its limits are documented in
[resume_fix/README.md](resume_fix/README.md). Original reports below are historical;
they do not certify the changed source. Server verification of the repair is PENDING.

## Exact mathematical and graph contract

Both configurations replace only graph node 17 with `DCCConv`, retaining 27 nodes and all original public state keys.
It inherits `Conv(c1,c2,k,s,p,g,d,act)` and executes `dcc(act(bn(conv(x))))`; the fused method executes
`dcc(act(conv(x)))` exactly once. Original `model.17.conv.*` / `model.17.bn.*` keys remain unchanged.
Added trainable state is only `model.17.dcc.{W_d,W_q,W_k,W_o}.weight`.

For X with 256 channels, r=32, h=4, d=8, N=H×W:

```text
Z = W_d(X); Zc = Z - mean_spatial(Z)
Q = reshape(W_q(Zc), B,4,8,N)
K = reshape(W_k(Zc), B,4,8,N)
V = reshape(Zc, B,4,8,N)
A = softmax(normalize(Q, spatial, eps=1e-6) @ normalize(K, spatial, eps=1e-6)^T / 1.0)
delta = W_o(reshape((A - I_8) @ V, B,32,H,W))
Y = X + delta
```

Four bias-free 1×1 projections; no V projection, learned temperature, extra scaling, normalization layer,
activation, spatial gate, DWConv or position encoding. A has shape B×4×8×8, never N×N.
The branch uses an autocast-disabled FP32 region and differentiable `weight.float()` functional convolutions;
only the final delta is cast to the input dtype. Identity is an on-device nonpersistent buffer.
W_d/W_q/W_k use PyTorch Conv2d default initialization; only W_o starts zero. CPU RNG isolation prevents
the new branch from consuming the parent graph's subsequent initialization sequence. Loading, EMA, forward
and fusion do not reset learned weights.

The main variant retains original CBR (rho=normal_fraction=.10), LIFDown, AIFI and Decoder. The single-module
variant retains original C2 downsampling and Decoder and contains no CBR/LIF. Decoder has hidden_dim256,
300 queries, 3 layers, eval_idx2. Concat order remains [16,17]; final node19 feeds node20 and Decoder26;
Decoder inputs remain [19,22,25]. Original CBR/LIF normalized SHA256 values are checked at runtime.

| Model (nc=1) | Unfused parameters | Fused parameters |
|---|---:|---:|
| Original CBR+LIF | 20,149,765 | 19,944,965 |
| CBR+LIF+DCC | 20,168,197 | 19,963,397 |
| Original C2 | 20,082,772 | 19,877,716 |
| C2+DCC | 20,101,204 | 19,896,148 |

These counts were measured on actual models. Both deltas are exactly **18,432**.
Automatic native THOP summaries are **PARTIAL**, not a complete audited DCC FLOP measurement: functional
convolutions and matrix operations are not guaranteed to be counted by the inherited profiler.
Parameter count does not establish memory capacity or speed.

## Initialization and training protocol

Public source SHA256: `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`.
The source is nc80, epoch=-1 with no optimizer/scaler/EMA trained state. The unchanged parent's initialization
tool constructs CBR/LIF; all public tensors and buffers are copied exactly. Actual RTDETRTrainer.get_model
constructs nc1; only the specified 9 class tensors adapt, with synchronized parent/target RNG and exact
post-adaptation equality. Both variants have identical initial DCC state. Checkpoints are saved in FP32,
immediately reloaded, and audited through native setup_model and the actual Model.train reconstruction dispatch
(stopped before any training). The audit records every key, shape and value hash.

`parent_actual_args.yaml` is the successful parent's actual recorded training configuration, read from the
user's result directory. All 109 fields match the attachment appendix. Recipe generation copies this entire
file and allows only model/data/output identity changes; data identity must match the parent's split and label
fingerprints in `parent_dataset_identity.json`. Key settings: 200 epochs, patience50, B16/640, nbs64,
seed42/workers8, AdamW lr0=.0005, weight_decay=.0001, warmup5, cosine LR, native AMP, original online augmentation.
No common weights are frozen; no auxiliary loss or candidate-selection changes are made.

`train_dcc.py plan` is read-only. `start` needs a fresh controlled init and matching current-environment engineering
and B16/640 preflight reports. `resume` needs the original launch identity and an unfinished native checkpoint,
including optimizer/scaler/EMA/epoch; it never treats load_weights as resume. Existing outputs are protected.
An experiment-specific directory reservation prevents accidental duplicate starts; it does not block another
experiment or inspect GPU occupancy as an idle requirement. Native automatic OOM batch reduction is disabled
for this experiment so B16/640 is not silently changed.

## What is tested and what remains environment-dependent

See the original JSON reports and `VALIDATION.md`. `check_dcc_math.py` tests actual relation-matrix shapes,
zero/constant/low-amplitude/rectangular B1/B2 inputs, nonzero delta, spatial mean, input-dependent A,
permutation equivariance, startup gradients, casts, learned-state reload, EMA and DCCConv fusion.

`check_dcc.py` uses two actual train images at 160 pixels, native detection loss, bounded CPU/CUDA FP32 and
native AMP updates, public parameter coverage, node hooks at640, initial parent equivalence and learned-state
lifecycle checks. Strict FP32 tolerance is declared in advance: atol2e-5 / rtol2e-4, with maximum absolute error,
relative L2 and exceeded fraction. Continuous P3/P4/P5, projections, encoder and candidate scores are checked
before final output. If candidate identities change, full native IDs/scores and fixed-candidate diagnostic
replay are recorded; diagnostic IDs are never installed in production. TF32 changes are scoped and restored.

The resume test invokes the DCC-specific save_model and native setup_model/resume_training. The current
policy retains native EMA-half saving while preserving FP32 optimizer states, with an explicit parameter-name
mapping. Complete restoration and AMP/scaler gradient replay are audited separately from independent CUDA
backward trajectories. Both checkpoint controls use the same saved state and freshly reconstructed ephemeral
anchor caches. Historical reports used native half optimizer moments and remain unchanged. Diagnostic epoch
metadata does not mean a formal epoch was completed; disposable updated models never become formal start weights.

`preflight_dcc.py` executes the real native Trainer using original train images and online augmentation,
B16/640, original warmup/accumulation and default GradScaler. At most16 batches, target at least2 effective
DCC updates; native scaler skips and warmup's initial zero-LR step are distinguished. It stops before epoch
validation/final_eval. It records GT/DN, gradients, loss, scales, updates, memory and elapsed time. This is not
a training launcher. Missing device/data/resources is PENDING; OOM or nonfinite forward is FAILED.

This host's PyTorch version differs from server2.1.2. The server must run the same engineering checks and
capacity preflight itself; local checks cannot satisfy the start gate on a different environment.

## Reference and historical SCCA comparison

The reference ZIP could not be found in bounded searches of the repository, attachment directory or provided
result resources. Its contents were **not read**. The complete attachment math contract was implemented
independently; ZIP correspondence remains PENDING. No package scripts were run or dependencies imported.

[XCiT (El-Nouby et al., 2021)](https://arxiv.org/abs/2106.09681) is a source for attention over channels using
cross-covariance rather than a token×token relation matrix. This implementation makes no originality claim.

Historical SCCA source was read at commit `f6e9dfda765046ae7691302cf5ec89d3f76cec5d`, file
`ultralytics-main/ultralytics/nn/modules/scca_aifi.py`; provenance hashes are in `provenance.json`.

| Property | Historical SCCA-AIFI | DCC |
|---|---|---|
| Position | Inside AIFI spatial-MHA residual | P3 lateral Conv/BN/act output, node17 |
| Input | Token input x plus spatial-attention output s | Single projected P3 feature X |
| Q/K | Q from normalized s, K from normalized x | Both from spatially centered reduced Zc |
| Relation | B×4×16×16 | B×4×8×8 |
| Value | Independent 256→64 linear V of x | Zc directly; no V projection |
| Residual | Output projection of A V, within AIFI norms | Output projection of (A-I)V, X+delta |
| Temperature | Four learned bounded temperatures | Fixed1.0 |

Both use channel relationships informed by spatial samples and zero output projection. Source-level
differences above are verified; they do not establish academic novelty. DCC alone is spatially permutation
equivariant in real arithmetic. Its delta has zero spatial mean in real arithmetic. A comes from the whole
image, so other positions can affect the current output through A: DCC is not strictly local. Zero mean does
not establish background suppression, pixelwise protection, better recall or a performance floor; half
rounding and final residual addition can change measured mean error.

## Evaluation and packaging

No final test has been run. Later use original best.pt selection, fix its SHA, then independently val and test
the same weights with `corrected_sorted_conf_mask_v1`: imgsz640,batch16,workers0,halfFalse,conf.001,iou.7,
max_det300,augmentFalse,rectFalse,seed42. The inherited native validator has a sorted-row/unsorted-mask bug;
the dedicated evaluator reuses the audited corrected sorted-confidence mask and adds no NMS. Training's
original validation/best selection is unchanged. JSON includes P/R, AP50/AP75/mAP50–95, all10 IoU APs,
speed and curves. Test requires the matching completed val report.

Optional `--fixed-precision` is val-only: native fixed-candidate-pool one-to-one IoU.50 TP flags, P≥.8656,
unique original score thresholds, equal-score groups included whole, ties resolved by maxRecall/maxPrecision/
highestThreshold. No attainable nonempty point gives NOT_ACHIEVED/null. It does not modify CLI iou.7,
rematch predictions per threshold, or choose thresholds on test. Parent raw predictions must be evaluated
separately; historical aggregate metrics cannot supply this diagnostic.

`pack_dcc_light.py` writes and verifies a <20MiB archive with source/configs, evidence, args/results,
curves, log tails and SHA manifests. It excludes weights, data, reference ZIP, large arrays/full predictions;
those remain on the host. It cannot trigger train/val/test or delete results.

Server commands are in `SERVER_COMMANDS.md`; the final handoff pins the post-commit SHA separately to avoid
rewriting test HEADs or creating self-referential commit loops.
