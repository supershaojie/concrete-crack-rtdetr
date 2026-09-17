# SDB-P3 v1

This experiment branches directly from the successful CBR + LIF-Down commit
`a0459d6a652cb702699087c88fa39a3e4c4087ec`. It does not inherit another experimental
branch. The original `cbr.py` and `lif_down.py` LF-normalized SHA256 contracts are
enforced by every initialization/topology audit.

The implementation contract is archived in `IMPLEMENTATION_CONTRACT.md`. The
requested reference ZIP was not visible in the bounded local searches recorded
in `source_audit.json`; its `SPDConv` implementation was **not read**. The explicit
mathematical contract supplies the implementation. The
[SPD-Conv paper](https://arxiv.org/abs/2208.03641) is only a space-to-depth reference,
not evidence of accuracy gains for this design.

## Structure

`model.19` is a `RepC3` subclass, receiving `[model.18, model.4]`. Its original
`cv1/cv2/m/cv3` paths and three internal RepConv blocks are unchanged. It computes
the original RepC3 output first, then adds the semantic-gated P2 detail residual.
Only `model.19.sdb.*` is new. P2 is retained in the native save list. The resulting
P3 feeds both `model.20` and decoder `model.26`; the decoder still receives
`[19,22,25]`. There is no P2 detection head.

The phase order is channel blocks **00,10,01,11**. Odd core P2 sizes receive
replicate padding only on the right/bottom. A mismatched P3 spatial shape raises
an error. No interpolation or detached bypass is used.

`W_o=0` gives exact identity at initialization. Other convolutions use Xavier
uniform initialization, `W_g.bias=0`, and GN affine state is one/zero. Construction
uses a fixed seed in an isolated CPU RNG scope. No load/forward/fuse path resets
learned state. GroupNorm remains GroupNorm during fusion, and the original LIF
pre-BN residual protection is retained.

| nc=1 model | Unfused parameters | Fused parameters |
|---|---:|---:|
| CBR + LIF parent | 20,149,765 | 19,944,965 |
| CBR + LIF + SDB | 20,177,861 | 19,973,061 |
| C2 parent | 20,082,772 | 19,877,716 |
| SDB only | 20,110,868 | 19,905,812 |

Each variant adds **28,096 parameters**, including 10 parameter tensors and no
buffers. At 640 input its convolution-only analytical cost is 178,790,400 MACs,
or 0.3575808 GFLOPs at two FLOPs per MAC. This excludes normalization, activations,
sigmoid and elementwise arithmetic; it is not a measured full-network FLOP count.

`tools/profile_sdb_p3.py` measures parents and targets with the same installed THOP
hooks at nc=1/640, separately before and after fusion. It records the registered
and unsupported operators and labels its results **PARTIAL**: functional
`grid_sample`, attention matrix operations and several normalizations/activations
are not fully counted. Its 2x-raw figures must not be presented as complete GFLOPs.

## Initialization and verification

`tools/init_sdb_p3.py` checks the exact untrained source SHA256, epoch and absence
of trained optimizer/EMA/scaler state. It reconstructs the successful controlled
parents and strictly transfers every public parameter and buffer with unchanged
paths. Both variants' new states must match exactly. Saving is exclusive and is
immediately followed by a strict reload.

The real `RTDETRTrainer.get_model` path is checked against a parent reconstruction
at the identical CPU RNG state. All nine nc=80-to-1 classification adaptations
must equal their nc=1 parent tensors exactly. Resume preserves all learned SDB
state; it does not demand a zero output projection.

`tools/check_sdb_p3_core.py` covers explicit phase values, core odd geometry,
identity, staged gradients, isolated residual-to-P2 differentiation, parser
repeats, exact counts and THOP integer/Tensor counter compatibility.
`tools/check_sdb_p3.py` additionally checks real detection loss and DN, matched
RNG/BN train/eval comparisons, nonzero save/reload, native optimizer/scaler/EMA
resume, and nonzero full-model fusion. Local engineering success leaves the
overall report **PENDING** until server B16/640/native AMP checks finish.

Low precision may change discrete encoder top-k choices. Diagnostics distinguish
native candidate identities from continuous feature/fixed-candidate comparisons;
a replay is not reported as native-output equivalence. FP32 fusion and explicit
half inference are separately recorded. In local checks the main model and its
matching parent both showed AMP/half native top-k set changes after fusion, while
FP32 full-output fusion passed. Reduced-precision native equality is consequently
labelled `NATIVE_TOPK_DRIFT`, not PASSED; finite inference, continuous neck features
and complete fixed-candidate replay are checked separately.

## Recipe, data and lifecycle

`parent_args.yaml` is the actual successful CBR + LIF args snapshot; all 109 fields
were compared with the attached appendix. Only model/output identities and a
verified equivalent data path may change. `parent_dataset_inventory.json` fixes
the split paths and label contents, not just image counts. The local inventory
matches train=6048, val=1728, test=864. Reading the test file inventory does not run
test inference.

The default variant is `cbr_lif_sdb_p3_v1`. The single-module YAML/init is available
for later controlled ablation. Server commands are in `SERVER_COMMANDS.md`; the
delivery copy has the complete pinned commit filled in. Initialization/preflight
never dispatches formal training. Explicit `plan` and `start` are separate; start
requires matching PASSED server evidence, code, data and initialization hashes.
Existing outputs are preserved. Resume uses an actual unfinished `last.pt`.

Training status records completed epochs independently of final-validation
success. At epochs 40/80, comparison is only against the parent's same-epoch val
record; patience and learning-rate policy are unchanged. Independent evaluation
uses `corrected_sorted_conf_mask_v1`, val first and then the exact same best.pt
hash for test. No extra NMS or test-driven model/threshold selection is introduced.

The lightweight packer includes code/configuration, metrics and bounded `.log`
tails as `.txt`, with a verified SHA256 manifest. It excludes weights, datasets
and reference archives and reports missing evidence and files exceeding the size
target. It does not start evaluations or download another experiment.

This delivery implements an experiment, not an accuracy result. Formal training
is **NOT_STARTED** and final test is **NOT_RUN**.
