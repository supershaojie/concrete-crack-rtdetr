# GRA v1: Guided Residual Alignment

This is an untrained experimental candidate, based directly on
`a0459d6a652cb702699087c88fa39a3e4c4087ec`, branch
`exp-rtdetr-r18-lite-gra-v1`. Formal training is **NOT_STARTED** and final test
is **NOT_RUN**. Engineering checks do not establish accuracy improvements or
academic priority. Public parameters remain trainable.

## Exact graph and mathematics

The two new YAML files preserve all 27 layer indices. Only node 18 changes from
Concat to GRAConcat with `from=[15,16,17]`: actual low-resolution Y4 (`H`), the
unchanged nearest upsample (`U`), and the unchanged backbone-P3 projection (`L`).
Output is `cat(U_corrected,L)` with 512 channels. Node 19 and its consumers
20 and 26 are unchanged. The main variant preserves original CBR and LIF;
`gra_v1` preserves the original C2 downsample and Decoder, without CBR/LIF.

Descriptors are `nearest(SiLU(Conv1x1(H)))` and `SiLU(Conv1x1(L))`, each with
16 channels. Their concatenation passes through a bias-free depthwise 3x3
convolution and SiLU, then a 32-to-8 offset head. Four contiguous channel groups
use interleaved `(dx,dy)` offsets. The fixed formula is

```
delta = 0.25 * tanh(offset_logits)
U_corrected = U + 0.5 * (sample(H, grid0 + delta) - sample(H, grid0))
grid0_x = 2*(j+0.5)/Wt - 1; grid0_y = 2*(i+0.5)/Ht - 1
grid_x = grid0_x + 2*dx/Ws; grid_y = grid0_y + 2*dy/Hs
```

Both samples use the same full H, bilinear interpolation, border padding, and
`align_corners=False`. Delta is in **source pixels**; positive dx reads further
right in H. The reference grid is bilinear; the original nearest result remains
in U. No offsets, features, or reference branches are detached. There is no
data-dependent zero shortcut. Sampling, tanh, subtraction, and residual addition
run in a local FP32 autocast-disabled region. The result returns to U's dtype.
No persistent grid cache, extra gate, BN, projection, or auxiliary loss is used.

Only the offset head starts at zero. The other convolutions retain PyTorch
default initialization under isolated CPU RNG, with CUDA RNG untouched. Thus
initial output is exactly the parent concatenation and the offset head has a
first-step gradient. Upstream descriptor gradients are initially zero by design,
and become active after the offset head learns. Fixed alpha is 0.5, not a second
zero gate. Constructor/load/fuse/EMA never reset learned offsets.

Parameter design: `256*16*2 + 32*3*3 + 32*8 + 8 = 8744`. The expected and
measured counts are kept separately in the check report. Conv-only added MACs
at 640 are 36,249,600; two sampling calls produce 3,276,800 scalar samples.
FLOPs coverage is **PARTIAL**, excluding sampling and elementwise/grid operations.
This is not a claim of zero extra computation. Server B16/640 cost remains
PENDING until the bounded real-data capacity preflight runs there.

## Initialization and verification

`tools/init_gra.py` reuses the original controlled parent initializers and the
public source SHA256
`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`.
It never reads historical trained best/last weights. Native Trainer adapts nine
classifier tensors from nc80 to nc1; the resulting parent and candidate values
are compared exactly. All public tensors, including buffers, are audited by
key, shape, value and digest. Only `model.18.*` is new. Both variants have exactly
the same new state. Saved initializations are reloaded and passed through native
Trainer model setup. Initialization audits live under `outputs/gra/<variant>/`.

`check_gra_module.py` tests non-square x/y ramps, constant channels, positive and
negative per-group offsets, border behavior, noninteger target sizes, gradients,
AMP/half and state preservation. `check_gra.py` checks actual train/eval hooks,
640 nc1 parameter counts, exact common initial features, two real-image detection
loss updates, native scaler behavior, optimizer coverage, EMA/save/reload/native
resume, half quantization and learned-state fusion. Fusion compares continuous
features at atol=2e-5/rtol=2e-4, separately records native top-k/output changes,
and replays fixed candidates only inside diagnostics. A precision note never
means native final rows were identical. Strict diagnostics restore TF32 flags.

The only non-GRA runtime repair changes AutoBackend warmup from uninitialized
`torch.empty` to finite `torch.zeros`; no training recipe or inference selection
policy changes. Original CBR and LIF source hashes remain the specified values.

## Execution and evidence

Run each tool with `--help` for its actual CLI. Initialization, structural checks,
bounded capacity preflight, plan, explicit start, native resume, independent
val/test, and light packing are separate entries. The capacity probe uses the
actual native Trainer loop and original recipe for at most 16 batches, stopping
after two effective optimizer updates. It never starts the 200-epoch run.
No GPU-idle gate, batch shrinking, AMP disabling, or package upgrade is allowed.

`parent_args.yaml` is the actual successful 200-epoch parent snapshot. The
generated recipe diff permits only model/output identity and verified equivalent
data locations. Data fingerprints contain image path inventories and label
hashes for the existing 6048/1728/864 splits; reading test identities is not test
evaluation. Standalone evaluation preserves `corrected_sorted_conf_mask_v1` and
the original one-to-one matcher. Optional fixed-P recall uses complete equal-score
groups from a fixed candidate pool on val only, never a smoothed/interpolated
fictional operating point. No test threshold search is performed.

Local full check reports and disposable updated weights remain under ignored
`outputs/gra/`; compact audit evidence is archived in this directory. Server
Torch 2.1 and B16/640 measurements require execution in that environment and must
remain PENDING until then. The final fixed-SHA handoff is generated after commit
under `outputs/gra/DELIVERY.md`, avoiding a self-referential commit cycle.

## Reference provenance and license

The requested ZIP (`c090a941-2c78-42c5-b566-27ec43639daf.zip`, historical alias
`dc2eda42-6ea1-467d-ab90-02d26882d4d0.zip`, expected SHA256
`b91dbaf5f74e35ff23079fe0854140e7fd82207c21455b564abc594636aba2cc`) was not
found in bounded searches of the project, attachment/result directories and
provided transfer/download locations. No package script was executed, no large
extra_modules file was imported, and no reference code was copied or relicensed.
Implementation follows the complete user-supplied mathematical contract.

Background sources verified from primary abstracts:
[DySample](https://arxiv.org/abs/2308.15085) formulates upsampling as point
sampling; [FADE](https://arxiv.org/abs/2207.10392) jointly uses encoder and decoder
features for upsampling. These are background, not attribution of this exact
double-sampling residual formula or evidence of gains on this dataset. The
reference package's FreqFusion sampler was not locally inspected. No claim of
first high-resolution-guided upsampling is made. New implementation carries the
repository's Ultralytics AGPL-3.0 license header.
