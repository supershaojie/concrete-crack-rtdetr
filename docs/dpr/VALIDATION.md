# DPR implementation validation

**Overall local engineering acceptance: FAILED / incomplete. No formal start
permission was generated.** This is an implemented experiment with honest
blocking evidence, not a claim that all preflight requirements passed.

Formal training **NOT_STARTED**; independent full val **NOT_RUN**; final test
**NOT_RUN**; server login **NOT_PERFORMED**. No ablation training was started.

## Environment and identity

Local: Windows, Python 3.9.25, PyTorch 2.7.1+cu118, RTX2060 6 GiB.
Required server: existing Linux/Python3.10/PyTorch2.1.2+cu121/RTX4090.
These environments are not interchangeable acceptance evidence.

Repository `supershaojie/concrete-crack-rtdetr`; experiment worktree was created
directly from `a0459d6a652cb702699087c88fa39a3e4c4087ec`. The original workspace's
user files and other experiment worktrees were preserved. Fixed parent blob
hashes, public source weight hash, original CBR/LIF hashes and actual successful
parent recipe were verified; see `source_audit.json`.

The public source SHA256 is
`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`.
No trained best/last checkpoint was used as an initialization source.

## Results by required scope

| Check | Result | Evidence / practical scope |
|---|---|---|
| Fixed location, four mappings, zero sum, off-diagonal preservation | PASSED | `module_checks.json`; nonzero asymmetric CPU FP64 reference |
| Output/input/W/four-group gradients and numerical gradcheck | PASSED | FP64 atol 1e-10, rtol 1e-9; structural CD/AD center zero gradients retained |
| Two graphs, 27 nodes, from indices, original CBR/LIF/C2 | PASSED | Exactly node 5 wrapper changes; 640 target B×128×80×80 |
| Zero initialization, P3/P4/P5/encoder/output equality | PASSED | Actual CPU FP32 640 models; no flipped candidates |
| Public initialization and immediate saved reload | PASSED | Complete reports `*_init.json.gz`, cross-variant zero hashes identical |
| Native nc1 get_model and actual train-entry rebuild | PASSED | Nine classification adaptations; original parent tensors exactly equal |
| Full 109-field recipe and data split/label identity | PASSED | `recipe_diff.json`, `source_audit.json`; train/val/test identities match parent |
| Real GT/DN loss and two effective updates | PASSED | Main variant, bounded B2/160 CPU FP32/CUDA FP32/native AMP; not capacity evidence |
| W_eff, W and four difference groups actually change | PASSED | Real finite gradients and effective updates in local preflight 03 |
| Checkpoint A CPU FP32 | PASSED | Original saved FP32 moments/quantized EMA oracle, full state, storage isolation, native replay, CPU continuation |
| Checkpoint A CUDA FP32 / native AMP | PASSED | Same-gradient native unscale/clip/step/scaler/EMA replay; AMP effective update and overflow skip |
| Independent backward B CPU | PASSED | Original atol 2e-5, rtol 2e-4 |
| Independent backward B CUDA FP32 | PRECISION_NOTE | Raw allclose remains false; same three inherited bias states plus EMA, complete per-tensor gradient evidence; A and DPR raw thresholds pass |
| Independent backward B CUDA native AMP | **FAILED** | 159 candidate gradient tensors exceed original tolerance, including all four DPR groups; not explained away by parent noise |
| Real learned CPU/CUDA FP32 fold/fuse/reload/AutoBackend | PASSED | Original norm retained, native fusion and repeated fusion, output comparisons |
| Explicit CUDA half comparisons | **FAILED** | Continuous feature discrepancies, not merely top-k index changes; exact saved half reload and half AutoBackend separately pass |
| Local native B16/640/AMP capacity | **TIMEOUT / PENDING** | Real dataset/AMP/optimizer setup reached; no completed batch or effective update; original B16/workers8 retained |
| Server B16/640 and complete acceptance | **PENDING** | No server login; local reports never authorize server start |
| Server Conda activation | **PENDING** | Shell syntax/help and set-u order reviewed; actual server activation not run |
| Admission failure cases | PASSED (unit scope) | `admission_fault_checks.json`; mocked evidence explicitly not a preflight pass |
| Evaluation sorted mask / one-to-one matching | PASSED (unit scope) | `evaluation_unit_checks.json`; no real val/test run |
| Light-package content validation | PASSED (unit scope) | `pack_unit_checks.json`; failed/mismatched/missing evidence cannot claim completion |

Real update counts in preflight 03: CPU FP32 2 batches / 2 updates; CUDA FP32
2 / 2; native AMP 6 / 2 with 4 overflow skips. The skip count is not treated as
successful optimizer steps. No AMP disabling, fixed scale=1, batch reduction,
or repeated seed/threshold search was used.

CUDA AMP DPR gradient max absolute differences were approximately
6.87e-5–9.16e-5, relative L2 .00154–.00186 and exceeded fractions .198–.299.
These raw failures block admission. They are not replaced by a precision note.

Half diagnostics also retain raw failures. In local preflight 02,
`native_half_vs_fp32` failed at target/P3/P4/P5/encoder; P5 exceeded fraction
was 1.83594%. `fp32_fold_then_half` passed target/P3/P4 but failed at 4 P5 and
3 encoder elements, alongside 403 changed candidate indices. Since continuous
features failed their predeclared thresholds, candidate changes alone cannot
justify acceptance. Final preflight 03 remains FAILED for half. No thresholds
were widened and no production query selection was changed.

## Parameters and complexity (actual measurements)

| Representation | CBR+LIF+DPR | C2+DPR |
|---|---:|---:|
| Parent, unfused | 20,149,765 | 20,082,772 |
| DPR training representation | **20,152,837** | **20,085,844** |
| Ordinary native fusion, DPR not folded | 19,948,037 | 19,880,788 |
| DPR fold + parent-equivalent native fusion | **19,944,965** | **19,877,716** |
| Parent under same native fusion | 19,944,965 | 19,877,716 |

Measured training increment is exactly 3,072 in each variant. Standard folding
keeps the original target BN and removes only the four DPR tensors. There is no
asymmetric extra backbone BN fusion or claimed new compression ratio.

THOP MAC×2, nc1, 640: main unfused 58.672512 GFLOPs, standard deployed
57.5649536; single 58.276608 and 57.1657728. Corresponding parents match these
convolution/BN totals. Raw THOP missed the functional dense convolution by
1.8874368 GFLOPs; the DPR hook corrects this. Kernel construction is separately
reported: difference reductions/maps, three delta additions, a 147,456-element
dense diagonal allocation and 147,456 dense-kernel additions per forward.
These operations are not falsely reported as zero or folded into THOP conv totals.

## Failures, corrections, and evidence preservation

The first initialization audit completed model checks but its runtime probe was
affected by native CPU device selection changing process CUDA visibility. The
audit now selects CPU without that environment side effect; the existing init
was preserved and exactly reverified. `init.failed-runtime.json` retains the
original failure. The first module check hit upstream path cleaning of a
single-quote system temporary directory; worktree-owned temporary directories
fixed that path issue without changing the mathematical checks.

Local preflight 01 exposed missing native `set_model_attributes()` in the
diagnostic trainer; formal Trainer setup already performed this step. Local 02
then exposed the diagnostic's missing native deterministic seed initialization.
Both were corrected through native APIs. One fixed-seed/batch/tolerance affected
rerun (local 03) followed; failures were not hidden by retries until passing.

Only the owned Windows capacity process tree was stopped after its wall-clock
budget; other Python/GPU tasks were not stopped. Future capacity runs use an
owned subprocess with a 600-second wall-clock cap as well as the 16-batch cap.
Timeout records preserve paths/logs and do not lower the original recipe.

The original preflight reports/logs and their lossless compressed copies retain
raw per-tensor outcomes. The probes ran before final Git commit, and gate/tool
audit code was refined during local 03 (`code_stable=false`). These are scoped
engineering diagnostics, not fixed-HEAD server admission. The unchanged module
mathematics evidence has separately been rechecked against current normalized
source/config hashes. The server must run its own complete fixed-commit checks.

`optimizer_fp32_v1` is an explicit storage policy change, with native half EMA
retained. It does not claim historical FP16-moment trajectory equivalence.
Checkpoint metadata also separates diagnostic and formal continuation so a
bounded diagnostic checkpoint cannot become a formal resume source.
The inherited Git reader returns no commit for a branch ref in a linked
worktree (it does not follow commondir). DPR now additionally saves the actual
`git rev-parse HEAD` as `DPR_checkpoint.code_head` and uses that field for
resume identity; native Git fields and training calculations remain intact.
This final metadata-only fix is separately checked without rerunning the model
precision matrix.

Reference ZIP: NOT_FOUND / NOT_READ. Implementation follows the supplied
mathematical contract; no third-party archive code was imported. Accuracy gains,
independent val/test metrics, server capacity, and full engineering acceptance
are not claimed.
