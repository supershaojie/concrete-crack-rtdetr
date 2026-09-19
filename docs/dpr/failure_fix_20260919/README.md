# DPR failure diagnostics and check corrections — 2026-09-19

**Formal admission: BLOCKED. Formal training: NOT_STARTED. Final test: NOT_RUN.**
The check corrections and bounded local evidence are delivered; unresolved AMP B
and whole-network half failures are retained. No production model defect was
established or patched. Exact-server validation and a fresh complete preflight
remain PENDING. No server login, dependency changes or B16 capacity rerun occurred.

## Identity and preserved work

Work began on the clean existing `exp-rtdetr-r18-lite-dpr-v1` worktree at
`2f4319281cac92a1e56663c3c4a48ba7bafc6ccb`. The separate main checkout and its
existing user files were preserved. The earlier THOP fix remains unchanged.
`server_failure_original.tar.gz` is the user's original 542,754-byte archive,
SHA256 `09df0e5cab2fecc8ce3c341ed758696ed927ced1c555a41e9179fb9c34b38aeb`.
Its 18 manifest entries and 12 reported source identities were verified.
The original 10,833,238-byte preflight JSON has SHA256
`37a8364cbc7406f22364d3a41b8a5f1f9c779e28869fe34d9bddf0ea19b0c740`.
Old reports are preserved verbatim; none is rewritten with the new statistics.

Production `ultralytics-main`, all model/data configurations, DPR forward/four
maps/folding, CBR/LIF, initialization, checkpoint A, native optimizer/GradScaler/
EMA order, B classifier, failure-name subset condition, new-DPR raw gradient
condition, and 109-field training recipe are unchanged. The insertion remains
`model.5.blocks.1.branch2b`, with 3,072 added parameters. Thresholds remain FP64
`1e-10/1e-9`, FP32 `2e-5/2e-4`, half `2e-3/2e-2`; all three FLOPs assertions
retain `1e-9`. `scope_checks.json.gz` records the source/AST audit.

## What changed and why

| Original report path / issue | Classification | Delivered correction / result |
| --- | --- | --- |
| `comparison(a,b).exceed_fraction` | Confirmed check defect | Use `abs(b)` as the relative reference, matching unchanged `torch.allclose(a,b)`. Report `allclose_right_reference_v2`; raw pass/fail never altered by this statistics fix. |
| `learned_inference_details.cpu_fp32.checks.autobackend_saved_best` | Original numeric failure remains unexplained | Observe the real `backend(image)` call, local/continuous tensors and actual decoder selections, independent reference/weights/file identities, fixed replay and original comparison. Local aligned output passes exactly; this does **not** explain the server's CPU failure retroactively. |
| CUDA `autobackend_saved_best` / `autobackend_half` | Confirmed comparison-condition mismatch | Independent reference now follows CPU file load → FP32 → CPU fuse → eval → device → inplace → dtype → frozen parameters. Keep original GPU-fused/half comparison and identity differences as `legacy_*`. Production loader is untouched. Aligned backend remains a raw-allclose hard gate. |
| `cuda_half.native_half_vs_fp32`, `fp32_fold_then_half` | Unresolved whole-network numeric failures | Keep original results; add U32/D32/N32/U16/D16/N16, same-effective-kernel ordinary ConvNormLayer, parent precision background and same-input/norm rounding-order checks. No threshold or dtype policy changes. |
| `checks.cuda_fp32.B`, `checks.cuda_native_amp.B` | Unresolved independent-backward differences | Same-backward, pre-clip, correctly unscaled effective-kernel/W/four-group evidence; independent analytic adjoints, delta propagation, frozen local replay. Original B classifier/conditions unchanged. No attribution to checkpoint A or proof of long-term equivalence. |

`tools/dpr_diagnostics.py` implements the observations. A private globals facade
binds the unchanged query-selection code to one decoder instance; it records
actual `topk` indices, callsite, dimension, count and hit count. It never globally
replaces `torch.topk`. Indices distinguish changed positions, set changes and
same-set reordering. Fixed replay is recorded even when continuous features fail;
it cannot erase that failure. Effective kernels come from the actual kernel call.
Normal/exception paths restore instance methods and hooks.

Backend evidence binds the original learned state to the checkpoint's selected
model/EMA before casting, checks file SHA before/after, records independent
storage and complete tensor/mode/dtype/device/input/cache identities, and proves
observed versus unobserved actual backend output is exact. Raw output, every
local/continuous leaf, deployment and retained norm are required. The shared
validator runs before the `details` branch; a forged PRECISION_NOTE or injected
details cannot bypass raw failure. Mathematical reports are v2; full reports
require `dpr_full_preflight_v2` plus the statistics version. The original
`dpr_acceptance_v1` contract is not relaxed. Diagnostic reports cannot enter it.

## Local results, final source

Windows / Python 3.9.25 / PyTorch 2.7.1+cu118 / RTX 2060 6 GB differs from the
server's Python 3.10.13 / PyTorch 2.1.2+cu121 / RTX 4090. Installed THOP 2.0.18
uses parent-replaces-subtree semantics (probe 7); an existing isolated official
THOP 2.1.6 copy uses cumulative semantics (probe 7+11=18). No environment was
upgraded/downgraded. Final `module_03_thop2018/216` runs cover **both variants at
640**, unfolded/deployed/repeated profile with hook/buffer/state preservation.

| THOP / configuration | Corrected unfolded GFLOPs = parent | Deployed GFLOPs = fused parent | Result |
| --- | ---: | ---: | --- |
| 2.1.6 / dpr_v1 | 59.42076 | 58.3099248 | PASSED |
| 2.1.6 / cbr_lif_dpr_v1 | 59.816664 | 58.7091056 | PASSED |
| 2.0.18 / dpr_v1 | 58.276608 | 57.1657728 | PASSED |
| 2.0.18 / cbr_lif_dpr_v1 | 58.672512 | 57.5649536 | PASSED |

Version totals differ because their subtree semantics also affect other custom
modules. Each comparison uses its own same-version parent; no total is hardcoded
to make different profiler versions agree.

Kernel construction remains separately documented, outside deployed convolution
FLOPs. Both existing controlled initializations were re-audited without rewriting
weights, including reload and native train-entry reconstruction stopped before
training: PASSED. FP64 mathematics, positive real-file backend observation,
right-reference statistics, actual-index capture, normal/exception cleanup,
deliberately wrong direction/scale, continuous failure, failed replay, nonfinite,
undeployed, missing schema, wrong SHA/state and diagnostic-admission rejection:
PASSED (`unit_03.json.gz`).

Final complete bounded collection is `diagnose_03`: 138.047 s, source stable,
all three requested modes collected, worker **exit 3 / BLOCKED**. It uses the
existing B2/160 real-GT/DN lifecycle procedure, not B16 capacity or formal training.
`diagnose_01` and `diagnose_02` preserve earlier iterations and their source
snapshots. `diagnose_01` returned 0 for completed collection despite blockers;
the corrected supervisor returns 3 for retained raw blockers. A deliberately
limited 10-second `timeout_01` returned 124, killed only its owned worker, recorded
the missing diagnostic JSON, and produced a light evidence package.

| Final local result | CPU FP32 | CUDA FP32 | CUDA native AMP |
| --- | --- | --- | --- |
| B parent/candidate failed state tensors | 0 / 0 | 6 / 6 | 56 / 48 |
| B parent/candidate failed gradient tensors | 0 / 0 | 0 / 0 | 160 / 159 |
| Original B classification | PASSED | FAILED | FAILED |
| Each run: four analytic adjoints; W=effective-kernel gradient | PASSED | PASSED | PASSED |
| Same frozen input/upstream/norm local replay; delta adjoints | PASSED | PASSED | PASSED |
| Aligned real FP32 file backend max_abs / raw_allclose | 0 / true | 0 / true | Outside inference scope |

Checkpoint A is explicitly NOT_RERUN_DIAGNOSTIC. The unchanged B classifier
therefore does not apply its A-dependent conditional precision note to the CUDA
FP32 state difference. Historical server A passed in all modes; that historical
pass cannot supply new-SHA admission. Original B tensors, complete failed names,
errors and independent controls remain in the package.

Final local AMP DPR raw gradient max_abs: CD `9.918212890625e-5`, HD
`8.7738037109375e-5`, VD `1.1444091796875e-4`, AD `7.62939453125e-5`;
all **raw_allclose=false** at original FP32 tolerance. Analytic/local/delta
checks passing supports propagation from each run's effective-kernel gradient;
it does not identify one CUDA operator as the cause or prove trajectory equality.

Final local original half outputs: `native_half_vs_fp32` max_abs
`0.8998786211013794`, `fp32_fold_then_half` `0.8562614917755127`, both raw false;
half deploy reload and aligned real half backend have max_abs 0, raw true.
U32→D32, D32→N32 and same-effective ordinary FP32 paths pass. U32→U16,
D32→D16, N32→N16 first exceed at `norm_output`; U16→D16 at P5;
D16→N16 at encoder_features. Changed index positions are respectively
411/404/399/387/387, not numbers of added/lost detections. Fixed output replay
passes in these comparisons but continuous failures remain. The ordinary-conv
and parent precision backgrounds also first exceed at norm output; neither
grants an exception. Local rounding-order kernel max_abs `6.103515625e-5`,
output `0.00390625`, both allclose true under the elementwise original half
tolerance; the two paths are not bitwise identical.

Original server evidence remains distinct: CPU backend max_abs 0.34705445,
raw false; native_fuse candidate replay passed but did not prove backend cause.
Original server half outputs 0.89977223 / 0.89947522 failed. Original server
FP32/AMP B had failed state counts 18/19 and 48/40; gradient counts 108/112 and
155/158. All four added gradients failed and the candidate failed-name subset
condition failed. Historical A and B16 capacity passed (5 batches, 2 effective
updates); the complete original preflight FAILED. No result was copied into a
new full-preflight permit.

## Evidence and next execution

`evidence_manifest.json` binds compressed artifacts and their original bytes.
Archives include JSON, logs and executed source snapshots; no weights, images or
dataset are committed. Raw replay tensors and learned diagnostic weights remain
only in their fresh local output directories with hashes in the JSON. The server
diagnostic likewise leaves its replay weights on that server and emits a small
`diagnostic_light.tar.gz` plus external member manifest.

Use the post-commit rendering of `SERVER_HANDOFF.template.md`: fixed-SHA sync,
two initialization/module audits, then the finite main-variant diagnosis. Exact
server THOP/PyTorch behavior and the original CPU backend failure remain PENDING
until those commands are executed. Run one new complete server preflight only
after the targeted blockers are resolved. Full preflight and formal start are
separate entries; this delivery runs neither. Any contract alternative is solely
in `CONTRACT_CHANGE_PROPOSAL.md`, UNAPPROVED and NOT_IMPLEMENTED.
