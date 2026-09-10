# Compatibility audit — 2026-09-10

This audit was written before production code changes. Sources are the supplied experiment archives, their nested source snapshots, local git objects/worktrees, C2 original args and the supplied module ZIP. See JSON for archive/file SHA256 and complete typed recipe differences. No applicable AGENTS.md was found; the current checkout has user-owned untracked artifacts, preserved untouched.

## Findings and evidence limits

H1/H2/H3 are supported as dependency risks. They are not demonstrated causal explanations of all AP loss. H4 is supported by historical single-run results, not a guarantee of additive gains. No trained-gradient angle or repeated-seed causal study was performed. The nonzero synthetic branch probes below establish real autograd paths; they do not establish opposing task-gradient directions.

* C17 `CSCEFv51` inherits v5 content mixing: lateral and upsampled Y4 each project to 32 channels, GroupNorm, concat/mix, depthwise convolution, SiLU. Scharr/32 confidence comes from projected lateral, is detached FP32, and v5.1 averages the complete nonlinear confidence over H/W per image. Semantic content remains differentiable in the residual. Its enhanced lateral is concatenated with the SAME upsampled Y4; the resulting P3 feeds PAN P4/P5.
* C24 `scca_channel(x,s)` uses Q from spatial MHA output s and K/V from x. Neither is detached. The added delta enters before norm1 (post-norm) or at the first residual (pre-norm); it changes Y5/Y4 and therefore both inputs to the subsequent C17 fusion route. Its additional gradient reaches src and MHA s.
* C19 wrapper reads x[0], final neck P3. CBR projects this for boundary samples and uses final query before a nonlinear position score. Both are differentiable. Sampling grid and geometry projection detach boxes; HOWEVER displacement = rho * original[w,w,h,h] * tanh(t) still differentiates through box size. The direct original-box addition also remains. rho=.10 bounds each side by 10% of original w/h; worst-case width/height change is ±20%, without clipping. Over-refinement is plausible, not established from AP alone.
* C17+C24 has the most direct serial semantic rewrite plus PAN propagation; C17+C19 uses a rewritten boundary reference plus extra P3/query gradients; C19+C24 couples indirectly through FPN/decoder distributions. This ordering is consistent with, but cannot uniquely explain, the observed degradation.
* All six innovation training args have 109 fields; only model/name/save_dir differ from C2. Initialization reports identify the same SHA-locked ImageNet/C2 source and public constructor equality; no mapping defect was found in the supplied reports. These are archived reports, not an independent replay of historical training.
* Evaluation caveat: C24/C25/C26 use corrected_sorted_conf_mask_v1 and record zero affected test images. C20/C25/C26 record effective seed=0, whereas C2 records 42. The new unified entry fixes seed=42 and explicitly records the same established corrected-mask policy; no thresholds are tuned. Historical comparisons retain these provenance differences.

## C2 semantic topology

Backbone stage3=P3/8, stage4=P4/16, stage5=P5/32. Original 1x1 projections produce stable P3_ref/P4_ref and projected P5. P5 projection→AIFI→Y5→upsample+P4_ref→FPN→Y4→upsample+P3_ref→FPN→P3_base→PAN(+Y4)→P4_base→PAN(+Y5)→P5_base→three-level decoder. At 640, spatial sizes are 80²/40²/20². Layer numbers are observations in the original YAML, not the semantic matching algorithm.

## Forward dependency graphs

Each arrow denotes value dependence; parentheses group multiple inputs.

```text
C2:  P5proj→AIFI→Y5→Y4; (upY4,P3ref)→P3→PAN(P4,P5)→Decoder
C17: (P3ref,upY4)→CSCEF→enh_lateral; (upY4,enh_lateral)→P3→PAN(P4,P5)→Decoder
C19: original Neck→Decoder→(query,box); (NeckP3,query,detach(box))→CBR_t; (box,CBR_t)→refined_box
C24: P5proj→MHA→s; (P5proj,s)→SCCA_delta; (P5proj,s,delta)→AIFIout→Y5→Y4→P3→PAN→Decoder
C17+C19: CSCEF→P3→PAN→Decoder; (CSCEF-modified P3,query,box)→CBR→final_box
C17+C24: SCCA→Y5→Y4→upY4→CSCEF_semantic→residual→enh_lateral; (upY4,enh_lateral)→P3→PAN→Decoder
C19+C24: SCCA→Y5/Y4→NeckP3→Decoder; (NeckP3,query,box)→CBR→final_box
```

## Gradient dependency graphs

Arrows below point in backward direction. stop means a blocked branch, not a detached main path.

```text
C2: loss→Decoder→PAN/P3→FPN→AIFI/backbone; references also receive normal FPN gradients
C17: loss→Decoder→PAN/P3→CSCEF_content→(P3ref,upY4); confidence→STOP; loss→concat→upY4 also
C19: loss→refined_box→original_box; loss→CBR→(NeckP3,query); grid/geometry→STOP; offset_size_scale→original_box[w,h]
C24: loss→AIFIout→main MHA/FFN; loss→delta→Q→s→MHA; loss→delta→K/V→src
C17+C19: loss→CBR→P3→CSCEF→(P3ref,upY4), and loss→CBR→query→Decoder; original box path retained
C17+C24: loss→CSCEF_content→upY4→Y4/Y5→SCCA→(src,s) plus ordinary concat and main AIFI gradients
C19+C24: loss→CBR→(NeckP3,query)→FPN/Decoder→SCCA→(src,s), plus ordinary box/main gradients
```

## Decision

Implement the specified stable-reference design: original C2 FPN/PAN completes first, DR-CSCEF changes only decoder P3 using detached projected backbone P3/P4; GI-SCCA preserves C24 values and isolates input derivatives; SR-CBR receives a fourth stable P3 reference and detached query/geometry, rho=.075. Explicitly detach offset scaling widths/heights to close the additional discovered box-condition derivative. Keep original box addition live. Keep old classes/YAML for regression; port only C19's optional return_final_query helper into C2, excluding unrelated ACR inherited by C24/C26. No fourth module, new loss, matcher, DN or recipe change.

## Module ZIP reference

Read CAFM.py (pooled channel relation and feature residual), FAAFusion.py (resolution alignment, FP32 core, grid sampling), FreqFusion.py (resampling and optional mmcv/CARAFE dependency), MAFusion/SFSFusion routing, and original transformer sampling conventions. These support explicit source routing, local dtype conversion and align_corners=False. No fusion/attention implementation or extra_modules tree is transplanted; no einops/mmcv/CUDA extension is added. Core algorithms are inherited from audited in-repository C17/C19/C24 sources.

## Historical test metrics (fractions)

|Model|P|R|mAP50|AP75|mAP50-95|
|---|---:|---:|---:|---:|---:|
|C2|0.833960007851|0.808044424433|0.858162985651|0.458360400552|0.469636231908|
|C17|0.844895323441|0.825003752064|0.879297454830|0.532829775858|0.512045085044|
|C19|0.849033844558|0.837311905432|0.885186356567|0.519995255215|0.503892419455|
|C24|0.855563273975|0.819900945520|0.878243875820|0.506461002298|0.499872058481|
|C20|0.837320605748|0.814948221522|0.866302563893|0.494810709544|0.487043012576|
|C25|0.793837773204|0.777127420081|0.818477876119|0.404977566059|0.425179978084|
|C26|0.842167659938|0.814948221522|0.860372099817|0.496019333872|0.490730335855|

## Actual nonzero branch derivatives

```json
{
  "scope": "synthetic nonzero branch dependency probe, NOT a trained-loss conflict/cosine measurement",
  "cscef": {
    "lateral_side_grad_l1": 5.955102920532227,
    "semantic_grad_l1": 6.175935745239258,
    "residual_rms": 0.010653483681380749
  },
  "scca": {
    "src_side_grad_l1": 16.903833389282227,
    "mha_side_grad_l1": 1.7848355770111084,
    "relation_shape": [
      1,
      4,
      16,
      16
    ]
  },
  "cbr": {
    "p3_side_grad_l1": 0.0055017475970089436,
    "query_side_grad_l1": 0.0032682709861546755,
    "box_residual_scale_grad_l1": 0.005145697388797998,
    "rho": 0.1,
    "normal_fraction": 0.1
  }
}
```

## Sources

- C2: `67c3078e54a657fd96d65fee657a75fbb1dae0d6`
- C17: `0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139`
- C19: `025997e3c51eaf6933534308a95da6ebf97bff53`
- C24: `f6e9dfda765046ae7691302cf5ec89d3f76cec5d`
- C20: `81d5f0175e0168e022f29feb1ceb54de8a52deb5`
- C25: `ac32e223a509981ee9e61be66a352f2b80c5bfd1`
- C26: `beedcfa307e250fb2de47587097c51f9c141123b`
