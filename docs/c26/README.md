# C26: C2 + original C19 CBR + original C24 SCCA-AIFI

This experiment tests whether two individually successful modules combine beneficially. No CBR-v2, rescue detach, tuning, trained-weight splice, or C25 dependency is used. C26 improvement is not assumed.

## Fixed sources and inventory

- Repository: `supershaojie/concrete-crack-rtdetr`.
- Parent: `codex/scca-aifi` at `f6e9dfda765046ae7691302cf5ec89d3f76cec5d` (C24 formal training source).
- CBR: `exp-rtdetr-r18-lite-cbr` at `025997e3c51eaf6933534308a95da6ebf97bff53` (C19 formal training source).
- Branch: `codex/c26-cbr-scca`; local worktree: `outputs/worktrees/c26`.
- Before creation, no C26 was found in local branch records, tracked tools/docs/experiment records, or the remote branch inventory. C25 is occupied and remains independent. Its existing worktree had uncommitted changes at inventory time; none were imported. No applicable `AGENTS.md` was found in the workspace/ancestors or the fixed source tree.
- Source hashes, file equality, and compatible interfaces are recorded in [source_trace.json](source_trace.json). Both module files match their pinned Git blobs after CRLF-to-LF normalization. The launcher also enforces these original module hashes.

## Structure and compatibility

YAML: `ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-scca.yaml`.

| Layer | C26 | Connection |
|---|---|---|
| 0–7 | Original C2 ResNet18-Lite backbone | Unchanged |
| 9 | `SCCAAIFI [1024, 8]` | Replaces AIFI only |
| 19 / 22 / 25 | Original Neck P3 / P4 / P5 | No inserted CSCEF layer |
| 26 | `RTDETRDecoderCBR [nc,256,300,4,8,3]` | `[19,22,25]`, slot 0 is Neck P3 |

The parsed YAML equals C19 with only layer 9 replaced, and equals C24 with only the last decoder replaced. At 640 input, the actual CBR input is `[B,256,80,80]`, not backbone S3. There are three decoder layers, 300 regular queries and the original DN, matching, losses and auxiliary outputs. No CSCEF, ACR or SALA module is active.

`scca_aifi.py` is unchanged: original spatial MHA/FFN, spatial-conditioned Q, input K/V, latent channel interaction, centered normalization, bounded temperature, FP32 core and zero output projection remain intact.

`cbr.py` is unchanged: 64-dimensional projection, 36 samples per query (four sides × three positions × inside/edge/outside), signed evidence, query conditioning, SiLU before position scoring/softmax, rho=0.1 and normal_fraction=0.1. Side displacements become `[(dl+dr)/2,(dt+db)/2,dr-dl,db-dt]`; they are not added directly as xywh. Sampling remains FP32 with border padding and align_corners=False. Sampling geometry and width/height conditioning retain their detach; P3, query and original boxes retain their main gradient paths. Only the final boxes are refined, identically for DN and regular queries. Scores and preceding layers stay unchanged; no feedback or extra clamp is added. `offset_out` retains its original zero initialization.

Necessary integration changes:

1. Append `return_final_query=False` **after** C24's existing `num_queries` and `dn_meta` parameters in `DeformableTransformerDecoder.forward`. Default calls return the original two tensors. Opt-in calls append the query for the last emitted layer, including early `eval_idx` in evaluation. Preserve `last_refined_bbox`, `refer_bbox.detach()`, refinement and auxiliary outputs verbatim. Existing C24/ACR keyword and positional callers remain compatible.
2. Register/export `CrackBoundaryRefinement` and `RTDETRDecoderCBR`; add the decoder to the native parser's channels-argument case. No full transformer/tasks file replacement.
3. Add separate C26 initialization, topology and loading checks. Old C24/C25 decoder assertions and old tools are unchanged. No forward hook or activation cache obtains the query.

## Initialization and recipe

Only `weights/rtdetr_r18_lite_imagenet_backbone_init.pt` from the main server repository is accepted, SHA256 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`.

| Model, nc=1, unfused | Parameters |
|---|---:|
| C2 | 20,082,772 |
| C19 | 20,128,661 |
| C24 | 20,148,312 |
| C26 | 20,194,201 |

Measured increment: 45,889 CBR + 65,540 SCCA = **111,429**. C26 has **552** state entries: **533** identity-mapped C2 entries + **14** CBR + **5** SCCA. No C25 layer shift is used. All common constructor states, loaded shapes/values and new initial states are checked. The native nc80→nc1 reconstruction loads **543/552** entries exactly and reinitializes only the nine classifier entries (DN embedding, encoder score weight/bias, three decoder score weight/bias pairs). Counts are derived from actual states, with per-key evidence in `evidence/mapping.json`.

Initialization is independently constructed in FP32 and round-trip checked. It has epoch=-1 and no optimizer, EMA, scaler, updates, metrics or training result state. Debug models/weights are temporary and deleted. Both modules use their original RNG-preserving construction; their initialized states also match independently constructed C19/C24 models. Native AdamW grouping is retained; all 19 added trainable tensors occur exactly once in the optimizer.

All **109** fields and their types are copied from the authoritative C2 args, with only `model`, `name`, `save_dir` changed. The launcher compares the actual server file against `docs/scca/c2_args.yaml`, then checks effective trainer args again. See [recipe_109_fields.json](recipe_109_fields.json).

The recipe is 200 epochs, patience50, 640, batch16, workers8, device0, seed42, deterministic=True, amp=True, AdamW/lr0=0.0005, lrf=0.01, weight_decay=0.0001, cosine scheduling and the archived warmup/online augmentations. `augment=False` remains in args; `RTDETRTrainer.build_dataset(mode="train")` enables the archived training augmentations independently of that inference flag. The dataset config, split and augmentation data are not regenerated.

## Results policy

Training validation chooses `best.pt` with the inherited C2 validator. Independent val/test use `corrected_sorted_conf_mask_v1` and the **CBR-refined final boxes**. Test requires a successful independent val of exactly the same checkpoint SHA, source commit, data config and evaluation settings. No historical reevaluation is a launch prerequisite.

| Historical test reference | mAP50–95 |
|---|---:|
| C2 | 46.9636% |
| C19 | 50.3892% |
| C24 | 49.9872% |

These numbers are the user's supplied historical results; C19/C24 source commits are listed above. They are not reevaluations in this delivery. C26 formal train/val/test results do not yet exist.

See [VALIDATION.md](VALIDATION.md) for actual local checks and limits, and [AUTODL.md](AUTODL.md) for manual commands.
