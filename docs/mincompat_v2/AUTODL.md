# Fixed-source AutoDL operation

No formal training was started during implementation. Use the final full 40-character SHA reported with the delivered commit. Existing main checkout, user results, environments and other worktrees are preserved.

First synchronization from the server main repository (the new script need not already exist in that checkout):

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR
MINCOMPAT_SHA=<FULL_SHA>
git fetch origin codex/rtdetr-mincompat-v2
git show "${MINCOMPAT_SHA}:tools/sync_mincompat_v2.sh" | bash -s -- "$MINCOMPAT_SHA"
cd /root/autodl-tmp/projects/Crack_RTDETR-mincompat-v2
bash tools/sync_mincompat_v2.sh "$MINCOMPAT_SHA"
```

Sync verifies origin, full SHA, branch ancestry and v1 parent ancestry, then creates an independent detached worktree. A dirty, attached or different-SHA existing worktree is preserved and refused. No reset/clean/stash, environment installation or training is performed.

The launcher activates the existing rtdetr environment. Preflight checks clean/pinned SHA, local module imports, CUDA, expected PyTorch2.1.2/CUDA12.1, nc1 topology and parameter count, fixed gamma0.25, original SCCA/CBR classes, the C2 initializer SHA, all109 typed C2 arguments, dataset paths and original dataset config. It repeats exact-forward/gradient, AMP/half, native loss/DN and optimizer checks before creating the tmux worker. Worker token and source/init/data/recipe hashes guard the dispatch. The real Trainer records exact args, AMP, original zero heads, nc1 load mapping and optimizer groups. OOM retries cannot silently reduce batch.

Round A: use two separate one-GPU AutoDL instances/containers, each exposing its assigned GPU as logical device0 so the archived device=0 recipe remains unchanged. Each uses the same SHA. The two commands are:

```bash
bash tools/autodl_mincompat_v2.sh cscef_v52_compat start-direct
bash tools/autodl_mincompat_v2.sh cscef_v52_scca_compat start-direct
```

Shared-host processes are isolated by variant run/init/session/lock paths. GPU allocation must be provided by the server/container; the launcher retains the fixed C2 device=0 argument and does not invent a different training recipe.

Round B: only after Round A supports compatibility, launch:

```bash
bash tools/autodl_mincompat_v2.sh triad_mincompat_v2 start-direct
```

For any one of those three variant IDs:

```bash
MINCOMPAT_VARIANT=cscef_v52_scca_compat
bash tools/autodl_mincompat_v2.sh "$MINCOMPAT_VARIANT" status
bash tools/autodl_mincompat_v2.sh "$MINCOMPAT_VARIANT" val
bash tools/autodl_mincompat_v2.sh "$MINCOMPAT_VARIANT" test
bash tools/autodl_mincompat_v2.sh "$MINCOMPAT_VARIANT" pack-complete
```

The mandatory order is train SUCCESS → val → test → pack. Test requires val of the identical checkpoint/source/config. Pack never implicitly trains or evaluates and refuses incomplete/altered evidence or existing destination files. Formal evaluation remains split=test, imgsz640, batch16, conf0.001, iou0.7, max_det300, halfFalse, augmentFalse, seed42; workers0 matches the established independent evaluator.

Artifacts: main `runs/c_series/mincompat_<variant>_rtdetr_r18_lite_e200_b16_onlineaug`; worktree `outputs/mincompat_v2/<variant>` for launch, preflight, common-weight mapping, gradient-edge evidence, source snapshot, parameter/config/environment records and separate val/test; complete archives at main `downloads/mincompat_v2/<variant>`. Each package includes training best/last, console and exit evidence, YAML/source/SHA, initialization/mapping, resolved and actual args, preflight validation, curves, confusion matrices, metrics, all-query/GT exports, MANIFEST and SHA256/inventory/verification sidecars.

Historical v1 variants are deprecated_for_performance_experiments and not accepted by this launcher. `c25_replay` is only a local regression configuration. No SCCA-v2, CSCEF-v6, stable-reference CBR, gamma scan or residual-scale experiment is launched by default.
