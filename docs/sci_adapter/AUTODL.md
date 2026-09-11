# Fixed-source AutoDL workflow

Base: `33535ab2a5d9acee4e62d34ae382df8b98b9bbde`.
C2: `67c3078e54a657fd96d65fee657a75fbb1dae0d6`.
Branch: `codex/rtdetr-sci-adapter`. Use the full release SHA reported with the commit.

```bash
export SCI_SHA=FULL_40_CHARACTER_RELEASE_SHA
export SCI_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
export SCI_WT=/root/autodl-tmp/projects/Crack_RTDETR-sci-adapter
git -C "$SCI_MAIN" fetch origin codex/rtdetr-sci-adapter
SCI_SYNC=$(mktemp /tmp/sync_sci_adapter.XXXXXX.sh)
git -C "$SCI_MAIN" show "$SCI_SHA:tools/sync_sci_adapter.sh" > "$SCI_SYNC"
bash "$SCI_SYNC" "$SCI_SHA" "$SCI_MAIN" "$SCI_WT"
cd "$SCI_WT"
test "$(git rev-parse HEAD)" = "$SCI_SHA"
```

The sync creates a separate detached worktree or reuses the same clean detached SHA.
It preserves dirty, attached or different-SHA directories and never reset/clean/stash.
The pin file must match HEAD; a later branch update cannot change this launch source.

Round A, two physical GPUs, each exposed as logical device 0:

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/autodl_sci_adapter.sh cscef_v51_sci_control start-direct
CUDA_VISIBLE_DEVICES=1 bash tools/autodl_sci_adapter.sh scca_sci_cscef_v51 start-direct
```

If using separate one-GPU instances, run each command there with CUDA_VISIBLE_DEVICES=0.
Run Round B only after interpreting Round A, and only on a free GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/autodl_sci_adapter.sh scca_sci_cscef_v51_cbr start-direct
```

Start requires the existing rtdetr conda environment, PyTorch 2.1.2/CUDA 12.1, CUDA,
correct local ultralytics path, clean fixed SHA, protected historical source audit,
exact topology/count/original module classes, SHA-locked public initialization,
zero heads, 109-field typed C2 recipe, dataset configuration, gradient audit, AMP
native loss/DN, true half, save/load and optimizer coverage. No package upgrade occurs.
Distinct variants have separate run, init, launch, lock and tmux paths. Dispatch follows
successful preflight. Existing artifacts, mismatched source, duplicate workers and
reduced-batch OOM retries are rejected. Worker inherits CUDA_VISIBLE_DEVICES.

```bash
V=cscef_v51_sci_control # or scca_sci_cscef_v51 / scca_sci_cscef_v51_cbr
bash tools/autodl_sci_adapter.sh "$V" status
CUDA_VISIBLE_DEVICES=0 bash tools/autodl_sci_adapter.sh "$V" val
CUDA_VISIBLE_DEVICES=0 bash tools/autodl_sci_adapter.sh "$V" test
bash tools/autodl_sci_adapter.sh "$V" pack-complete
```

Training recipe: epochs=200, patience=50, batch=16, imgsz=640, workers=8, device=0,
AdamW lr0=.0005 lrf=.01 weight_decay=.0001 warmup_epochs=5 cos_lr=True amp=True,
seed=42 deterministic=True close_mosaic=10. All augmentation fields come from the
archived real C2 args; only model/name/save_dir change.

Independent val/test use imgsz=640,batch=16,conf=.001,iou=.7,max_det=300,half=False,
augment=False,seed=42. Test requires completed val of the same best checkpoint, data,
source and settings. The established corrected sorted-confidence policy and all 300
final-query/GT export are retained. Pack never implicitly evaluates or trains; it
requires completed successful training+val+test, all curves/confusion matrices,
initialization/mapping/config/environment/source/console/SCI statistics. Archive
manifest, per-member hashes, SHA256, inventory and bytewise read-back verification
are emitted under MAIN/downloads/sci_adapter/<variant>/.

No formal training or complete dataset val/test was executed in this implementation.
