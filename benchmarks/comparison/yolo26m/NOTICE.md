# YOLO26m configurable COCO comparison

Implementation derives from YOLO11m at `a08b072a4608dfaa92e7f30a5b3dd0c1301d2a87`
and YOLOv8m at `a7b3d842df60adf28e5cf35c086c58ef752674f8`.
YOLO26m scratch at `6466680f8d1848b2817a6fe6a88764b1800179c6` supplies compatible
native interfaces and loss-state recovery.
Initialization is **official COCO detection pretrained**, never scratch or another run's best.
The committed first recipe is the planned `y26m_musgd_aug_x13_01`, copied from the user-supplied
`v8_aug_x13_01` values without another multiplication. It has not been formally trained.

The detector, dual branches, feature detach, assigners, E2ELoss, MuSGD and EMA are
official Ultralytics AGPL-3.0 v8.4.0 / `f2d3aed634a5b0e4828024718d4a61ab2f83fb19`.
`reg_max=1` uses image-normalized L1 in the `dfl_loss` slot, with gain 1.5.
Native one-to-one output is decoded `[B,N,6]`; confidence filtering and a 300-box
limit apply, with no IoU NMS. Training and public metrics are distinct.
The compatibility patch, source digest, license and actual official weight digest are in
`upstream.lock.json`. Loading is limited by the entry points to the checksum-verified
official asset or checksum-verified checkpoints belonging to the same frozen run.
The copied YAML avoids native nc assignment mutating the original COCO model's identity.

CutMix is the AGPL-3.0 excerpt from the locked upstream v8.4.0 commit, with its separate
license in `LICENSE.cutmix.txt` and attribution/hashes in `augmentation.lock.json`.
Its implementation has not been computationally modified. MotherHSV derives from the
RT-DETR mother source at a0459d6a652cb702699087c88fa39a3e4c4087ec.
`augmentation_reference.py` contains verbatim detection classes from official
v8.3.20 / `f4d8f7765a490f3920e2d14c592a2967e347f185`, preserving reference v8
numerical operations while running the newer model. Source/class hashes are locked.
Native CutMix stays disabled; the reference adapter owns it. No other project's
data or recipe is introduced. Each model/run has its own label/header cache.

The narrow source patch preserves local paths, identifies bias warmup by actual
parameter roles, and retains pending gradients and the optimizer-step index during
recovery. Native grouping and 3x LR matching remain intact. MuSGD keeps its fixed
muon=0.5/sgd=0.5 coefficients, Nesterov and native bfloat16 Newton-Schulz calculations.

Public AP uses `../evaluation/evaluate.py` unchanged and its `corrected_sorted_conf_mask_v1`
protocol hash. Native training metrics and final public metrics are kept separate.
Epoch-boundary resume restores FP32 live model, optimizer, EMA, scaler, scheduler,
E2ELoss schedule, both mixed momentum buffers, pending gradients, last-step index,
random streams, loader generators, early stopping, augmentation phase and counters.
Worker RNG, prefetch contents, sampler cursor and Mosaic buffers are not serialized;
bitwise continuation is not claimed. Public export checks FP32 model/input/attention
and disables TF32 and autocast, preserving padding-only false positives.

CPU and CUDA verification scripts use synthetic samples and explicitly identify
`SMOKE_ONLY`. The CUDA script uses 2 epochs, batch2, nbs2, 64 pixels, workers0 and a
smoke-only initial loss scale of 1 to exercise actual momentum-state updates in four
batches. This hook exists only in the verification script; the formal adapter retains
native AMP initial scaling/backoff. Preflight checks FP32 and AMP backward on a copy
using unit loss scaling and one native update on a copy, preserves RNG, and performs
no formal optimizer updates.
