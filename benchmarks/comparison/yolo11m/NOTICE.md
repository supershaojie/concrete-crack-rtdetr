# YOLO11m configurable COCO comparison

Implementation derives from the verified YOLOv8m configurable lifecycle at
`a7b3d842df60adf28e5cf35c086c58ef752674f8`. The scratch YOLO11m implementation at
`b18ae4e97b4a4bc9e17b0f4543705cf10be26d95` supplies structural cross-checks only.
Initialization is **official COCO detection pretrained**, never scratch or another run's best.
The committed first recipe is the planned `y11_aug_x13_01`, copied from the user-supplied
`v8_aug_x13_01` values without another multiplication. It has not been formally trained.

The detector, nonlegacy depthwise Detect classification branches, `v8DetectionLoss`,
TaskAlignedAssigner, DFL, native optimizer/warmup/accumulation/scheduler and EMA remain
the pinned official Ultralytics implementation at v8.3.20 / f4d8f7765a490f3920e2d14c592a2967e347f185.
The compatibility patch, source digest, license and actual official weight digest are in
`upstream.lock.json`. Loading is limited by the entry points to the checksum-verified
official asset or checksum-verified checkpoints belonging to the same frozen run.
The copied YAML avoids native nc assignment mutating the original COCO model's identity.

CutMix is the AGPL-3.0 excerpt from the locked upstream v8.4.0 commit, with its separate
license in `LICENSE.cutmix.txt` and attribution/hashes in `augmentation.lock.json`.
Its implementation has not been computationally modified. MotherHSV derives from the
RT-DETR mother source at a0459d6a652cb702699087c88fa39a3e4c4087ec.
The original `yolov8m_header_labels_v1` cache format identifier is deliberately retained
to permit verified read-only v8 label inventory reuse; it does not identify the model.
Patch context containing upstream example model names is retained to keep the patch hash intact.

Public AP uses `../evaluation/evaluate.py` unchanged and its `corrected_sorted_conf_mask_v1`
protocol hash. Native training metrics and final public metrics are kept separate.
Epoch-boundary resume restores FP32 live model, optimizer, EMA, scaler, scheduler,
random streams, early-stopping selection, closed augmentation and shared counters.
Prefetched worker batches and cross-epoch pending gradients are not replayed bit for bit.

CPU and CUDA verification scripts use synthetic samples and explicitly identify
`SMOKE_ONLY`. The CUDA script uses 2 epochs, batch2, nbs2, 64 pixels, workers0 and a
smoke-only initial loss scale of 1 to exercise actual momentum-state updates in four
batches. This hook exists only in the verification script; the formal adapter retains
native AMP initial scaling/backoff. Preflight checks FP32 and AMP backward on a copy
using unit loss scaling, preserves RNG, and performs no formal optimizer updates.
