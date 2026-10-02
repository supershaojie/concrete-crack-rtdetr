# Faster R-CNN (ResNet-50-FPN), random initialization

This adapter is independent of the existing YOLO trainers. The detector is the
standard TorchVision `fasterrcnn_resnet50_fpn`, pinned to v0.16.2 / commit
`c6f39778e636ec40a69bdbc74386818c57a65af3`, paired with torch 2.1.2. Both
`weights=None` and `weights_backbone=None` are explicit. Construction intercepts
weight downloads, `torch.load` and `Module.load_state_dict`; native trainable BN,
all backbone parameters, FPN/RPN/RoIAlign/TwoMLPHead/FastRCNNPredictor are retained.
See [NOTICE.md](NOTICE.md) for sources, licenses and adapter patches.

The branch starts from the frozen YOLOv8m-scratch reference
`61c386bc722b11208453cab0808d8f6edb15385d`. It shares the light data identity and
public evaluator, without changing their implementation. `data.py` reads the
real YAML and checks paths, image byte sizes and label contents against the
existing frozen identity; only necessary image headers are read. It never
claims a full image-content audit. No train COCO conversion or source repair is
performed. All label/size caches and unused image-cache paths are run-local.

## Model input and data operations

Training uses the pinned Ultralytics v8.3.20 **data** pipeline matching the frozen
adapter: square stretch, native Mosaic, RandomPerspective, native MixUp,
additive mother HSV, vertical flip, horizontal flip, then RGB CHW float /255.
CopyPaste, Albumentations, CutMix and classification augmentations are disabled.
The native MixUp Beta(32,32) image blend concatenates detection boxes. Workers
report actual Mosaic/MixUp calls; fresh nonpersistent worker copies each epoch
make the zero-based epoch-190 close boundary effective. Empty targets are legal.

Source YOLO class 0 becomes TorchVision foreground label 1; foreground 1 becomes
public category 1 directly. Targets contain float32 pixel xyxy boxes, int64
labels/image_id/iscrowd, and area. Batches are image/target lists.

Validation and final export use the same fixed 640 square letterbox (scaleup
enabled, integer centered padding, no auto stride padding). Each record stores
original size, rounded resized size, per-axis actual scale and integer padding.
TorchVision applies its own ImageNet mean/std exactly once. A backbone pre-hook
rejects any hidden input enlargement. The native model output is inverted once
through the data letterbox, using those actual scales. No extra clipping,
rounding or NMS occurs after this inverse; finite positive boxes outside original
bounds are retained as original-pixel coordinates, as accepted by the public
schema. This is disclosed rather than assuming YOLO's postprocessing is identical.

RPN train 2000/test 1000 proposal caps and NMS 0.7 remain native. Final ROI score
threshold 0.001, class-aware NMS 0.7 and max300 operate **inside** the detector.

## Training and numerical recipe

`recipe.yaml` is the expanded, consumed recipe. SGD has one parameter group, as
the official reference with `norm_weight_decay=None`: BN and biases also decay
at 1e-4. Real batch16, one update per batch, momentum .9, LR .02, no Nesterov,
EMA, clipping or YOLO nbs scaling. Native classifier/box/objectness/RPN-box losses
are summed and logged individually. Nonfinite loss or unscaled gradient fails
the run before stepping; no skipped batches are counted as successful steps.

At 6048 images there are 378 optimizer steps per epoch. For zero-based step s,
steps 0..1889 linearly interpolate .00002 to .02; steps 1890..75599 follow cosine
from .02 to .0002. Both phase endpoints are explicit, with no additional native
first-epoch warmup. The saved step count resumes this same planned schedule.

**Pre-run numerical amendment, 2026-10-03:** the local torch 2.7.1 / torchvision
0.22.1 synthetic probe had finite FP32 gradients (maximum about 7.80) but
nonfinite gradients with the default AMP loss scale 65536, despite four finite
losses. Before any formal training or model ranking, the recipe explicitly fixes
`amp_init_scale=128` and `amp_growth_interval=1000000000`. Thus the scale does not
grow during the 75,600-step budget; there is no automatic failed-batch skip or
adaptive recipe change. Scale128 passed FP32/AMP forward and backward checks.
The target 2.1.2/0.16.2 runtime must repeat the disposable preflight on the server.

Determinism uses seed42, deterministic cuDNN and
`torch.use_deterministic_algorithms(True, warn_only=True)`. Native CUDA RoIAlign
can warn about nondeterministic backward; it is not replaced. Bitwise replay is
not promised. Preflight uses a different process and model; formal training
re-seeds and rebuilds model, optimizer, scaler and loader states.

Every epoch exports real FP32 val predictions in memory to
`corrected_sorted_conf_mask_v1`. Unrounded mAP50–95 selects best; `>=` updates the
best epoch even on equality and resets patience50. Epoch metrics and state are
retained, without 200 repeated full prediction caches. Test is used only after
selection finishes. Final val/test use the same saved best state_dict and save
full precision JSONL.GZ records, including explicit empty-image records.

## State, environment and outputs

Bootstrap creates `.envs/fasterrcnn-r50-fpn-scratch` with `--copies`; it can
read-only inherit a compatible base torch/vision, otherwise installs the pinned
CUDA11.8 wheel pair into a fresh private environment. Python3.9–3.11 is required.
`sys.executable` is made absolute without following its symlink; `sys.prefix`,
the recorded interpreter and every subprocess must agree. Pinned clean upstream
trees, normalized source hashes, installed wheel Python source correspondence,
binary extension SHA, wheel build metadata and native CUDA NMS/RoIAlign are checked.

Runtime output paths are under this worktree. A run UUID and identities bind
model, random initialization, recipe, code, dataset, upstream sources and runtime
ABI. New training never auto-resumes. `--resume` accepts only this run's own
atomic last/best files with matching index hashes, optimizer/scaler/scheduler,
completed/best epochs, patience and RNG/loader-generator states. Original
initialization receipts are not overwritten. Mid-epoch work is replayed from
the last completed checkpoint. An interruption between separate best/last/index
atomic writes is detected and requires inspection; files are not silently repaired.

Main artifacts: `manifest.json`, `input_checksums.json`, `run_id.json`,
`preflight_model.json`, `identity.json`, `environment.json`, `initialization.json`,
`expanded_recipe.json`, `augmentation.json`, `actual_training_setup.json`,
`epoch_trace.jsonl`, `training_progress.json`, `epoch_metrics/`, `checkpoints/`,
`predictions/`, `metrics/`, stage statuses and `summary.json`.

`measure` reports actual parameters and observed proposal counts on a single
640 FP32 input. Torch profiler's counted FLOPs are explicitly **partial**;
unaccounted/zero-flop operators are listed and total GFLOPs remains null.
Training and deployment use the same unfused architecture. `speed` is a separate
opt-in exclusive-GPU measurement; it is never a training prerequisite.

Local checks cover real worker augmentation/closure, class/coordinate mapping,
shared-cache protection, schedule boundaries, forbidden checkpoint identities,
child/tee error codes, own-session/run protection and tmux gate ordering using a
mock. The synthetic GPU lifecycle trains, deliberately interrupts, resumes and
exports/evaluates the actual best. These are not paper metrics or formal capacity
tests. Real Linux tmux/process signals, server target runtime, formal data
training and concurrent batch16/640 memory capacity remain server checks.

RT-DETR reference initialization is ImageNet backbone; this experiment is random.
The comparison does not claim all models have identical initialization.
