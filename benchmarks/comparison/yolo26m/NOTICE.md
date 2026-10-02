# YOLO26m scratch comparison provenance

Official architecture, end-to-end head, assignment and E2ELoss come from
[Ultralytics v8.4.0](https://github.com/ultralytics/ultralytics/tree/f2d3aed634a5b0e4828024718d4a61ab2f83fb19),
commit `f2d3aed634a5b0e4828024718d4a61ab2f83fb19`, AGPL-3.0.
The official LICENSE remains in the ignored vendor checkout and its digest is locked.
No pretrained asset, official source tree or dependency binary is committed.

The adapter reuses this repository's YOLOv8m scratch orchestration, light data checks,
isolated cache and public schema at `61c386bc722b11208453cab0808d8f6edb15385d`.
Model construction, validator/predictor contracts, loss schedule, AMP comparison and
resume state were adapted for v8.4.0. The HSV mapping derives from the mother source
at `a0459d6a652cb702699087c88fa39a3e4c4087ec` (AGPL-3.0).

The upstream patch preserves real local paths containing apostrophes and adds three
training-loop continuity hooks: restore the last optimization step, preserve verified
partial accumulated gradients, and record each optimization step. No architecture,
loss, assigner or postprocessing changes. Native torch_load already specifies
weights_only=False; the old v8 checkpoint-loading patch is not reused.

Checkpoint serialization follows the official BaseTrainer design with extra FP32
training weights, optimizer/scaler/scheduler, E2ELoss, early-stop and RNG/loader state.
Native implicit NaN recovery becomes an explicit failure, preventing loss-state resets.

Only locked overlay packages are installed. Compatible Torch/TorchVision and base
packages are inherited read-only, probed and recorded in each run's pip_freeze.txt.
Dependency licenses and provenance remain with their distributions and the vendor tree.
