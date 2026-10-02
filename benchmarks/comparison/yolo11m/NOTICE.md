# Source and license

Official source: https://github.com/ultralytics/ultralytics

Frozen implementation: v8.3.20, commit `f4d8f7765a490f3920e2d14c592a2967e347f185`.
The upstream AGPL-3.0 license remains in the downloaded source tree's `LICENSE`.
Pinned license source: https://github.com/ultralytics/ultralytics/blob/f4d8f7765a490f3920e2d14c592a2967e347f185/LICENSE

The trainer checkpoint serializer follows that official implementation. The additive HSV
formula follows this project's mother source at `a0459d6a652cb702699087c88fa39a3e4c4087ec`.
These derived adapters and the compatibility patch retain the upstream AGPL-3.0 terms.
The model architecture, detection loss, DFL and task-aligned assigner are the pinned upstream code.

YOLO11m uses the official YAML with scale=m and nc=1, with no external weights.
No weights or third-party source checkout are committed. Bootstrap verifies the upstream
commit, source content, license and the two compatibility edits in the patch.
The patch only changes explicit trusted-checkpoint loading and apostrophes in local paths.
It does not change C3k2, C2PSA, Detect.legacy=False, loss or TaskAlignedAssigner.
