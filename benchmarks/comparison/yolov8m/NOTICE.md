# Source and license

Official source: https://github.com/ultralytics/ultralytics

Frozen implementation: v8.3.20, commit `f4d8f7765a490f3920e2d14c592a2967e347f185`.
The upstream AGPL-3.0 license remains in the downloaded source tree's `LICENSE`.
Pinned license source: https://github.com/ultralytics/ultralytics/blob/f4d8f7765a490f3920e2d14c592a2967e347f185/LICENSE

The trainer checkpoint serializer follows that official implementation. The additive HSV
formula follows this project's mother source at `a0459d6a652cb702699087c88fa39a3e4c4087ec`.
These derived adapters and the compatibility patch retain the upstream AGPL-3.0 terms.
The model architecture, detection loss, DFL and task-aligned assigner are the pinned upstream code.

COCO initialization is the official detection asset in `upstream.lock.json`. No weights or
third-party source checkout are committed to this project. Bootstrap preserves the official
license and verifies the fixed commit, source contents, patch and weight checksum.

`cutmix_reference.py` contains the three detection CutMix methods derived solely from
Ultralytics v8.4.0 commit `f2d3aed634a5b0e4828024718d4a61ab2f83fb19`,
`ultralytics/data/augment.py` (full file SHA256 in `augmentation.lock.json`).
The class name/imports and removed documentation are the only excerpt changes.
`augment_b19.py` adapts the old Instances/partner interfaces and adds worker-shared
counters without changing those methods. This derived code remains AGPL-3.0:
[fixed source](https://github.com/ultralytics/ultralytics/blob/f2d3aed634a5b0e4828024718d4a61ab2f83fb19/ultralytics/data/augment.py),
[fixed license](https://github.com/ultralytics/ultralytics/blob/f2d3aed634a5b0e4828024718d4a61ab2f83fb19/LICENSE).
No newer model, loss, assigner, optimizer or whole augmentation module is imported.
