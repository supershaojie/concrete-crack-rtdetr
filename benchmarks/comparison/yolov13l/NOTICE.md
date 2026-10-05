# Sources and license

Model, native attention, loss, assigner and trainer are the official iMoonLab
YOLOv13 source at `73289949533efac82bb5f72ec19b746618656bd2` (AGPL-3.0).
The checked-out author license stays in `.vendor/.../yolov13/LICENSE`.
Source/patch/YAML/license and original COCO weight identities are checked by
`upstream.lock.json`; no upstream checkout, environment, data or .pt is committed.

The native backend selector, trusted-local checkpoint compatibility patch and
ArithmeticAudit derive from this repository's YOLOv13-L scratch model code
`38b49c547ad1fd93d7b85ef9917fa812b1568eeb`. Scratch loading prohibitions,
scratch initialization, old augmentation and automatic test were not retained.

Configuration/trainer/data/export lifecycle helpers derive from YOLOv8m
`a7b3d842df60adf28e5cf35c086c58ef752674f8` and the verified configuration
extensions in YOLO11m `07bb47b03f5066f72ab4e2e1baa9cf5209f43942`.
Only helpers are reused; actual model identity/transfer/precision are measured
for this author's YOLOv13-L and never reuse another architecture's counts.
Checkpoint serialization preserves the pinned author's optimizer and EMA
semantics, with separate FP32 state for resume/selected EMA evaluation.
MotherHSV derives from the mother source at
`a0459d6a652cb702699087c88fa39a3e4c4087ec`. Derived adapters retain AGPL-3.0.

`cutmix_reference.py` is the same computational excerpt used in the v8 reference,
from Ultralytics v8.4.0 `f2d3aed634a5b0e4828024718d4a61ab2f83fb19`,
`ultralytics/data/augment.py`. Source/excerpt/license hashes are in
`augmentation.lock.json`, with AGPL-3.0 text in `LICENSE.cutmix.txt`.
Only the bounded train partner/Instances interface, counting and integration
are adapted; no newer model, loss or optimizer implementation is imported.

Primary references:

- [Pinned author source](https://github.com/iMoonLab/yolov13/tree/73289949533efac82bb5f72ec19b746618656bd2)
- [Official COCO asset](https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13l.pt)
- [CutMix source](https://github.com/ultralytics/ultralytics/blob/f2d3aed634a5b0e4828024718d4a61ab2f83fb19/ultralytics/data/augment.py)
- [CutMix license](https://github.com/ultralytics/ultralytics/blob/f2d3aed634a5b0e4828024718d4a61ab2f83fb19/LICENSE)
