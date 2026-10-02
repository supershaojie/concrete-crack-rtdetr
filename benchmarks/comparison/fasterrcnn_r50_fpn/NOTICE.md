# Sources and scope

Detector: official TorchVision v0.16.2, commit
`c6f39778e636ec40a69bdbc74386818c57a65af3`, BSD-3-Clause.
The compatible official torch 2.1.2 / torchvision 0.16.2 wheel pair is used;
bootstrap keeps a clean upstream checkout and compares installed Python sources
against this commit. Wheel metadata, binary extension SHA256, package versions,
CUDA build and successful native CUDA operator probes are recorded. No detector
source, head, loss, anchors, proposal sampler or initialization is patched.

Data operations: official Ultralytics v8.3.20, commit
`f4d8f7765a490f3920e2d14c592a2967e347f185`, AGPL-3.0.
`data_adapter.py` uses its detection augmentation components; no Ultralytics
detector/trainer/predictor/loss is instantiated. Its package initializer imports
model class definitions but no model instance or weights. MotherHSV, source
protection and cache logic derive from the project's frozen YOLOv8m adapter at
`61c386bc722b11208453cab0808d8f6edb15385d` and mother source at
`a0459d6a652cb702699087c88fa39a3e4c4087ec` (AGPL-3.0-derived portions).
The upstream license files remain in each `.vendor` checkout. The upstream
package itself is unmodified; changes are explicit adapter classes.

`data.py` preserves the frozen light-check algorithm and public GT rules from
that reference adapter. `stage.py` preserves its subprocess/tee supervision.
The shared evaluator `corrected_sorted_conf_mask_v1` is called directly, without
a second AP implementation. The copied helpers are independent of the YOLO model.

All source text identities normalize CRLF to LF. The Ultralytics license's
normalized SHA256 is `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0`.
The old reference lock's `e0eedba6...` is the same license with Windows CRLF;
using that raw hash on Linux would incorrectly reject the official checkout.

No model weights, data, environments, vendor repositories or full prediction
caches are committed. Distribution/use of the AGPL-derived data adapter must
retain the applicable source/license obligations.
