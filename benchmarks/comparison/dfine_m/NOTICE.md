# D-FINE-M project adapter

Official repository: https://github.com/Peterande/D-FINE (Apache-2.0), locked full commit in upstream.lock.json. Its native HGNetv2/HybridEncoder/DFINETransformer/HungarianMatcher/DFINECriterion/ModelEMA files execute unchanged in a private import namespace. The adapter skips eager imports for unrelated datasets/profilers/solvers; there is no architecture, FDR or GO-LSD patch.

Project differences: COCO-only strict class-count fine-tuning; YOLO26 reference image/box augmentation; fixed640 stretch train and letterbox public evaluation; explicit native0/public1 mapping; project LR/update schedule; continuous EMA without stage restarts; public FP32 val best selection and full resume.

The image augmentation excerpts and Bboxes/Instances/coordinate/IOA helpers derive from Ultralytics v8.3.20 (f4d8f7765a490f3920e2d14c592a2967e347f185), with detection CutMix from v8.4.0 (f2d3aed634a5b0e4828024718d4a61ab2f83fb19), under AGPL-3.0. They preserve the project a25e6379 reference computational classes; imports are narrowed to data-only components. Source/excerpt/class hashes are in augmentation.lock.json. MotherHSV and NoAlbumentations are the two isolated classes from the recorded project adapter. No Ultralytics package is imported. See LICENSE.augmentation.txt and LICENSE.D-FINE.txt.

The official README Model Zoo COCO M asset is 79,108,938 bytes, SHA256 b44a7586bf490858c7b8bce9e44bd025cb88724df9a07a8deb3ae1c12e608195. It contains model (no ema); 1042 compatible tensors load and 11 class-count keys initialize anew. The original official solver's broad mismatch handling is replaced by an exact class-only whitelist and strict final state load. This is fresh fine-tuning, never resume.

The private bootstrap lock and run locks belong only to this model. No GPU-wide lock, remote execution, global pip upgrade, other worktree changes, or process termination is performed. Server-specific paths, capacity and final metrics are observations written by each actual run, not asserted from development.
