Torchvision standard Faster R-CNN ResNet50-FPN: BSD-3-Clause, reference v0.16.2 / c6f39778e636ec40a69bdbc74386818c57a65af3.

Detection transform class bodies originate from Ultralytics v8.3.20 (f4d8f7765a490f3920e2d14c592a2967e347f185). Detection CutMix comes from v8.4.0 (f2d3aed634a5b0e4828024718d4a61ab2f83fb19), preserved from the executed YOLO26 comparison at a25e6379be06970f1aa02fbb85d2c9a614a6436f. Standalone geometry helpers are extracted verbatim from the mother a0459d6a652cb702699087c88fa39a3e4c4087ec. These components are AGPL-3.0; see LICENSE.cutmix.txt and augmentation.lock.json. Imports are narrowed to local image/box helpers; no Ultralytics detector/trainer/optimizer/loss is imported.

Light data checks and checkpoint lifecycle derive from the scratch branch 4aeedd2ae6e527d83dcfca868b82b5652b4bbb3e. This experiment uses a distinct COCO configuration, runtime, identity and checkpoint format; historical scratch state is never reused.
