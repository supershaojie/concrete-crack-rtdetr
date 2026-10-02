# YOLOv13l comparison adapter

Official source: [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13), commit
73289949533efac82bb5f72ec19b746618656bd2, AGPL-3.0. Source and license are fetched
into an ignored vendor checkout; no weights or datasets are distributed.

The workflow derives from this project's verified YOLOv8m scratch implementation
at 61c386bc722b11208453cab0808d8f6edb15385d. Additive HSV derives from the AGPL-3.0
mother implementation at a0459d6a652cb702699087c88fa39a3e4c4087ec.
Checkpoint handling derives from the pinned official BaseTrainer and adds FP32
training state, optimizer, EMA, scaler, scheduler, RNG and comparison identity.

The three-file patch only controls trusted-local checkpoint loading, preserves
local paths containing apostrophes, and selects an existing attention branch.
Model equations, initialization, topology, loss, assigner, EMA update and gradient
clipping remain official.
