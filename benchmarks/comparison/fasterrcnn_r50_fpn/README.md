# Faster R-CNN ResNet-50-FPN configurable

See `docs/comparison/FASTERRCNN_R50_FPN_CONFIGURABLE_SERVER_COMMANDS.md` for complete Chinese server commands.

Default: official COCO_V1 detection pretrained, replace the 91-class predictor with background0/crack1; standard FasterRCNN architecture, native anchors/RPN/ROIAlign/four losses, FrozenBatchNorm preserved. This project recipe uses 200 epochs, batch16/640, SGD0.02, five-epoch step warmup then cosine to 0.01 of lr0, no automatic LR batch scaling, no EMA and no default early stopping.

Standalone locked image/box augmentation reuses the executed YOLO26 comparison: stretch -> Mosaic -> RandomPerspective -> MixUp -> detection CutMix -> additive HSV -> flips -> RGB/255. There are no Ultralytics imports. Mixing closes in fresh workers at epoch195. Public FP32 evaluation uses fixed letterbox and one precise outer inverse, and the unchanged shared `corrected_sorted_conf_mask_v1` evaluator selects best only from val, with later epochs winning ties. Native COCOeval is auxiliary.

Commands: print-config, preflight, full, train --resume, export, evaluate, redraw, pack, summary. Defaults < YAML < --set; a prepared run is immutable. Configuration clones only recipes. Checkpoint/run/code/config/data/initialization/environment identities are checked. Actual configured training capacity is measured in an isolated subprocess before formal training, without global GPU locks or changing another experiment.

`test_core.py` contains focused regression tests. `smoke.py` is explicitly synthetic/local-only and uses 3 epochs, batch2/64 for a CUDA/AMP lifecycle check; these results must never enter the paper. Target torch2.1.2/vision0.16.2 and batch16/640 are checked on the server. Outputs, environments, caches and weights are ignored by Git.

`visualization.load_for_visualization(run_config_directory, best_checkpoint, device)` returns a strict recovered gradient-capable model plus RGB preprocessing, exact inverse and class map. No global no_grad/inference_mode is imposed. Final pack contains actual layer/shape/path handoff, metrics, predictions, GT, curves, source snapshots and SHA manifest, strictly below 100,000,000 bytes; weights remain on the server.

Source licenses and pinned versions are in NOTICE.md, LICENSE.cutmix.txt, upstream.lock.json and augmentation.lock.json.
