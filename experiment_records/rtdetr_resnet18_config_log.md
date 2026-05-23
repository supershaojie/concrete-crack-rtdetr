# RT-DETR-ResNet18 Config Log

## Date
2026-05-24

## Purpose
Add a lightweight RT-DETR-ResNet18 baseline configuration in the Ultralytics RT-DETR framework.

## Changes
- Added `rtdetr-resnet18.yaml`
- Modified ResNetBlock to support BasicBlock behavior when expansion is 1
- Modified model parsing logic for ResNetLayer output channels
- Added a model construction smoke test script

## Test Result
- `model.info()` passed for RT-DETR-ResNet18
- `model.info()` passed for RT-DETR-ResNet50
- Full training test was not performed because the final `crack_det` dataset has not been generated yet.