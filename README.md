# Concrete Crack Detection with RT-DETR

This project focuses on concrete crack detection using the Ultralytics implementation of RT-DETR.

## Current Stage

- Self-built concrete crack dataset: 1440 images
- Annotation in progress
- Planned offline augmentation to approximately 10000 images
- Baseline model: RT-DETR from Ultralytics

## Project Structure

- `ultralytics-main/`: Ultralytics RT-DETR source code
- `tools/`: dataset checking, preprocessing, splitting, augmentation and visualization scripts
- `configs/`: dataset and model configuration files
- `docs/`: annotation rules and dataset processing notes
- `experiment_records/`: experiment logs and result records
- `datasets/`: local dataset directory, not tracked by Git
- `logs/`: local running logs, not tracked by Git

## Notes

Large files such as datasets, training outputs, logs and model weights are not tracked by Git.

## Chapter 4 visualization

The isolated eight-model detection comparison and Grad-CAM++ pipeline is documented in
[docs/chapter4_visualization.md](docs/chapter4_visualization.md), including server checkout,
CPU preflight, tmux, resumable caches, visual case review and a strictly sub-100MB review ZIP.
The default matrix uses the same five images as columns, Faster R-CNN first and ours last.
