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

## Lightweight experiment preflight

Future experiment preflights should check the changed path with one fixed small input first, then run one bounded native train/save/resume/validation check. For BLC, start and resume share a maximum of 16 training batches at the original B16/640/AMP settings; native half EMA validation consumes one real batch. Keep required checks and their original tolerances, record failures and unrun stages, and do not retry automatically. Test checkpoints belong in owned temporary directories and are removed on success or catchable interruption. Retain small JSON/logs, targeting at most 10 MiB per invocation; preserve all prior results and formal weights. See [the BLC implementation and evidence](docs/blc_v1/light_preflight_fix/README.md).
