# Dataset Processing Notes

## Dataset Overview

- Original dataset size: 1440 images
- Task: concrete crack detection
- Annotation format: YOLO detection format
- Planned augmented dataset size: approximately 10000 images

## Processing Pipeline

1. Collect raw images.
2. Rename images.
3. Check image-label pairs.
4. Select valid images.
5. Annotate cracks.
6. Split dataset into train/val/test sets.
7. Apply offline data augmentation.
8. Visualize labels for quality checking.

## Data Storage

The dataset is stored locally under `datasets/` and is not tracked by Git.