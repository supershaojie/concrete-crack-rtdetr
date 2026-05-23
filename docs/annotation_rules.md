# Annotation Rules for Concrete Crack Detection

## Task Type

The current task is object detection for concrete cracks using bounding boxes.

## Class Definition

- `crack`: visible concrete crack regions.

## Annotation Rules

1. Only visible cracks are annotated.
2. Background textures, stains, shadows and water marks are not annotated as cracks.
3. For long continuous cracks, use one bounding box if the crack is visually continuous.
4. For separated cracks, annotate them as separate objects.
5. For crossed cracks, annotate according to visible crack instances and keep the box tight.
6. Extremely unclear cracks should be reviewed before annotation.

## Notes

The annotation rules may be updated during dataset cleaning.