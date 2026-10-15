"""Actual rounded letterbox geometry and full-map inverse transforms."""
from __future__ import annotations
import numpy as np


def geometry(width, height, kind="letterbox", size=640):
    if min(width, height) <= 0 or kind not in ("letterbox", "stretch"):
        raise ValueError("Invalid geometry")
    if kind == "stretch":
        rw = rh = size
    else:
        ratio = min(size / width, size / height)
        rw, rh = round(width * ratio), round(height * ratio)
    left, top = round((size - rw) / 2 - .1), round((size - rh) / 2 - .1)
    return dict(kind=kind, width=width, height=height, size=size, resized_width=rw,
                resized_height=rh, left=left, top=top, right=size-rw-left, bottom=size-rh-top,
                gain_x=rw/width, gain_y=rh/height, interpolation="OpenCV INTER_LINEAR")


def inverse_boxes(boxes, info):
    # This function accepts model-input coordinates only. Cache Results.xyxy never enters it.
    result = boxes.clone() if hasattr(boxes, "clone") else np.array(boxes, dtype=np.float64, copy=True)
    result[..., [0, 2]] = (result[..., [0, 2]] - info["left"]) / info["gain_x"]
    result[..., [1, 3]] = (result[..., [1, 3]] - info["top"]) / info["gain_y"]
    return result


def resize_float(array, size):
    from PIL import Image
    return np.asarray(Image.fromarray(np.asarray(array, dtype=np.float32)).resize(
        size, Image.Resampling.BILINEAR), dtype=np.float32)


def inverse_cam(native, info):
    # Align to full model input FIRST, then remove real integer padding once.
    square = resize_float(native, (info["size"], info["size"]))
    crop = square[info["top"]:info["top"]+info["resized_height"],
                  info["left"]:info["left"]+info["resized_width"]]
    if crop.shape != (info["resized_height"], info["resized_width"]):
        raise ValueError("Incorrect CAM padding inverse")
    return resize_float(crop, (info["width"], info["height"]))


def normalize_cam(cam, floor=1e-12):
    cam = np.asarray(cam, dtype=np.float32)
    if not np.isfinite(cam).all() or cam.min() < 0:
        raise ValueError("CAM must be finite and nonnegative")
    low, high = float(cam.min()), float(cam.max())
    if high <= floor or high - low <= floor:
        raise ValueError("Zero/near-zero or spatially constant CAM; no colored fallback")
    return (cam-low)/(high-low), dict(algorithm="positive-map min-max v1", minimum=low,
                                    maximum=high, floor=floor, epsilon_added=False)
