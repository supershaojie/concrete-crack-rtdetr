"""Read-only activation probe of the first 128 real C18 val images; no metrics, training or interventions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from init_rtdetr_r18_lite_gsdr_aifi_v3_controlled import ROOT, require, sha256, verify_module
from ultralytics import RTDETR
from ultralytics.utils import YAML


def val_images(data_path):
    """Read image directories/lists without constructing a label dataset or writing caches."""
    config = YAML.load(data_path)
    root = Path(config.get("path", data_path.parent))
    if not root.is_absolute():
        root = data_path.parent / root
    entries = config["val"]
    entries = entries if isinstance(entries, list) else [entries]
    images = []
    formats = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    for entry in entries:
        path = Path(entry)
        path = path if path.is_absolute() else root / path
        if path.is_dir():
            images.extend(p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in formats)
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = Path(line.strip())
                images.append((item if item.is_absolute() else path.parent / item).resolve())
        else:
            raise FileNotFoundError(f"Val image source unavailable: {path}")
    require(images and all(p.is_file() for p in images), "Val image files are missing.")
    return sorted(set(images), key=lambda p: p.as_posix())


def activation_statistics(x, delta, state):
    """Return per-image measures; undefined zero-energy ratios are null, never evidence of nondegeneracy."""
    require(x.shape[0] == delta.shape[0] == 1, "Diagnostic processes exactly one real image per forward.")
    x, delta = x.detach().float(), delta.detach().float()
    pixels, query, weights = (state[k].detach().float() for k in ("pixels", "query", "weights"))
    height, width = delta.shape[-2:]
    energy = delta.square().mean().item()
    input_energy = x.square().mean().item()
    displacement = pixels[0] - query[:, None, None]
    l2 = displacement.square().sum(-1).sqrt()
    size = pixels.new_tensor((width, height))
    outside = ((pixels < -1e-5) | (pixels > size-1+1e-5)).any(-1)
    entropy = -(weights * weights.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1) / math.log(4)
    return {
        "feature_shape": list(x.shape), "branch_rms": math.sqrt(energy), "input_rms": math.sqrt(input_energy),
        "branch_rms_over_input_rms": math.sqrt(energy/input_energy) if input_energy else None,
        "spatial_constant_energy_fraction": delta.mean((-2, -1)).square().mean().item()/energy if energy else None,
        "zero_delta": energy == 0, "normalized_point_weight_entropy": entropy.mean().item(),
        "outside_valid_pixel_centers_fraction": outside.float().mean().item(), "domain_tolerance_cells": 1e-5,
        "displacement_mean_abs_xy_cells": displacement.abs().mean((0, 1, 2)).tolist(),
        "displacement_max_abs_axis_cells": displacement.abs().max().item(),
        "displacement_mean_l2_cells": l2.mean().item(),
        "per_query": [{"xy": query[q].tolist(), "mean_signed_xy_cells": displacement[q].mean((0, 1)).tolist(),
                       "mean_abs_xy_cells": displacement[q].abs().mean((0, 1)).tolist(),
                       "mean_l2_cells": l2[q].mean().item()} for q in range(height*width)],
    }


def diagnose(weights, data, output, device, limit):
    require(1 <= limit <= 128, "--limit must be in [1, 128].")
    require(not output.exists(), f"Refusing diagnostic report overwrite: {output}")
    before = sha256(weights)
    data_hash = sha256(data)
    files = val_images(data)
    require(len(files) >= limit, f"Only {len(files)} val images available; requested {limit}.")
    if device != "cpu":
        require(torch.cuda.is_available(), "Requested CUDA is unavailable.")
    model = RTDETR(str(weights)).model.float().to(device).eval()
    verify_module(model, require_zero=False)
    relation = model.model[9].sparse_relation
    zero_boundary = torch.count_nonzero(relation.output_proj.weight).item() == 0
    rows = []

    def observe(module, inputs, delta):
        # Recompute only the local sampling state, under inference_mode, without storing training activations.
        _, state = module.relation_features(inputs[0])
        rows.append(activation_statistics(inputs[0], delta, state))

    handle = relation.register_forward_hook(observe)
    try:
        with torch.inference_mode():
            for path in files[:limit]:
                image = cv2.imread(str(path))
                require(image is not None, f"Cannot read {path}")
                # Matches RTDETRDataset.load_image(rect_mode=False): direct INTER_LINEAR stretch, RGB / 255.
                image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
                tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)))
                count = len(rows)
                model(tensor.unsqueeze(0).to(device=device, dtype=torch.float32) / 255)
                require(len(rows) == count + 1, "Expected exactly one local-branch call per real image.")
                rows[-1]["image"] = str(path)
                print(f"Probed {len(rows)}/{limit}: {path.name}", flush=True)
    finally:
        handle.remove()
    require(sha256(weights) == before and sha256(data) == data_hash, "Checkpoint/data config changed during read-only probe.")
    scalar_fields = ("branch_rms", "input_rms", "branch_rms_over_input_rms", "spatial_constant_energy_fraction",
                     "normalized_point_weight_entropy", "outside_valid_pixel_centers_fraction", "displacement_mean_l2_cells")
    summary = {}
    for key in scalar_fields:
        values = [row[key] for row in rows if row[key] is not None]
        summary[key] = {"mean": float(np.mean(values)) if values else None, "defined_images": len(values)}
    report = {"status": "completed", "weights": str(weights.resolve()), "weights_sha256": before,
              "data": str(data.resolve()), "data_sha256": data_hash, "split": "val", "available_val_images": len(files),
              "real_images": len(rows), "warmup_images": 0, "dtype": "FP32", "device": device,
              "zero_output_projection": zero_boundary, "summary": summary, "per_image": rows,
              "note": "Activation observations only; no mAP, retraining, intervention, threshold-driven change, or claim of training benefit. "
                      "Spatial constancy is within each image; null fractions mean zero delta, not proof against degeneration. "
                      "Radius is per axis; a 2x2 displacement may have L2 up to sqrt(8) cells. At 640 each P5 cell is 32 input pixels."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Read-only diagnostic report: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Future trained C18 best.pt; never overwritten.")
    parser.add_argument("--data", type=Path, default=Path("/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/gsdr_aifi_v3/val_probe_128.json")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu; diagnostics use FP32.")
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    diagnose(args.weights, args.data, args.output, args.device, args.limit)


if __name__ == "__main__":
    main()
