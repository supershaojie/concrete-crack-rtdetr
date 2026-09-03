"""Run read-only, post-training validation diagnostics for one CSCEF-v2 checkpoint.

The tool deliberately reloads the checkpoint for every intervention, never saves a model,
and accepts only the validation split. Quantiles are estimated from bounded deterministic
samples while moments and threshold fractions are accumulated over every observed value.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gc
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import random
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from types import MethodType
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "ultralytics-main"
VARIANTS = (
    "original",
    "layer_scale_zero",
    "gate_const_05",
    "gate_const_1",
    "shared_norm_bias_zero",
    "edge_norm_bias_zero",
    "both_norm_bias_zero",
    "raw_cosine_gate",
)
METRIC_FIELDS = (
    "variant",
    "precision",
    "recall",
    "mAP50",
    "mAP50-95",
    "fitness",
    "weights_sha256",
    "data_sha256",
    "elapsed_seconds",
    "status",
    "error",
)
FULL_QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
RATIO_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
POST_TRAINING_WARNING = (
    "这些干预是在训练完成后的分布外修改，只能用于原因诊断，不能作为正式消融结果或论文性能结果。"
)
CUDA_CONTEXT_ERROR_MARKERS = (
    "device-side assert",
    "indexkernel.cu",
    "illegal memory access",
    "unspecified launch failure",
)


class InvalidBatchIndexError(ValueError):
    """Raised after invalid GT batch indices are detected safely on CPU."""


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest without changing the file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    """Serialize paths and NumPy scalar values in manifests."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, value: Any) -> None:
    """Write stable, human-readable JSON."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def parse_variants(value: str) -> list[str]:
    """Parse ``all`` or a comma-separated, duplicate-free variant list."""
    names = list(VARIANTS) if value.strip().lower() == "all" else [item.strip() for item in value.split(",")]
    if not names or any(not item for item in names):
        raise ValueError("--variants must be 'all' or a non-empty comma-separated list.")
    unknown = [item for item in names if item not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown diagnostic variants: {unknown}. Valid variants: {list(VARIANTS)}")
    if len(set(names)) != len(names):
        raise ValueError("--variants must not contain duplicates.")
    return names


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without importing Ultralytics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="C10 best.pt checkpoint (read only).")
    parser.add_argument("--data", type=Path, required=True, help="Detection dataset YAML.")
    parser.add_argument("--split", default="val", help="Dataset split; only 'val' is accepted.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, required=True, help="Parent directory for diagnostic runs.")
    parser.add_argument("--name", default="c10_cscef_v2_val_diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stats-max-batches",
        type=int,
        default=0,
        help="Activation-statistics batch limit; 0 means the complete validation set.",
    )
    parser.add_argument("--variants", default="all", help="Comma-separated variants or 'all'.")
    parser.add_argument("--plots", action="store_true", help="Create compact diagnostic plots at 180 DPI.")
    parser.add_argument("--overwrite", action="store_true", help="Replace this exact named diagnostic directory.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Reject unsafe or invalid inputs before loading a checkpoint."""
    if str(args.split).lower() != "val":
        raise ValueError("Only --split val is permitted; test/train splits are rejected for model diagnosis.")
    if not Path(args.weights).is_file():
        raise FileNotFoundError(f"Weights file does not exist: {Path(args.weights).resolve()}")
    if not Path(args.data).is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {Path(args.data).resolve()}")
    if args.imgsz <= 0 or args.batch <= 0:
        raise ValueError("--imgsz and --batch must be positive.")
    if args.workers < 0 or args.stats_max_batches < 0:
        raise ValueError("--workers and --stats-max-batches must be non-negative.")
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("--name must be one non-empty directory name, not a path.")
    parse_variants(args.variants)


def prepare_output_directory(project: Path, name: str, overwrite: bool) -> Path:
    """Create one exact run directory, replacing it only after explicit opt-in."""
    project = project.expanduser().resolve()
    output = (project / name).resolve()
    if output.parent != project:
        raise ValueError("Diagnostic output must be a direct child of --project.")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists (use --overwrite explicitly): {output}")
        if output == Path(output.anchor) or output == project:
            raise RuntimeError(f"Refusing to remove broad output path: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output


def find_unique_cscef_v2(model: nn.Module) -> nn.Module:
    """Find exactly one CSCEFv2 by reliable class identity/name, never by layer index."""
    matches = [module for module in model.modules() if module.__class__.__name__ == "CSCEFv2"]
    if not matches:
        raise RuntimeError("Checkpoint model contains no CSCEFv2 module.")
    if len(matches) != 1:
        raise RuntimeError(f"Checkpoint model contains {len(matches)} CSCEFv2 modules; expected exactly one.")
    return matches[0]


def count_cscef_v2_modules(model: nn.Module) -> int:
    """Count CSCEFv2 instances without assuming a graph-layer index."""
    return sum(module.__class__.__name__ == "CSCEFv2" for module in model.modules())


def find_unique_edge_group_norm(module: nn.Module) -> nn.GroupNorm:
    """Locate the sole edge-calibration GroupNorm by type."""
    edge_calibration = getattr(module, "edge_calibration", None)
    if not isinstance(edge_calibration, nn.Module):
        raise RuntimeError("CSCEFv2 has no edge_calibration module.")
    matches = [item for item in edge_calibration.modules() if isinstance(item, nn.GroupNorm)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one edge-calibration GroupNorm, found {len(matches)}.")
    return matches[0]


def _constant_gate(module: nn.Module, lateral_embedding: torch.Tensor, semantic_embedding: torch.Tensor, value: float) -> torch.Tensor:
    """Create a dynamic FP32 gate on the embedding's device."""
    if lateral_embedding.shape != semantic_embedding.shape:
        raise ValueError("Projected feature shapes differ while creating a constant gate.")
    return torch.full(
        (lateral_embedding.shape[0], 1, lateral_embedding.shape[-2], lateral_embedding.shape[-1]),
        value,
        dtype=torch.float32,
        device=lateral_embedding.device,
    )


def constant_gate_05(module: nn.Module, lateral_embedding: torch.Tensor, semantic_embedding: torch.Tensor) -> torch.Tensor:
    """Return a shape/device-aware gate equal to 0.5."""
    return _constant_gate(module, lateral_embedding, semantic_embedding, 0.5)


def constant_gate_1(module: nn.Module, lateral_embedding: torch.Tensor, semantic_embedding: torch.Tensor) -> torch.Tensor:
    """Return a shape/device-aware gate equal to 1.0."""
    return _constant_gate(module, lateral_embedding, semantic_embedding, 1.0)


def raw_cosine_gate(module: nn.Module, lateral_embedding: torch.Tensor, semantic_embedding: torch.Tensor) -> torch.Tensor:
    """Apply the requested raw-cosine intervention without centering, standardizing, or clipping."""
    if lateral_embedding.shape != semantic_embedding.shape:
        raise ValueError("Projected feature shapes differ while computing the raw-cosine gate.")
    consistency = F.cosine_similarity(
        lateral_embedding.float(), semantic_embedding.float(), dim=1, eps=module.eps
    ).unsqueeze(1)
    return torch.sigmoid(module._effective_temperature() * consistency)


@contextmanager
def temporary_intervention(module: nn.Module, variant: str) -> Iterator[None]:
    """Apply one memory-only intervention and restore all touched state on exit."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if variant == "original":
        yield
        return

    tensor_snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
    had_instance_gate = "_semantic_gate" in module.__dict__
    previous_instance_gate = module.__dict__.get("_semantic_gate")

    def snapshot_and_zero(tensor: torch.Tensor | None, label: str) -> None:
        if tensor is None:
            raise RuntimeError(f"Required intervention tensor is missing: {label}")
        tensor_snapshots.append((tensor, tensor.detach().clone()))
        with torch.no_grad():
            tensor.zero_()

    try:
        if variant == "layer_scale_zero":
            snapshot_and_zero(getattr(module, "layer_scale_raw", None), "layer_scale_raw")
        elif variant == "gate_const_05":
            module._semantic_gate = MethodType(constant_gate_05, module)
        elif variant == "gate_const_1":
            module._semantic_gate = MethodType(constant_gate_1, module)
        elif variant == "shared_norm_bias_zero":
            snapshot_and_zero(getattr(getattr(module, "shared_norm", None), "bias", None), "shared_norm.bias")
        elif variant == "edge_norm_bias_zero":
            snapshot_and_zero(find_unique_edge_group_norm(module).bias, "edge_calibration GroupNorm.bias")
        elif variant == "both_norm_bias_zero":
            snapshot_and_zero(getattr(getattr(module, "shared_norm", None), "bias", None), "shared_norm.bias")
            snapshot_and_zero(find_unique_edge_group_norm(module).bias, "edge_calibration GroupNorm.bias")
        elif variant == "raw_cosine_gate":
            module._semantic_gate = MethodType(raw_cosine_gate, module)
        yield
    finally:
        with torch.no_grad():
            for target, original in tensor_snapshots:
                target.copy_(original)
        if variant in {"gate_const_05", "gate_const_1", "raw_cosine_gate"}:
            if had_instance_gate:
                module.__dict__["_semantic_gate"] = previous_instance_gate
            else:
                module.__dict__.pop("_semantic_gate", None)


class OnlineTensorStats:
    """Accumulate exact moments/extrema and a bounded deterministic quantile sample."""

    def __init__(self, sample_capacity: int = 200_000, sample_per_update: int = 4096, seed: int = 42) -> None:
        self.sample_capacity = sample_capacity
        self.sample_per_update = sample_per_update
        self.rng = random.Random(seed)
        self.count = 0
        self.finite_count = 0
        self.nan_count = 0
        self.posinf_count = 0
        self.neginf_count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.samples: list[float] = []
        self.sample_candidates_seen = 0

    def update(self, values: torch.Tensor | np.ndarray | list[float]) -> None:
        """Update without retaining the full input activation."""
        tensor = torch.as_tensor(values).detach().to(dtype=torch.float32).reshape(-1)
        self.count += tensor.numel()
        self.nan_count += int(torch.isnan(tensor).sum().item())
        self.posinf_count += int(torch.isposinf(tensor).sum().item())
        self.neginf_count += int(torch.isneginf(tensor).sum().item())
        finite = tensor[torch.isfinite(tensor)]
        if not finite.numel():
            return
        self.finite_count += finite.numel()
        finite64 = finite.to(dtype=torch.float64)
        self.total += float(finite64.sum().item())
        self.total_square += float(finite64.square().sum().item())
        self.minimum = min(self.minimum, float(finite.min().item()))
        self.maximum = max(self.maximum, float(finite.max().item()))

        take = min(finite.numel(), self.sample_per_update)
        if take == finite.numel():
            candidates = finite.cpu().tolist()
        else:
            # Integer arithmetic is essential here. A float32 linspace can round ``length - 1`` up to ``length``
            # once an activation exceeds 2**24 elements (C10: 16*256*80*80 = 26,214,400), causing a CUDA
            # IndexKernel device-side assert at ``finite[indices]``.
            indices = evenly_spaced_integer_indices(finite.numel(), take, finite.device)
            candidates = finite[indices].cpu().tolist()
        for candidate in candidates:
            self.sample_candidates_seen += 1
            if len(self.samples) < self.sample_capacity:
                self.samples.append(float(candidate))
            else:
                index = self.rng.randrange(self.sample_candidates_seen)
                if index < self.sample_capacity:
                    self.samples[index] = float(candidate)

    def summary(self, quantiles: tuple[float, ...] = FULL_QUANTILES) -> dict[str, Any]:
        """Return finite statistics and explicit non-finite counts."""
        result: dict[str, Any] = {
            "count": self.count,
            "finite_count": self.finite_count,
            "nan_count": self.nan_count,
            "positive_infinity_count": self.posinf_count,
            "negative_infinity_count": self.neginf_count,
            "quantile_sample_count": len(self.samples),
            "quantiles_are_bounded_sample_estimates": True,
        }
        if not self.finite_count:
            result.update({"mean": None, "std": None, "min": None, "max": None})
            result.update({_quantile_name(q): None for q in quantiles})
            return result
        mean = self.total / self.finite_count
        variance = max(0.0, self.total_square / self.finite_count - mean * mean)
        result.update({"mean": mean, "std": math.sqrt(variance), "min": self.minimum, "max": self.maximum})
        sample = np.asarray(self.samples, dtype=np.float64)
        for quantile in quantiles:
            result[_quantile_name(quantile)] = float(np.quantile(sample, quantile))
        return result


def evenly_spaced_integer_indices(length: int, count: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return monotonically increasing in-range sample indices using only int64 arithmetic."""
    if length <= 0:
        raise ValueError("Sampling length must be positive.")
    if count <= 0 or count > length:
        raise ValueError("Sampling count must satisfy 0 < count <= length.")
    if count == 1:
        return torch.zeros(1, dtype=torch.int64, device=device)
    positions = torch.arange(count, dtype=torch.int64, device=device)
    return torch.div(positions * (length - 1), count - 1, rounding_mode="floor")


def _quantile_name(quantile: float) -> str:
    return f"q{int(round(quantile * 100)):02d}"


def summarize_small_tensor(tensor: torch.Tensor, quantiles: tuple[float, ...]) -> dict[str, Any]:
    """Summarize a parameter-sized tensor exactly."""
    values = tensor.detach().float().cpu().reshape(-1)
    stats = OnlineTensorStats(sample_capacity=max(1, values.numel()), sample_per_update=max(1, values.numel()))
    stats.update(values)
    return stats.summary(quantiles)


def validate_batch_indices_on_cpu(
    batch_indices: torch.Tensor,
    batch_size: int,
    image_paths: list[str] | tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Copy GT batch indices to CPU and reject out-of-range values before any tensor indexing."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    indices = torch.as_tensor(batch_indices).detach().long().cpu().reshape(-1)
    invalid_mask = (indices < 0) | (indices >= batch_size)
    if invalid_mask.any():
        positions = torch.nonzero(invalid_mask, as_tuple=False).reshape(-1).tolist()
        details = [{"object_position": position, "batch_idx": int(indices[position])} for position in positions]
        paths = [] if image_paths is None else [str(path) for path in image_paths]
        raise InvalidBatchIndexError(
            f"GT batch_idx outside [0, {batch_size}): {details}; current_batch_paths={paths}"
        )
    return indices


def build_gt_masks(
    bboxes_xywhn: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    image_paths: list[str] | tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, list[int], int]:
    """Map normalized xywh boxes to unions of intersecting gate cells for their matching images."""
    if batch_size <= 0 or height <= 0 or width <= 0:
        raise ValueError("Mask dimensions must be positive.")
    boxes = torch.as_tensor(bboxes_xywhn).detach().float().cpu().reshape(-1, 4)
    indices = validate_batch_indices_on_cpu(batch_indices, batch_size, image_paths)
    if boxes.shape[0] != indices.shape[0]:
        raise ValueError("bboxes and batch_indices must contain the same number of objects.")
    masks = torch.zeros((batch_size, height, width), dtype=torch.bool)
    valid_boxes_per_image = [0] * batch_size
    invalid_boxes = 0
    for box, image_index in zip(boxes, indices):
        image_index = int(image_index.item())
        if not torch.isfinite(box).all():
            invalid_boxes += 1
            continue
        cx, cy, box_width, box_height = (float(item) for item in box)
        x0 = max(0, min(width, math.floor((cx - box_width / 2) * width)))
        y0 = max(0, min(height, math.floor((cy - box_height / 2) * height)))
        x1 = max(0, min(width, math.ceil((cx + box_width / 2) * width)))
        y1 = max(0, min(height, math.ceil((cy + box_height / 2) * height)))
        if box_width <= 0 or box_height <= 0 or x1 <= x0 or y1 <= y0:
            invalid_boxes += 1
            continue
        masks[image_index, y0:y1, x0:x1] = True
        valid_boxes_per_image[image_index] += 1
    return masks, valid_boxes_per_image, invalid_boxes


class ActivationCollector:
    """Collect original-model activation statistics with strict validator-batch alignment."""

    def __init__(self, module: nn.Module, max_batches: int = 0, seed: int = 42, variant: str = "original") -> None:
        self.module = module
        self.max_batches = max_batches
        self.variant = variant
        names = (
            "raw_consistency",
            "standardized_similarity",
            "gate",
            "scharr_magnitude",
            "calibrated_edge",
            "output_projection",
            "effective_layer_scale",
            "unscaled_residual_l2_over_lateral_l2",
            "scaled_delta_l2_over_lateral_l2",
            "scaled_delta_abs_mean_over_lateral_abs_mean",
            "cosine_delta_lateral",
        )
        self.stats = {name: OnlineTensorStats(seed=seed + index) for index, name in enumerate(names)}
        self.per_image_raw_mean = OnlineTensorStats(sample_capacity=100_000, sample_per_update=100_000, seed=seed)
        self.per_image_raw_std = OnlineTensorStats(sample_capacity=100_000, sample_per_update=100_000, seed=seed + 1)
        self.per_image_gate_mean = OnlineTensorStats(sample_capacity=100_000, sample_per_update=100_000, seed=seed + 2)
        self.per_image_gate_std = OnlineTensorStats(sample_capacity=100_000, sample_per_update=100_000, seed=seed + 3)
        self.inside_gate = OnlineTensorStats(seed=seed + 4)
        self.outside_gate = OnlineTensorStats(seed=seed + 5)
        self.per_image_inside = OnlineTensorStats(sample_capacity=100_000, sample_per_update=100_000, seed=seed + 6)
        self.per_image_outside = OnlineTensorStats(sample_capacity=100_000, sample_per_update=100_000, seed=seed + 7)
        self.threshold_counts = {
            "standardized_at_negative_clip": 0,
            "standardized_at_positive_clip": 0,
            "gate_lt_0.10": 0,
            "gate_lt_0.25": 0,
            "gate_between_0.45_and_0.55": 0,
            "gate_gt_0.75": 0,
            "gate_gt_0.90": 0,
            "per_image_gate_mean_between_0.45_and_0.55": 0,
        }
        self.per_image_rows: list[dict[str, Any]] = []
        self.valid_images = 0
        self.valid_gt_boxes = 0
        self.invalid_gt_boxes = 0
        self.skipped_no_gt = 0
        self.skipped_no_inside = 0
        self.skipped_no_background = 0
        self.started_batches = 0
        self.completed_batches = 0
        self._active_batch: int | None = None
        self._capture_enabled = False
        self._pending: dict[str, Any] | None = None
        self.runtime_context: dict[str, Any] = {
            "variant": variant,
            "current_batch_index": None,
            "current_image_paths": [],
            "current_batch_size": None,
            "gate_shape": None,
            "gt_batch_idx_min": None,
            "gt_batch_idx_max": None,
        }
        self._hook = module.register_forward_hook(self._forward_hook)

    def close(self) -> None:
        """Remove the model hook and discard any batch-sized pending activation."""
        self._hook.remove()
        self._pending = None
        self._active_batch = None

    def begin_batch(self, batch_index: int, batch: dict[str, Any] | None = None) -> None:
        """Arm the hook immediately after this exact validation batch is preprocessed."""
        if self._active_batch is not None:
            raise RuntimeError("Activation collector saw a new batch before the previous batch was finalized.")
        self._active_batch = int(batch_index)
        self._capture_enabled = self.max_batches == 0 or batch_index < self.max_batches
        self._pending = None
        self.started_batches += 1
        self.runtime_context.update(
            {
                "current_batch_index": int(batch_index),
                "current_image_paths": [] if batch is None else [str(path) for path in batch.get("im_file", [])],
                "current_batch_size": None if batch is None else int(batch["img"].shape[0]),
                "gate_shape": None,
                "gt_batch_idx_min": None,
                "gt_batch_idx_max": None,
            }
        )
        if batch is not None:
            indices = torch.as_tensor(batch.get("batch_idx", torch.empty(0))).detach().long().cpu().reshape(-1)
            if indices.numel():
                self.runtime_context["gt_batch_idx_min"] = int(indices.min().item())
                self.runtime_context["gt_batch_idx_max"] = int(indices.max().item())
            validate_batch_indices_on_cpu(
                indices,
                self.runtime_context["current_batch_size"],
                self.runtime_context["current_image_paths"],
            )

    def _forward_hook(self, module: nn.Module, hook_inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        if self._active_batch is None or not self._capture_enabled:
            return None  # Ignores validator warmup and batches beyond --stats-max-batches.
        if self._pending is not None:
            raise RuntimeError("CSCEFv2 ran more than once for one validation batch; alignment is ambiguous.")
        if len(hook_inputs) != 1 or not isinstance(hook_inputs[0], (list, tuple)) or len(hook_inputs[0]) != 2:
            raise RuntimeError("Unexpected CSCEFv2 hook input structure.")
        x_lateral, x_semantic = hook_inputs[0]
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Unexpected non-tensor CSCEFv2 output.")

        with torch.no_grad():
            lateral_embedding, semantic_embedding = module._project_features(x_lateral, x_semantic)
            raw = F.cosine_similarity(
                lateral_embedding.float(), semantic_embedding.float(), dim=1, eps=module.eps
            ).unsqueeze(1)
            centered = raw - raw.mean(dim=(-2, -1), keepdim=True)
            variance = centered.square().mean(dim=(-2, -1), keepdim=True)
            standardized_unclipped = centered * torch.rsqrt(variance + module.eps)
            standardized = standardized_unclipped.clamp(-module.similarity_clip, module.similarity_clip)
            gate = torch.sigmoid(module._effective_temperature() * standardized)
            self.runtime_context["gate_shape"] = list(gate.squeeze(1).shape)
            edge = module._scharr_magnitude(lateral_embedding)
            calibrated_edge = module.edge_calibration(edge)
            residual = module.output_projection(calibrated_edge * gate.to(calibrated_edge.dtype))
            delta = output - x_lateral
            layer_scale = module._effective_layer_scale()

            self.stats["raw_consistency"].update(raw)
            self.stats["standardized_similarity"].update(standardized)
            self.stats["gate"].update(gate)
            self.stats["scharr_magnitude"].update(edge)
            self.stats["calibrated_edge"].update(calibrated_edge)
            self.stats["output_projection"].update(residual)
            self.stats["effective_layer_scale"].update(layer_scale)

            finite_standardized = torch.isfinite(standardized_unclipped)
            self.threshold_counts["standardized_at_negative_clip"] += int(
                ((standardized_unclipped <= -module.similarity_clip) & finite_standardized).sum().item()
            )
            self.threshold_counts["standardized_at_positive_clip"] += int(
                ((standardized_unclipped >= module.similarity_clip) & finite_standardized).sum().item()
            )
            finite_gate = torch.isfinite(gate)
            self.threshold_counts["gate_lt_0.10"] += int(((gate < 0.10) & finite_gate).sum().item())
            self.threshold_counts["gate_lt_0.25"] += int(((gate < 0.25) & finite_gate).sum().item())
            self.threshold_counts["gate_between_0.45_and_0.55"] += int(
                ((gate >= 0.45) & (gate <= 0.55) & finite_gate).sum().item()
            )
            self.threshold_counts["gate_gt_0.75"] += int(((gate > 0.75) & finite_gate).sum().item())
            self.threshold_counts["gate_gt_0.90"] += int(((gate > 0.90) & finite_gate).sum().item())

            batch_size = x_lateral.shape[0]
            raw_flat = raw.reshape(batch_size, -1)
            gate_flat = gate.reshape(batch_size, -1)
            raw_means = raw_flat.mean(1)
            raw_stds = raw_flat.std(1, unbiased=False)
            gate_means = gate_flat.mean(1)
            gate_stds = gate_flat.std(1, unbiased=False)
            self.per_image_raw_mean.update(raw_means)
            self.per_image_raw_std.update(raw_stds)
            self.per_image_gate_mean.update(gate_means)
            self.per_image_gate_std.update(gate_stds)
            self.threshold_counts["per_image_gate_mean_between_0.45_and_0.55"] += int(
                ((gate_means >= 0.45) & (gate_means <= 0.55) & torch.isfinite(gate_means)).sum().item()
            )

            eps = float(module.eps)
            lateral_vector = x_lateral.float().reshape(batch_size, -1)
            residual_vector = residual.float().reshape(batch_size, -1)
            delta_vector = delta.float().reshape(batch_size, -1)
            unscaled_ratio = residual_vector.norm(dim=1) / (lateral_vector.norm(dim=1) + eps)
            scaled_ratio = delta_vector.norm(dim=1) / (lateral_vector.norm(dim=1) + eps)
            abs_ratio = delta_vector.abs().mean(1) / (lateral_vector.abs().mean(1) + eps)
            delta_cosine = F.cosine_similarity(delta_vector, lateral_vector, dim=1, eps=eps)
            self.stats["unscaled_residual_l2_over_lateral_l2"].update(unscaled_ratio)
            self.stats["scaled_delta_l2_over_lateral_l2"].update(scaled_ratio)
            self.stats["scaled_delta_abs_mean_over_lateral_abs_mean"].update(abs_ratio)
            self.stats["cosine_delta_lateral"].update(delta_cosine)

            # Only this batch-sized gate and per-image scalars cross the hook/callback boundary.
            self._pending = {
                "batch_index": self._active_batch,
                "gate": gate.squeeze(1).detach().float().cpu(),
                "raw_means": raw_means.detach().float().cpu(),
                "raw_stds": raw_stds.detach().float().cpu(),
                "gate_means": gate_means.detach().float().cpu(),
                "gate_stds": gate_stds.detach().float().cpu(),
            }
        return None  # A forward hook returning None cannot replace or modify the model output.

    def finish_batch(self, batch_index: int, batch: dict[str, Any]) -> None:
        """Join the captured gate to labels from the same validator local batch object."""
        if self._active_batch != batch_index:
            raise RuntimeError(f"Validator/collector batch mismatch: {self._active_batch} != {batch_index}")
        try:
            if not self._capture_enabled:
                return
            if self._pending is None or self._pending["batch_index"] != batch_index:
                raise RuntimeError("No uniquely aligned CSCEFv2 activation was captured for this validation batch.")
            gates = self._pending["gate"]
            batch_size, height, width = gates.shape
            paths = batch.get("im_file", [""] * batch_size)
            masks, box_counts, invalid_boxes = build_gt_masks(
                batch["bboxes"], batch["batch_idx"], batch_size, height, width, paths
            )
            self.invalid_gt_boxes += invalid_boxes
            for image_index in range(batch_size):
                gate = gates[image_index]
                mask = masks[image_index]
                inside_pixels = int(mask.sum().item())
                outside_pixels = int((~mask).sum().item())
                inside_mean: float | None = None
                outside_mean: float | None = None
                status = "valid"
                if box_counts[image_index] == 0:
                    self.skipped_no_gt += 1
                    status = "no_valid_gt"
                elif inside_pixels == 0:
                    self.skipped_no_inside += 1
                    status = "no_positive_gate_cells"
                elif outside_pixels == 0:
                    self.skipped_no_background += 1
                    status = "no_background_gate_cells"
                else:
                    inside = gate[mask]
                    outside = gate[~mask]
                    self.inside_gate.update(inside)
                    self.outside_gate.update(outside)
                    inside_mean = float(inside.mean().item())
                    outside_mean = float(outside.mean().item())
                    self.per_image_inside.update([inside_mean])
                    self.per_image_outside.update([outside_mean])
                    self.valid_images += 1
                    self.valid_gt_boxes += box_counts[image_index]
                self.per_image_rows.append(
                    {
                        "image_index": len(self.per_image_rows),
                        "batch_index": batch_index,
                        "image_path": str(paths[image_index]),
                        "raw_consistency_spatial_mean": float(self._pending["raw_means"][image_index]),
                        "raw_consistency_spatial_std": float(self._pending["raw_stds"][image_index]),
                        "gate_spatial_mean": float(self._pending["gate_means"][image_index]),
                        "gate_spatial_std": float(self._pending["gate_stds"][image_index]),
                        "valid_gt_boxes": box_counts[image_index],
                        "inside_pixels": inside_pixels,
                        "outside_pixels": outside_pixels,
                        "gate_inside_gt_mean": inside_mean,
                        "gate_outside_gt_mean": outside_mean,
                        "inside_minus_outside": None
                        if inside_mean is None or outside_mean is None
                        else inside_mean - outside_mean,
                        "inside_div_outside": None
                        if inside_mean is None or outside_mean is None
                        else inside_mean / (outside_mean + float(self.module.eps)),
                        "status": status,
                    }
                )
            self.completed_batches += 1
        finally:
            self._pending = None
            self._active_batch = None
            self._capture_enabled = False

    def as_dict(self) -> dict[str, Any]:
        """Build the activation artifact after collection is complete."""
        if self._active_batch is not None:
            raise RuntimeError("Cannot finalize activation statistics with an unfinished validation batch.")
        raw = self.stats["raw_consistency"].summary(FULL_QUANTILES)
        standardized = self.stats["standardized_similarity"].summary(FULL_QUANTILES)
        gate = self.stats["gate"].summary(FULL_QUANTILES)
        standardized_count = max(1, self.stats["standardized_similarity"].finite_count)
        gate_count = max(1, self.stats["gate"].finite_count)
        image_count = max(1, self.per_image_gate_mean.finite_count)
        standardized["fraction_at_negative_clip"] = (
            self.threshold_counts["standardized_at_negative_clip"] / standardized_count
        )
        standardized["fraction_at_positive_clip"] = (
            self.threshold_counts["standardized_at_positive_clip"] / standardized_count
        )
        gate.update(
            {
                "fraction_lt_0.10": self.threshold_counts["gate_lt_0.10"] / gate_count,
                "fraction_lt_0.25": self.threshold_counts["gate_lt_0.25"] / gate_count,
                "fraction_between_0.45_and_0.55": self.threshold_counts["gate_between_0.45_and_0.55"]
                / gate_count,
                "fraction_gt_0.75": self.threshold_counts["gate_gt_0.75"] / gate_count,
                "fraction_gt_0.90": self.threshold_counts["gate_gt_0.90"] / gate_count,
            }
        )
        inside_mean = self.inside_gate.summary()["mean"]
        outside_mean = self.outside_gate.summary()["mean"]
        return {
            "collection": {
                "started_batches": self.started_batches,
                "completed_stat_batches": self.completed_batches,
                "stats_max_batches": self.max_batches,
                "alignment": (
                    "The custom validator arms the module hook only after preprocessing a specific batch, then "
                    "joins the one captured gate to that same local batch object before detection metrics update. "
                    "Warmup forwards are ignored and zero/multiple captures fail explicitly."
                ),
                "retention": "Online moments plus bounded quantile samples; only one batch-sized gate is pending.",
            },
            "raw_consistency": raw,
            "raw_consistency_per_image_spatial_mean": self.per_image_raw_mean.summary(FULL_QUANTILES),
            "raw_consistency_per_image_spatial_std": self.per_image_raw_std.summary(FULL_QUANTILES),
            "standardized_similarity": standardized,
            "gate": gate,
            "gate_per_image_spatial_mean": {
                **self.per_image_gate_mean.summary(FULL_QUANTILES),
                "fraction_between_0.45_and_0.55": self.threshold_counts[
                    "per_image_gate_mean_between_0.45_and_0.55"
                ]
                / image_count,
            },
            "gate_per_image_spatial_std": self.per_image_gate_std.summary(FULL_QUANTILES),
            "gate_inside_vs_outside_gt": {
                "gate_inside_gt_mean": inside_mean,
                "gate_outside_gt_mean": outside_mean,
                "inside_minus_outside": None
                if inside_mean is None or outside_mean is None
                else inside_mean - outside_mean,
                "inside_div_outside": None
                if inside_mean is None or outside_mean is None
                else inside_mean / (outside_mean + float(self.module.eps)),
                "valid_images": self.valid_images,
                "valid_gt_boxes": self.valid_gt_boxes,
                "invalid_gt_boxes": self.invalid_gt_boxes,
                "skipped_no_gt": self.skipped_no_gt,
                "skipped_no_positive_region": self.skipped_no_inside,
                "skipped_no_background_region": self.skipped_no_background,
                "per_image_inside_mean": self.per_image_inside.summary(FULL_QUANTILES),
                "per_image_outside_mean": self.per_image_outside.summary(FULL_QUANTILES),
            },
            "edge_and_residual": {
                "scharr_magnitude": self.stats["scharr_magnitude"].summary(FULL_QUANTILES),
                "calibrated_edge": self.stats["calibrated_edge"].summary(FULL_QUANTILES),
                "output_projection_unscaled_residual": self.stats["output_projection"].summary(FULL_QUANTILES),
                "effective_layer_scale": self.stats["effective_layer_scale"].summary(FULL_QUANTILES),
                "unscaled_residual_l2_over_lateral_l2": self.stats[
                    "unscaled_residual_l2_over_lateral_l2"
                ].summary(RATIO_QUANTILES),
                "scaled_delta_l2_over_lateral_l2": self.stats["scaled_delta_l2_over_lateral_l2"].summary(
                    RATIO_QUANTILES
                ),
                "scaled_delta_abs_mean_over_lateral_abs_mean": self.stats[
                    "scaled_delta_abs_mean_over_lateral_abs_mean"
                ].summary(RATIO_QUANTILES),
                "cosine_delta_lateral": self.stats["cosine_delta_lateral"].summary(RATIO_QUANTILES),
            },
        }


def parameter_audit(module: nn.Module) -> dict[str, Any]:
    """Audit trained CSCEF-v2 parameters without mutation."""
    edge_norm = find_unique_edge_group_norm(module)
    shared_norm = getattr(module, "shared_norm")
    layer_scale = module._effective_layer_scale().detach().float()
    layer_stats = summarize_small_tensor(layer_scale, FULL_QUANTILES)
    finite_layer = layer_scale[torch.isfinite(layer_scale)]
    denominator = max(1, finite_layer.numel())
    layer_stats.update(
        {
            "negative_fraction": float((finite_layer < 0).sum().item()) / denominator,
            "near_zero_fraction": float((finite_layer.abs() <= 1e-6).sum().item()) / denominator,
            "near_bound_fraction": float(
                (finite_layer.abs() >= 0.95 * float(module.layer_scale_max)).sum().item()
            )
            / denominator,
            "layer_scale_max": float(module.layer_scale_max),
        }
    )

    def norm_stats(tensor: torch.Tensor) -> dict[str, Any]:
        result = summarize_small_tensor(tensor, (0.05, 0.50, 0.95))
        result["l2_norm"] = float(tensor.detach().float().norm().item())
        return result

    calibration_convs = [item for item in module.edge_calibration.modules() if isinstance(item, nn.Conv2d)]
    if len(calibration_convs) != 1:
        raise RuntimeError(f"Expected one edge-calibration Conv2d, found {len(calibration_convs)}.")

    def projection_stats(projection: nn.Module) -> dict[str, Any]:
        parameters = list(projection.parameters())
        return {
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "weight_l2_norm": float(projection.weight.detach().float().norm().item()),
        }

    return {
        "temperature": {
            "raw_temperature": float(module.raw_temperature.detach().float().item()),
            "effective_temperature": float(module._effective_temperature().detach().float().item()),
            "temperature_min": float(module.temperature_min),
            "temperature_max": float(module.temperature_max),
        },
        "effective_layer_scale": layer_stats,
        "shared_group_norm": {
            "weight": norm_stats(shared_norm.weight),
            "bias": norm_stats(shared_norm.bias),
        },
        "edge_calibration_group_norm": {
            "weight": norm_stats(edge_norm.weight),
            "bias": norm_stats(edge_norm.bias),
        },
        "projections": {
            "shared_projection": projection_stats(module.shared_projection),
            "depthwise_calibration_conv": projection_stats(calibration_convs[0]),
            "output_projection": projection_stats(module.output_projection),
        },
        "cscef_v2_total_parameters": sum(parameter.numel() for parameter in module.parameters()),
    }


def write_per_image_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write aligned per-image gate statistics."""
    fields = (
        "image_index",
        "batch_index",
        "image_path",
        "raw_consistency_spatial_mean",
        "raw_consistency_spatial_std",
        "gate_spatial_mean",
        "gate_spatial_std",
        "valid_gt_boxes",
        "inside_pixels",
        "outside_pixels",
        "gate_inside_gt_mean",
        "gate_outside_gt_mean",
        "inside_minus_outside",
        "inside_div_outside",
        "status",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_metric_artifacts(output: Path, rows: list[dict[str, Any]]) -> None:
    """Write complete metric schemas to CSV and JSON, retaining nulls for failures."""
    with (output / "variant_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in METRIC_FIELDS} for row in rows])
    write_json(output / "variant_metrics.json", [{key: row.get(key) for key in METRIC_FIELDS} for row in rows])


def _git(command: list[str]) -> str:
    """Run a read-only Git query and return text or an explicit error marker."""
    try:
        result = subprocess.run(
            ["git", *command], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
        )
        return result.stdout.rstrip()
    except Exception as error:  # Git metadata is diagnostic context, not a reason to hide validation results.
        return f"ERROR: {error}"


def git_state() -> dict[str, str]:
    """Capture branch, HEAD, remote, and status without changing repository state."""
    return {
        "branch": _git(["branch", "--show-current"]),
        "head": _git(["rev-parse", "HEAD"]),
        "origin": _git(["remote", "get-url", "origin"]),
        "status_short": _git(["status", "--short"]),
    }


def import_local_ultralytics() -> tuple[Any, type[nn.Module], type[Any]]:
    """Import and verify the repository-local Ultralytics runtime."""
    sys.path.insert(0, str(ULTRALYTICS_ROOT))
    import ultralytics
    from ultralytics import RTDETR
    from ultralytics.models.rtdetr.val import RTDETRValidator

    imported_path = Path(ultralytics.__file__).resolve()
    if not imported_path.is_relative_to(ULTRALYTICS_ROOT.resolve()):
        raise RuntimeError(
            f"Imported Ultralytics from {imported_path}, not repository source {ULTRALYTICS_ROOT.resolve()}."
        )
    return ultralytics, RTDETR, RTDETRValidator


def require_rtdetr_validator(validator_class: type[Any], rtdetr_validator_class: type[Any]) -> None:
    """Fail before GPU inference unless the selected validator is RTDETRValidator or a subclass."""
    if not isinstance(validator_class, type) or not issubclass(validator_class, rtdetr_validator_class):
        name = getattr(validator_class, "__name__", repr(validator_class))
        raise TypeError(f"Refusing validation with {name}; an RTDETRValidator subclass is required.")


def make_diagnostic_validator(
    base_class: type[Any],
    collector: ActivationCollector | None,
    rtdetr_validator_class: type[Any] | None = None,
) -> type[Any]:
    """Create a validator that provides an unambiguous gate/GT batch join."""
    if rtdetr_validator_class is not None:
        require_rtdetr_validator(base_class, rtdetr_validator_class)

    class CSCEFDiagnosticValidator(base_class):
        def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
            processed = super().preprocess(batch)
            if collector is not None:
                collector.begin_batch(self.batch_i, processed)
            return processed

        def update_metrics(self, preds: Any, batch: dict[str, Any]) -> None:
            if collector is not None:
                collector.finish_batch(self.batch_i, batch)
            return super().update_metrics(preds, batch)

    CSCEFDiagnosticValidator.__name__ = "CSCEFDiagnosticValidator"
    return CSCEFDiagnosticValidator


def model_class_count(model: nn.Module) -> int | None:
    """Read the detector class count without making graph-index assumptions."""
    names = getattr(model, "names", None)
    if isinstance(names, (dict, list, tuple)):
        return len(names)
    for module in reversed(list(model.modules())):
        value = getattr(module, "nc", None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def dataset_class_count(data_path: Path, ultralytics: Any) -> int | None:
    """Read dataset nc/names with the repository's YAML loader."""
    data = ultralytics.utils.YAML.load(data_path)
    names = data.get("names")
    if isinstance(names, (dict, list, tuple)):
        return len(names)
    value = data.get("nc")
    return int(value) if isinstance(value, int) else None


def build_runtime_audit(
    wrapper: Any,
    validator_class: type[Any],
    rtdetr_validator_class: type[Any],
    dataset_nc: int | None,
    variant: str,
) -> dict[str, Any]:
    """Record and enforce the RT-DETR validation stack before GPU validation."""
    require_rtdetr_validator(validator_class, rtdetr_validator_class)
    cscef_count = count_cscef_v2_modules(wrapper.model)
    if cscef_count != 1:
        raise RuntimeError(f"Expected exactly one CSCEFv2 module, found {cscef_count}.")
    return {
        "variant": variant,
        "wrapper_class": wrapper.__class__.__name__,
        "wrapper_module": wrapper.__class__.__module__,
        "model_class": wrapper.model.__class__.__name__,
        "model_module": wrapper.model.__class__.__module__,
        "validator_class": validator_class.__name__,
        "validator_module": validator_class.__module__,
        "validator_is_rtdetr": issubclass(validator_class, rtdetr_validator_class),
        "task": getattr(wrapper, "task", None),
        "model_nc": model_class_count(wrapper.model),
        "dataset_nc": dataset_nc,
        "cscef_v2_instance_count": cscef_count,
    }


def is_unrecoverable_cuda_error(error: BaseException | str) -> bool:
    """Return whether an error indicates that the current CUDA context must not be reused."""
    message = str(error).lower()
    return any(marker in message for marker in CUDA_CONTEXT_ERROR_MARKERS)


def extract_metrics(metrics: Any) -> dict[str, float]:
    """Extract the exact requested detection metrics from Ultralytics DetMetrics."""
    values = metrics.results_dict
    mapping = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)",
        "mAP50-95": "metrics/mAP50-95(B)",
        "fitness": "fitness",
    }
    missing = [source for source in mapping.values() if source not in values]
    if missing:
        raise RuntimeError(f"Ultralytics metrics are missing required fields: {missing}")
    result = {target: float(values[source]) for target, source in mapping.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError(f"Validation returned non-finite metrics: {result}")
    return result


def validation_arguments(args: argparse.Namespace, output: Path, variant: str) -> dict[str, Any]:
    """Return the fixed C10-compatible validation configuration."""
    return {
        "data": str(Path(args.data).resolve()),
        "split": "val",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": str(args.device),
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "augment": False,
        "seed": args.seed,
        "project": str(output / "variants"),
        "name": variant,
        "exist_ok": True,
        "plots": False,
        "save_json": False,
        "save_txt": False,
        "save_conf": False,
        "verbose": True,
    }


def run_variant(
    variant: str,
    args: argparse.Namespace,
    output: Path,
    weights_hash: str,
    data_hash: str,
    rtdetr_class: type[nn.Module],
    rtdetr_validator_class: type[Any],
    dataset_nc: int | None,
) -> tuple[dict[str, Any], ActivationCollector | None, dict[str, Any], bool]:
    """Reload the checkpoint and validate through ``RTDETR.val`` with a verified RT-DETR validator."""
    started = time.perf_counter()
    collector: ActivationCollector | None = None
    wrapper = None
    runtime_audit: dict[str, Any] = {"variant": variant}
    unrecoverable_cuda_error = False
    row = {field: None for field in METRIC_FIELDS}
    row.update(
        {
            "variant": variant,
            "weights_sha256": weights_hash,
            "data_sha256": data_hash,
            "status": "failed",
            "error": None,
        }
    )
    try:
        wrapper = rtdetr_class(str(Path(args.weights).resolve()))
        target = find_unique_cscef_v2(wrapper.model)
        target.eval()
        official_validator_class = wrapper.task_map[wrapper.task]["validator"]
        require_rtdetr_validator(official_validator_class, rtdetr_validator_class)
        if variant == "original":
            collector = ActivationCollector(
                target, max_batches=args.stats_max_batches, seed=args.seed, variant=variant
            )
            selected_validator_class = make_diagnostic_validator(
                official_validator_class, collector, rtdetr_validator_class
            )
        else:
            selected_validator_class = official_validator_class
        runtime_audit = build_runtime_audit(
            wrapper, selected_validator_class, rtdetr_validator_class, dataset_nc, variant
        )
        logging.getLogger("cscef_v2_diagnostic").info("Runtime audit: %s", runtime_audit)
        with temporary_intervention(target, variant), torch.no_grad():
            # Use the RTDETR wrapper's official Model.val entry. The original variant supplies a strict
            # RTDETRValidator subclass solely to join side-channel activations to the same validator batch.
            if variant == "original":
                metrics = wrapper.val(
                    validator=selected_validator_class, **validation_arguments(args, output, variant)
                )
            else:
                metrics = wrapper.val(**validation_arguments(args, output, variant))
        row.update(extract_metrics(metrics))
        row["status"] = "success"
    except Exception as error:
        row["error"] = traceback.format_exc()
        unrecoverable_cuda_error = is_unrecoverable_cuda_error(error) or is_unrecoverable_cuda_error(row["error"])
        runtime_audit["unrecoverable_cuda_error"] = unrecoverable_cuda_error
        if collector is not None:
            runtime_audit["failure_context"] = dict(collector.runtime_context)
    finally:
        row["elapsed_seconds"] = time.perf_counter() - started
        if collector is not None:
            collector.close()
        del wrapper
        gc.collect()
        # Any CUDA API call after a device-side assert can raise again and prevent failure artifacts from being saved.
        if not unrecoverable_cuda_error and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as error:
                cleanup_traceback = traceback.format_exc()
                row["status"] = "failed"
                row["error"] = (row["error"] or "") + "\nCUDA cleanup failure:\n" + cleanup_traceback
                unrecoverable_cuda_error = is_unrecoverable_cuda_error(error) or is_unrecoverable_cuda_error(
                    cleanup_traceback
                )
                runtime_audit["unrecoverable_cuda_error"] = unrecoverable_cuda_error
    return row, collector, runtime_audit, unrecoverable_cuda_error


def environment_info(ultralytics: Any) -> dict[str, Any]:
    """Describe the exact software and accelerator runtime."""
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "ultralytics_path": str(Path(ultralytics.__file__).resolve()),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": gpu_names,
    }


def write_environment_text(path: Path, environment: dict[str, Any]) -> None:
    path.write_text("\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n", encoding="utf-8")


def write_git_text(path: Path, state: dict[str, str]) -> None:
    path.write_text("\n\n".join(f"[{key}]\n{value}" for key, value in state.items()) + "\n", encoding="utf-8")


def _format_number(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def generate_report(
    output: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    parameters: dict[str, Any] | None,
    activations: dict[str, Any] | None,
) -> None:
    """Generate an objective, non-causal Markdown summary."""
    successes = {row["variant"]: row for row in rows if row["status"] == "success"}
    original = successes.get("original")
    lines = [
        "# CSCEF-v2 validation diagnostic",
        "",
        POST_TRAINING_WARNING,
        "",
        "## Experiment identity",
        "",
        f"- Weights: `{manifest['weights']}`",
        f"- Weights SHA256: `{manifest['weights_sha256_before']}`",
        f"- Data: `{manifest['data']}`",
        f"- Data SHA256: `{manifest['data_sha256']}`",
        f"- Git branch / HEAD: `{manifest['git']['branch']}` / `{manifest['git']['head']}`",
        "",
        "## Validation configuration",
        "",
        "`split=val, conf=0.001, iou=0.7, max_det=300, augment=False, "
        f"imgsz={manifest['arguments']['imgsz']}, batch={manifest['arguments']['batch']}, "
        f"workers={manifest['arguments']['workers']}, device={manifest['arguments']['device']}, "
        f"seed={manifest['arguments']['seed']}`",
        "",
        "## Full-validation metrics",
        "",
        "| Variant | Precision | Recall | mAP50 | mAP50-95 | Fitness | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {_format_number(row['precision'])} | {_format_number(row['recall'])} | "
            f"{_format_number(row['mAP50'])} | {_format_number(row['mAP50-95'])} | "
            f"{_format_number(row['fitness'])} | {row['status']} |"
        )
    lines += ["", "The `original` row is the unmodified C10 checkpoint; `layer_scale_zero` is not a baseline model."]
    if original is not None:
        lines += [
            "",
            "## Original C10",
            "",
            f"Precision={original['precision']:.6f}, Recall={original['recall']:.6f}, "
            f"mAP50={original['mAP50']:.6f}, mAP50-95={original['mAP50-95']:.6f}.",
        ]
    if activations is not None:
        gate_mean = activations["gate_per_image_spatial_mean"]
        gt = activations["gate_inside_vs_outside_gt"]
        residual = activations["edge_and_residual"]
        lines += [
            "",
            "## Activation observations (original C10 only)",
            "",
            f"- Per-image gate mean: mean={_format_number(gate_mean['mean'])}, "
            f"std={_format_number(gate_mean['std'])}; fraction in [0.45, 0.55]="
            f"{_format_number(gate_mean['fraction_between_0.45_and_0.55'])}.",
            f"- GT/background gate: inside={_format_number(gt['gate_inside_gt_mean'])}, "
            f"outside={_format_number(gt['gate_outside_gt_mean'])}, "
            f"inside-minus-outside={_format_number(gt['inside_minus_outside'])}, "
            f"valid images={gt['valid_images']}, valid boxes={gt['valid_gt_boxes']}.",
            "- Real scaled residual L2/lateral L2: "
            f"mean={_format_number(residual['scaled_delta_l2_over_lateral_l2']['mean'])}, "
            f"median={_format_number(residual['scaled_delta_l2_over_lateral_l2']['q50'])}.",
            "- Gate/label alignment uses the same preprocessed validator batch and fails on missing or duplicate captures; "
            "details and skip counts are in `activation_stats.json`.",
        ]
    lines += [
        "",
        "## Normalization bias state",
        "",
    ]
    if parameters is None:
        lines.append("- Parameter audit was unavailable because the run failed before it completed.")
    else:
        shared_bias = parameters["shared_group_norm"]["bias"]
        edge_bias = parameters["edge_calibration_group_norm"]["bias"]
        lines += [
            f"- Shared GroupNorm bias: mean={_format_number(shared_bias['mean'])}, "
            f"std={_format_number(shared_bias['std'])}, L2={_format_number(shared_bias['l2_norm'])}.",
            f"- Edge-calibration GroupNorm bias: mean={_format_number(edge_bias['mean'])}, "
            f"std={_format_number(edge_bias['std'])}, L2={_format_number(edge_bias['l2_norm'])}.",
        ]
    lines += [
        "",
        "## Runtime validation audit",
        "",
    ]
    for audit in manifest.get("runtime_audits", []):
        lines.append(
            f"- `{audit.get('variant')}`: wrapper=`{audit.get('wrapper_class')}`, "
            f"model=`{audit.get('model_class')}`, validator=`{audit.get('validator_module')}."
            f"{audit.get('validator_class')}`, task=`{audit.get('task')}`, model_nc={audit.get('model_nc')}, "
            f"dataset_nc={audit.get('dataset_nc')}, CSCEF-v2 count={audit.get('cscef_v2_instance_count')}."
        )
    lines += [
        "",
        "## Hypothesis evidence",
        "",
        "The following are diagnostic associations, not causal conclusions:",
        "",
    ]
    if activations is None:
        lines.append("- Original activation statistics were not collected because `original` was not successfully run.")
    else:
        gate_mean = activations["gate_per_image_spatial_mean"]
        gt = activations["gate_inside_vs_outside_gt"]
        lines += [
            "- Spatial-standardization / near-0.5 gate hypothesis: compare the recorded per-image gate mean "
            f"distribution (mean {_format_number(gate_mean['mean'])}, [0.45,0.55] fraction "
            f"{_format_number(gate_mean['fraction_between_0.45_and_0.55'])}) with `raw_cosine_gate` metrics.",
            "- Shared-bias hypothesis: compare `shared_norm_bias_zero` with `original`, together with raw cosine and "
            "the shared-bias statistics above.",
            "- Edge-bias hypothesis: compare `edge_norm_bias_zero` and `both_norm_bias_zero` with `original`, together "
            "with calibrated-edge distributions.",
            "- Entire-residual-harm hypothesis: compare `layer_scale_zero` with `original` while remembering that this "
            "is an out-of-distribution post-training edit, not the R18-Lite baseline.",
            "- Edge-localization / ineffective-gate hypothesis: jointly inspect inside-minus-outside "
            f"({_format_number(gt['inside_minus_outside'])}), fixed-gate variants, and residual ratios.",
        ]
    failures = [row for row in rows if row["status"] != "success"]
    if failures:
        lines += ["", "## Failures", ""]
        lines.extend(f"- `{row['variant']}` failed; full traceback is in `variant_metrics.json`." for row in failures)
    lines += [
        "",
        "No intervention checkpoint was saved. Quantiles are bounded-sample estimates; global moments and threshold "
        "fractions cover every finite collected activation.",
    ]
    (output / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_plots(
    output: Path,
    collector: ActivationCollector | None,
    rows: list[dict[str, Any]],
) -> None:
    """Create the requested finite-only, white-background diagnostic plots."""
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)

    def finite_sample(stats: OnlineTensorStats) -> np.ndarray:
        values = np.asarray(stats.samples, dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError("A non-finite sampled value reached diagnostic plotting.")
        return values

    def histogram(stats: OnlineTensorStats, filename: str, title: str, xlabel: str) -> None:
        values = finite_sample(stats)
        if not values.size:
            return
        fig, axis = plt.subplots(figsize=(7.2, 4.5), facecolor="white")
        axis.hist(values, bins=60, color="#4C78A8", alpha=0.85)
        axis.set(title=title, xlabel=xlabel, ylabel="Sample count")
        axis.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180, facecolor="white")
        plt.close(fig)

    if collector is not None:
        histogram(collector.stats["raw_consistency"], "raw_cosine_histogram.png", "Raw cosine similarity", "Cosine")
        histogram(
            collector.stats["standardized_similarity"],
            "standardized_similarity_histogram.png",
            "Standardized similarity",
            "Standardized cosine",
        )
        histogram(collector.stats["gate"], "gate_histogram.png", "Semantic gate", "Gate value")
        histogram(
            collector.per_image_gate_mean,
            "per_image_gate_mean_histogram.png",
            "Per-image gate mean",
            "Spatial mean",
        )
        histogram(
            collector.stats["scaled_delta_l2_over_lateral_l2"],
            "residual_ratio_histogram.png",
            "Scaled residual relative L2 norm",
            "||delta||2 / (||lateral||2 + eps)",
        )
        inside = finite_sample(collector.per_image_inside)
        outside = finite_sample(collector.per_image_outside)
        if inside.size and outside.size:
            fig, axis = plt.subplots(figsize=(6.4, 4.5), facecolor="white")
            axis.boxplot([inside, outside], labels=["Inside GT", "Outside GT"], showfliers=False)
            axis.set(title="Per-image gate inside vs outside GT", ylabel="Mean gate")
            axis.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            fig.savefig(plot_dir / "gate_inside_vs_outside_gt.png", dpi=180, facecolor="white")
            plt.close(fig)

    successes = [row for row in rows if row["status"] == "success"]
    if successes:
        metrics = ("precision", "recall", "mAP50", "mAP50-95")
        positions = np.arange(len(successes))
        width = 0.19
        colors = ("#4C78A8", "#F58518", "#54A24B", "#B279A2")
        fig, axis = plt.subplots(figsize=(max(9.0, len(successes) * 1.25), 5.2), facecolor="white")
        for index, (metric, color) in enumerate(zip(metrics, colors)):
            axis.bar(positions + (index - 1.5) * width, [row[metric] for row in successes], width, label=metric, color=color)
        axis.set_xticks(positions, [row["variant"] for row in successes], rotation=30, ha="right")
        axis.set(title="Post-training diagnostic validation metrics", ylabel="Metric value")
        axis.legend(ncols=4)
        axis.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / "variant_metrics_comparison.png", dpi=180, facecolor="white")
        plt.close(fig)


def failed_metric_row(
    variant: str,
    weights_hash: str,
    data_hash: str,
    error: str,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    """Create a schema-complete failure row without inventing metric values."""
    row = {field: None for field in METRIC_FIELDS}
    row.update(
        {
            "variant": variant,
            "weights_sha256": weights_hash,
            "data_sha256": data_hash,
            "elapsed_seconds": elapsed_seconds,
            "status": "failed",
            "error": error,
        }
    )
    return row


def run(args: argparse.Namespace) -> int:
    """Execute diagnostics and persist terminal artifacts even when validation fails."""
    validate_args(args)
    args.weights = Path(args.weights).expanduser().resolve()
    args.data = Path(args.data).expanduser().resolve()
    args.project = Path(args.project).expanduser().resolve()
    selected_variants = parse_variants(args.variants)
    output = prepare_output_directory(args.project, args.name, args.overwrite)
    log_path = output / "diagnostic.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger("cscef_v2_diagnostic")

    weights_hash_before = sha256_file(args.weights)
    data_hash = sha256_file(args.data)
    ultralytics, rtdetr_class, rtdetr_validator_class = import_local_ultralytics()
    state = git_state()
    environment = environment_info(ultralytics)
    dataset_nc = dataset_class_count(args.data, ultralytics)
    arguments = vars(args).copy()
    arguments["weights"] = str(args.weights)
    arguments["data"] = str(args.data)
    arguments["project"] = str(args.project)
    arguments["variants"] = selected_variants
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repository_root": str(ROOT.resolve()),
        "weights": str(args.weights),
        "weights_sha256_before": weights_hash_before,
        "data": str(args.data),
        "data_sha256": data_hash,
        "arguments": arguments,
        "fixed_validation": {"conf": 0.001, "iou": 0.7, "max_det": 300, "augment": False, "split": "val"},
        "git": state,
        "environment": environment,
        "dataset_nc": dataset_nc,
        "runtime_audits": [],
        "warning": POST_TRAINING_WARNING,
    }
    write_json(output / "run_manifest.json", manifest)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
    write_environment_text(output / "environment.txt", environment)
    write_git_text(output / "git_state.txt", state)

    rows: list[dict[str, Any]] = []
    parameters: dict[str, Any] | None = None
    original_collector: ActivationCollector | None = None
    activation_data: dict[str, Any] | None = None
    current_variant: str | None = None
    orchestration_error: str | None = None
    fatal_cuda_error = False
    weights_hash_after: str | None = None
    (output / "variants").mkdir(exist_ok=True)
    try:
        logger.info("Loading checkpoint once for module discovery and parameter audit (no validation).")
        audit_wrapper = rtdetr_class(str(args.weights))
        audit_module = find_unique_cscef_v2(audit_wrapper.model)
        parameters = parameter_audit(audit_module)
        write_json(output / "module_parameter_stats.json", parameters)
        del audit_module, audit_wrapper
        gc.collect()

        for variant in selected_variants:
            current_variant = variant
            manifest["current_variant"] = variant
            write_json(output / "run_manifest.json", manifest)
            logger.info("Starting variant %s from a fresh checkpoint load.", variant)
            row, collector, runtime_audit, variant_fatal_cuda = run_variant(
                variant,
                args,
                output,
                weights_hash_before,
                data_hash,
                rtdetr_class,
                rtdetr_validator_class,
                dataset_nc,
            )
            rows.append(row)
            manifest["runtime_audits"].append(runtime_audit)
            write_metric_artifacts(output, rows)
            write_json(output / "run_manifest.json", manifest)
            if variant == "original" and row["status"] == "success" and collector is not None:
                original_collector = collector
                activation_data = collector.as_dict()
                write_json(output / "activation_stats.json", activation_data)
                write_per_image_csv(output / "per_image_gate_stats.csv", collector.per_image_rows)
            if row["status"] == "success":
                logger.info("Variant %s completed successfully.", variant)
            else:
                logger.error("Variant %s failed:\n%s", variant, row["error"])
            if variant_fatal_cuda:
                fatal_cuda_error = True
                logger.critical(
                    "Stopping immediately after %s because a CUDA device-side failure made the context untrustworthy.",
                    variant,
                )
                break

        if args.plots and not fatal_cuda_error:
            create_plots(output, original_collector, rows)
    except Exception as error:
        orchestration_error = traceback.format_exc()
        fatal_cuda_error = fatal_cuda_error or is_unrecoverable_cuda_error(error) or is_unrecoverable_cuda_error(
            orchestration_error
        )
        logger.exception("Diagnostic orchestration failed.")
        if current_variant is not None and all(row["variant"] != current_variant for row in rows):
            rows.append(failed_metric_row(current_variant, weights_hash_before, data_hash, orchestration_error))
    finally:
        # These artifacts use CPU/file operations only and are safe to write after a poisoned CUDA context.
        write_metric_artifacts(output, rows)
        if parameters is None:
            write_json(
                output / "module_parameter_stats.json",
                {"status": "not_available", "error": orchestration_error},
            )
        if activation_data is None:
            failure_context = next(
                (
                    audit.get("failure_context")
                    for audit in reversed(manifest.get("runtime_audits", []))
                    if audit.get("failure_context") is not None
                ),
                None,
            )
            write_json(
                output / "activation_stats.json",
                {
                    "status": "not_collected",
                    "reason": "The original variant was not selected or did not succeed.",
                    "failure_context": failure_context,
                },
            )
            write_per_image_csv(output / "per_image_gate_stats.csv", [])
        try:
            weights_hash_after = sha256_file(args.weights)
        except Exception:
            logger.exception("Could not calculate the final checkpoint hash.")
        failures = [row["variant"] for row in rows if row["status"] != "success"]
        manifest.update(
            {
                "weights_sha256_after": weights_hash_after,
                "weights_unchanged": weights_hash_after == weights_hash_before,
                "failed_variants": failures,
                "fatal_cuda_context_error": fatal_cuda_error,
                "orchestration_error": orchestration_error,
                "finished_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "status": "failed"
                if failures or orchestration_error or weights_hash_after != weights_hash_before
                else "success",
            }
        )
        write_json(output / "run_manifest.json", manifest)
        try:
            generate_report(output, manifest, rows, parameters, activation_data)
        except Exception:
            report_error = traceback.format_exc()
            logger.exception("Could not generate the full diagnostic report.")
            orchestration_error = orchestration_error or report_error
            manifest["report_error"] = report_error
            manifest["orchestration_error"] = orchestration_error
            manifest["status"] = "failed"
            write_json(output / "run_manifest.json", manifest)
            (output / "diagnostic_report.md").write_text(
                "# CSCEF-v2 validation diagnostic\n\n"
                f"{POST_TRAINING_WARNING}\n\n"
                "The full report failed to render. See `run_manifest.json`, `variant_metrics.json`, and "
                "`diagnostic.log` for preserved failure details.\n",
                encoding="utf-8",
            )

    exit_code = 0
    if weights_hash_after != weights_hash_before:
        logger.critical("Checkpoint SHA256 changed or could not be verified; refusing success.")
        exit_code = 2
    elif fatal_cuda_error:
        exit_code = 3
    elif orchestration_error:
        exit_code = 2
    else:
        failures = [row["variant"] for row in rows if row["status"] != "success"]
        if failures:
            logger.error("Diagnostics completed with failed variants: %s", failures)
            exit_code = 1
        else:
            logger.info("Diagnostics completed; checkpoint SHA256 is unchanged.")
    logging.shutdown()
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """CLI entry point with concise fatal-error reporting."""
    parser = build_parser()
    try:
        return run(parser.parse_args(argv))
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
