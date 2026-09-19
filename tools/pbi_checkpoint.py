"""PBI-only native checkpoint policy: keep optimizer moments in FP32.

The parent serializer rounds every FP32 optimizer tensor other than ``step`` to
FP16. Small AdamW second moments can therefore become zero. This subclass keeps
the native EMA-half checkpoint and every native field/file-selection rule, but
serializes an independent optimizer state copy without that lossy conversion.
It does not change the live model, optimizer, scaler, or shared Ultralytics code.

The native serializer AST is pinned below. If upstream saving semantics change,
this local override must be reviewed instead of silently omitting new behavior.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime
import hashlib
import inspect
import io
import textwrap

import torch

from ultralytics import __version__
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import GIT
from ultralytics.utils.torch_utils import unwrap_model

OPTIMIZER_POLICY = "optimizer_fp32_v1"
NATIVE_SERIALIZER_SOURCE_SHA256 = "5f2b57d19a7b7ea4441ff2a7f922a59e202c38a86141c4c4aa907f5b709eec71"
NATIVE_SERIALIZER_AST_SHA256 = "b5b321397df68dafb67ff2c9eb763c491f6c177acf527a46cbbbcbc5d8551de3"


def native_serializer_identity():
    """Verify the reviewed native saving semantics and return source identities."""
    source = inspect.getsource(RTDETRTrainer.save_model).replace("\r\n", "\n")
    semantic = ast.dump(ast.parse(textwrap.dedent(source)), include_attributes=False)
    identity = {
        "native_serializer_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "native_serializer_ast_sha256": hashlib.sha256(semantic.encode("utf-8")).hexdigest(),
    }
    # Exact reviewed source is interpreter-independent. AST is an alternative
    # for comment/whitespace-only changes; newer Python AST representations can
    # add optional fields, so do not reject byte-identical reviewed source.
    if (identity["native_serializer_sha256"] != NATIVE_SERIALIZER_SOURCE_SHA256 and
            identity["native_serializer_ast_sha256"] != NATIVE_SERIALIZER_AST_SHA256):
        raise RuntimeError("Native save_model semantics changed; review PBI checkpoint override before saving")
    return identity


def optimizer_param_names(model, optimizer):
    """Map serialized optimizer group/parameter order to unwrapped model names.

    PyTorch restores optimizer states by group order, not by parameter name. The
    explicit mapping lets a resume audit verify identity before comparing values.
    """
    unwrapped = unwrap_model(model)
    names = {id(parameter): name for name, parameter in unwrapped.named_parameters()}
    trainable = {id(parameter) for parameter in unwrapped.parameters() if parameter.requires_grad}
    result, seen = [], set()
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in names:
                raise RuntimeError("Optimizer contains a parameter absent from the current model")
            if identity in seen:
                raise RuntimeError("Optimizer contains a duplicated parameter")
            seen.add(identity)
            group_names.append(names[identity])
        result.append(group_names)
    if seen != trainable:
        raise RuntimeError("Optimizer must cover every trainable model parameter exactly once, without frozen extras")
    return result


def require_checkpoint_policy(checkpoint):
    """Validate original checkpoint tensors and positional optimizer identity.

    Call immediately after loading, before any ``.float()`` or optimizer restore.
    A historical FP16 optimizer state cannot regain lost information by casting;
    absence of the policy or an FP16 source moment is an explicit rejection.
    This function only inspects values and never mutates the checkpoint or EMA.
    """
    metadata = checkpoint.get("pbi_checkpoint", {})
    if not isinstance(metadata, dict) or metadata.get("policy") != OPTIMIZER_POLICY:
        raise RuntimeError("Checkpoint lacks optimizer_fp32_v1 policy; historical FP16 optimizer information is irreversibly lost")
    optimizer_state = checkpoint.get("optimizer")
    if not isinstance(optimizer_state, dict) or not isinstance(optimizer_state.get("state"), dict) or not optimizer_state["state"]:
        raise RuntimeError("FP32 resume requires nonempty original optimizer state")
    groups, names = optimizer_state.get("param_groups"), metadata.get("optimizer_param_names")
    if not isinstance(groups, list) or not groups or not isinstance(names, list) or len(groups) != len(names):
        raise RuntimeError("Checkpoint optimizer group/name metadata differs")
    ema = checkpoint.get("ema")
    if not isinstance(ema, torch.nn.Module):
        raise RuntimeError("Native PBI resume requires its original EMA model")
    parameters = dict(unwrap_model(ema).named_parameters())
    id_to_name, seen_names = {}, set()
    for group, group_names in zip(groups, names):
        ids = group.get("params") if isinstance(group, dict) else None
        if not isinstance(ids, list) or not isinstance(group_names, list) or len(ids) != len(group_names):
            raise RuntimeError("Checkpoint optimizer param-ID/name lengths differ")
        for parameter_id, name in zip(ids, group_names):
            if type(parameter_id) is not int or parameter_id in id_to_name:
                raise RuntimeError("Checkpoint optimizer parameter IDs must be unique integers")
            if not isinstance(name, str) or name in seen_names or name not in parameters:
                raise RuntimeError("Checkpoint optimizer names must be unique original EMA parameter names")
            id_to_name[parameter_id] = name
            seen_names.add(name)
    if seen_names != set(parameters):
        raise RuntimeError("Checkpoint optimizer names omit model parameters")
    if (any(type(key) is not int for key in optimizer_state["state"]) or
            not set(optimizer_state["state"]).issubset(id_to_name)):
        raise RuntimeError("Checkpoint optimizer state has an unknown parameter ID")
    moment_tensors = 0
    for parameter_id, state in optimizer_state["state"].items():
        if not isinstance(state, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
            raise RuntimeError("Incomplete AdamW checkpoint state")
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name not in state:
                continue
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
                raise RuntimeError("Original AdamW checkpoint moment is not FP32; casting cannot recover lost information")
            if value.shape != parameters[id_to_name[parameter_id]].shape or not bool(torch.isfinite(value).all()):
                raise RuntimeError("Original AdamW checkpoint moment has invalid shape or nonfinite values")
            moment_tensors += 1
    return dict(policy=OPTIMIZER_POLICY, optimizer_param_names=deepcopy(names),
                parameter_ids_by_group=[list(group["params"]) for group in groups],
                state_entries=len(optimizer_state["state"]), fp32_moment_tensors=moment_tensors,
                original_tensors_inspected_before_cast=True, checkpoint_unmodified=True)


class PBICheckpointTrainer(RTDETRTrainer):
    """RTDETRTrainer with native checkpoint semantics and FP32 optimizer state."""

    def save_model(self):
        """Save native fields and destinations; only optimizer precision differs."""
        identity = native_serializer_identity()
        names = optimizer_param_names(self.model, self.optimizer)
        # The copy owns its tensors. Never cast or replace active optimizer state.
        optimizer_state = deepcopy(self.optimizer.state_dict())
        for state in optimizer_state["state"].values():
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = state.get(name)
                if value is not None and (not isinstance(value, torch.Tensor) or value.dtype != torch.float32):
                    raise RuntimeError(
                        "PBI requires active FP32 AdamW moments; old FP16 checkpoint information cannot be recovered by casting"
                    )
        metadata = dict(policy=OPTIMIZER_POLICY, optimizer_param_names=names, **identity)

        # Kept field-for-field and route-for-route with the guarded native method.
        # Intentional differences: no optimizer FP16 conversion; audit metadata.
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(unwrap_model(self.ema.ema)).half(),
                "updates": self.ema.updates,
                "optimizer": optimizer_state,
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
                "pbi_checkpoint": metadata,
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)
