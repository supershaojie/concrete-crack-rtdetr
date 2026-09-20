"""Bounded, read-only evidence from the existing native half EMA validation forward."""
from collections import Counter, deque
import json

import torch


NAME_LIMIT = 20
EVENT_LIMIT = 8
RAW_NAMES = ("dec_bboxes", "dec_scores", "enc_bboxes", "enc_scores", "dn_meta")


def tensors(value, path):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            yield from tensors(child, f"{path}[{i}]")
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensors(child, f"{path}.{key}")


def tensor_stats(value, path):
    """JSON scalars only. Integer/bool buffers have no NaN/Inf and are not cast to half."""
    x = value.detach()
    mask = torch.isfinite(x)
    all_finite = bool(mask.all())
    is_float = x.is_floating_point()
    good = x if all_finite else x[mask]
    # Floating abs/max stays in the existing dtype; avoid a full FP64 activation copy.
    maximum = float((good if is_float else good.to(torch.float64)).abs().max()) if good.numel() else None
    return dict(path=path, shape=list(x.shape), dtype=str(x.dtype), numel=x.numel(),
                nan=int(torch.isnan(x).sum()) if is_float and not all_finite else 0,
                posinf=int(torch.isposinf(x).sum()) if is_float and not all_finite else 0,
                neginf=int(torch.isneginf(x).sum()) if is_float and not all_finite else 0,
                finite=all_finite, finite_max_abs=maximum)


def state_stats(model):
    groups, index = {}, {}
    for kind, items in (("parameters", model.named_parameters()), ("buffers", model.named_buffers())):
        row = dict(tensor_items=0, nonfinite_items=0, nonfinite_names=[], details=[],
                   integer_items=0, fp16_range_exceeded_items=0, fp16_range_exceeded_names=[], dtypes={})
        dtypes = Counter()
        for name, value in items:
            stat = tensor_stats(value, name)
            index[kind + "." + name] = stat["finite"]
            row["tensor_items"] += 1
            row["integer_items"] += int(not value.is_floating_point())
            dtypes[stat["dtype"]] += 1
            if not stat["finite"]:
                row["nonfinite_items"] += 1
                if len(row["details"]) < NAME_LIMIT:
                    row["nonfinite_names"].append(name)
                    row["details"].append(stat)
            if value.is_floating_point() and stat["finite_max_abs"] is not None and stat["finite_max_abs"] > 65504:
                row["fp16_range_exceeded_items"] += 1
                if len(row["fp16_range_exceeded_names"]) < NAME_LIMIT:
                    row["fp16_range_exceeded_names"].append(name)
        row["dtypes"] = dict(dtypes)
        row["finite"] = row["nonfinite_items"] == 0
        groups[kind] = row
    return dict(finite=all(g["finite"] for g in groups.values()), **groups), index


def output_stats(out):
    """Keep y AND every returned raw tensor; raw logits are consumed by native val loss."""
    if isinstance(out, (tuple, list)) and len(out) == 2:
        entries = list(tensors(out[0], "y"))
        raw = out[1]
        if isinstance(raw, (tuple, list)) and len(raw) == len(RAW_NAMES):
            for name, value in zip(RAW_NAMES, raw):
                entries.extend(tensors(value, "raw." + name))
        else:
            entries.extend(tensors(raw, "raw"))
    else:
        entries = list(tensors(out, "out"))
    if len(entries) > 64:
        raise RuntimeError("Unexpected output tensor count exceeds diagnostic cap")
    rows = [tensor_stats(value, path) for path, value in entries]
    return dict(tensors=rows, nonfinite_paths=[r["path"] for r in rows if not r["finite"]],
                scope="complete returned tensors; y for metrics and raw for native validation loss")


class HalfEMADiagnostic:
    """Observe module boundaries, not individual functional operators or a proven root cause.

    All callbacks/hooks return None and catch their own diagnostic failures. A native exception
    remains primary. Missing/failed evidence blocks a successful diagnostic on normal exit.
    """
    def __init__(self, model, validator, row, *, amp, ema_updates):
        self.model, self.validator = model, validator
        self.handles, self.stack = [], []
        self.recent = deque(maxlen=3)
        self.before_index = {}
        self.start_callback = self.on_val_start
        self.data = row["diagnostic"] = dict(
            schema=1, scope="same native half EMA batch; no replay or extra forward", root_cause="UNDETERMINED",
            amp=bool(amp), ema_updates=int(ema_updates), forward_calls=0, module_boundary_events=0,
            first_nonfinite=None, anomalies=[], semantic_masks=[], nonfinite_events=0, diagnostic_errors=[],
            outputs=None, hooks_removed=False,
            coverage="pre/post hooks on all named modules including backbone, neck and decoder internals; canonical names for shared modules",
            limits=dict(anomaly_events=EVENT_LIMIT, names_or_tensors_per_event=NAME_LIMIT, preceding_finite_events=3),
            interpretation="First observed boundary is not the root cause. Functional operations are only bracketed; internal mask/anchor Inf does not excuse nonfinite returned tensors.")

    def safe(self, where, function, *args):
        try:
            function(*args)
        except Exception as error:
            if len(self.data["diagnostic_errors"]) < NAME_LIMIT:
                self.data["diagnostic_errors"].append(dict(where=where, error=repr(error)[:1000]))

    def before_half(self):
        self.data["ema_before_half"], self.before_index = state_stats(self.model)
        self.data["model_training_before_validation"] = bool(self.model.training)

    def after_half(self):
        state, after = state_stats(self.model)
        new = [n for n, ok in after.items() if not ok and self.before_index.get(n) is True]
        state.update(new_nonfinite_items=len(new), new_nonfinite_names=new[:NAME_LIMIT],
                     model_training=bool(self.model.training), validator_training=bool(self.validator.training),
                     validator_half=bool(self.validator.args.half))
        self.data["ema_after_native_half"] = state

    def on_val_start(self, validator):
        # Native validator has already done model.half() and model.eval(), before any batch forward.
        self.safe("after_native_half", self.after_half)

    def root_input(self, args, kwargs):
        self.data["forward_calls"] += 1
        image = args[0] if args else kwargs["x"]
        self.data["input"] = dict(tensor_stats(image, "input.img"), actual_batch_size=int(image.shape[0]),
                                  model_training=bool(self.model.training), grad_enabled=torch.is_grad_enabled(),
                                  cuda_autocast=torch.is_autocast_enabled(), cpu_autocast=torch.is_autocast_cpu_enabled(),
                                  cuda_autocast_dtype=str(torch.get_autocast_gpu_dtype()),
                                  inference_mode=torch.is_inference_mode_enabled())

    def anchors(self):
        # Read an existing cache only. Never regenerate anchors or infer selected anchor provenance.
        head = getattr(self.model, "model", [None])[-1]
        anchor, valid = getattr(head, "anchors", None), getattr(head, "valid_mask", None)
        if isinstance(anchor, torch.Tensor) and isinstance(valid, torch.Tensor) and anchor.numel():
            expected = (~valid).expand_as(anchor)
            self.data["anchor_cache"] = dict(tensor_stats(anchor, "decoder.anchors"),
                positive_inf_matches_invalid_mask=bool(torch.equal(torch.isposinf(anchor), expected)),
                invalid_anchor_elements=int(expected.sum()),
                interpretation="Cached invalid anchors intentionally use +Inf before sigmoid. Selected reference provenance is not observed.")

    def boundary(self, name, module, side, value, inputs_finite=None):
        self.data["module_boundary_events"] += 1
        entries = list(tensors(value, side))
        bad, masks = [], []
        for path, tensor in entries:
            if bool(torch.isfinite(tensor).all()):
                continue
            # Only explicit attention mask arguments with -Inf (no NaN/+Inf) have known sentinel semantics.
            mask_path = (path in {"input.kwargs.attn_mask", "input.kwargs.key_padding_mask", "input.kwargs.padding_mask"}
                         or (type(module).__name__ == "MultiheadAttention" and path in {"input.args[3]", "input.args[5]"}))
            if mask_path and not bool(torch.isnan(tensor).any()) and not bool(torch.isposinf(tensor).any()):
                masks.append((path, tensor))
            else:
                bad.append((path, tensor))
        event = dict(sequence=self.data["module_boundary_events"], module=name or "<root>",
                     module_type=type(module).__name__, side=side,
                     input_finite=(not bad and not masks) if side == "input" else inputs_finite,
                     enclosing_module=self.stack[-1][0] if self.stack else None)
        if masks and len(self.data["semantic_masks"]) < EVENT_LIMIT:
            self.data["semantic_masks"].append(dict(event, meaning="explicit attention mask -Inf sentinel, not activation overflow",
                                                  tensors=[tensor_stats(t, p) for p, t in masks[:NAME_LIMIT]]))
        if bad:
            self.data["nonfinite_events"] += 1
            if len(self.data["anomalies"]) < EVENT_LIMIT:
                event.update(nonfinite_tensor_items=len(bad), tensors=[tensor_stats(t, p) for p, t in bad[:NAME_LIMIT]],
                             preceding_finite=list(self.recent), root_cause="UNDETERMINED")
                if type(module).__name__ == "DeformableTransformerDecoder" and side == "input":
                    event["reference_note"] = "Unactivated reference logits may carry invalid-anchor +Inf; this does not explain returned output anomalies."
                self.data["anomalies"].append(event)
                if self.data["first_nonfinite"] is None:
                    self.data["first_nonfinite"] = event
        elif not masks:
            self.recent.append(dict(event, finite=True, tensor_items=len(entries),
                                    tensors=[dict(path=p, shape=list(t.shape), dtype=str(t.dtype)) for p, t in entries[:3]]))
        return not bad and not masks  # Actual finiteness, even when mask Inf is intentional.

    def pre(self, name, module, args, kwargs):
        if name == "":
            self.safe("root_input", self.root_input, args, kwargs)
        if type(module).__name__ == "DeformableTransformerDecoder":
            self.safe("anchor_cache", self.anchors)
        ok = self.boundary(name, module, "input", dict(args=args, kwargs=kwargs))
        self.stack.append((name or "<root>", ok))

    def post(self, name, module, out):
        if name == "":
            self.data["outputs"] = output_stats(out)
        _, ok = self.stack.pop() if self.stack else (None, None)
        self.boundary(name, module, "output", out, ok)

    def __enter__(self):
        self.safe("before_half", self.before_half)
        try:
            self.validator.add_callback("on_val_start", self.start_callback)
            for name, module in self.model.named_modules():
                def pre(m, a, kw, n=name):
                    self.safe("pre:" + n, self.pre, n, m, a, kw)
                def post(m, a, kw, out, n=name):
                    self.safe("post:" + n, self.post, n, m, out)
                self.handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
                self.handles.append(module.register_forward_hook(post, with_kwargs=True))
            self.data["hook_count"] = len(self.handles)
        except BaseException:
            self.cleanup()
            raise
        return self

    def cleanup(self):
        for handle in self.handles:
            self.safe("remove_hook", handle.remove)
        self.handles.clear()
        callbacks = self.validator.callbacks.get("on_val_start", [])
        if self.start_callback in callbacks:
            callbacks.remove(self.start_callback)
        self.stack.clear()
        self.data["hooks_removed"] = not any(e["where"] == "remove_hook" for e in self.data["diagnostic_errors"])

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.data["native_exception"] = repr(exc)[:1000]
        self.cleanup()
        # Always retain native failure; never turn missing/broken diagnostics into successful evidence.
        if exc is None:
            if self.data["diagnostic_errors"] or self.data["forward_calls"] != 1 or self.data["outputs"] is None:
                raise RuntimeError("Half EMA diagnostic incomplete; inspect diagnostic_errors")
            if not all(self.data.get(k, {}).get("finite") for k in ("ema_before_half", "ema_after_native_half")):
                raise RuntimeError("Nonfinite or unobserved EMA parameters/buffers before/after native half conversion")
        return False


def brief(report):
    """Small failure summary for server handoff; missing evidence explicitly stays unlocated."""
    d = report.get("half_ema_val", {}).get("diagnostic", {})
    return dict(status=report.get("status"), phase=report.get("phase"), error=report.get("error"),
                first_nonfinite=d.get("first_nonfinite"), output_nonfinite_paths=(d.get("outputs") or {}).get("nonfinite_paths"),
                ema_before_half=d.get("ema_before_half"), ema_after_native_half=d.get("ema_after_native_half"),
                diagnostic_errors=d.get("diagnostic_errors"),
                localization=("INCOMPLETE/UNLOCATED: diagnostic errors present" if d.get("diagnostic_errors") else
                              "observed boundary only; root cause UNDETERMINED" if d.get("first_nonfinite") else "UNLOCATED"))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    path = Path(sys.argv[1])
    report = json.loads(path.read_text()) if path.is_file() else dict(status="NOT_WRITTEN", phase="entry/local_checks")
    print(json.dumps(dict(report=str(path), **brief(report)), indent=2, allow_nan=False))
