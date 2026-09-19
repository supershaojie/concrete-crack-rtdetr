"""Nonzero BLC persistence diagnostics, native half checkpoint and optimizer binding."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from blc_common import ROOT, KEYS, require, rebuild, verify_model, tensor_hash, write_json, paths, stamp
from check_blc import difference
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks import load_checkpoint
from blc_probe_io import temporary_probe
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.patches import torch_load


def compare_networks(first, second, sample, amp=False):
    """Record natural results; fixed-index replay diagnoses near-tie permutations only."""
    from unittest.mock import patch
    records, predictions = [], []
    for net in (first, second):
        record, hooks = {}, []
        try:
            for index in (5, 6, 7):
                hooks.append(net.model[index].register_forward_hook(
                    lambda m, a, y, index=index: record.update({str(index): y.detach().clone()})))
            hooks.append(net.model[-1].enc_score_head.register_forward_hook(
                lambda m, a, y: record.update(scores=y.detach().clone())))
            with torch.no_grad(), torch.autocast(device_type=sample.device.type, enabled=amp):
                predictions.append(net(sample)[0])
        finally:
            for hook in hooks:
                hook.remove()
        record["indices"] = record["scores"].max(-1).values.topk(300, dim=1).indices
        records.append(record)
    natural = difference(*predictions)
    features = {k: difference(records[0][k], records[1][k]) for k in ("5", "6", "7", "scores")}
    idx1, idx2 = records[0]["indices"], records[1]["indices"]
    result = dict(natural=natural, features=features, forward_count=2, differing_candidate_positions=int((idx1 != idx2).sum()),
                  same_candidate_set=bool(torch.equal(idx1.sort(1).values, idx2.sort(1).values)),
                  indices_sha256=[tensor_hash(idx1), tensor_hash(idx2)], status="PASSED" if natural["allclose"] else "PENDING")
    if not natural["allclose"]:
        original_topk = torch.topk
        fixed = idx1
        def replay(input, k, dim=None, largest=True, sorted=True, *, out=None):
            actual = original_topk(input, k, dim=dim, largest=largest, sorted=sorted, out=out)
            if input.ndim == 2 and k == 300 and dim == 1 and input.shape == records[0]["scores"].shape[:2]:
                return type(actual)((input.gather(1, fixed), fixed))
            return actual
        with patch("torch.topk", replay), torch.no_grad(), torch.autocast(device_type=sample.device.type, enabled=amp):
            replayed = second(sample)[0]
        result["fixed_indices_diagnostic_only"] = difference(predictions[0], replayed)
        result["forward_count"] += 1
        if natural["finite"] and all(v["allclose"] and v["finite"] for v in features.values()) and \
                result["differing_candidate_positions"] > 0 and result["fixed_indices_diagnostic_only"]["allclose"]:
            result["status"] = "PRECISION_NOTE"
            result["note"] = "Raw natural output differs after near-equal encoder score top-k ordering; fixed-index replay is diagnostic only."
    return result


def groups(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    result = [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]
    flat = [n for group in result for n in group]
    require(len(flat) == len(set(flat)) and set(flat) == set(names.values()), "Optimizer parameters missing/duplicated")
    return result


def cold_decoder(model):
    # Test copies only: let the native decoder rebuild anchors for this input.
    model.model[-1].shapes = []


def state_independence(first, second):
    a, b = first.state_dict(), second.state_dict()
    require(a.keys() == b.keys(), "Independent loader state keys differ")
    equal = all(a[k].dtype == b[k].dtype and a[k].device == b[k].device and torch.equal(a[k], b[k]) for k in a)
    separate = all(a[k].numel() == 0 or a[k].data_ptr() != b[k].data_ptr() for k in a)
    return dict(tensors=len(a), exact=equal, shared_storage=not separate)


def observed_forward(call, model, inp, amp, record):
    record.update(calls=0, increment_norms=[], completed=False)
    def capture(module, args, output):
        record["calls"] += 1
        record["increment_norms"].append(float((output - args[0]).float().norm()))
    hook = model.model[5].blc.register_forward_hook(capture)
    try:
        with torch.no_grad(), torch.autocast(device_type=inp.device.type, enabled=amp):
            result = call(inp)[0]
        record["completed"] = True
        return result
    finally:
        hook.remove()


def file_precision_paths(checkpoint_path, sample, variant, rows, same_path_only=False):
    """One input; two independent native file loads per existing precision mode."""
    device = sample.device
    modes = ["FP32", "AMP", "half"] if device.type == "cuda" else ["FP32"]
    rows.update({mode: dict(status="NOT_RUN", stage="not_started") for mode in ("FP32", "AMP", "half")})
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for mode in modes:
            row = rows[mode]
            row.update(status="RUNNING", stage="load_reference", device=str(device),
                       dtype="torch.float16" if mode == "half" else "torch.float32",
                       autocast=mode == "AMP", input_shape=list(sample.shape), input_sha256=tensor_hash(sample),
                       checkpoint=str(checkpoint_path), fusion=dict(status="NOT_RUN"),
                       autobackend=dict(status="NOT_RUN"),
                       reference_path="independent load_checkpoint(file, fuse=True): CPU FP32 fuse -> eval -> device -> requested dtype",
                       backend_path="independent AutoBackend(file, fuse=True): same native loader -> requested dtype",
                       cache_policy="cold decoder shapes on both independent instances; natural anchors/top-k")
            try:
                direct, _ = load_checkpoint(checkpoint_path, device=device, fuse=True)
                direct.half() if mode == "half" else direct.float()
                direct.eval().requires_grad_(False)
                row["stage"] = "load_autobackend"
                backend = AutoBackend(str(checkpoint_path), device=device, fp16=mode == "half", fuse=True, verbose=False)
                backend.eval()
                cold_decoder(direct)
                cold_decoder(backend.model)
                verify_model(direct, variant, fused=True)
                verify_model(backend.model, variant, fused=True)
                row["independent_state"] = state_independence(direct, backend.model)
                require(row["independent_state"]["exact"] and not row["independent_state"]["shared_storage"],
                        "Native/AutoBackend independent state differs or shares storage")
                inp = sample.half() if mode == "half" else sample
                row["stage"] = "direct_forward"
                row["direct_blc"] = {}
                expected = observed_forward(direct, direct, inp, mode == "AMP", row["direct_blc"])
                row["stage"] = "autobackend_forward"
                row["backend_blc"] = {}
                actual = observed_forward(backend, backend.model, inp, mode == "AMP", row["backend_blc"])
                # Persist raw observations BEFORE any assertion, including the failing mode.
                row["autobackend"] = {**difference(actual, expected), "status": "RUNNING"}
                row["stage"] = "same_path_assertion"
                require(row["autobackend"]["allclose"] and row["autobackend"]["finite"],
                        "Same-path AutoBackend output differs")
                for key in ("direct_blc", "backend_blc"):
                    require(row[key]["calls"] == 1 and row[key]["increment_norms"][0] > 0,
                            "Native file path bypassed nonzero BLC")
                row["autobackend"]["status"] = "PASSED"
                # Separate unfused vs CPU-fused native file comparison. Keep its
                # original natural-output criteria and diagnostic-only top-k replay.
                if not same_path_only:
                    row["stage"] = "unfused_fused_comparison"
                    unfused, _ = load_checkpoint(checkpoint_path, device=device, fuse=False)
                    unfused.half() if mode == "half" else unfused.float()
                    unfused.eval()
                    cold_decoder(unfused)
                    cold_decoder(direct)
                    row["fusion"] = compare_networks(unfused, direct, inp, amp=mode == "AMP")
                    require(row["fusion"]["natural"]["finite"], "Nonfinite fused output")
                    row["status"] = row["fusion"]["status"]
                    del unfused
                else:
                    row["status"] = "PASSED"
                row["stage"] = "complete"
                del direct, backend, expected, actual
            except BaseException as error:
                row.update(status="FAILED", exception_stage=row["stage"], error=repr(error))
                if row["autobackend"]["status"] == "RUNNING":
                    row["autobackend"]["status"] = "FAILED"
                raise
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32


def lifecycle(model, optimizer, ema, variant, args, device, scaler=None, checkpoint_path=None, report=None):
    report = report if report is not None else {}
    report.update(status="RUNNING", stage="initialization", paths={
        mode: dict(status="NOT_RUN") for mode in ("FP32", "AMP", "half")})
    try:
        return _lifecycle(model, optimizer, ema, variant, args, device, scaler, report, checkpoint_path)
    except BaseException as error:
        report.update(status="FAILED", exception_stage=report["stage"], error=repr(error))
        raise
    finally:
        write_json(paths(variant)["evidence"]/("lifecycle-"+stamp()+".json"), report)


def _lifecycle(model, optimizer, ema, variant, args, device, scaler, report, checkpoint_path):
    """No synthetic optimizer replays: actual native resume audits that state once."""
    device = torch.device(device)
    require(torch.count_nonzero(model.model[5].blc.Wo.weight) > 0, "Lifecycle needs a nonzero BLC")
    model.eval()
    sample = torch.linspace(0, 1, 3 * 160 * 160, device=device).reshape(1, 3, 160, 160)
    report.update(scope="one fixed input; nonzero temporary copy; native file diagnostics", device=str(device))
    with temporary_probe(ROOT / "outputs/blc_v1/temporary", "lifecycle", report) as tmp:
        report["stage"] = "state_dict_full_model"
        sd_path, full_path = tmp/"state.pt", tmp/"model.pt"
        torch.save(model.state_dict(), sd_path)
        torch.save(model, full_path)
        sd = torch_load(sd_path, map_location=device)
        full = torch_load(full_path, map_location=device)
        restored = deepcopy(model)
        restored.load_state_dict(sd, strict=True)
        cold_decoder(model); cold_decoder(restored); cold_decoder(full)
        report["live_blc"] = {}
        original = observed_forward(model, model, sample, False, report["live_blc"])
        with torch.no_grad():
            a, b = restored(sample)[0], full(sample)[0]
        report["serialization"] = dict(state_dict=difference(a, original), full_model=difference(b, original))
        report["state_dict_and_full_model_exact"] = bool(torch.equal(original, a) and torch.equal(original, b))
        require(report["state_dict_and_full_model_exact"], "Same-dtype serialization output changed")
        delta = report["live_blc"]["increment_norms"]
        report["nonzero_increment_norm"] = delta[0] if delta else None
        require(len(delta) == 1 and delta[0] > 0, "Nonzero branch lost")
        del sd, full, restored, a, b
        # Capacity passes its one native checkpoint; standalone legacy diagnostics
        # may still create one native checkpoint inside this temporary directory.
        report["stage"] = "native_checkpoint"
        if checkpoint_path is None:
            writer = RTDETRTrainer.__new__(RTDETRTrainer)
            writer.args = IterableSimpleNamespace(**args)
            writer.ema, writer.optimizer = ema, optimizer
            writer.scaler = scaler or torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
            writer.epoch, writer.best_fitness, writer.fitness = 0, 0., 0.
            writer.metrics = {}
            writer.csv = tmp/"results.csv"
            writer.wdir, writer.last, writer.best, writer.save_period = tmp, tmp/"last.pt", tmp/"best.pt", -1
            writer.save_model()
            checkpoint_path = writer.last
        checkpoint = torch_load(checkpoint_path, map_location="cpu")
        report["native_storage"] = dict(model=checkpoint["model"],
                                       ema_dtype=str(next(checkpoint["ema"].parameters()).dtype),
                                       optimizer="native convert_optimizer_state_dict_to_fp16", updates=checkpoint["updates"])
        quantized = deepcopy(checkpoint["ema"]).float().to(device).eval()
        cold_decoder(quantized)
        with torch.no_grad():
            q = quantized(sample)[0]
        report["live_vs_file_quantization"] = difference(original, q)
        del checkpoint, quantized, original, q
        report["stage"] = "precision_paths"
        file_precision_paths(checkpoint_path, sample, variant, report["paths"])
        report["status"] = "PASSED" if all(row["status"] in ("PASSED", "PRECISION_NOTE")
                                           for row in report["paths"].values() if row["status"] != "NOT_RUN") else "PENDING"
        report["stage"] = "complete"
    return report
