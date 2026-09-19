"""Nonzero BLC persistence diagnostics, native half checkpoint and optimizer binding."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile

import torch
from blc_common import ROOT, KEYS, require, rebuild, verify_model, tensor_hash, write_json, paths, stamp
from check_blc import difference
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import ModelEMA


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
    result = dict(natural=natural, features=features, differing_candidate_positions=int((idx1 != idx2).sum()),
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


def restore(checkpoint, variant, device):
    """Native reconstruction + resume_training, with fresh optimizer/scaler/EMA."""
    weights = deepcopy(checkpoint["ema"]).float()
    model, _ = rebuild(weights.yaml, weights, variant)
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.model = model.to(device).train()
    trainer.args = IterableSimpleNamespace(**checkpoint["train_args"])
    trainer.optimizer = trainer.build_optimizer(trainer.model, "AdamW", .0005, .937, .0001)
    trainer.scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    trainer.ema = ModelEMA(trainer.model)
    trainer.resume = True
    trainer.epochs = trainer.args.epochs
    trainer.start_epoch = 0
    trainer.resume_training(deepcopy(checkpoint))
    require(trainer.start_epoch == checkpoint["epoch"] + 1, "Epoch did not resume")
    require(trainer.ema.updates == checkpoint["updates"], "EMA updates not restored")
    require(trainer.scaler.state_dict() == checkpoint["scaler"], "Scaler not restored")
    file_model = checkpoint["ema"].state_dict()
    require(all(torch.equal(v.cpu(), trainer.model.state_dict()[k].cpu().to(v.dtype)) for k, v in file_model.items()), "File EMA/model state lost")
    for saved, restored in zip(checkpoint["optimizer"]["param_groups"], trainer.optimizer.param_groups):
        for ident, parameter in zip(saved["params"], restored["params"]):
            for key, value in checkpoint["optimizer"]["state"].get(ident, {}).items():
                actual = trainer.optimizer.state[parameter][key]
                require(torch.equal(actual.cpu(), value.cpu().to(actual.dtype)) if torch.is_tensor(value) else actual == value,
                        "Optimizer moment/step lost: " + key)
    return trainer


def lifecycle(model, optimizer, ema, variant, args, device, scaler=None):
    report = {}
    try:
        return _lifecycle(model, optimizer, ema, variant, args, device, scaler, report)
    except BaseException as error:
        report.update(status="FAILED", error=repr(error))
        raise
    finally:
        write_json(paths(variant)["evidence"]/("lifecycle-"+stamp()+".json"), report)


def _lifecycle(model, optimizer, ema, variant, args, device, scaler, report):
    """Consumes a nonzero test model; never writes controlled_init or formal runs."""
    device = torch.device(device)
    require(torch.count_nonzero(model.model[5].blc.Wo.weight) > 0, "Lifecycle needs a nonzero BLC")
    model.eval()
    sample = torch.rand(1, 3, 160, 160, device=device)
    report.update(status="PASSED", scope="nonzero test copy / native file diagnostics", device=str(device))
    tmp_root = ROOT / "outputs/blc_v1/temporary"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lifecycle-", dir=tmp_root) as tmp:
        tmp = Path(tmp)
        sd_path, full_path = tmp/"state.pt", tmp/"model.pt"
        torch.save(model.state_dict(), sd_path)
        torch.save(model, full_path)
        sd = torch_load(sd_path, map_location=device)
        full = torch_load(full_path, map_location=device)
        restored = deepcopy(model)
        restored.load_state_dict(sd, strict=True)
        with torch.no_grad():
            original = model(sample)[0]
            a, b = restored(sample)[0], full(sample)[0]
        require(torch.equal(original, a) and torch.equal(original, b), "Same-dtype serialization output changed")
        report["state_dict_and_full_model_exact"] = True
        delta = []
        hook = model.model[5].blc.register_forward_hook(lambda m, x, y: delta.append(float((y-x[0]).norm())))
        try:
            with torch.no_grad():
                model(sample)
        finally:
            hook.remove()
        require(len(delta) == 1 and delta[0] > 0, "Nonzero branch lost")
        report["nonzero_increment_norm"] = delta[0]
        writer = RTDETRTrainer.__new__(RTDETRTrainer)
        writer.args = IterableSimpleNamespace(**args)
        writer.ema, writer.optimizer = ema, optimizer
        writer.scaler = scaler or torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        writer.epoch, writer.best_fitness, writer.fitness = 0, 0., 0.
        writer.metrics = {}
        writer.csv = tmp/"results.csv"
        writer.wdir, writer.last, writer.best, writer.save_period = tmp, tmp/"last.pt", tmp/"best.pt", -1
        writer.save_model()
        checkpoint = torch_load(writer.last, map_location="cpu")
        report["native_storage"] = dict(model=checkpoint["model"], ema_dtype=str(next(checkpoint["ema"].parameters()).dtype),
                                         optimizer="native convert_optimizer_state_dict_to_fp16", updates=checkpoint["updates"])
        first, second = [restore(checkpoint, variant, device) for _ in range(2)]
        require(groups(model, optimizer) == groups(first.model, first.optimizer) == groups(second.model, second.optimizer), "Native parameter-name grouping changed")
        report["optimizer_groups"] = groups(first.model, first.optimizer)
        for (name, p1), (_, p2) in zip(first.model.named_parameters(), second.model.named_parameters()):
            require(p1.data_ptr() != p2.data_ptr(), "Model copies share storage")
            for key, value in first.optimizer.state.get(p1, {}).items():
                if torch.is_tensor(value):
                    require(value.data_ptr() != second.optimizer.state[p2][key].data_ptr(), "Optimizer copies share storage")
        quantized = deepcopy(checkpoint["ema"]).float().to(device).eval()
        first.model.eval(); second.model.eval()
        report["file_quantized_forward"] = compare_networks(quantized, first.model, sample)
        require(report["file_quantized_forward"]["status"] in ("PASSED", "PRECISION_NOTE"),
                "File-quantized same-path forward differs without numerical localization")
        with torch.no_grad():
            q = quantized(sample)[0]
        report["live_vs_file_quantization"] = difference(original, q)
        # External identical gradients diagnose restore/binding; NOT independent training.
        for trainer in (first, second):
            for p in trainer.model.parameters():
                p.grad = torch.full_like(p, .001)
            trainer.scaler.scale(torch.zeros((), device=device, requires_grad=True)).backward()
            # Scale gradients exactly as native scaler expects, preserving restored scale.
            for p in trainer.model.parameters():
                p.grad.mul_(trainer.scaler.get_scale())
            trainer.optimizer_step()
        require(all(torch.equal(p, dict(second.model.named_parameters())[n]) for n, p in first.model.named_parameters()),
                "Same file + same external gradient update differs")
        report["dual_restore_same_gradient_native_update_exact"] = True
        report["resume_state_exact"] = dict(epoch=first.start_epoch, scaler=True, optimizer_moments_steps=True,
                                              ema=True, updates=checkpoint["updates"], shared_storage=False)
        # Fusion paths are separate numerical experiments; do not compare across
        # precision as if they were checkpoint-state equality.
        report["paths"] = {}
        tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            for mode in (["FP32", "AMP", "half"] if device.type == "cuda" else ["FP32"]):
                net = deepcopy(quantized).eval()
                if mode == "half":
                    net.half()
                inp = sample.half() if mode == "half" else sample
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=mode == "AMP"):
                    ref = net(inp)[0]
                # Native load_checkpoint/AutoBackend order is FP32 -> fuse ->
                # optional half. Calling the parent's fuse() on an already-half
                # model is a different, unsupported path (mixed bias dtypes).
                fused = deepcopy(quantized).fuse(verbose=False)
                if mode == "half":
                    fused.half()
                verify_model(fused, variant, fused=True)
                captured, hooks = [], []
                try:
                    hooks.append(fused.model[5].blc.register_forward_hook(lambda m, a, y: captured.append(float((y-a[0]).float().norm()))))
                    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=mode == "AMP"):
                        pred = fused(inp)[0]
                finally:
                    for hook in hooks:
                        hook.remove()
                require(len(captured) == 1 and captured[0] > 0, "Fusion bypassed nonzero BLC")
                diagnostics = compare_networks(net, fused, inp, amp=mode == "AMP")
                d = difference(pred, ref)
                require(d["finite"], "Nonfinite fused output")
                # Same fused file path via AutoBackend must preserve natural output.
                auto_path = tmp/(mode+".pt")
                torch.save(dict(model=deepcopy(net), ema=None, train_args=args), auto_path)
                backend = AutoBackend(str(auto_path), device=device, fp16=mode == "half", fuse=True, verbose=False)
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=mode == "AMP"):
                    auto = backend(inp)[0]
                same = difference(auto, pred)
                require(same["allclose"] and same["finite"], "Same-path AutoBackend output differs")
                report["paths"][mode] = dict(fusion=d, diagnostics=diagnostics, blc_calls=1,
                                             nonzero_increment=captured[0], autobackend=same, status=diagnostics["status"],
                                             fusion_order="native FP32 fuse then requested dtype")
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        if any(row["status"] not in ("PASSED", "PRECISION_NOTE") for row in report["paths"].values()):
            report["status"] = "PENDING"
    return report
