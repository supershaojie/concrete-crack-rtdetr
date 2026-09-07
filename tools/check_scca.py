"""Optional small-batch validation; never required by start-direct and never starts 200 epochs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import time

import torch
from init_scca import ROOT, VARIANTS, build, controlled_models, initialize, runtime, verify_model, write_json, require
from train_scca import rebuild_audit, recipe
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer


def tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in tensors(v)]
    if isinstance(value, dict):
        return [t for v in value.values() for t in tensors(v)]
    return []


def comparison(a, b):
    aa, bb = tensors(a), tensors(b)
    require(len(aa) == len(bb), "Output tensor count differs")
    maximum = 0.
    for x, y in zip(aa, bb):
        x, y = x.detach().cpu(), y.detach().cpu()
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
        if x.numel():
            maximum = max(maximum, float((x - y).abs().max()))
    return maximum


def rebuild(source, variant):
    # Exercise exactly the function called by RTDETR.train() during nc=1 rebuilding.
    trainer = object.__new__(RTDETRTrainer)
    trainer.data = dict(nc=1, channels=3)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        target = trainer.get_model(cfg=source.yaml, weights=source, verbose=False)
    return target, rebuild_audit(source, target, variant)


def native_updates(model, device, amp):
    model = model.to(device).train()
    model.nc = model.model[-1].nc  # RTDETRTrainer.set_model_attributes supplies this in normal training.
    trainer = object.__new__(RTDETRTrainer)
    opt = trainer.build_optimizer(model, name="AdamW", lr=.0005, momentum=.937, decay=.0001)
    added = {n: p for n, p in model.named_parameters() if ".scca_" in n}
    counts = {n: sum(id(p) == id(v) for g in opt.param_groups for v in g["params"]) for n, p in added.items()}
    require(all(v == 1 for v in counts.values()), "Missing/duplicated optimizer entries")
    # Two steps, then a last partial batch; native RT-DETR loss and DN with bounded synthetic targets.
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=128.)
    records = []
    for step, batch_size in enumerate((2, 1)):
        batch = dict(img=torch.rand(batch_size, 3, 160, 160, device=device),
                     cls=torch.zeros(batch_size, 1, device=device), batch_idx=torch.arange(batch_size, device=device),
                     bboxes=torch.tensor([[.5, .5, .2, .3]], device=device).repeat(batch_size, 1))
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=amp):
            prediction = model.predict(batch["img"], batch=dict(cls=batch["cls"].flatten().long(), bboxes=batch["bboxes"],
                                       batch_idx=batch["batch_idx"].long(), gt_groups=[1] * batch_size))
            dn = prediction[-1]
            require(dn is not None and dn["dn_num_split"][0] > 0, "Native DN inactive")
            loss, items = model.loss(batch, prediction)
        require(torch.isfinite(loss).all(), "Nonfinite native loss")
        scaler.scale(loss.sum()).backward()
        scaler.unscale_(opt)
        grads = {n: float(p.grad.abs().max()) if p.grad is not None else None for n, p in added.items()}
        require(all(v is not None and v >= 0 and v < float("inf") for v in grads.values()), "Nonfinite/missing SCCA gradients")
        require(grads["model.9.scca_o.weight"] > 0, "O has no gradient")
        upstream = [v for k, v in grads.items() if k != "model.9.scca_o.weight"]
        require(all(v == 0 for v in upstream) if step == 0 else all(v > 0 for v in upstream), "Zero-init gradient staging failed")
        old = {n: p.detach().clone() for n, p in added.items()}
        scaler.step(opt)
        scaler.update()
        records.append(dict(step=step, batch=batch_size, image_size=160, loss=float(loss.sum()), loss_items=items.detach().cpu().tolist(),
                            dn_num_split=dn["dn_num_split"], max_gradients=grads,
                            max_updates={n: float((p.detach() - old[n]).abs().max()) for n, p in added.items()}))
    return dict(amp=amp, debug_scaler_init_scale=128, optimizer_occurrences=counts, steps=records)


def run(source, output):
    report = dict(runtime=runtime(), formal_training_started=False, full_server_preflight="NOT_RUN", variants={},
                  scope="Synthetic native loss/DN at 160, batch2 then batch1; FP32 network inference at 640 and non-square 160x192")
    output.mkdir(parents=True, exist_ok=False)
    for variant in VARIANTS:
        row = report["variants"][variant] = {}
        reference, target, loading = controlled_models(source, variant)
        write_json(output / f"{variant}_mapping.json", loading)
        row["parameters_nc80"] = sum(p.numel() for p in target.parameters())
        row["increment"] = sum(p.numel() for p in target.parameters()) - sum(p.numel() for p in reference.parameters())
        target, row["nc1_loading"] = rebuild(target, variant)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            trainer = object.__new__(RTDETRTrainer)
            trainer.data = dict(nc=1, channels=3)
            reference = trainer.get_model(cfg=reference.yaml, weights=reference, verbose=False)
        row["parameters_nc1"] = sum(p.numel() for p in target.parameters())
        row["baseline_parameters_nc1"] = sum(p.numel() for p in reference.parameters())
        target.eval(); reference.eval()
        row["fp32_max_abs"] = {}
        with torch.no_grad():
            for h, w in ((640, 640), (160, 192)):
                x = torch.rand(1, 3, h, w)
                row["fp32_max_abs"][f"1x3x{h}x{w}"] = comparison(reference(x), target(x))
        # Save/restore updated model below; formal initialization is freshly and independently generated.
        row["cpu_native"] = native_updates(deepcopy(target), "cpu", False)
        if torch.cuda.is_available():
            row["cuda_amp_native"] = native_updates(deepcopy(target), "cuda", True)
            half = deepcopy(target).eval().cuda().half()
            with torch.no_grad():
                predicted = half(torch.rand(1, 3, 160, 192, device="cuda", dtype=torch.float16))
            require(all(torch.isfinite(t).all() for t in tensors(predicted)), "Half model failed")
            row["real_model_half"] = "passed, batch1 160x192"
            del half, predicted
            # Tiny measured inference workload; not a throughput claim for the server or batch16.
            row["cost_batch1_160"] = {}
            for name, cpu_model in (("baseline", reference), ("scca", target)):
                m = deepcopy(cpu_model).cuda().eval()
                x = torch.rand(1, 3, 160, 160, device="cuda")
                with torch.no_grad(), torch.autocast("cuda"):
                    for _ in range(3): m(x)
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    for _ in range(10): m(x)
                    torch.cuda.synchronize()
                row["cost_batch1_160"][name] = dict(mean_ms=(time.perf_counter()-start)*100,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
                del m, x
                torch.cuda.empty_cache()
        else:
            row["cuda_amp_native"] = row["real_model_half"] = "NOT_RUN: CUDA unavailable"
        formal = output / f"{variant}_fresh_init.pt"
        row["initialization"] = initialize(source, formal, variant)
        restored = RTDETR(str(formal)).model
        verify_model(restored, variant, zero=True)
        recipe_args, diff = recipe(ROOT / "docs/scca/c2_args.yaml", variant, ROOT / "weights" / f"{variant}_scca_controlled_init.pt")
        write_json(output / f"{variant}_parameter_diff.json", diff)
        row["recipe_changed_fields"] = [r["field"] for r in diff if r["changed"]]
        row["status"] = "passed"
        del reference, target, restored
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/scca_check")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    print(json.dumps(run(args.source, args.output), indent=2, default=str))
