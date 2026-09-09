"""Bounded synthetic EVC checks; no formal training or full-dataset evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import torch
from init_evc import (ROOT, build, controlled_models, initialize, is_added, require,
                      runtime, sha256, verify_model, write_json)
from train_evc import rebuild_audit, recipe, optimizer_groups
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules.evc_deform import EVCMSDeformAttn, sample_deformable_values
from ultralytics.nn.modules.transformer import MSDeformAttn
from ultralytics.utils import YAML


def compare(a, b, atol=2e-6, rtol=2e-5):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return float((a - b).abs().max()) if a.numel() else 0.


def formulas():
    torch.manual_seed(42)
    module = EVCMSDeformAttn()
    q = torch.randn(2, 7, 256)
    samples = torch.randn(2, 7, 8, 3, 4, 32)
    z = torch.randn(2, 7, 8, 3, 4) * 3
    w, e = module.evidence_weights(q, samples, z)
    baseline = z.flatten(-2).softmax(-1).view_as(z)
    result = dict(zero_identity_max_abs=compare(w, baseline), normalization_max_abs=compare(w.sum((-2, -1)), torch.ones_like(w[..., 0, 0])))
    require((w >= 0).all(), "Negative attention weight")
    with torch.no_grad():
        module.evc_a_p.fill_(.7)
    wp, _ = module.evidence_weights(q, samples, z)
    result["point_only_scale_share_max_abs"] = compare(wp.sum(-1), baseline.sum(-1))
    with torch.no_grad():
        module.evc_a_s.fill_(.5)
    w, e = module.evidence_weights(q, samples, z)
    p0 = z.softmax(-1)
    p = (z + module.evc_a_p.tanh()[None, None, :, None, None] * (e - e.mean(-1, keepdim=True))).softmax(-1)
    pi = (z.logsumexp(-1) + module.evc_a_s.tanh()[None, None, :, None] * (p0 * e).sum(-1)).softmax(-1)
    result["original_p0_scale_formula_max_abs"] = compare(w, pi[..., None] * p)
    altered = samples.clone()
    altered[0, 0, 0, 1, 2] = -altered[0, 0, 0, 1, 2]
    ww, ee = module.evidence_weights(q, altered, z)
    result["fixed_query_coordinate_evidence_delta"] = float((ee - e).abs().max())
    result["fixed_query_coordinate_weight_delta"] = float((ww - w).abs().max())
    result["original_query_only_weight_delta"] = compare(baseline, z.flatten(-2).softmax(-1).view_as(z))
    require(result["fixed_query_coordinate_evidence_delta"] > 0 and result["fixed_query_coordinate_weight_delta"] > 0, "Evidence ignores sampled content")
    wz, ez = module.evidence_weights(q * 0, samples * 0, z * 1000)
    require(torch.isfinite(wz).all() and not torch.count_nonzero(ez), "Zero vector or extreme logits nonfinite")
    return result


def attention_pair(device, refdim=4):
    torch.manual_seed(43)
    old, new = MSDeformAttn(n_levels=3).to(device), EVCMSDeformAttn().to(device)
    # Nonuniform logits exercise the actual sampling path beyond default uniform initialization.
    with torch.no_grad():
        old.attention_weights.weight.normal_(0, .04)
    new.load_state_dict({**new.state_dict(), **old.state_dict()}, strict=True)
    shapes = [(7, 9), (4, 5), (2, 3)]
    q = torch.randn(2, 17, 256, device=device)
    v = torch.randn(2, 89, 256, device=device)
    ref = torch.rand(2, 17, 1, refdim, device=device) * 1.4 - .2
    mask = torch.zeros(2, 89, dtype=torch.bool, device=device); mask[:, ::9] = True
    inputs_a = [x.detach().clone().requires_grad_() for x in (q, v, ref)]
    inputs_b = [x.detach().clone().requires_grad_() for x in (q, v, ref)]
    out_a = old(inputs_a[0], inputs_a[2], inputs_a[1], shapes, mask)
    original_grid_sample = torch.nn.functional.grid_sample
    with patch("torch.nn.functional.grid_sample", wraps=original_grid_sample) as sampling:
        out_b = new(inputs_b[0], inputs_b[2], inputs_b[1], shapes, mask)
        require(sampling.call_count == 3, "Must sample once per level, not sample keys again")
    error = compare(out_a, out_b)
    upstream = torch.randn_like(out_a)
    (out_a * upstream).sum().backward(); (out_b * upstream).sum().backward()
    grads = {n: compare(p.grad, dict(new.named_parameters())[n].grad, atol=5e-5, rtol=3e-4) for n, p in old.named_parameters()}
    inputs = {n: compare(x.grad, y.grad, atol=5e-5, rtol=3e-4) for n, x, y in zip(("query", "value", "reference"), inputs_a, inputs_b)}
    require(new.evc_a_p.grad.abs().max() > 0 and new.evc_a_s.grad.abs().max() > 0, "Coefficients cannot open")
    return dict(device=device, reference_dim=refdim, output_max_abs=error, public_gradient_max_abs=grads,
                input_gradient_max_abs=inputs, grid_sample_calls=3, Q=17, padding_and_out_of_bounds=True)


def native_updates(model, device, amp, temporary):
    model = model.to(device).train(); model.nc = 1
    trainer = object.__new__(RTDETRTrainer)
    optimizer = trainer.build_optimizer(model, name="AdamW", lr=.0005, momentum=.937, decay=.0001)
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    require(all(ids.count(id(p)) == 1 for p in model.parameters()), "Common/new optimizer occurrence !=1")
    added = {n: p for n, p in model.named_parameters() if is_added(n)}
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=128.)
    records = []
    for step, size in enumerate((2, 1, 2)):
        torch.manual_seed(42 + step)
        batch = dict(img=torch.rand(size, 3, 160, 160, device=device), cls=torch.zeros(size, 1, device=device),
                     batch_idx=torch.arange(size, device=device), bboxes=torch.tensor([[.5, .5, .2, .3]], device=device).repeat(size, 1))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=amp):
            pred = model.predict(batch["img"], batch=dict(cls=batch["cls"].flatten().long(), bboxes=batch["bboxes"],
                                 batch_idx=batch["batch_idx"].long(), gt_groups=[1] * size))
            require(len(pred) == 5 and pred[0].shape[:2] == (3, size), "Native train outputs changed")
            require(pred[-1]["dn_num_split"][1] == 300 and pred[-1]["dn_num_split"][0] > 0, "DN/query contract changed")
            loss, items = model.loss(batch, pred)
        require(torch.isfinite(loss).all(), "Loss nonfinite")
        scaler.scale(loss.sum()).backward(); scaler.unscale_(optimizer)
        require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), "Nonfinite gradient")
        grads = {n: float(p.grad.abs().max()) for n, p in added.items()}
        require(all(v > 0 for n, v in grads.items() if n.endswith(("a_p", "a_s"))), "Coefficients not learning")
        if step == 0:
            require(all(v == 0 for n, v in grads.items() if n.endswith(("evc_q", "evc_v"))), "Wrong zero coefficient gradient staging")
        else:
            require(all(v > 0 for v in grads.values()), "Scoring projections did not open after coefficient update")
        before = {n: p.detach().clone() for n, p in added.items()}
        scaler.step(optimizer); scaler.update()
        changes = {n: float((p.detach() - before[n]).abs().max()) for n, p in added.items()}
        records.append(dict(step=step, batch=size, imgsz=160, loss=float(loss.sum()), loss_items=items.tolist(),
                            dn_num_split=pred[-1]["dn_num_split"], added_gradient_max=grads, added_update_max=changes))
    model = model.cpu().eval()
    probe = torch.rand(1, 3, 160, 192)
    with torch.no_grad(): expected = model(probe)[0]
    path = temporary / (device + ("_amp" if amp else "_fp32") + ".pt")
    torch.save(dict(epoch=-1, model=deepcopy(model).float(), train_args={"task": "detect"}), path)
    restored = RTDETR(str(path)).model.eval()
    verify_model(restored)
    require(all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()), "Reload state mismatch")
    with torch.no_grad(): error = compare(expected, restored(probe)[0])
    return dict(device=device, amp=amp, steps=records, all_optimizer_parameters_exactly_once=True,
                optimizer_groups=optimizer_groups(model, optimizer), reload_max_abs=error)


def native_api(initialization):
    records = {}
    class ProbeComplete(Exception): pass
    class RebuildOnly(RTDETRTrainer):
        def __init__(self, overrides, _callbacks): self.data = dict(nc=1, channels=3)
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, False)
            records.update(rebuild_audit(weights, model, "evc_deform"))
            return model
        def train(self):
            verify_model(self.model, zero=True)
            raise ProbeComplete()
    with patch("ultralytics.utils.checks.check_pip_update_available"):
        try: RTDETR(str(initialization)).train(trainer=RebuildOnly, data="no-dataset-needed", epochs=200)
        except ProbeComplete: records["public_train_API"] = "passed; stopped before trainer setup"
        else: raise RuntimeError("Probe did not stop")
    return records


def synthetic_evaluation(model, temporary):
    """Two generated images per split through the real fixed-config export entry."""
    import numpy as np
    from PIL import Image
    from evc_results import evaluate, verify_archive
    dataset = temporary / "dataset"
    for split in ("val", "test"):
        for kind in ("images", "labels"): (dataset / kind / split).mkdir(parents=True)
        for i in range(2):
            image = np.zeros((160, 192, 3), dtype=np.uint8); image[:, 40 + i * 20:43 + i * 20] = 220
            Image.fromarray(image).save(dataset / "images" / split / f"{i}.jpg")
            (dataset / "labels" / split / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.3\n")
    config = dataset / "data.yaml"
    YAML.save(config, dict(path=str(dataset), train="images/val", val="images/val", test="images/test", names={0: "crack"}))
    checkpoint = temporary / "evaluation_debug.pt"
    torch.save(dict(epoch=-1, model=deepcopy(model).float().eval(), train_args={"task": "detect"}), checkpoint)
    reports = {}
    for split in ("val", "test"):
        reports[split] = evaluate(checkpoint, config, split, temporary / ("evaluation_" + split), device="cpu",
                                 val_report=temporary / "evaluation_val/metrics.json" if split == "test" else None)
    require(reports["val"]["images"] == reports["test"]["images"] == 2, "Synthetic split coverage")
    return reports


def run(source, output, c2_args):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(runtime=runtime(), full_server_preflight="NOT_RUN", formal_training="NOT_RUN", formal_val="NOT_RUN", formal_test="NOT_RUN")
    report["formulas"] = formulas()
    report["attention_paths"] = [attention_pair("cpu", d) for d in (2, 4)]
    if torch.cuda.is_available(): report["attention_paths"] += [attention_pair("cuda", d) for d in (2, 4)]
    reference, target, mapping = controlled_models(source)
    write_json(output / "mapping.json", mapping)
    report["parameters_unfused"] = mapping["parameters_unfused"]
    reference.eval(); target.eval()
    with torch.no_grad():
        x = torch.rand(1, 3, 640, 640)
        report["network_cpu_fp32_zero_max_abs"] = compare(reference(x)[0], target(x)[0])
    del reference; gc.collect()
    with tempfile.TemporaryDirectory(prefix="evc_debug_", dir=ROOT / "outputs") as d:
        temporary = Path(d)
        report["cpu_fp32_updates"] = native_updates(deepcopy(target), "cpu", False, temporary)
        if torch.cuda.is_available():
            report["cuda_fp32_updates"] = native_updates(deepcopy(target), "cuda", False, temporary)
            report["cuda_amp_updates"] = native_updates(deepcopy(target), "cuda", True, temporary)
            half = deepcopy(target).cuda().half().eval()
            with torch.no_grad():
                y = half(torch.rand(1, 3, 160, 192, device="cuda", dtype=torch.float16))[0]
                require(y.dtype == torch.float16 and torch.isfinite(y).all() and y.shape == (1, 300, 5), "Actual half failure")
            report["cuda_actual_half"] = dict(shape=list(y.shape), dtype=str(y.dtype), finite=True)
            del half; torch.cuda.empty_cache()
        else:
            report["cuda_fp32_updates"] = report["cuda_amp_updates"] = report["cuda_actual_half"] = "NOT_RUN: CUDA unavailable"
        fresh = temporary / "fresh_init.pt"
        report["fresh_initialization"] = initialize(source, fresh)
        report["native_train_api"] = native_api(fresh)
        report["synthetic_evaluation"] = synthetic_evaluation(target, temporary)
    _, rows = recipe(c2_args, "evc_deform", Path("/root/autodl-tmp/projects/Crack_RTDETR-evc/weights/evc_deform_controlled_init.pt"))
    write_json(output / "parameter_diff.json", rows)
    report.update(status="passed", actual_c2_args=str(c2_args), actual_c2_args_sha256=sha256(c2_args), recipe_fields=len(rows),
                  debug_files_discarded=True, latency="NOT_MEASURED; parameter increment is not a latency claim")
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--c2-args", type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(4)
    run(args.source, args.output, args.c2_args)
