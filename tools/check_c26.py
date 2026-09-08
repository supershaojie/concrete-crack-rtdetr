"""Bounded C26 local integration checks; debug models/files are discarded, never formal initialization."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
from init_c26 import (ROOT, PARAMETERS, build, controlled_models, initialize, is_added,
                      require, runtime, verify_model, write_json)
from train_c26 import rebuild_audit, recipe, optimizer_groups
from check_scca import comparison, tensors
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer


def rebuild(source):
    trainer = object.__new__(RTDETRTrainer)
    trainer.data = dict(nc=1, channels=3)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return trainer.get_model(cfg=source.yaml, weights=source, verbose=False)


def native_api(initialization):
    records = {}

    class ProbeComplete(Exception):
        pass

    class RebuildOnly(RTDETRTrainer):
        def __init__(self, overrides, _callbacks):
            self.data = dict(nc=1, channels=3)

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, False)
            records.update(rebuild_audit(weights, model, "c26"))
            return model

        def train(self):
            verify_model(self.model, zero=True)
            raise ProbeComplete()

    model = RTDETR(str(initialization))
    with patch("ultralytics.utils.checks.check_pip_update_available"):
        try:
            model.train(trainer=RebuildOnly, data="rebuild-only-no-dataset", epochs=200)
        except ProbeComplete:
            records["public_RTDETR_train_rebuild"] = "passed; stopped before training setup"
        else:
            raise RuntimeError("Native API probe did not stop before training")
    return records


def updates(model, device, amp, temporary):
    model = model.to(device).train()
    model.nc = 1  # Normally supplied by RTDETRTrainer.set_model_attributes.
    trainer = object.__new__(RTDETRTrainer)
    opt = trainer.build_optimizer(model, name="AdamW", lr=.0005, momentum=.937, decay=.0001)
    added = {n: p for n, p in model.named_parameters() if is_added(n)}
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    counts = {n: ids.count(id(p)) for n, p in added.items()}
    require(all(v == 1 for v in counts.values()), "Added parameters missing/duplicated")
    require(all(ids.count(id(p)) == 1 for p in model.parameters() if p.requires_grad), "Trainable parameter missing/duplicated")
    groups = optimizer_groups(model, opt)
    for g in groups:
        for name in g["parameters"]:
            if is_added(name):
                require(g["weight_decay"] == (0. if "bias" in name else .0001), "New parameter native grouping changed")
    projections = ("model.9.scca_o.weight", "model.26.cbr.offset_out.weight", "model.26.cbr.offset_out.bias")
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=128.)
    records = []
    for step, batch_size in enumerate((2, 2, 1)):
        batch = dict(img=torch.rand(batch_size, 3, 160, 160, device=device), cls=torch.zeros(batch_size, 1, device=device),
                     batch_idx=torch.arange(batch_size, device=device),
                     bboxes=torch.tensor([[.5, .5, .2, .3]], device=device).repeat(batch_size, 1))
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=amp):
            pred = model.predict(batch["img"], batch=dict(cls=batch["cls"].flatten().long(), bboxes=batch["bboxes"],
                                 batch_idx=batch["batch_idx"].long(), gt_groups=[1] * batch_size))
            require(len(pred) == 5 and pred[0].shape[:2] == (3, batch_size), "Train/auxiliary output changed")
            require(pred[-1]["dn_num_split"][1] == 300 and pred[-1]["dn_num_split"][0] > 0, "DN or regular queries changed")
            loss, items = model.loss(batch, pred)
        require(torch.isfinite(loss).all(), "Nonfinite loss")
        scaler.scale(loss.sum()).backward()
        scaler.unscale_(opt)
        require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), "Nonfinite network gradients")
        grads = {n: float(p.grad.abs().max()) if p.grad is not None else None for n, p in added.items()}
        require(all(g is not None and 0 <= g < float("inf") for g in grads.values()), "Missing/nonfinite added gradient")
        require(all(grads[n] > 0 for n in projections), "An output projection is not learning")
        if step == 0:
            require(all(g == 0 for n, g in grads.items() if n not in projections), "Unexpected zero-init gradient staging")
        before = {n: p.detach().clone() for n, p in added.items()}
        scaler.step(opt)
        scaler.update()
        changes = {n: float((p.detach() - before[n]).abs().max()) for n, p in added.items()}
        require(all(changes[n] > 0 for n in projections), "Output projection update skipped")
        records.append(dict(step=step, batch=batch_size, imgsz=160, loss=float(loss.sum()),
                            loss_items=items.detach().cpu().tolist(), dn_num_split=pred[-1]["dn_num_split"],
                            added_gradient_max=grads, added_update_max=changes))
    model = model.cpu().eval()
    probe = torch.rand(1, 3, 160, 192)
    with torch.no_grad():
        expected = model(probe)
    path = temporary / (device + ("_amp" if amp else "_fp32") + "_debug.pt")
    torch.save(dict(epoch=-1, model=deepcopy(model).float(), train_args={"task": "detect"}), path)
    restored = RTDETR(str(path)).model.eval()
    verify_model(restored)
    require(all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()), "Updated checkpoint state mismatch")
    with torch.no_grad():
        reload_error = comparison(expected, restored(probe))
        ablated = deepcopy(restored)
        ablated.model[9].scca_o.weight.zero_()
        scca_effect = float((restored(probe)[0] - ablated(probe)[0]).abs().max())
        ablated = deepcopy(restored)
        ablated.model[-1].cbr.offset_out.weight.zero_()
        ablated.model[-1].cbr.offset_out.bias.zero_()
        without_cbr = ablated(probe)[0]
        actual = restored(probe)[0]
        cbr_effect = float((actual[..., :4] - without_cbr[..., :4]).abs().max())
        require(torch.equal(actual[..., 4:], without_cbr[..., 4:]), "CBR changed scores")
    require(scca_effect > 0 and cbr_effect > 0, "Reloaded module/output contribution missing")
    return dict(device=device, amp=amp, debug_scaler_initial_scale=128., optimizer_occurrences=counts,
                optimizer_groups=groups, steps=records, reload_max_abs=reload_error,
                learned_scca_effect=scca_effect, learned_cbr_box_effect=cbr_effect, cbr_scores_unchanged=True)


def run(source, output):
    require(not output.exists(), "Preserve prior local validation evidence")
    output.mkdir(parents=True)
    report = dict(runtime=runtime(), full_server_preflight="NOT_RUN", server_batch16_smoke="NOT_RUN",
                  formal_training="NOT_RUN", formal_val="NOT_RUN", formal_test="NOT_RUN", parameters={})
    for variant, expected in PARAMETERS.items():
        m = build(variant, nc=1)
        count = sum(p.numel() for p in m.parameters())
        require(count == expected, f"{variant} parameter count mismatch")
        report["parameters"][variant] = count
        del m
    reference80, target80, loading = controlled_models(source)
    write_json(output / "mapping.json", loading)
    target = rebuild(target80)
    report["nc1_loading"] = rebuild_audit(target80, target, "c26")
    reference = rebuild(reference80)
    require(all(torch.equal(v, target.state_dict()[k]) for k, v in reference.state_dict().items()), "nc1 common state mismatch")
    del target80, reference80
    target.eval(); reference.eval()
    report["zero_fp32_max_abs"] = {}
    with torch.no_grad():
        for h, w in ((640, 640), (160, 192)):
            x = torch.rand(1, 3, h, w)
            # Assert CBR slot 0 really receives Neck P3 at 80x80 for 640, without a hook/cache in production.
            original = target.model[-1].cbr.forward
            def inspect(p3, query, boxes, **kwargs):
                require(p3.shape == (1, 256, h // 8, w // 8), "CBR is not using Neck P3")
                require(query.shape == (1, 300, 256), "CBR final query shape changed")
                return original(p3, query, boxes, **kwargs)
            with patch.object(target.model[-1].cbr, "forward", side_effect=inspect):
                report["zero_fp32_max_abs"][f"1x3x{h}x{w}"] = comparison(reference(x), target(x))
    with tempfile.TemporaryDirectory(prefix="c26_debug_", dir=ROOT / "outputs") as d:
        temporary = Path(d)
        report["cpu_fp32_updates"] = updates(deepcopy(target), "cpu", False, temporary)
        if torch.cuda.is_available():
            report["cuda_amp_updates"] = updates(deepcopy(target), "cuda", True, temporary)
            torch.cuda.empty_cache()
        else:
            report["cuda_amp_updates"] = "NOT_RUN: CUDA unavailable"
        # Independently reconstructed AFTER debug updates; no debug tensors can enter this checkpoint.
        fresh = temporary / "fresh_initialization.pt"
        report["fresh_initialization"] = initialize(source, fresh)
        report["native_training_api"] = native_api(fresh)
    _, rows = recipe(ROOT / "docs/scca/c2_args.yaml", "c26", Path("/root/autodl-tmp/projects/Crack_RTDETR-c26/weights/c26_cbr_scca_controlled_init.pt"))
    write_json(output / "parameter_diff.json", rows)
    report.update(status="passed", debug_files_discarded=True, recipe_fields=len(rows),
                  recipe_changed=[r["field"] for r in rows if r["changed"]])
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    result = run(args.source, args.output)
    print(json.dumps({k: result[k] for k in ("status", "parameters", "nc1_loading", "zero_fp32_max_abs", "native_training_api")}, indent=2))
