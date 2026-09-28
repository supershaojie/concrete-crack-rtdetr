"""Executable mathematical, routing and bounded local model checks; never test/long train."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile

from dtr_v1_common import *
import torch
from ultralytics.models.rtdetr.dtr_loss import DTRDetectionLoss, tolerance_loss, phi, ramp
from ultralytics.models.rtdetr.dtr_model import DTRDetectionModel
from ultralytics.models.rtdetr.dtr_trainer import DTRTrainer, aggregate
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.models.utils.ops import get_cdn_group
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh


def pair(q, g):
    return torch.tensor(q, dtype=torch.long), torch.tensor(g, dtype=torch.long)


def close(actual, expected, atol=1e-7, rtol=1e-5):
    torch.testing.assert_close(torch.as_tensor(actual).double(), torch.as_tensor(expected).double(), atol=atol, rtol=rtol)


def fails(fn):
    try:
        fn()
    except (ValueError, RuntimeError, AssertionError):
        return
    raise AssertionError("Expected explicit error")


def example(left=106., refs=(101., 102., 103.)):
    x = torch.tensor([[[left, 20., 120., 40.], [150., 25., 160., 40.]]], requires_grad=True)
    g = torch.tensor([[100., 20., 120., 40.]], requires_grad=True)
    d = torch.tensor([[[r, 20., 120., 40.] for r in refs]], requires_grad=True)
    result, diagnostics = tolerance_loss(x, g, [pair([0], [0])], [1], d,
        [pair(list(range(len(refs))), [0] * len(refs))], (1000, 1000), True)
    return result, diagnostics, x, g, d


def mathematics():
    raw, s, x, g, d = example()
    close(raw, .0390388203)
    close(.2 * raw, .0078077641)
    grad = torch.autograd.grad(.2 * raw, (x, g, d), allow_unused=True)
    close(grad[0][0, 0, 0], .2 / 4 * (.2 / (.2 ** 2 + .05 ** 2) ** .5) / 20)
    assert grad[1] is None and grad[2] is None and torch.count_nonzero(grad[0][0, 1]) == 0
    for left in (101., 102.):
        loss, _, student, _, _ = example(left)
        loss.backward()
        assert float(loss) == 0 and torch.count_nonzero(student.grad) == 0
    neg, _, student, _, _ = example(94.)
    neg.backward()
    close(neg, raw)
    assert student.grad[0, 0, 0] < 0
    for refs in ((98., 102.), (101., 103.)):
        loss, stats, *_ = example(refs=refs)
        close(loss, raw)
        close(stats["radius_over_scale_sum"][0], .1)
    single, _, *_ = example(refs=(102.,))
    close(single, raw)
    zero_teacher, _, *_ = example(refs=(100.,))
    close(zero_teacher, phi(torch.tensor(.3)) / 4)
    # Fixed detached teacher; finite difference only in student coordinate.
    eps = .01
    numerical = (example(106 + eps)[0] - example(106 - eps)[0]) / (2 * eps)
    close(numerical, grad[0][0, 0, 0] / .2, atol=2e-6, rtol=4e-4)
    close([ramp(e) for e in (0, 5, 6, 19, 20, 30)], [0, 0, 1 / 15, 14 / 15, 1, 1])
    values = torch.tensor([0., 1e-8, 1e-5, .001, .2, 2.])
    close(phi(values), (values.square() + .05 ** 2).sqrt() - .05, atol=2e-7)
    assert torch.isfinite(phi(values)).all()
    scaled, _ = tolerance_loss(x / 1000, g / 1000, [pair([0], [0])], [1], d / 1000,
                              [pair([0, 1, 2], [0, 0, 0])], (1000, 1000))
    close(scaled, raw)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        amp_loss, _ = tolerance_loss(x, g, [pair([0], [0])], [1], d,
                                    [pair([0, 1, 2], [0, 0, 0])], (1000, 1000))
    assert amp_loss.dtype == torch.float32
    close(amp_loss, raw)
    # Rectangular 100x200 input, subpixel widths: LR floor=1/200, TB=1/100.
    gt = torch.tensor([[.5, .5, .5001, .5001]], requires_grad=True)
    before = gt.detach().clone()
    student = (gt.detach() + .02).view(1, 1, 4).requires_grad_()
    loss, stat = tolerance_loss(student, gt, [pair([0], [0])], [1], gt.detach().view(1, 1, 4),
                                [pair([0], [0])], (100, 200), True)
    close(stat["error_over_scale_sum"], [4, 2, 4, 2], atol=1e-5)
    assert torch.equal(gt, before)
    # Denominator includes a second match without a reference.
    xx = x.detach().clone().requires_grad_()
    gg = torch.cat((g.detach(), torch.tensor([[140., 25., 160., 40.]])))
    half, stat = tolerance_loss(xx, gg, [pair([0, 1], [0, 1])], [2], d.detach(),
                               [pair([0, 1, 2], [0, 0, 0])], (1000, 1000), True)
    close(half, raw / 2)
    assert stat["M"] == 2 and stat["without_dn"] == 1
    empty, _ = tolerance_loss(xx, gg, [pair([], [])], [2], None, None, (100, 200))
    empty.backward()
    assert torch.isfinite(empty) and torch.count_nonzero(xx.grad) == 0
    no_dn, _ = tolerance_loss(xx, gg, [pair([0], [0])], [2], None, None, (100, 200))
    assert no_dn.requires_grad and no_dn.item() == 0
    fails(lambda: tolerance_loss(xx, gg, [pair([0], [3])], [2], None, None, (100, 200)))
    return dict(status="PASS", raw=float(raw.detach()), weighted=float(.2 * raw.detach()),
                analytic_gradient=float(grad[0][0, 0, 0]), finite_difference=float(numerical.detach()),
                cases="reference, signs, inactive/boundary, odd/even/K1, zero teacher, tiny-u, rectangular floor, denominator, empty/noDN")


def mapping():
    gt = torch.tensor([[.2, .25, .06, .08], [.7, .75, .04, .05], [.4, .45, .02, .03]])
    batch = dict(cls=torch.zeros(3, dtype=torch.long), bboxes=gt, batch_idx=torch.tensor([0, 0, 2]), gt_groups=[2, 0, 1])
    _, encoded, mask, meta = get_cdn_group(batch, 1, 300, torch.ones(1, 8), num_dn=6,
                                          cls_noise_ratio=0., box_noise_scale=0., training=True)
    matches = RTDETRDetectionLoss.get_dn_match_indices(meta["dn_pos_idx"], meta["dn_num_group"], batch["gt_groups"])
    assert meta["dn_num_group"] == 3
    for i, (q, g) in enumerate(matches):
        assert torch.equal(encoded[i, q], gt[g])
        assert len(q) == batch["gt_groups"][i] * 3
    assert matches[2][1].tolist() == [2, 2, 2]  # flat IDs, never local GT0
    # Identifiable same-local-ID references, negatives/padding poisoned but finite.
    dn = torch.full((3, 12, 4), 1000.)
    gg = xywh2xyxy(gt)
    for i, (q, g) in enumerate(matches):
        for k, (qi, gi) in enumerate(zip(q.tolist(), g.tolist())):
            dn[i, qi] = gg[gi] + (gi + 1) * .001 * (k // max(batch["gt_groups"][i], 1) + 1)
    ordinary = torch.zeros(3, 2, 4)
    ordinary[0] = gg[:2] + .01
    ordinary[2, 0] = gg[2] + .01
    raw, stats = tolerance_loss(ordinary.requires_grad_(), gg, [pair([0, 1], [0, 1]), pair([], []), pair([0], [2])],
                                [2, 0, 1], dn, matches, (640, 640), True)
    assert stats["M"] == 3 and stats["K"] == {"3": 3} and raw > 0
    # Relocate queries arbitrarily and preserve explicit metadata. No reshape guess.
    permutation = torch.tensor([7, 2, 9, 0, 11, 5, 4, 8, 3, 1, 10, 6])
    inverse = permutation.argsort()
    remapped = [(inverse[q], g) for q, g in matches]
    alternate, _ = tolerance_loss(ordinary, gg, [pair([0, 1], [0, 1]), pair([], []), pair([0], [2])],
                                  [2, 0, 1], dn[:, permutation], remapped, (640, 640))
    close(alternate, raw)
    fails(lambda: tolerance_loss(ordinary, gg, [pair([0], [2]), pair([], []), pair([], [])],
                                 [2, 0, 1], dn, matches, (640, 640)))
    return dict(status="PASS", positive_indices=[q.tolist() for q, _ in matches],
                flat_gt_indices=[g.tolist() for _, g in matches], negative_padding_excluded=True, permutation_invariant=True)


def fixture():
    torch.manual_seed(42)
    batch = dict(cls=torch.zeros(3, dtype=torch.long), bboxes=torch.tensor([[.25, .3, .1, .2], [.6, .65, .15, .12], [.3, .4, .1, .2]]),
                 batch_idx=torch.tensor([0, 0, 1]), gt_groups=[2, 1])
    boxes = (torch.rand(4, 2, 5, 4) * .4 + .2).requires_grad_()
    scores = torch.randn(4, 2, 5, 1, requires_grad=True)
    dn = (torch.rand(3, 2, 8, 4) * .4 + .2).requires_grad_()
    ds = torch.randn(3, 2, 8, 1, requires_grad=True)
    meta = dict(dn_pos_idx=[torch.tensor([0, 1, 2, 3]), torch.tensor([0, 2])], dn_num_group=2, dn_num_split=[8, 5])
    return (boxes, scores), batch, dn, ds, meta


def routing():
    preds, batch, dn, ds, meta = fixture()
    mother, dtr = RTDETRDetectionLoss(nc=1, use_vfl=True), DTRDetectionLoss(nc=1, use_vfl=True)
    assert mother.vfl.alpha == dtr.vfl.alpha == .25 and mother.vfl.gamma == dtr.vfl.gamma == 1.5
    assert mother.matcher.alpha == dtr.matcher.alpha == .25 and mother.matcher.gamma == dtr.matcher.gamma == 2.
    assert mother.matcher.cost_gain == dtr.matcher.cost_gain == {"class": 2, "bbox": 5, "giou": 2}
    original = mother(preds, batch, dn, ds, meta)
    for enabled, epoch in ((False, 20), (True, 5)):
        dtr.enabled, dtr.epoch = enabled, epoch
        state = torch.get_rng_state().clone()
        actual = dtr(preds, batch, dn, ds, meta)
        assert torch.equal(state, torch.get_rng_state()) and set(actual) == set(original)
        assert all(torch.equal(actual[k], v) for k, v in original.items())
        a = torch.autograd.grad(sum(actual.values()), (*preds, dn, ds), retain_graph=True)
        b = torch.autograd.grad(sum(original.values()), (*preds, dn, ds), retain_graph=True)
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    calls = []
    hook = dtr.matcher.register_forward_hook(lambda m, a, o: calls.append(a[0].shape))
    dtr.enabled, dtr.epoch, dtr.input_hw, dtr.sample = True, 20, (320, 640), True
    actual = dtr(preds, batch, dn, ds, meta)
    hook.remove()
    assert len(calls) == 4 and set(actual) - set(original) == {"loss_dtr"}
    assert all(torch.equal(actual[k], v) for k, v in original.items())
    assert dtr._final_matches is None and not dtr._capture_final
    gradients = torch.autograd.grad(actual["loss_dtr"], (*preds, dn, ds), allow_unused=True, retain_graph=True)
    assert gradients[0][-1].abs().sum() > 0 and torch.count_nonzero(gradients[0][:-1]) == 0
    assert gradients[1:] == (None, None, None)
    original_dn_grad = torch.autograd.grad(sum(v for k, v in actual.items() if "_dn" in k), dn)[0]
    assert original_dn_grad.abs().sum() > 0
    none = dtr(preds, batch)
    assert "loss_dtr_dn" not in none and none["loss_dtr"].item() == 0
    malformed = deepcopy(meta); malformed["dn_pos_idx"][0][0] = 999
    fails(lambda: dtr(preds, batch, dn, ds, malformed))
    malformed = deepcopy(meta); malformed["dn_pos_idx"][0] = torch.tensor([0])
    fails(lambda: dtr(preds, batch, dn, ds, malformed))
    broken = dn.detach().clone(); broken[-1, 0, 0, 0] = float("nan")
    fails(lambda: dtr(preds, batch, broken, ds, meta))
    assert dtr._final_matches is None and not dtr._capture_final
    fails(lambda: DTRDetectionLoss(nc=2))
    empty_batch = dict(cls=torch.zeros(0, dtype=torch.long), bboxes=torch.zeros(0, 4),
                       batch_idx=torch.zeros(0, dtype=torch.long), gt_groups=[0, 0])
    empty_losses = dtr(preds, empty_batch)
    assert empty_losses["loss_dtr"].requires_grad and empty_losses["loss_dtr"].item() == 0
    assert all(torch.isfinite(v) for v in empty_losses.values())
    return dict(status="PASS", matcher_calls=len(calls), original_keys=list(original), added_keys=["loss_dtr"],
                VFL=dict(alpha=.25, gamma=1.5), matcher=dict(alpha=.25, gamma=2., costs=mother.matcher.cost_gain),
                all_L0_and_DN_equal=True, gradients="final matched ordinary only; original DN remains trainable")


def reload_checkpoint(path):
    from ultralytics.utils.patches import torch_load
    saved = torch_load(path, map_location="cpu")
    model = saved["model"]
    assert isinstance(model, DTRDetectionModel)
    model.eval()
    with torch.no_grad():
        output = model(saved["input"])[0]
    assert torch.equal(output, saved["output"])
    assert isinstance(model.init_criterion(), DTRDetectionLoss)
    write_json(path.with_suffix(".json"), dict(status="PASS", new_process_class=type(model).__name__, exact_output=True))


def local_val(checkpoint, output, device):
    """One small real val batch; this cannot qualify the formal B16 preflight."""
    from ultralytics.models.rtdetr.dtr_val import DTRValidator
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.utils.torch_utils import init_seeds
    import time
    output = output / ("real_val_" + time.strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True)
    validator = DTRValidator(args=dict(model=str(checkpoint), data=str(data_config()), split="val",
        device=device, batch=2, imgsz=160, workers=0, half=False, conf=.001, iou=.7, max_det=300,
        rect=False, plots=False, seed=42), save_dir=output)
    validator.one_batch = True
    validator.export_path = output / "val_predictions_gt.jsonl.gz"
    validator.export_identity = dict(scope="local B2/160 one val batch; not formal evaluation")
    warmup, original = [], AutoBackend.warmup
    def observed(backend, imgsz=(1, 3, 640, 640)):
        handle = backend.model.register_forward_pre_hook(lambda m, a: warmup.append(dict(
            shape=list(a[0].shape), finite=bool(torch.isfinite(a[0]).all()), zero=bool(torch.count_nonzero(a[0]) == 0),
            dtype=str(a[0].dtype))))
        try:
            return original(backend, imgsz)
        finally:
            handle.remove()
    AutoBackend.warmup = observed
    try:
        init_seeds(42, deterministic=True)
        metrics = validator(model=str(checkpoint))
    finally:
        AutoBackend.warmup = original
    assert len(validator.seen_ids) == 2 and validator.query_count == 600
    if device != "cpu":
        assert warmup and all(r["finite"] and r["zero"] and r["dtype"] == "torch.float32" for r in warmup)
    report = dict(status="PASS", scope="one real local val batch B2/160 FP32; no test", device=device, warmup=warmup,
                  images=2, queries=600, metrics=metrics, export=file_info(validator.export_path), runtime=runtime())
    write_json(output / "checks.json", report)
    print("PASS local FP32 warmup / one real val batch / all-query+GT export", flush=True)


def integration(output, device):
    from init_c19_lif_v1 import source_contract, initialize, build_training_model, verify_model
    from ultralytics.utils.patches import torch_load
    source_contract()
    require(SOURCE.is_file() and sha256(SOURCE) == SOURCE_SHA256, "Public source absent/wrong")
    init = output / "public_init.pt"
    if not init.exists():
        write_json(output / "initialization.json", initialize(SOURCE, init))
    weights = torch_load(init, map_location="cpu")["model"]
    torch.manual_seed(42)
    state = torch.get_rng_state().clone()
    mother, mapping_report = build_training_model(str(MODEL), weights, dict(nc=1, channels=3))
    torch.set_rng_state(state)
    trainer = DTRTrainer.__new__(DTRTrainer); trainer.data = dict(nc=1, channels=3)
    student = trainer.get_model(str(MODEL), weights, verbose=False)
    assert sum(p.numel() for p in student.parameters()) == 20149765
    assert [(n, p.shape) for n, p in mother.named_parameters()] == [(n, p.shape) for n, p in student.named_parameters()]
    assert set(mother.state_dict()) == set(student.state_dict())
    assert all(torch.equal(v, student.state_dict()[k]) for k, v in mother.state_dict().items())
    model_inventory = [dict(name=n, shape=list(p.shape), numel=p.numel()) for n, p in student.named_parameters()]
    write_json(output / "parameter_inventory.json", model_inventory)
    write_json(output / "class_mapping.json", mapping_report)
    # Native Trainer.set_model_attributes supplies nc after get_model.
    mother.nc = student.nc = 1
    mother, student = mother.to(device).train(), student.to(device).train()
    image = torch.rand(2, 3, 128, 128, device=device)
    batch = dict(img=image, cls=torch.zeros(3, 1, device=device), batch_idx=torch.tensor([0, 0, 1], device=device),
                 bboxes=torch.tensor([[.3, .4, .1, .2], [.7, .6, .2, .1], [.4, .5, .15, .2]], device=device))
    targets = dict(cls=batch["cls"].long().flatten(), bboxes=batch["bboxes"], batch_idx=batch["batch_idx"], gt_groups=[2, 1])
    # Real stochastic DN forward, restored CPU and CUDA RNG before each model.
    rng = torch.get_rng_state(); cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    pred = mother.predict(image, batch=targets)
    torch.set_rng_state(rng)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)
    actual_pred = student.predict(image, batch=targets)
    assert all(torch.equal(a, b) for a, b in zip(pred[:4], actual_pred[:4]))
    original, items = mother.loss(batch, pred)
    student.dtr_epoch = 5
    zero, zero_items = student.loss(batch, actual_pred)
    assert torch.equal(zero, original) and torch.equal(zero_items, items)
    selected_m = [p for n, p in mother.named_parameters() if n.endswith("cbr.offset_out.weight") or n.endswith("dec_bbox_head.2.layers.2.weight")]
    selected_s = [p for n, p in student.named_parameters() if n.endswith("cbr.offset_out.weight") or n.endswith("dec_bbox_head.2.layers.2.weight")]
    cuda_check = str(device).startswith("cuda")
    gm = torch.autograd.grad(original, [p for p in mother.parameters() if p.requires_grad],
                             allow_unused=True, retain_graph=cuda_check)
    control = torch.autograd.grad(original, [p for p in mother.parameters() if p.requires_grad], allow_unused=True) if cuda_check else None
    gs = torch.autograd.grad(zero, [p for p in student.parameters() if p.requires_grad], allow_unused=True, retain_graph=True)
    gradient_differences, control_differences = [], []
    for (name, _), a, b in zip([(n, p) for n, p in mother.named_parameters() if p.requires_grad], gm, gs):
        assert (a is None) == (b is None), name
        if a is not None and not torch.equal(a, b):
            gradient_differences.append(dict(name=name, max_abs=float((a - b).abs().max()),
                magnitude=float(a.abs().max()), close=bool(torch.allclose(a, b, rtol=1e-5, atol=1e-6))))
    if control is not None:
        for (name, _), a, b in zip([(n, p) for n, p in mother.named_parameters() if p.requires_grad], gm, control):
            assert (a is None) == (b is None), name
            if a is not None and not torch.equal(a, b):
                control_differences.append(dict(name=name, max_abs=float((a - b).abs().max()),
                    magnitude=float(a.abs().max()), close=bool(torch.allclose(a, b, rtol=1e-5, atol=1e-6))))
    all_close = all(row["close"] for row in gradient_differences)
    gradient_report = dict(status="PASS" if all_close else "PENDING", compared=len(gm),
        rtol=1e-5, atol=1e-6, differences=gradient_differences, native_same_graph_repeat_differences=control_differences,
        note="A CUDA discrepancy is retained as PENDING, not hidden by relaxing tolerance; native repeat measures backend reproducibility.")
    write_json(output / "gradient_comparison.json", gradient_report)
    if not cuda_check:
        assert all_close, gradient_differences[:5]
    del control
    del gm, gs
    assert set(student.state_dict()) == set(mother.state_dict())
    student.dtr_epoch, student.dtr_sample = 20, True
    captured = {}
    hook = student.criterion.register_forward_hook(lambda m, a, o: captured.update(o))
    active, _ = student.loss(batch, actual_pred)
    hook.remove()
    close(active, sum(captured.values()))
    assert captured["loss_dtr"].requires_grad and torch.isfinite(active)
    assert student.criterion.input_hw is None and student.criterion._final_matches is None
    # One actual local FP32 optimizer use; isolated from formal initialization.
    before = selected_s[0].detach().clone()
    opt = torch.optim.AdamW(student.parameters(), lr=.0005, weight_decay=.0001)
    active.backward()
    assert all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None)
    opt.step()
    assert not torch.equal(before, selected_s[0])
    diagnostic = student.criterion.last_diagnostics
    # Inference identity uses same weights, not the independently updated mother.
    student.eval()
    compare = deepcopy(student); compare.__class__ = RTDETRDetectionModel
    with torch.no_grad():
        a, b = student(image)[0], compare(image)[0]
    assert torch.equal(a, b)
    from ultralytics.models.rtdetr.dtr_val import postprocess
    assert all(torch.equal(p[k], q[k]) for p, q in zip(postprocess(a, 128, .001), postprocess(b, 128, .001)) for k in p)
    # Save importable model with no transient criterion context/graphs.
    student.cpu()
    cpu_input = image.cpu()
    with torch.no_grad():
        expected = student(cpu_input)[0]
    path = output / "reload.pt"
    torch.save(dict(model=student, input=cpu_input, output=expected), path)
    code = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--reload", str(path)], cwd=ROOT, timeout=120).returncode
    assert code == 0
    return dict(status="PASS" if all_close else "PENDING", device=str(device), scope="local B2/128 FP32; not formal B16/640 AMP",
                module_hashes=source_contract(), parameters=20149765, state_keys=len(student.state_dict()),
                same_RNG_real_DN=True, real_L0_exact=True, gradients_allclose=gradient_report,
                optimizer_update=True,
                inference_equal=True, fresh_process_reload=read_json(path.with_suffix(".json")), diagnostic=diagnostic)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--math-only", action="store_true")
    p.add_argument("--integration", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, default=OUT / "local_checks")
    p.add_argument("--reload", type=Path)
    p.add_argument("--one-val", type=Path, help="new-process one local B2/160 val batch using a diagnostic checkpoint")
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.reload:
        reload_checkpoint(args.reload)
        return
    if args.one_val:
        local_val(args.one_val, args.output, args.device)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(runtime=runtime(), mathematics=mathematics(), mapping=mapping(), routing=routing())
    if args.integration:
        report["integration"] = integration(args.output, args.device)
    report["formal_server_B16_640_AMP"] = "PENDING; run delivered preflight"
    report["DDP"] = "PENDING: not tested; first experiment is single device=0"
    write_json(args.output / "checks.json", report)
    print("PASS:", ", ".join(k for k in report if isinstance(report[k], dict) and report[k].get("status") == "PASS"), flush=True)


if __name__ == "__main__":
    main()
