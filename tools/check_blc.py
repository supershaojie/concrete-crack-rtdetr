"""BLC-specific math, topology, native loss and checkpoint checks; no final test."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import traceback

import torch
from torch.nn import functional as F
from blc_common import (ROOT, VARIANTS, KEYS, COUNTS, build, verify_model, rebuild,
                        require, write_json, stamp, runtime, code_identity, paths, sha256)
from ultralytics.nn.modules import BLC
from ultralytics.utils.patches import torch_load


def reference_responses(z):
    """Independent direct indexing; no padding, response helper or shifted averages."""
    h, w = z.shape[-2:]
    yy = torch.arange(h, device=z.device)
    xx = torch.arange(w, device=z.device)
    directions = [(0, 1, 1, 0), (1, 0, 0, 1), (1, 1, 1, -1), (1, -1, 1, 1)]
    out = []
    for ty, tx, ny, nx in directions:
        for s in [1, 2]:
            lines = []
            for side in [0, 1, -1]:
                samples = []
                for along in [-1, 0, 1]:
                    y = (yy + along * ty + side * s * ny).clamp(0, h - 1)
                    x = (xx + along * tx + side * s * nx).clamp(0, w - 1)
                    samples.append(z[:, :, y[:, None], x[None, :]])
                lines.append((samples[0] + samples[1] + samples[2]) / 3)
            dp, dm = lines[0] - lines[1], lines[0] - lines[2]
            out.append(torch.minimum(dp.relu(), dm.relu()) - torch.minimum((-dp).relu(), (-dm).relu()))
    return out


def reference(x, wd, wg, bias, wo):
    dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        xw = x.to(dtype)
        z = F.conv2d(xw, wd.to(dtype))
        rs = reference_responses(z)
        gate = torch.cat([z] + [r.abs().mean(1, keepdim=True) for r in rs], 1)
        a = F.conv2d(gate, wg.to(dtype), bias.to(dtype)).softmax(1)
        y = xw + F.conv2d(sum(a[:, i:i+1] * r for i, r in enumerate(rs)), wo.to(dtype))
        return y.to(x.dtype)


def difference(a, b, atol=2e-5, rtol=2e-4):
    a, b = a.detach().double(), b.detach().double()
    return dict(max_abs=float((a-b).abs().max()), relative_L2=float((a-b).norm() / b.norm().clamp_min(1e-30)),
                allclose=bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()), atol=atol, rtol=rtol)


def math_checks():
    torch.manual_seed(481)
    cases = {}
    for h, w in [(1, 1), (2, 5), (5, 3), (7, 9), (11, 13)]:
        z = torch.randn(2, 32, h, w, dtype=torch.float64)
        a, b = BLC.responses(z), reference_responses(z)
        error = max(float((x-y).abs().max()) for x, y in zip(a, b))
        require(error < 1e-8, "Final-coordinate clamp/candidate order mismatch")
        cases[f"{h}x{w}"] = error
    patterns = {}
    for name in ["flat", "ramp", "step", "positive1", "positive2", "positive3", "negative1", "negative2", "negative3"]:
        z = torch.zeros(1, 32, 15, 17, dtype=torch.float64)
        if name == "flat":
            z.fill_(3)
        elif name == "ramp":
            z += torch.arange(15, dtype=z.dtype)[None, None, :, None] * 3
            z += torch.arange(17, dtype=z.dtype)[None, None, None, :] * 2
        elif name == "step":
            z[:, :, 7:] = 2
        else:
            z[:, :, 7:7+int(name[-1])] = 2 if name.startswith("positive") else -2
        actual, expected = BLC.responses(z), reference_responses(z)
        a, b = torch.stack(actual), torch.stack(expected)
        mask = torch.ones_like(a, dtype=torch.bool)
        mask[..., 3:-3, 3:-3] = False
        patterns[name] = dict(interior_max_error=float((a-b)[..., 3:-3, 3:-3].abs().max()),
                              boundary_max_error=float((a-b)[mask].abs().max()),
                              horizontal_s1_min=float(a[0].min()), horizontal_s1_max=float(a[0].max()))
        require(torch.equal(a, b), "Ideal pattern mismatch")
        if name in ("flat", "ramp"):
            require(torch.count_nonzero(a[..., 3:-3, 3:-3]) == 0, "Flat/ramp interior must vanish")
        if name == "step":
            require(torch.count_nonzero(a[:2]) == 0, "Horizontal response on one-sided step must vanish")
        if "positive" in name:
            require(a[1, :, :, 7:7+int(name[-1]), 3:-3].max() > 0, "Positive signed line lost")
        if "negative" in name:
            require(a[1, :, :, 7:7+int(name[-1]), 3:-3].min() < 0, "Negative signed line lost")
    precision = {}
    for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
        m = BLC().to(dtype=dtype)
        x = torch.randn(1, 128, 5, 7).to(dtype).requires_grad_()
        require(torch.equal(m(x), x), "Zero Wo must be exact identity")
        with torch.no_grad():
            m.Wo.weight.normal_(0, .04)
            m.Wg.weight.normal_(0, .03)
            m.Wg.bias.normal_(0, .02)
        actual = m(x)
        expected = reference(x, m.Wd.weight, m.Wg.weight, m.Wg.bias, m.Wo.weight)
        atol, rtol = (1e-8, 1e-6) if dtype == torch.float64 else (2e-5, 2e-4)
        error = difference(actual, expected, atol, rtol)
        require(error["allclose"] and error["finite"], "Nonzero forward differs")
        parameters = [x, m.Wd.weight, m.Wg.weight, m.Wg.bias, m.Wo.weight]
        probe = torch.randn_like(actual)
        ag = torch.autograd.grad((actual * probe).sum(), parameters)
        bg = torch.autograd.grad((expected * probe).sum(), parameters)
        grad = [difference(a, b, atol, rtol) for a, b in zip(ag, bg)]
        entry = dict(forward=error, gradients=grad)
        if dtype in (torch.float16, torch.bfloat16):
            # Diagnose native low-dtype leaf-gradient rounding without relaxing
            # the strict FP32/FP64 reference tolerances. The raw comparison stays.
            mf = deepcopy(m).float()
            xf = x.detach().float().requires_grad_()
            pf = [xf, mf.Wd.weight, mf.Wg.weight, mf.Wg.bias, mf.Wo.weight]
            af = torch.autograd.grad((mf(xf) * probe.float()).sum(), pf)
            bf = torch.autograd.grad((reference(*pf) * probe.float()).sum(), pf)
            fp32 = [difference(a, b) for a, b in zip(af, bf)]
            cast_exact = all(torch.equal(a, f.to(dtype)) for a, f in zip(ag, af))
            ref_cast_exact = all(torch.equal(b, f.to(dtype)) for b, f in zip(bg, bf))
            require(all(g["allclose"] and g["finite"] for g in fp32) and cast_exact and ref_cast_exact,
                    "Low-dtype gradient discrepancy not explained by native leaf rounding")
            entry.update(fp32_same_quantized_values=fp32, native_leaf_cast_exact=cast_exact,
                         reference_leaf_cast_exact=ref_cast_exact,
                         status="PRECISION_NOTE" if not all(g["allclose"] for g in grad) else "PASSED")
        else:
            require(all(v["allclose"] and v["finite"] for v in grad), "Reference gradient mismatch")
        precision[str(dtype)] = entry
    # Directional gradcheck uses a fixed random smooth point and fast analytical
    # vs central-difference probes. No tolerance adaptation or resampling.
    from torch.func import functional_call
    m = BLC().double()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, .08)
    x = torch.randn(1, 128, 5, 7, dtype=torch.float64, requires_grad=True)
    names, params = zip(*m.named_parameters())
    checked = torch.autograd.gradcheck(lambda *v: functional_call(m, dict(zip(names, v[1:])), (v[0],)),
                                     (x, *params), fast_mode=True, eps=1e-6, atol=1e-5, rtol=1e-3)
    # Explicit ninth-choice test: its logits suppress all signed branches.
    with torch.no_grad():
        m.Wg.weight.zero_(); m.Wg.bias.zero_()
        ordinary = m(x)
        m.Wg.bias[8] = 30
        bypass = m(x)
        ratio = float((bypass-x).norm() / (ordinary-x).norm())
        require(ratio < 1e-10, "Zero response option was renormalized away")
    return dict(status="PASSED", rectangles=cases, ideal_patterns=patterns, precision=precision,
                gradcheck=checked, gradcheck_tolerance="eps=1e-6 atol=1e-5 rtol=1e-3 (numerical derivative)",
                ninth_choice_residual_ratio=ratio)


def topology_checks(variant, initialized):
    loaded = torch_load(initialized, map_location="cpu")["model"].float()
    candidate, audit = rebuild(loaded.yaml, loaded, variant)
    parent = build(variant, parent=True)
    parent.load_state_dict({k: v for k, v in candidate.state_dict().items() if k not in KEYS}, strict=True)
    parent.eval(); candidate.eval()
    x = torch.rand(1, 3, 640, 640)
    captures = [{}, {}]
    consumers, calls, hooks = {}, [], []
    try:
        for model, capture in zip((parent, candidate), captures):
            for i in (4, 5, 6, 7):
                hooks.append(model.model[i].register_forward_hook(lambda m, a, y, i=i, c=capture: c.update({i: y.detach()})))
        for i in (6, 17):
            hooks.append(candidate.model[i].register_forward_pre_hook(lambda m, a, i=i: consumers.update({i: a[0].data_ptr()})))
        hooks.append(candidate.model[5].blc.register_forward_hook(lambda m, a, y: calls.append(y.data_ptr())))
        with torch.no_grad():
            a, b = parent(x)[0], candidate(x)[0]
        require(len(calls) == 1 and consumers[6] == consumers[17] == calls[0], "BLC consumer routing/call count")
        require(torch.equal(a, b), "Initial native output differs")
        require(all(torch.equal(captures[0][i], captures[1][i]) for i in captures[0]), "Initial backbone features differ")
        require(tuple(captures[1][4].shape) == (1, 64, 160, 160) and
                tuple(captures[1][5].shape) == (1, 128, 80, 80), "P3 shape changed")
    finally:
        for hook in hooks:
            hook.remove()
    fused = deepcopy(candidate).fuse(verbose=False)
    verify_model(fused, variant, fused=True)
    # Compare post-construction RNG states without build()'s isolation masking it.
    from ultralytics.nn.tasks import RTDETRDetectionModel
    from blc_common import MODEL_DIR
    rng, states = [], []
    for cfg in VARIANTS[variant]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(923)
            model = RTDETRDetectionModel(str(MODEL_DIR/cfg), nc=1, verbose=False)
            rng.append(torch.get_rng_state())
            states.append(model.state_dict())
    require(torch.equal(*rng), "BLC consumes later common-layer CPU RNG")
    require(all(torch.equal(v, states[0][k]) for k, v in states[1].items()), "Public constructor differs")
    return dict(status="PASSED", native_get_model=audit, exact_initial_features=True, exact_native_output=True,
                blc_calls=1, consumers=[6, 17], public_rng_unchanged=True,
                unfused_parameters=COUNTS[variant][0], fused_parameters=COUNTS[variant][1], added_parameters=8561)


def loss_checks(variant, initialized, dataset, device):
    from c19_lif_v1_data import real_batch
    from ultralytics.models.rtdetr.train import RTDETRTrainer
    from ultralytics.utils import IterableSimpleNamespace
    from ultralytics.utils.torch_utils import ModelEMA
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(42, deterministic=True)
    loaded = torch_load(initialized, map_location="cpu")["model"].float()
    model, _ = rebuild(loaded.yaml, loaded, variant)
    model = model.to(device).train()
    model.args = {**DEFAULT_CFG_DICT}
    model.nc, model.names = 1, {0: "crack"}  # Native Trainer.set_model_attributes contract.
    batch, images = real_batch(Path(dataset), size=160, count=2)
    batch = {k: v.to(device) for k, v in batch.items()}
    require(len(batch["cls"]) > 0, "Native detection loss needs valid GT")
    dummy = RTDETRTrainer.__new__(RTDETRTrainer)
    dummy.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    optimizer = dummy.build_optimizer(model, "AdamW", lr=.0005, momentum=.937, decay=.0001)
    ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
    require(len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters()}, "Optimizer binding")
    ema, rows = ModelEMA(model), []
    for step in range(3):
        before = {k: p.detach().clone() for k, p in model.named_parameters() if k in KEYS}
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(batch)
        loss.sum().backward()
        require(torch.isfinite(loss).all(), "Nonfinite native detection loss")
        require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), "Nonfinite gradient")
        grads = {k: float(p.grad.norm()) for k, p in model.named_parameters() if k in KEYS}
        require(grads["model.5.blc.Wo.weight"] > 0, "Wo gradient did not start")
        if step == 0:
            require(all(v == 0 for k, v in grads.items() if "Wo." not in k), "Upstream first-step gradient should be zero")
        else:
            require(all(v > 0 for v in grads.values()), "Nonzero branch upstream gradients missing")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step(); ema.update(model)
        rows.append(dict(step=step, loss=float(loss.sum()), new_gradient_norms=grads,
                         parameter_delta={k: float((p-before[k]).norm()) for k, p in model.named_parameters() if k in KEYS}))
    from blc_lifecycle import lifecycle
    from blc_common import recipe
    write_json(paths(variant)["evidence"]/("loss-gradient-"+str(device).replace(":", "_")+"-"+stamp()+".json"),
               dict(status="PASSED", seed=42, device=str(device), images=images, updates=rows,
                    scope="B2/160 FP32 real-GT native loss, not capacity", code=code_identity()))
    life = lifecycle(model, optimizer, ema, variant, recipe(variant)[0], device)
    return dict(status=life["status"], device=str(device), scope="real GT, B2/160 FP32 native loss smoke; NOT B16/640 capacity",
                seed=42, deterministic=True, images=images, updates=rows, optimizer_all_parameters_once=True,
                ema_updates=ema.updates, lifecycle=life)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=VARIANTS, default="cbr_lif_blc_v1")
    p.add_argument("--initialized", type=Path)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--only", choices=["math", "topology", "loss"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    report = dict(status="FAILED", scope=args.only, variant=args.variant, runtime=runtime(), code=code_identity())
    try:
        if args.only == "math":
            report["checks"] = math_checks()
        elif args.only == "topology":
            report["checks"] = topology_checks(args.variant, args.initialized)
        else:
            report["checks"] = loss_checks(args.variant, args.initialized, args.dataset, args.device)
        report["status"] = report["checks"]["status"]
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(args.output, report)
        print(json.dumps({k: report[k] for k in ("status", "scope", "variant")}), flush=True)


if __name__ == "__main__":
    main()
