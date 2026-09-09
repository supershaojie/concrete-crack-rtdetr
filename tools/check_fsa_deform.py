"""Bounded synthetic FSA validation; never train a dataset or report detection performance."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import io
import os
from pathlib import Path
import time
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F
from init_fsa_deform import (ROOT, MODEL_DIR, VARIANTS, controlled_models, initialize, require, runtime,
                             sha256, verify_model, write_json, RTDETRDecoderFSA)
from train_fsa_deform import recipe
from ultralytics.nn.modules.fsa_deform import FSADeformAttn, support_geometry, multi_scale_deformable_attn_fsa_pytorch
from ultralytics.nn.modules.transformer import MSDeformAttn
from ultralytics.nn.modules.utils import multi_scale_deformable_attn_pytorch
from ultralytics.utils.patches import torch_load


def errors(a, b):
    a, b = a.detach().float(), b.detach().float()
    require(torch.isfinite(a).all() and torch.isfinite(b).all(), "Nonfinite equivalence tensor")
    diff = (a - b).abs()
    return dict(max_abs_error=diff.max().item(), max_rel_error=(diff / a.abs().clamp_min(1e-8)).max().item())


def equivalent(a, b, atol=2e-5, rtol=2e-5):
    result = errors(a, b)
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return result


def geometry_and_fields():
    raw = torch.zeros(1, 1, 8, 3)
    box = torch.tensor([[[[.5, .5, .2, .1]]]])
    shapes = [(20, 40), (10, 20), (5, 10)]
    radius, delta, eta = support_geometry(raw, box, shapes, 4)
    expected = torch.tensor([[1., .25], [.5, .25], [.25, .25]])
    torch.testing.assert_close(radius[0, 0, 0], expected)
    signs = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    torch.testing.assert_close(delta[0, 0, 0, 0], signs * torch.tensor([1., .25]) / (3 ** .5 * torch.tensor([40., 20.])))
    for wh, logit, expected_radius in [(100., 0., 1.5), (1e-9, 0., .25), (100., 100., 2.), (1e-9, -100., .125)]:
        b = box.clone(); b[..., 2:] = wh
        r = raw.clone(); r[..., :2] = logit
        actual = support_geometry(r, b, shapes, 4)[0]
        torch.testing.assert_close(actual, torch.full_like(actual, expected_radius))
    raw[..., 0], raw[..., 1] = 1., -1.
    independent = support_geometry(raw, box, shapes, 4)[0]
    require(not torch.equal(independent[..., 0], independent[..., 1]), "x/y tied")
    # Read an exact feature-grid pixel center and off-cell nonlinear neighborhoods.
    height, width = 11, 13
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    location = torch.tensor([[[[[[(6.5) / width, (5.5) / height]]]]]])
    weights = torch.ones(1, 1, 1, 1, 1)
    raw = torch.zeros(1, 1, 1, 3)
    box = torch.tensor([[[[.5, .5, 1., 1.]]]])
    radius, delta, _ = support_geometry(raw, box, [(height, width)], 4)
    result = dict(base_radius=expected.tolist(), signs=signs.tolist(), clamps="passed", independent_xy=True)
    for name, field in [("constant", torch.ones_like(xx) * 3), ("affine", xx * 2 + yy * 3 + 1), ("nonlinear", (xx - 6).square() + (yy - 5).square())]:
        value = field.float().reshape(1, height * width, 1, 1)
        center = multi_scale_deformable_attn_pytorch(value, [(height, width)], location, weights)
        outputs = {}
        captured = []
        native = F.grid_sample
        def sample(feature, grid, **kwargs):
            captured.append(grid.detach().clone())
            return native(feature, grid, **kwargs)
        for a in (0., 1., -1., 100., -100.):
            eta = .5 * torch.tensor([[[a]]]).tanh()
            with patch("ultralytics.nn.modules.fsa_deform.F.grid_sample", side_effect=sample):
                output = multi_scale_deformable_attn_fsa_pytorch(value, [(height, width)], location, weights, delta, eta)
            outputs[str(a)] = output.item()
            require(abs(eta.item()) <= .5, "eta range")
            if a == 0 or name != "nonlinear":
                equivalent(center, output, atol=1e-5)
        if name == "nonlinear":
            require(outputs["1.0"] > outputs["0.0"] > outputs["-1.0"], "Signed eta behavior wrong")
            require(outputs["100.0"] > .1, "Neighborhood difference missing")
        positions = torch.cat((location[0, 0, 0, 0, 0].unsqueeze(0), location[0, 0, 0, 0, 0] + delta[0, 0, 0, 0]), 0)
        torch.testing.assert_close(captured[0][0, 0], 2 * positions - 1)
        require(len(captured) == 5, "More than one grid_sample per level")
        result[name] = dict(center=center.item(), outputs=outputs)
    # All levels/head dimensions, padding, out-of-range centers, and finite learned support.
    torch.manual_seed(19)
    q, value = torch.randn(2, 17, 256), torch.randn(2, 83, 256)
    refs = torch.rand(2, 17, 3, 4); refs[..., :2] = refs[..., :2] * 2 - .5
    shapes = [(7, 9), (4, 4), (2, 2)]
    mask = torch.rand(2, 83) > .6
    base, fsa = MSDeformAttn(256, 3, 8, 4), FSADeformAttn()
    fsa.load_state_dict({**fsa.state_dict(), **base.state_dict()}, strict=True)
    result["masked_out_of_bounds_zero"] = equivalent(base(q, refs, value, shapes, mask), fsa(q, refs, value, shapes, mask))
    with torch.no_grad(): fsa.support_fc2.weight.normal_(0, .1)
    # Independent nonzero oracle: four separate native helper reads (test-only), then signed per-head residual.
    projected = fsa.value_proj(value).masked_fill(mask[..., None], 0).view(2, 83, 8, 32)
    offsets = fsa.sampling_offsets(q).view(2, 17, 8, 3, 4, 2)
    attention = fsa.attention_weights(q).view(2, 17, 8, 12).softmax(-1).view(2, 17, 8, 3, 4)
    centers = refs[:, :, None, :, None, :2] + offsets / 4 * refs[:, :, None, :, None, 2:] * .5
    _, displacements, eta = support_geometry(fsa.support_parameters(q), refs, shapes, 4)
    v0 = multi_scale_deformable_attn_pytorch(projected, shapes, centers, attention)
    area = sum(multi_scale_deformable_attn_pytorch(projected, shapes, centers + displacements[..., i, :].unsqueeze(-2), attention)
               for i in range(4)) / 4
    oracle = fsa.output_proj((v0.view(2,17,8,32) + eta.unsqueeze(-1) * (area-v0).view(2,17,8,32)).flatten(-2))
    result["nonzero_vectorized_oracle"] = equivalent(oracle, fsa(q, refs, value, shapes, mask))
    with patch.object(fsa, "support_parameters", side_effect=AssertionError("2D fallback used support")):
        result["2d_fallback"] = equivalent(base(q, refs[..., :2], value, shapes, mask), fsa(q, refs[..., :2], value, shapes, mask), atol=0, rtol=0)
    return result


def staged_gradients():
    torch.manual_seed(28)
    model = FSADeformAttn()
    q, value = torch.randn(2, 37, 256), torch.randn(2, 336, 256)
    refs = torch.rand(2, 37, 1, 4) * .3 + .3
    shapes = [(16, 16), (8, 8), (4, 4)]
    probe = torch.randn_like(q)
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    steps = []
    for step in range(3):
        optimizer.zero_grad()
        (model(q, refs, value, shapes) * probe).mean().backward()
        grads = model.support_fc2.weight.grad.reshape(8, 3, 16)
        row = dict(step=step, tx=grads[:, 0].abs().sum().item(), ty=grads[:, 1].abs().sum().item(),
                   a=grads[:, 2].abs().sum().item(), fc1=model.support_fc1.weight.grad.abs().sum().item())
        require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), "Nonfinite gradients")
        steps.append(row)
        optimizer.step()
    require(steps[0]["a"] > 0 and steps[0]["tx"] == steps[0]["ty"] == steps[0]["fc1"] == 0, "Zero-init stage wrong")
    require(all(steps[1][k] > 0 for k in ("tx", "ty", "a", "fc1")), "Support did not activate")
    buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
    restored = FSADeformAttn(); restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    require(all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()), "Learned support reload failed")
    return dict(steps=steps, learned_state_dict_reload="exact")


def native_training_step(fsa):
    """One whole-model native matching/loss/backward + ordinary AdamW step on synthetic inputs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = deepcopy(fsa).to(device).train()
    model.nc = model.model[-1].nc  # RTDETRTrainer.set_model_attributes normally sets this before loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.0001)
    torch.manual_seed(44)
    batch = dict(img=torch.randn(2,3,160,160,device=device), cls=torch.zeros(2,1,device=device),
                 bboxes=torch.tensor([[.5,.5,.2,.1],[.3,.3,.1,.2]],device=device),batch_idx=torch.tensor([0,1],device=device))
    with torch.autocast(device, enabled=device=="cuda"):
        loss, components = model.loss(batch)
    loss.backward()
    added = model.model[-1].decoder.layers[1].cross_attn
    gradient = added.support_fc2.weight.grad.view(8,3,16)[:,2].abs().sum().item()
    require(torch.isfinite(loss) and torch.isfinite(components).all() and gradient > 0, "Native loss/eta activation failed")
    require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), "Native full backward nonfinite")
    require({id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in model.parameters()}, "Optimizer dropped state")
    optimizer.step()
    require(torch.count_nonzero(added.support_fc2.weight), "AdamW did not update FSA")
    return dict(device=device, amp=device=="cuda", synthetic_batch=2, imgsz=160, steps=1, loss=loss.item(),
                components=components.tolist(), a_gradient_sum=gradient, optimizer="AdamW; all parameters once", finite_backward=True)


def full_equivalence(base, fsa, device="cpu", train=False):
    base, fsa = base.to(device), fsa.to(device)
    base.train(train); fsa.train(train)
    torch.manual_seed(41)
    image = torch.randn(2 if train else 1, 3, 160, 160, device=device)
    batch = dict(cls=torch.zeros(3, dtype=torch.long, device=device), bboxes=torch.tensor([[.5,.5,.2,.1],[.3,.3,.15,.2],[.7,.6,.1,.1]], device=device),
                 batch_idx=torch.tensor([0, 0, 1], device=device), gt_groups=[2,1]) if train else None
    records, outputs = [], []
    for model in (base, fsa):
        seen = {}; hooks = []
        for index in (1, 2):
            def pre(module, args, index=index):
                seen[f"layer{index}_input"] = args[0].detach().clone()
                seen[f"layer{index}_reference_requires_grad"] = args[1].requires_grad
                if len(args) > 5 and args[5] is not None: seen[f"layer{index}_mask"] = args[5].detach().clone()
            def post(module, args, output, index=index): seen[f"layer{index}_output"] = output.detach().clone()
            hooks.extend([model.model[-1].decoder.layers[index].register_forward_pre_hook(pre), model.model[-1].decoder.layers[index].register_forward_hook(post)])
        torch.manual_seed(42)
        with torch.no_grad(): output = model.predict(image.clone(), batch=batch)
        for hook in hooks: hook.remove()
        records.append(seen); outputs.append(output)
    result = {k: equivalent(v, records[1][k]) for k, v in records[0].items() if isinstance(v, torch.Tensor) and v.dtype != torch.bool}
    for k, v in records[0].items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.bool: require(torch.equal(v, records[1][k]), "DN attention mask changed")
    if train:
        for index, key in enumerate(("bbox", "score", "enc_bbox", "enc_score")):
            result[key] = equivalent(outputs[0][index], outputs[1][index])
        a, b = outputs[0][-1], outputs[1][-1]
        require(a.keys() == b.keys(), "DN meta keys")
        require(a["dn_num_split"] == b["dn_num_split"] and a["dn_num_group"] == b["dn_num_group"], "DN metadata changed")
        require(all(torch.equal(x,y) for x,y in zip(a["dn_pos_idx"],b["dn_pos_idx"])), "DN query order changed")
        result["queries"] = records[1]["layer1_input"].shape[1]
        require(result["queries"] > 300, "DN queries absent")
        result["reference_detached"] = not records[1]["layer1_reference_requires_grad"]
        result["dn_meta_and_mask"] = "exact"
    else:
        result["bbox"] = equivalent(outputs[0][0][..., :4], outputs[1][0][..., :4])
        result["score"] = equivalent(outputs[0][0][..., 4:], outputs[1][0][..., 4:])
    return result


def precision_and_detach(fsa):
    if not torch.cuda.is_available(): return dict(status="SKIPPED: CUDA unavailable")
    results = {}
    model = deepcopy(fsa.model[-1]).cuda().train()
    model.shapes = None  # invalidate native unregistered anchor cache after this test-only device move
    # A learned support branch exercises geometric backward as well as eta backward.
    with torch.no_grad(): model.decoder.layers[1].cross_attn.support_fc2.weight.normal_(0, .02)
    batch = dict(cls=torch.zeros(2, dtype=torch.long, device="cuda"), bboxes=torch.tensor([[.5,.5,.2,.1],[.3,.3,.1,.2]],device="cuda"),
                 batch_idx=torch.tensor([0,1],device="cuda"),gt_groups=[1,1])
    for amp in (False, True):
        model.zero_grad(set_to_none=True)
        features = [torch.randn(2,256,h,w,device="cuda",requires_grad=True) for h,w in ((20,20),(10,10),(5,5))]
        refs = []
        hook = model.decoder.layers[1].cross_attn.register_forward_pre_hook(lambda m,a: refs.append(a[1].requires_grad))
        with torch.autocast("cuda", enabled=amp):
            output = model(features, batch)
            loss = output[0].square().mean() + output[1].square().mean()
        loss.backward(); hook.remove()
        require(torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), "Precision backward nonfinite")
        require(refs == [False], "FSA added previous bbox gradient path")
        require(all(x.grad is not None and torch.isfinite(x.grad).all() for x in features), "Feature gradient missing")
        results["CUDA_AMP" if amp else "CUDA_FP32"] = dict(loss=loss.item(), backward="finite", reference_detached=True,
                          fc1_grad=model.decoder.layers[1].cross_attn.support_fc1.weight.grad.abs().sum().item())
    del model
    model = deepcopy(fsa).cuda().half().eval()
    model.model[-1].shapes = None
    with torch.no_grad():
        model.model[-1].decoder.layers[1].cross_attn.support_fc2.weight.normal_(0,.02)
        output = model(torch.randn(1,3,160,160,device="cuda",dtype=torch.float16))[0]
    require(output.dtype == torch.float16 and torch.isfinite(output).all(), "True-half whole model inference failed")
    results["true_model_half"] = dict(dtype=str(output.dtype), shape=list(output.shape), finite=True)
    buffer=io.BytesIO(); torch.save(dict(model=model.float().cpu()),buffer); buffer.seek(0)
    restored=torch_load(buffer,map_location="cpu")["model"]
    require(all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items()), "Learned full model serialization changed state")
    results["learned_full_model_save_load"] = "exact; support_fc2 remains nonzero"
    return results


def benchmark(base, fsa):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = dict(device=device, scope="synthetic whole-model eval, B1 160x160, 5 warmups + 20 forwards; local coarse estimate only", runs={})
    base.cpu(); fsa.cpu(); gc.collect()
    if device == "cuda": torch.cuda.empty_cache()
    for name, original in (("C2", base), ("FSA", fsa)):
        model = deepcopy(original).to(device).eval()
        model.model[-1].shapes = None
        image = torch.randn(1,3,160,160,device=device)
        with torch.no_grad():
            for _ in range(5): model(image)
            if device == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); before=torch.cuda.memory_allocated()
            start=time.perf_counter()
            for _ in range(20): model(image)
            if device == "cuda": torch.cuda.synchronize()
            elapsed=(time.perf_counter()-start)*1000/20
        result["runs"][name]=dict(forward_ms=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=="cuda" else None,
                                  incremental_peak_bytes=torch.cuda.max_memory_allocated()-before if device=="cuda" else None)
        del model,image; gc.collect()
        if device == "cuda": torch.cuda.empty_cache()
    return result


def run(source, output):
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    base, fsa, mapping = controlled_models(source)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    write_json(output/"weight_mapping.json",mapping)
    report=dict(runtime=runtime(), scope="synthetic only; no formal training/full val/test", deterministic="True warn_only=True (native C2 CUDA grid_sample backward is nondeterministic)",
                parameters={"baseline":sum(p.numel() for p in base.parameters()),"fsa":sum(p.numel() for p in fsa.parameters())},
                structure=[type(l.cross_attn).__name__ for l in fsa.model[-1].decoder.layers])
    report["parameters"]["delta"]=report["parameters"]["fsa"]-report["parameters"]["baseline"]
    report["geometry_eta_fields_fallback"]=geometry_and_fields(); print("geometry passed",flush=True)
    report["staged_gradient"]=staged_gradients(); print("gradients passed",flush=True)
    report["cpu_eval_zero"]=full_equivalence(base,fsa)
    report["cpu_dn_zero"]=full_equivalence(base,fsa,train=True); print("CPU whole model / DN equivalence passed",flush=True)
    if torch.cuda.is_available(): report["cuda_eval_zero"]=full_equivalence(base,fsa,device="cuda")
    report["precision"]=precision_and_detach(fsa.cpu()); base.cpu(); print("precision passed",flush=True)
    report["native_synthetic_training_step"]=native_training_step(fsa)
    report["runtime_benchmark"]=benchmark(base,fsa)
    args,rows=recipe(ROOT/"docs/fsa_deform/c2_args.yaml","fsa_deform",Path("/root/autodl-tmp/projects/Crack_RTDETR-fsa-deform/weights/fsa_deform_controlled_init.pt"))
    report["recipe_fields"]=len(rows); write_json(output/"recipe_diff.json",rows)
    try: RTDETRDecoderFSA(ndl=2)
    except ValueError: report["non_three_layers_rejected"]=True
    else: raise AssertionError("Non-C2 decoder accepted")
    init=ROOT/"weights/fsa_deform_controlled_init.pt"
    if not init.exists(): write_json(output/"initialization.json",initialize(source,init))
    report["status"]="passed"
    report["unverified"]=["formal 200 epoch training", "complete dataset val/test and mAP", "AutoDL RTX 4090 batch16 640 performance", "export/ONNX/custom backends"]
    write_json(output/"validation.json",report)
    print(report["parameters"],report["runtime_benchmark"],flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,default=ROOT/"docs/fsa_deform")
    a=p.parse_args(); run(a.source,a.output)
