"""Read-only, single-backbone-graph CBR conditional-gradient probe. No optimizer."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import random
import sys

import numpy as np
import torch

from cbr_rescue_common import ROOT, require
sys.path.insert(0, str(ROOT / "ultralytics-main"))
from ultralytics.nn.modules.cbr import RTDETRDecoderCBR
from ultralytics.nn.modules.cscef_v51 import CSCEFv51

EPS = 1e-12
GRAD_ATOL = 1e-7
GRAD_RTOL = 1e-5
NOISE_MULTIPLIER = 10.0


def tensor_record(t):
    t = t.detach().contiguous().cpu()
    if t.is_floating_point():
        require(bool(torch.isfinite(t).all()), "Nonfinite tensor in probe.")
    return {"shape": list(t.shape), "dtype": str(t.dtype), "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest()}


def tree_record(x):
    if isinstance(x, torch.Tensor):
        return tensor_record(x)
    if isinstance(x, dict):
        return {k: tree_record(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [tree_record(v) for v in x]
    return x


def tree_clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()  # graph retained; no shared mutable loss-input storage
    if isinstance(x, dict):
        return {k: tree_clone(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(tree_clone(v) for v in x)
    return x


def state_record(model):
    return {"parameters": {n: tensor_record(p) for n, p in model.named_parameters()},
            "buffers": {n: tensor_record(b) for n, b in model.named_buffers()},
            "grad_attributes": {n: None if p.grad is None else tensor_record(p.grad) for n, p in model.named_parameters()}}


def rng_record():
    return {"python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            "numpy": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
            "torch_cpu": tensor_record(torch.get_rng_state()),
            "torch_cuda": [tensor_record(s) for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def seed_batch(index):
    seed = 42 + index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def objects(model):
    found = [(n, m) for n, m in model.named_modules() if isinstance(m, RTDETRDecoderCBR)]
    require(len(found) == 1, "Exactly one actual RTDETRDecoderCBR required.")
    name, head = found[0]
    require(model.model[-1] is head, "CBR must be the actual terminal head.")
    require(isinstance(head.f, list) and len(head.f) == 3 and head.f[0] >= 0, "Unexpected P3 connection.")
    cs = [(n, m) for n, m in model.named_modules() if isinstance(m, CSCEFv51)]
    require(len(cs) <= 1, "Unexpected CSCEF objects.")
    require(head.cbr.rho == .10 and head.cbr.normal_fraction == .10, "CBR constants changed.")
    roles = {"neck_p3": model.model[head.f[0]], "decoder_last": head.decoder.layers[-1],
             "cbr": head.cbr, "box_head_last": head.dec_bbox_head[-1],
             "cscef_projection": cs[0][1].output_projection if cs else None,
             "cscef_all": cs[0][1] if cs else None}
    module_names = {id(m): n for n, m in model.named_modules()}
    parameter_names = {id(p): n for n, p in model.named_parameters()}
    description = {role: None if m is None else {"module": module_names[id(m)],
                   "type": type(m).__name__, "parameters": [parameter_names[id(p)] for p in m.parameters()]}
                   for role, m in roles.items()}
    return head, roles, description


@contextmanager
def mixed_graph_mode(model):
    """Enable only native routing flags; keep every other module, including BN/MHA/dropout, eval."""
    head, _, _ = objects(model)
    modes = {m: m.training for m in model.modules()}
    flags = {p: p.requires_grad for p in model.parameters()}
    try:
        model.eval()
        model.training = head.training = head.decoder.training = True
        for p in model.parameters():
            if p.is_floating_point():
                p.requires_grad_(True)
        stochastic = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd,
                      torch.nn.MultiheadAttention, torch.nn.RNNBase)
        require(all(not m.training for m in model.modules() if isinstance(m, stochastic)), "Stateful/random module in train mode.")
        yield {"training_objects": [n or "<root>" for n, m in model.named_modules() if m.training],
               "all_other_modules_eval": True,
               "requires_grad_before": {n: flags[p] for n, p in model.named_parameters()},
               "requires_grad_during": {n: p.requires_grad for n, p in model.named_parameters()}}
    finally:
        for m, mode in modes.items():
            m.training = mode
        for p, flag in flags.items():
            p.requires_grad_(flag)


@contextmanager
def capture_forward(head, roles, captured):
    handles = []
    def cbr_input(module, args):
        require("inputs" not in captured, "Unexpected second native CBR call.")
        captured["inputs"] = args[:3]
    def p3_output(module, args, output):
        captured["p3_object"] = output
    def query_output(module, args, output):
        captured["query_object"] = output
    def decoder_input(module, args, kwargs):
        captured["decoder_calls"] = captured.get("decoder_calls", 0) + 1
        captured["decoder_embed"] = args[0]
        captured["decoder_reference"] = args[1]
        captured["attention_mask"] = kwargs.get("attn_mask")
    try:
        handles.append(head.cbr.register_forward_pre_hook(cbr_input))
        handles.append(roles["neck_p3"].register_forward_hook(p3_output))
        handles.append(roles["decoder_last"].register_forward_hook(query_output))
        handles.append(head.decoder.register_forward_pre_hook(decoder_input, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


def index_lists(indices):
    return [[a.detach().cpu().tolist(), b.detach().cpu().tolist()] for a, b in indices]


@contextmanager
def loss_observer(criterion, report):
    """Observe native matching, actual quality targets and all loss terms, without replacing their math."""
    handles, replaced = [], []
    report.update(hungarian=[], dn_matches=[], classification_targets=[], layer_losses=[], terms={})
    def replace(name, wrap):
        existed, value = name in criterion.__dict__, criterion.__dict__.get(name)
        original = getattr(criterion, name)
        setattr(criterion, name, wrap(original))
        replaced.append((name, existed, value))
    def matcher(module, args, output):
        report["hungarian"].append({"prediction_shape": list(args[0].shape), "indices": index_lists(output)})
    def dn_wrapper(original):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            report["dn_matches"].append(index_lists(result))
            return result
        return wrapped
    def class_wrapper(original):
        def wrapped(scores, targets, gt_scores, num_gts, postfix=""):
            report["classification_targets"].append({"postfix": postfix, "num_gts": num_gts,
                "targets": tensor_record(targets), "quality": tensor_record(gt_scores), "scores": tensor_record(scores)})
            return original(scores, targets, gt_scores, num_gts, postfix)
        return wrapped
    def layer_wrapper(original):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            report["layer_losses"].append({k: float(v.detach()) for k, v in result.items()})
            return result
        return wrapped
    def terms_hook(module, args, output):
        report["terms"] = {k: float(v.detach()) for k, v in output.items()}
    try:
        handles.append(criterion.matcher.register_forward_hook(matcher))
        handles.append(criterion.register_forward_hook(terms_hook))
        replace("get_dn_match_indices", dn_wrapper)
        replace("_get_loss_class", class_wrapper)
        replace("_get_loss", layer_wrapper)
        yield
    finally:
        for handle in handles:
            handle.remove()
        for name, existed, value in reversed(replaced):
            if existed:
                setattr(criterion, name, value)
            else:
                delattr(criterion, name)


def loss_call(model, batch, prediction):
    before = tree_record((prediction, batch))
    copied = tree_clone(prediction), tree_clone(batch)
    copied_before = tree_record(copied)
    report = {}
    with loss_observer(model.criterion, report):
        total, displayed = model.loss(copied[1], preds=copied[0])
    require(before == tree_record((prediction, batch)), "Native loss mutated shared prediction/batch.")
    require(copied_before == tree_record(copied), "Unexpected native loss input mutation; investigate before comparison.")
    require(bool(torch.isfinite(total)), "Nonfinite native loss.")
    report.update(total=float(total.detach()), displayed=displayed.detach().cpu().tolist(), input_unchanged=True)
    return total, report


def gradient_read(loss, inputs, retain_graph):
    grads = torch.autograd.grad(loss, inputs, retain_graph=retain_graph, create_graph=False, allow_unused=True)
    require(all(g is None or bool(torch.isfinite(g).all()) for g in grads), "Nonfinite autograd result, including repeat noise read.")
    return [None if g is None else g.detach().cpu() for g in grads]


def stats_for_group(indices, tensors, ga, gb, noise=None, reference_noise=None):
    """FP64 reductions, no cross-model dot products; unused tensors count as zero only for arithmetic."""
    aa = bb = cc = dot = noise_sq = max_abs = 0.0
    zero_a = zero_b = unused_a = unused_b = 0
    elements = 0
    equality = True
    for i in indices:
        a, b = ga[i], gb[i]
        shape = tensors[i].shape
        unused_a += int(a is None)
        unused_b += int(b is None)
        a = torch.zeros(shape, dtype=torch.float64) if a is None else a.double()
        b = torch.zeros(shape, dtype=torch.float64) if b is None else b.double()
        require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), "Nonfinite gradient.")
        c = a - b
        elements += a.numel()
        aa += float(a.square().sum()); bb += float(b.square().sum()); cc += float(c.square().sum())
        dot += float((c * b).sum())
        zero_a += int(a.count_nonzero() == 0); zero_b += int(b.count_nonzero() == 0)
        if a.numel():
            max_abs = max(max_abs, float(c.abs().max()))
        equality &= bool(torch.all(c.abs() <= GRAD_ATOL + GRAD_RTOL * torch.maximum(a.abs(), b.abs())))
        if noise is not None and noise[i] is not None:
            noise_sq += float(noise[i].double().square().sum())
    na, nb, nc = math.sqrt(aa), math.sqrt(bb), math.sqrt(cc)
    nr = math.sqrt(noise_sq) if reference_noise is None else reference_noise
    floor = max(EPS, NOISE_MULTIPLIER * nr, 1e-7 * max(na, nb))
    meaningful = nc > floor and nb > floor
    return {"parameters_or_tensors": len(indices), "elements": elements,
            "norm_all": na, "norm_keep": nb, "norm_condition": nc,
            "condition_over_keep": nc / (nb + EPS), "dot_condition_keep": dot,
            "cosine_condition_keep": dot / (nc * nb) if meaningful else None,
            "cosine_meaningful": meaningful, "cosine_noise_floor": floor,
            "negative_projection_over_keep": -dot / (bb + EPS), "repeat_A_noise_norm": nr,
            "condition_over_noise": nc / (nr + EPS), "max_abs_all_minus_keep": max_abs,
            "all_keep_close": equality, "finite": True, "unused_all": unused_a, "unused_keep": unused_b,
            "zero_tensors_all": zero_a, "zero_tensors_keep": zero_b, "eps": EPS}


def probe_batch(model, batch, batch_index, first_noise=None):
    """Exactly one model.predict and two native model.loss calls; first batch repeats A gradient once."""
    head, roles, descriptions = objects(model)
    require(model.training and head.training and head.decoder.training, "Mixed native training route required.")
    require(not torch.is_inference_mode_enabled() and torch.is_grad_enabled(), "Probe needs autograd enabled.")
    captured, hooks_before = {}, {id(m): (len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()}
    seed = seed_batch(batch_index)
    rng_before = rng_record()
    groups = [(batch["batch_idx"] == i).sum().item() for i in range(batch["img"].shape[0])]
    targets = {"cls": batch["cls"].long().flatten(), "bboxes": batch["bboxes"],
               "batch_idx": batch["batch_idx"].long().flatten(), "gt_groups": groups}
    with capture_forward(head, roles, captured):
        native, details_a = model.predict(batch["img"], batch=targets, cbr_diagnostics=True)
    require(captured["decoder_calls"] == 1, "More than one backbone/decoder forward.")
    p3, query, boxes = captured["inputs"]
    require(p3 is captured["p3_object"] and query is captured["query_object"], "Captured tensors differ from role objects.")
    require(boxes is details_a["before"] and boxes.requires_grad, "Original boxes graph must be preserved.")
    rng_forward = rng_record()
    refined_b, details_b = head.cbr(p3.detach(), query.detach(), boxes, return_diagnostics=True)
    require(details_b["before"] is boxes, "B detached/replaced original boxes.")
    require(tree_record(details_a) == tree_record(details_b), "A/B CBR forward diagnostics differ.")
    dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta = native
    alternative = (torch.cat((dec_boxes[:-1], refined_b.unsqueeze(0)), dim=0), dec_scores, enc_boxes, enc_scores, dn_meta)
    output_a, output_b = tree_record(native), tree_record(alternative)
    require(output_a == output_b, "A/B raw boxes/logits/encoder/DN are not bitwise equal.")
    loss_a, report_a = loss_call(model, batch, native)
    loss_b, report_b = loss_call(model, batch, alternative)
    require(report_a == report_b, "A/B native loss, matching or quality targets differ.")
    require(rng_record() == rng_forward, "B or loss generated additional randomness/DN.")
    for m in model.modules():
        if id(m) in hooks_before:
            require(hooks_before[id(m)] == (len(m._forward_hooks), len(m._forward_pre_hooks)), "Observer hook leaked.")
    ndn = dn_meta["dn_num_split"][0] if dn_meta else 0
    dn = {"metadata": tree_record(dn_meta), "queries_per_image": ndn, "gt_groups": groups,
          "no_gt_no_dn_expected": sum(groups) == 0, "decoder_forward_calls": captured["decoder_calls"],
          "initial_noisy_boxes": tensor_record(captured["decoder_reference"][:, :ndn]),
          "initial_dn_embeddings": tensor_record(captured["decoder_embed"][:, :ndn]),
          "attention_mask": tree_record(captured["attention_mask"]), "reused_same_DN_object": native[4] is alternative[4]}
    require(ndn > 0 or sum(groups) == 0, "GT exists but native DN is missing.")
    tensors, ids, group_indices = [], {}, {}
    for role, module in roles.items():
        group_indices[role] = []
        if module is None:
            continue
        for p in module.parameters():
            require(p.requires_grad, "Observed parameter incorrectly frozen.")
            if id(p) not in ids:
                ids[id(p)] = len(tensors)
                tensors.append(p)
            group_indices[role].append(ids[id(p)])
    box_index = len(tensors)
    tensors.append(boxes)
    group_indices["input_boxes"] = [box_index]
    ga = gradient_read(loss_a, tensors + [details_a["after"]], retain_graph=True)
    repeat_noise = None
    if batch_index == 0:
        repeat = gradient_read(loss_a, tensors + [details_a["after"]], retain_graph=True)
        repeat_noise = [(None if a is None and b is None else
                         (torch.zeros_like(b) if a is None else a) - (torch.zeros_like(a) if b is None else b))
                        for a, b in zip(ga, repeat)]
        del repeat
    gb = gradient_read(loss_b, tensors + [refined_b], retain_graph=False)
    statistics = {}
    for role, indices in group_indices.items():
        if not indices:
            statistics[role] = {"applicable": False, "reason": "C19 has no CSCEF"}
            continue
        noise = None if first_noise is None else first_noise[role]
        statistics[role] = {"applicable": True, **stats_for_group(indices, tensors, ga, gb, repeat_noise, noise)}
        statistics[role]["noise_source_batch"] = 0
    # CBR and the raw-box head have no conditional-input path to cut: enforce gradient agreement.
    for role in ("cbr", "box_head_last", "input_boxes"):
        require(statistics[role]["all_keep_close"], f"A/B {role} gradients differ beyond fixed FP32 tolerance.")
    # Explicit local Jacobian check proves identity AND original w/h displacement scaling remain.
    v = gb[-1]
    u = details_b["tanh_offsets"].detach().cpu()
    gbox = gb[box_index]
    jacobian = {"applicable": gbox is not None and v is not None, "rho": head.cbr.rho}
    if jacobian["applicable"]:
        left, right, top, bottom = u.unbind(-1)
        rho = head.cbr.rho
        expected = torch.stack((v[..., 0], v[..., 1],
            v[..., 2] * (1 + rho * (right-left)) + v[..., 0] * .5 * rho * (left+right),
            v[..., 3] * (1 + rho * (bottom-top)) + v[..., 1] * .5 * rho * (top+bottom)), -1)
        torch.testing.assert_close(gbox, expected, atol=GRAD_ATOL, rtol=GRAD_RTOL)
        jacobian.update(passed=True, max_abs_error=float((gbox-expected).abs().max()),
                        gradient_by_coordinate={axis: float(gbox[..., j].double().norm()) for j, axis in enumerate(("cx", "cy", "w", "h"))})
    else:
        require(sum(groups) == 0, "Raw box path unexpectedly unused with GT.")
        jacobian.update(passed=True, reason="Empty GT has no native box loss; not evidence of freezing.")
    learning = {role: statistics[role]["norm_keep"] > EPS for role in ("cbr", "box_head_last", "input_boxes")}
    cbr_parameters = {name: stats_for_group([i], tensors, ga, gb, repeat_noise)
                      for name, i in zip(descriptions["cbr"]["parameters"], group_indices["cbr"])}
    return {"batch": batch_index, "seed": seed, "images": list(batch["im_file"]), "gt_groups": groups,
            "mode": "native training routes, all other modules eval, FP32", "roles": descriptions,
            "outputs_equal_bitwise": True, "raw_output_hashes": output_a,
            "details_equal_bitwise": True, "loss_and_matching_equal": True, "loss_A": report_a, "loss_B": report_b,
            "DN": dn, "RNG_before": rng_before, "RNG_after_forward": rng_forward, "RNG_after_B_and_loss": rng_record(),
            "gradient_groups": statistics, "box_path_jacobian": jacobian, "nonzero_B_gradients": learning,
            "cbr_parameter_gradients": cbr_parameters, "statistics_dtype": "float64",
            "repeated_A_gradient_reads": int(batch_index == 0), "all_hooks_removed": True,
            "gradient_tolerances": {"atol": GRAD_ATOL, "rtol": GRAD_RTOL},
            "overlap_note": "CSCEF projection is contained in CSCEF-all; groups are never summed; no cross-model cosine."}


# These are preregistered engineering triage rules, not a universal cosine/conflict criterion.
DECISION_RULE = {"primary_roles": ["neck_p3", "decoder_last"], "min_batches": 3,
                 "condition_over_keep_min": .10, "negative_projection_over_keep_min": .05,
                 "noise_multiplier": NOISE_MULTIPLIER, "C20_minus_C19_projection_min": .05,
                 "meaning": "Conservative local evidence for ONE trial; no causal claim or promised training gain."}


def decide(reports):
    evidence, qualifies = {}, []
    for role in DECISION_RULE["primary_roles"]:
        rows = []
        for i in range(4):
            c19, c20 = (reports[name][i]["gradient_groups"][role] for name in ("C19", "C20"))
            delta = c20["negative_projection_over_keep"] - c19["negative_projection_over_keep"]
            noise_relative = sum(s["cosine_noise_floor"] / (s["norm_keep"] + EPS) for s in (c19, c20))
            # The noise floor already includes 10x the one measured same-A repeat difference.
            hit = (c20["cosine_meaningful"] and c20["condition_over_keep"] >= .10
                   and c20["negative_projection_over_keep"] >= .05 and delta >= .05
                   and c19["norm_keep"] > c19["cosine_noise_floor"] and delta > noise_relative)
            rows.append({"batch": i, "C19": c19, "C20": c20, "paired_projection_difference": delta,
                         "paired_relative_noise_screen": noise_relative,
                         "meets_preregistered_screen": hit})
        evidence[role] = rows
        if sum(r["meets_preregistered_screen"] for r in rows) >= 3:
            qualifies.append(role)
    eligible = all(any(r["nonzero_B_gradients"][role] for r in reports[name])
                   for name in ("C19", "C20") for role in ("cbr", "box_head_last", "input_boxes"))
    if not eligible:
        return {"category": "诊断实现/输入问题", "reason": "四批中仍无法确认 CBR/原框主路径有非零梯度；不使用方向统计决定补救。",
                "rule": DECISION_RULE, "evidence": evidence}
    return {"category": "有尝试依据" if qualifies else "没有明确依据", "qualifying_roles": qualifies,
            "rule": DECISION_RULE, "evidence": evidence,
            "recommendation": ("仅建议将 P3/query 条件分支限制回传作为唯一一次 CBR-v2 组合训练候选，保持 CSCEF 和 CBR 前向不变。"
                               if qualifies else "按预先固定的小样本标准，没有清晰且区别于 C19 的线索；建议停止 CBR 补救，保留 C17。"),
            "limits": "固定16图与局部梯度不能证明200轮训练下降的原因；负余弦不等于梯度冲突，不保证改后涨点；未启动训练。"}
