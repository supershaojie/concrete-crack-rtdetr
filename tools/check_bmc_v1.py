"""Bounded BMC-specific checks; CPU and CUDA interface evidence are separate from capacity."""
from __future__ import annotations

import argparse
from copy import deepcopy
import itertools
import os
from pathlib import Path
import tempfile

from bmc_v1_common import ROOT, OUT, MODEL, require, runtime, write
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics.models.utils.bmc import BMCConfig, BMCLoss, BMCDetectionModel, bounded_consensus
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.models.utils.ops import HungarianMatcher
from ultralytics.nn.tasks import RTDETRDetectionModel


def pair(queries, offset=0):
    return torch.tensor(queries, dtype=torch.long), torch.arange(len(queries)) + offset


def equal_pair(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def reference(c, x2, x3, budget=.02, floor=.001):
    # Independent adjacency/BFS reference (no union/find), with linear percentile by hand.
    q, g = c.shape
    graph = {}
    for j in range(g):
        for i in (x2[j], x3[j]):
            graph.setdefault(("q", i), set()).add(("g", j))
            graph.setdefault(("g", j), set()).add(("q", i))
    def percentile(column, p):
        ordered = sorted(column)
        t = (len(ordered)-1)*p
        low = int(t)
        return ordered[low] + (ordered[min(low+1, len(ordered)-1)]-ordered[low])*(t-low)
    result, seen = list(x2), set()
    for j in range(g):
        node = ("g", j)
        if node in seen:
            continue
        pending, component = [node], set()
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(graph[node]-component)
        seen |= component
        targets = [n for kind, n in component if kind == "g"]
        if all(float(c[x3[n], n]-c[x2[n], n]) <= budget*max(
            percentile(c[:, n].tolist(), .75)-percentile(c[:, n].tolist(), .25), floor) for n in targets):
            for n in targets:
                result[n] = x3[n]
    return result


def algorithm_checks():
    cases = [
        ("same", [[0., 1.], [1., 0.], [4., 4.], [8., 8.]], [0, 1], [0, 1]),
        ("near_cycle", [[0., .001], [.001, 0.], [4., 4.], [8., 8.]], [0, 1], [1, 0]),
        ("over_budget_cycle", [[0., .001], [2., 0.], [4., 4.], [8., 8.]], [0, 1], [1, 0]),
        ("alternating_path", [[0., 1.], [.001, 0.], [5., .001], [8., 8.]], [0, 1], [1, 2]),
        ("partial_components", [[0., 8.], [.001, 8.], [8., 0.], [8., 2.]], [0, 2], [1, 3]),
        ("negative", [[-8., -7.999], [-7.999, -8.], [-4., -4.], [0., 0.]], [0, 1], [1, 0]),
        ("iqr_zero", [[-2., -2.], [-2., -2.]], [0, 1], [1, 0]),
        ("negative_delta", [[0., 0.], [-.01, -.01], [4., 4.], [8., 8.]], [0, 1], [1, 0]),
    ]
    records = []
    for name, matrix, x2, x3 in cases:
        c = torch.tensor(matrix)
        p = pair(x2)
        actual, report = bounded_consensus([p], [pair(x3)], [c], [len(x2)], detail=True)
        want = reference(c, x2, x3)
        got = dict(zip(actual[0][1].tolist(), actual[0][0].tolist()))
        require([got[i] for i in range(len(x2))] == want, name)
        require(len(set(got.values())) == len(want), "query reused")
        if want == x2:
            require(actual[0] is p, "Unchanged image must retain ordering/object")
        records.append(dict(name=name, assignment=want, stats=report))
    empty = pair([])
    out, _ = bounded_consensus([empty], [empty], [torch.empty(3, 0)], [0])
    require(out[0] is empty, "Empty path changed")
    over = (torch.tensor([0, 1]), torch.tensor([0, 2]))
    out, _ = bounded_consensus([over], [(torch.tensor([1, 0]), torch.tensor([1, 2]))], [torch.ones(2, 3)], [3])
    require(out[0] is over, "G>Q must fall back entirely")
    # Different GT counts, empty middle image, and an overfull image; offsets are batch-global.
    costs = [torch.tensor(cases[1][1]), torch.empty(4, 0), torch.zeros(4, 1), torch.ones(1, 2)]
    a2 = [pair([0, 1]), pair([], 2), pair([0], 2), (torch.tensor([0]), torch.tensor([4]))]
    a3 = [pair([1, 0]), pair([], 2), pair([3], 2), (torch.tensor([0]), torch.tensor([3]))]
    out, _ = bounded_consensus(a2, a3, costs, [2, 0, 1, 2])
    require(equal_pair(out[2], pair([3], 2)) and out[3] is a2[3], "Batch GT offsets/fallback")
    a3[0] = a2[0]
    isolated, _ = bounded_consensus(a2, a3, costs, [2, 0, 1, 2])
    require(all(equal_pair(a, b) for a, b in zip(out[1:], isolated[1:])), "Cross-image contamination")
    try:
        bounded_consensus([pair([0])], [pair([1], 1)], [torch.ones(3, 1)], [1])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid global GT offset was accepted")
    zero, _ = bounded_consensus([pair([0, 1])], [pair([1, 0])], [torch.zeros(2, 2)], [2],
                                BMCConfig(budget_iqr_fraction=0))
    require(not equal_pair(zero[0], pair([0, 1])), "Zero budget incorrectly disabled consensus")
    return dict(status="PASS", cases=records, empty=True, g_gt_q=True, batch_isolation=True, zero_budget_active=True)


def crafted(device):
    torch.manual_seed(17)
    boxes = torch.rand(4, 1, 300, 4, device=device) * .6 + .2
    gt = torch.tensor([[.4, .5, .1, .1], [.4001, .5, .1, .1]], device=device)
    boxes[:, 0, :2] = gt
    boxes[3, 0, :2] = gt.flip(0)
    boxes.requires_grad_()
    scores = torch.full((4, 1, 300, 1), -2., device=device, requires_grad=True)
    batch = dict(cls=torch.zeros(2, dtype=torch.long, device=device), bboxes=gt, gt_groups=[2])
    dn_box = gt.repeat(3, 1, 1, 1).clone().requires_grad_()
    dn_score = torch.zeros(3, 1, 2, 1, device=device, requires_grad=True)
    meta = dict(dn_pos_idx=[torch.tensor([0, 1])], dn_num_group=1)
    return boxes, scores, batch, dn_box, dn_score, meta


def loss_checks(device):
    boxes, scores, batch, db, ds, dm = crafted(device)
    native = RTDETRDetectionLoss(nc=1, use_vfl=True)
    expected = native((boxes, scores), batch, db, ds, dm)
    params = (boxes, scores, db, ds)
    gradients = torch.autograd.grad(sum(expected.values()), params, retain_graph=True)
    for config, epoch in ((BMCConfig(enabled=False), 20), (BMCConfig(), 19)):
        tested = BMCLoss(config=config, epoch=epoch)
        actual = tested((boxes, scores), batch, db, ds, dm)
        require(actual.keys() == expected.keys() and all(torch.equal(actual[k], v) for k, v in expected.items()), "Disabled loss mismatch")
        grads = torch.autograd.grad(sum(actual.values()), params, retain_graph=True)
        require(all(torch.equal(a, b) for a, b in zip(grads, gradients)), "Disabled gradient mismatch")
    routes = {}
    active = BMCLoss(epoch=20)
    active.route_observer = lambda slot, indices: routes.update({slot: indices})
    actual = active((boxes, scores), batch, db, ds, dm)
    require(active.last_report["images"][0]["changed"] == 2, "Crafted mechanism never activated")
    for slot in (0, 1, 3):
        want = native.matcher(boxes[slot], scores[slot], batch["bboxes"], batch["cls"], [2])
        require(all(equal_pair(a, b) for a, b in zip(routes[f"ordinary_{slot}"], want)), "Non-target route changed")
    require(equal_pair(routes["ordinary_2"][0], routes["ordinary_3"][0]), "Target layer did not change")
    require(all(torch.equal(actual[k], v) for k, v in expected.items() if not k.endswith("_aux")), "Main or DN loss changed")
    grads = torch.autograd.grad(sum(actual.values()), params)
    require(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads), "Nonfinite/missing prediction gradient")
    indices, costs = native.matcher(boxes[2], scores[2], batch["bboxes"], batch["cls"], [2], return_costs=True)
    require(not costs[0].requires_grad, "Cost has gradient")
    i, j = linear_sum_assignment(costs[0])
    require(equal_pair(indices[0], (torch.tensor(i), torch.tensor(j))), "C2 is not the matrix used by SciPy")
    bad = boxes[2].detach().clone()
    bad[0, 0, 0] = float("nan")
    try:
        native.matcher(bad, scores[2], batch["bboxes"], batch["cls"], [2], return_costs=True)
    except ValueError as error:
        require("Nonfinite Hungarian cost" in str(error), "Missing cost context")
    else:
        raise AssertionError("BMC silently sanitized a nonfinite cost")
    return dict(status="PASS", changed_gt=2, only_slot_2=True, dn_unchanged=True, disabled_exact_loss_and_grad=True,
                cost_detached=True, scipy_exact_cost=True, epoch19_disabled_epoch20_enabled=True)


def model_checks(device):
    torch.manual_seed(42)
    model = BMCDetectionModel(str(MODEL), nc=1, verbose=False).to(device)
    model.set_bmc_epoch(19)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        native = RTDETRDetectionModel(str(MODEL), nc=1, verbose=False).to(device)
    native.nc = model.nc = 1  # normally installed by native Trainer.set_model_attributes
    from bmc_v1_training import compare_states
    inventory = compare_states(native, model)
    require(inventory["states"] == 552 and inventory["parameters"] == 20149765, "Mother count mismatch")
    batch = dict(img=torch.rand(2, 3, 160, 160, device=device),
        cls=torch.zeros(3, 1, device=device), bboxes=torch.tensor([[.4, .5, .1, .2], [.7, .2, .1, .1], [.5, .6, .2, .1]], device=device),
        batch_idx=torch.tensor([0, 0, 1], device=device))
    state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None
    reference_loss = native(batch)[0]
    reference_loss.backward()
    torch.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    actual_loss = model(batch)[0]
    actual_loss.backward()
    require(torch.equal(actual_loss, reference_loss), "Full model disabled forward/loss differs")
    max_gradient_difference = 0.
    max_gradient_scaled_error = 0.
    for (n, p), (m, q) in zip(native.named_parameters(), model.named_parameters()):
        difference = float((p.grad-q.grad).abs().max()) if p.grad is not None and q.grad is not None else 0.
        magnitude = float(p.grad.abs().max()) if p.grad is not None else 0.
        close = difference <= 2e-6 + 2e-5*magnitude
        require(n == m and ((p.grad is None and q.grad is None) or
                (p.grad is not None and q.grad is not None and close)),
                f"Public gradient differs beyond FP32 tolerance: {n}, max_diff="
                f"{float((p.grad-q.grad).abs().max()) if p.grad is not None and q.grad is not None else 'missing'}")
        if p.grad is not None:
            max_gradient_difference = max(max_gradient_difference, float((p.grad-q.grad).abs().max()))
            max_gradient_scaled_error = max(max_gradient_scaled_error, difference/(1+magnitude))
    del native
    model.zero_grad(set_to_none=True)
    model.set_bmc_epoch(20)
    active_loss = model(batch)[0]
    active_loss.backward()
    require(torch.isfinite(active_loss), "Nonfinite real model loss")
    names = ("model.0.conv.weight", "model.20.O_proj.weight", "model.26.dec_bbox_head.1.layers.2.weight", "model.26.cbr.offset_out.weight")
    gradients = {n: dict(finite=bool(torch.isfinite(p.grad).all()), norm=float(p.grad.norm()))
                 for n, p in model.named_parameters() if n in names and p.grad is not None}
    require(len(gradients) == len(names) and all(v["finite"] and v["norm"] > 0 for v in gradients.values()), "Original parameter gradient missing")
    with torch.no_grad():
        model.model[-1].cbr.offset_out.weight.fill_(.01)
        model.model[20].O_proj.weight.fill_(.002)
    model.eval()
    class Sentinel:
        def record(self, *args):
            raise AssertionError("BMC ran outside training")
    model.criterion.collector = Sentinel()
    with torch.no_grad():
        y = model.predict(batch["img"])
        eval_loss = model.loss(batch, y)[0]
    require(torch.isfinite(y[0]).all() and torch.isfinite(eval_loss), "Nonfinite eval")
    require(model.criterion.last_report["state"] == "EVAL_NATIVE", "Eval routing wrong")
    return dict(status="PASS", scope="B2/160 interface, not B16/640 capacity", inventory=inventory,
                model_type=type(model).__name__, criterion_type=type(model.criterion).__name__,
                disabled_full_forward_loss_exact=True, gradient_atol=2e-6, gradient_rtol=2e-5,
                gradient_rule="per-tensor infinity-norm bound: max_abs_diff <= atol + rtol * reference_max_abs (CUDA grid_sample atomic sums)",
                maximum_scaled_gradient_error=max_gradient_scaled_error,
                maximum_gradient_difference=max_gradient_difference, real_gradients=gradients,
                nontrivial_lif_cbr_eval_predict_isolation=True)


def run(device="cpu", full=True):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    report = dict(status="FAIL", runtime=runtime(), device=device)
    try:
        report["algorithm"] = algorithm_checks()
        report["loss"] = loss_checks(device)
        if full:
            report["model"] = model_checks(device)
        report["status"] = "PASS"
        return report
    finally:
        write(OUT / ("checks_" + device.replace(":", "_") + ".json"), report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--algorithm-only", action="store_true")
    args = parser.parse_args()
    run(args.device, not args.algorithm_only)
