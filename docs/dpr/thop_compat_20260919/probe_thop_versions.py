"""Independent tiny runtime proof; run in an isolated process per THOP version."""
import hashlib
from copy import deepcopy
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import random

import numpy as np

import torch
from torch import nn
import thop


class Child(nn.Module):
    def forward(self, x):
        return x


class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = Child()

    def forward(self, x):
        return self.child(x)


def count_parent(module, inputs, output):
    module.total_ops += 7


def count_child(module, inputs, output):
    module.total_ops += 11


def cleanup_fixture(repo):
    sys.path.insert(0, str(Path(repo) / "ultralytics-main"))
    from ultralytics.nn.modules.dpr import (
        DPRConvNormLayer, _probe_thop_semantics, profile_dpr, thop_dpr_semantics,
    )
    from ultralytics.nn.modules.block import ConvNormLayer

    _probe_thop_semantics.cache_clear()
    py_rng = random.getstate()
    np_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    semantics = thop_dpr_semantics()
    assert py_rng == random.getstate()
    now_numpy = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(np_rng, now_numpy))
    assert torch.equal(torch_rng, torch.get_rng_state())

    class UnsupportedContainer(nn.Module):
        def __init__(self):
            super().__init__()
            self.parent = Parent()

        def forward(self, x):
            return self.parent(x)

    report = {"rng_unchanged_after_cold_probe": True, "semantics": semantics,
              "dpr_source_sha256": hashlib.sha256(Path(inspect.getfile(profile_dpr)).read_bytes()).hexdigest(),
              "cases": {}}
    for failure in (False, True):
        model = UnsupportedContainer()
        model.train()
        model.parent.eval()
        model.parent.child.train()
        model.parent.register_buffer("total_ops", torch.tensor([123.0]), persistent=False)
        model.parent.child.register_buffer("total_params", torch.tensor([456.0]), persistent=True)
        user_hook = model.parent.register_forward_hook(lambda m, args, kwargs, output: None,
                                                       with_kwargs=True, always_call=True)
        attributes = ("_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called")
        snapshots = [(m, m.training, dict(m._buffers), set(m._non_persistent_buffers_set),
                      {key: dict(getattr(m, key)) for key in attributes}) for m in model.modules()]
        state = {key: value.clone() for key, value in model.state_dict().items()}

        def maybe_fail(module, inputs, output):
            count_parent(module, inputs, output)
            if failure:
                raise RuntimeError("deliberate counting failure")

        results = []
        for repeat in range(2):
            caught = False
            try:
                ops, params = profile_dpr(model, inputs=(torch.zeros(1),),
                                          custom_ops={Parent: maybe_fail, Child: count_child}, verbose=False)
            except RuntimeError as error:
                assert failure and str(error) == "deliberate counting failure", error
                caught = True
            assert caught == failure
            for module, training, buffers, non_persistent, hooks in snapshots:
                assert module.training == training
                assert set(module._buffers) == set(buffers)
                assert all(module._buffers[key] is value for key, value in buffers.items())
                assert module._non_persistent_buffers_set == non_persistent
                assert all(dict(getattr(module, key)) == values for key, values in hooks.items())
                assert "total_ops" not in module.__dict__
                assert "total_params" not in module.__dict__
            assert set(model.state_dict()) == set(state)
            assert all(torch.equal(model.state_dict()[key], value) for key, value in state.items())
            results.append({"repeat": repeat + 1, "exception_caught": caught,
                            "all_hooks_buffers_state_and_modes_restored": True,
                            "ops": None if failure else ops})
        user_hook.remove()
        report["cases"]["exception" if failure else "success"] = results

    parent = ConvNormLayer(128, 128, 3, 1, act=None).eval()
    candidate = DPRConvNormLayer(deepcopy(parent)).eval()
    inputs = (torch.zeros(1, 128, 8, 8),)
    parent_ops = profile_dpr(parent, inputs=inputs, verbose=False)[0]
    report["dpr_layout_cases"] = []
    for deployed in (False, True):
        for nested in (False, True):
            target = deepcopy(candidate)
            if deployed:
                target.switch_to_deploy()
            model = nn.Sequential(target) if nested else target
            state = {key: value.clone() for key, value in model.state_dict().items()}
            expected_buffers = {name: set(module._buffers) for name, module in model.named_modules()}
            expected_modes = {name: module.training for name, module in model.named_modules()}
            for repeat in range(2):
                ops = profile_dpr(model, inputs=inputs, verbose=False)[0]
                assert ops == parent_ops, (deployed, nested, ops, parent_ops)
                assert set(model.state_dict()) == set(state)
                assert all(torch.equal(model.state_dict()[key], value) for key, value in state.items())
                assert all(set(module._buffers) == expected_buffers[name] for name, module in model.named_modules())
                assert all(module.training == expected_modes[name] for name, module in model.named_modules())
                assert all(not module._forward_hooks for module in model.modules())
                assert all("total_ops" not in module.__dict__ for module in model.modules())
                report["dpr_layout_cases"].append({"deployed": deployed, "nested": nested,
                                                   "repeat": repeat + 1, "ops": ops,
                                                   "parent_ops": parent_ops, "difference": ops - parent_ops,
                                                   "hooks_buffers_state_modes_unchanged": True})
    return report


def main():
    result = {
        "fixture_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "torch": torch.__version__,
        "thop_module_version": thop.__version__,
        "thop_distribution_version": importlib.metadata.version("ultralytics-thop"),
        "thop_file": str(Path(thop.__file__).resolve()),
        "profile_file": inspect.getfile(thop.profile),
        "profile_source_sha256": hashlib.sha256(Path(inspect.getfile(thop.profile)).read_bytes()).hexdigest(),
        "expected_parent_ops": 7,
        "expected_child_ops": 11,
        "cases": {},
    }
    for name, model in [("nested", nn.Sequential(Parent())), ("standalone", Parent())]:
        values = []
        for _ in range(2):
            ops, params = thop.profile(model, inputs=(torch.zeros(1),),
                                       custom_ops={Parent: count_parent, Child: count_child}, verbose=False)
            values.append({"ops": ops, "params": params,
                           "remaining_buffers": {n: list(m._buffers) for n, m in model.named_modules()},
                           "remaining_hooks": {n: len(m._forward_hooks) for n, m in model.named_modules()}})
        result["cases"][name] = values
    result["observed_aggregation"] = {7.0: "replace_subtree", 18.0: "accumulate_children"}.get(result["cases"]["nested"][0]["ops"], "unexpected")
    hooks = importlib.import_module("thop.vision.basic_hooks")
    result["count_convNd_source"] = inspect.getsource(hooks.count_convNd)
    result["count_normalization_source"] = inspect.getsource(hooks.count_normalization)
    result["calculate_conv2d_flops_source"] = inspect.getsource(hooks.calculate_conv2d_flops)
    profile_source = inspect.getsource(thop.profile)
    result["profile_source"] = profile_source
    if len(sys.argv) > 2:
        result["dpr_profile_cleanup"] = cleanup_fixture(sys.argv[2])
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("_source")}, indent=2))


if __name__ == "__main__":
    main()
