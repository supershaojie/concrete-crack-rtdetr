"""Bounded CPU FP32 checks; never train a full model or run a dataset evaluation."""
from __future__ import annotations

from copy import deepcopy
import time
import traceback

import torch
from torch.nn import functional as F
from rdm_common import (RDM, ROOT, MODEL_DIR, VARIANTS, NEW, require, paths, controlled_models,
                        training_rebuild, verify_model, runtime, write_json, code_identity)


def core_checks():
    before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    module = RDM()
    require(torch.equal(before, torch.random.get_rng_state()), "RDM consumed global CPU RNG")
    require(all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all() if cuda_before else [])), "RDM changed CUDA RNG")
    rows = []
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(79)
        for h, w in ((16, 16), (13, 19)):
            x = torch.randn(2, 128, h, w, requires_grad=True)
            require(torch.equal(module(x), x), "Zero Wo is not identity")
            module.zero_grad()
            (module(x) * torch.randn_like(x)).sum().backward()
            require(torch.isfinite(module.Wo.weight.grad).all() and module.Wo.weight.grad.abs().sum() > 0, "Wo did not start gradients")
            require(all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for n, p in module.named_parameters() if n != "Wo.weight"), "Unexpected upstream gradient before Wo update")
            active = deepcopy(module)
            with torch.no_grad():
                active.Wo.weight.add_(-.001 * module.Wo.weight.grad)
            active.zero_grad()
            x.grad = None
            real = active(x)
            # Independent explicit formula: tensor convolutions and reshape mean pooling.
            ref_x = x.detach().clone().requires_grad_(True)
            state = {n: p.detach().clone().requires_grad_(True) for n, p in active.named_parameters()}
            z = F.conv2d(ref_x, state["Wd.weight"])
            local = F.silu(F.conv2d(z, state["Dl.weight"], padding=1, groups=32))
            q = z[:, :, :h // 4 * 4, :w // 4 * 4].reshape(2, 32, h // 4, 4, w // 4, 4).mean((3, 5))
            c = F.silu(F.conv2d(F.conv2d(q, state["Dc.weight"], padding=1, groups=32), state["Pc.weight"]))
            c = F.interpolate(c, size=(h, w), mode="bilinear", align_corners=False)
            gate = torch.sigmoid(F.conv2d(torch.cat((z, c, abs(z - c)), 1), state["Wg.weight"], state["Wg.bias"]))
            reference = ref_x + F.conv2d(gate * local, state["Wo.weight"])
            require(real.shape == x.shape and torch.allclose(real, reference, atol=2e-5, rtol=2e-4), "Explicit formula mismatch")
            probe = torch.randn_like(real)
            (real * probe).mean().backward()
            (reference * probe).mean().backward()
            max_grad = 0.
            for n, p in active.named_parameters():
                require(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, "Inactive/invalid branch gradient: " + n)
                require(torch.allclose(p.grad, state[n].grad, atol=2e-5, rtol=2e-4), "Formula gradient mismatch: " + n)
                max_grad = max(max_grad, float((p.grad - state[n].grad).abs().max()))
            altered_gate = torch.sigmoid(F.conv2d(torch.cat((z, c + .25, abs(z - c - .25)), 1), state["Wg.weight"], state["Wg.bias"]))
            require((altered_gate - gate).abs().max() > 1e-5, "Coarse context cannot affect gate")
            require(torch.isfinite(x.grad).all() and torch.allclose(x.grad, ref_x.grad, atol=2e-5, rtol=2e-4), "Formula input gradient mismatch")
            rows.append(dict(shape=list(x.shape), pool_shape=list(q.shape), identity_exact=True,
                             forward_max_abs=float((real - reference).abs().max()), gradient_max_abs=max_grad,
                             context_gate_max_abs=float((altered_gate - gate).abs().max()), all_active_gradients_finite_nonzero=True))
    for shape in ((3, 8), (8, 3)):
        try:
            module(torch.zeros(1, 128, *shape))
        except ValueError:
            pass
        else:
            raise RuntimeError("Undersized input silently accepted")
    return dict(status="PASSED", parameter_count=sum(p.numel() for p in module.parameters()),
                state_items=len(module.state_dict()), tensor_buffers=0, isolated_cpu_cuda_rng=True, cases=rows,
                explicit_formula_tolerance=dict(atol=2e-5, rtol=2e-4), small_input_rejected=True)


def model_checks(source, report, progress):
    initial = None
    for variant in VARIANTS:
        progress("model." + variant)
        parent, target, common = controlled_models(source, variant)
        additions = {k: v for k, v in target.state_dict().items() if k in NEW}
        if initial is None:
            initial = additions
        else:
            require(all(torch.equal(initial[k], v) and initial[k].data_ptr() != v.data_ptr() for k, v in additions.items()), "Variant RDM init/storage mismatch")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            target, trainer = training_rebuild(str(MODEL_DIR / VARIANTS[variant][0]), target, dict(nc=1, channels=3), variant)
        # Same native nc adaptation RNG for the controlled parent.
        from init_c19_lif_v1 import native_rebuild
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            parent = native_rebuild(str(MODEL_DIR / VARIANTS[variant][1]), parent, 1, 3)
        target.eval(); parent.eval()
        observed = dict(calls=0)
        handles = []
        def stage_hook(module, args, out):
            observed["stage_pointer"] = out.data_ptr()
            observed["shape"] = list(out.shape)
        def rdm_hook(module, args, out):
            observed["calls"] += 1
        def consumer(index):
            def hook(module, args):
                require(args[0].data_ptr() == observed["stage_pointer"], f"model.{index} did not consume enhanced P3")
                observed[str(index)] = True
            return hook
        handles += [target.model[5].register_forward_hook(stage_hook), target.model[5].rdm.register_forward_hook(rdm_hook)]
        handles += [target.model[i].register_forward_pre_hook(consumer(i)) for i in (6, 17)]
        with torch.no_grad():
            prediction = target(torch.zeros(1, 3, 640, 640))
        for handle in handles:
            handle.remove()
        require(observed["calls"] == 1 and observed["shape"] == [1, 128, 80, 80], "RDM placement/call/shape mismatch")
        require(torch.isfinite(prediction[0]).all(), "Nonfinite model forward")
        counts = dict(parent=sum(p.numel() for p in parent.parameters()), target=sum(p.numel() for p in target.parameters()))
        require(counts["target"] - counts["parent"] == 12896, "Actual delta differs")
        observed.pop("stage_pointer")
        row = dict(status="RUNNING", initialization=common, trainer_rebuild=trainer, connection=observed, parameters_unfused=counts)
        report[variant] = row
        # One existing THOP count per parent/target, same nc1/640/unfused scope.
        try:
            import thop
        except ImportError:
            row["counter"] = dict(status="PENDING", reason="Existing THOP unavailable; no installation")
        else:
            with torch.no_grad():
                values = [thop.profile(deepcopy(m), inputs=(torch.zeros(1, 3, 640, 640),), verbose=False)[0] * 2 / 1e9 for m in (parent, target)]
            row["counter"] = dict(status="PASSED", library=thop.__version__, parent_GFLOPs=values[0], target_GFLOPs=values[1], delta_GFLOPs=values[1]-values[0],
                                  scope="THOP supported operators only; nc1, B1, 640, unfused; functional interpolate/abs/SiLU/sigmoid/multiply/add not all covered")
        learned = deepcopy(target.model[5].rdm.state_dict())
        target.fuse(verbose=False); parent.fuse(verbose=False)
        verify_model(target, variant)
        require(all(torch.equal(v, target.model[5].rdm.state_dict()[k]) for k, v in learned.items()), "Fusion changed RDM state")
        row["parameters_fused"] = dict(parent=sum(p.numel() for p in parent.parameters()), target=sum(p.numel() for p in target.parameters()))
        row["fusion_preserved_rdm"] = True
        row["status"] = "PASSED"
    report["variants_equal_independent_rdm_storage"] = True


def local_checks(source, destination):
    started = time.monotonic()
    report = dict(status="RUNNING", phase="environment", core={"status": "NOT_RUN"}, models={},
                  server={"status": "PENDING", "reason": "Separate RTX4090 / torch2.1.2+cu121 B16/640 real-data lifecycle required"},
                  formal_training="NOT_STARTED", final_test="NOT_RUN", temporary_checkpoints_created=0)
    def progress(phase):
        report["phase"] = phase
        write_json(destination, report)
    try:
        progress("environment")
        report["environment"] = runtime()
        report["code"] = code_identity()
        progress("core")
        report["core"] = core_checks()
        if not source.is_file():
            report["models"] = dict(status="PENDING", reason="Specified public source missing")
            report["status"] = "PENDING"
        else:
            model_checks(source, report["models"], progress)
            report["status"] = "PASSED"
    except BaseException as error:
        report.update(status="FAILED", error=repr(error), traceback=traceback.format_exc(limit=8))
        raise
    finally:
        report.update(seconds=time.monotonic()-started, temporary_files_cleaned=True)
        write_json(destination, report)
    return report
