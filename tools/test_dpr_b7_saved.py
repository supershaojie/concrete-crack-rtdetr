"""Fault tests for saved-state diagnostic plumbing; no detector forwards/updates."""
from copy import deepcopy
import ast
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
import torch
import compare_dpr_b7 as task
import dpr_diagnostics as observation

task.torch = torch


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1., 2.]))
        self.register_buffer("counter", torch.tensor(7))
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        return image


def fixture(model):
    state = {n:v.clone() for n,v in model.state_dict().items()}
    return dict(actual_post=dict(model=state, buffers={n:v.clone() for n,v in model.named_buffers()}),
        model_modes={n:True for n,_ in model.named_modules()},
        parameter_requires_grad={n:p.requires_grad for n,p in model.named_parameters()},
        pre={"model":{"wrong":torch.tensor([999.])}}, ema={"wrong":torch.tensor([998.])})


def inventory(model):
    return dict(parameters=list(dict(model.named_parameters())), buffers=list(dict(model.named_buffers())))


def trace(value=0.):
    values = {k:torch.tensor([value]) for k in task.STAGES}
    values["candidate_scores"] = torch.full((2,525,1), value)
    values["output"] = torch.full((2,300,5), value)
    values["candidate_indices"] = torch.arange(300).repeat(2,1)
    return values


class SavedStateTests(unittest.TestCase):
    def test_select_actual_only_and_preserve_bytes(self):
        model = Tiny()
        data = fixture(model)
        data["actual_post"]["model"]["weight"].fill_(3.)
        result = task.restore_post(data, model, inventory(model))
        self.assertEqual(result["selected"], "actual_post.model")
        self.assertTrue(torch.equal(model.weight, torch.tensor([3.,3.])))
        self.assertEqual(model.calls, 0)
        del data["actual_post"]
        with self.assertRaisesRegex(RuntimeError, "fallback forbidden"):
            task.restore_post(data, model, inventory(model))

    def test_dtype_buffer_names_and_modes_fail_closed(self):
        for fault in ("dtype", "buffer", "missing", "mode", "grad"):
            model = Tiny(); data = fixture(model)
            if fault == "dtype": data["actual_post"]["model"]["weight"] = model.weight.half()
            if fault == "buffer": data["actual_post"]["buffers"]["counter"] += 1
            if fault == "missing": del data["actual_post"]["model"]["counter"]
            if fault == "mode": data["model_modes"][""] = False
            if fault == "grad": data["parameter_requires_grad"] = {}
            with self.assertRaises(RuntimeError, msg=fault):
                task.restore_post(data, model, inventory(model))

    def test_both_native_constructor_inventories_without_forward(self):
        from init_dpr import build
        for parent in (True, False):
            model = build("cbr_lif_dpr_v1", nc=1, baseline=parent)
            model.nc = 1
            model.criterion = model.init_criterion()
            self.assertFalse(model.criterion.state_dict())
            self.assertEqual(model.model[-1].shapes, [])
            self.assertEqual(model.model[-1].anchors.numel(), 0)
            data = fixture(model)
            result = task.restore_post(data, model, inventory(model))
            self.assertTrue(result["exact_bytes"])
            # No model(...) invocation and no optimizer/backward in this test.
            del model, data

    def test_limits_deadline_and_eval_only(self):
        model = Tiny().eval()
        budget = task.Budget(time.monotonic()+60)
        for group in (m+"/"+r for m in task.MODES for r in task.ROLES):
            for _ in range(4): budget.invoke(group, model, None)
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                budget.invoke(group, model, None)
        self.assertEqual(sum(budget.counts.values()), 16)
        expired = task.Budget(time.monotonic()-1)
        with self.assertRaisesRegex(RuntimeError, "expired"):
            expired.invoke("cuda_fp32/parent", model, None)
        with self.assertRaisesRegex(RuntimeError, "eval"):
            task.Budget(time.monotonic()+60).invoke("cuda_fp32/parent", Tiny(), None)

    def test_original_direction_metrics_coverage_indices(self):
        left, right = trace(1.), trace(1.0001)
        result = task.raw_rows(left, right)
        for k in task.STAGES:
            self.assertEqual(result["comparisons"][k], observation.metric(left[k], right[k], 2e-5, 2e-4))
        right["target"] += 1.
        result = task.raw_rows(left, right)
        self.assertEqual(result["first_exceeded"], "target")
        right["candidate_indices"].zero_()
        with self.assertRaisesRegex(RuntimeError, "top-k"):
            task.raw_rows(left, right)
        del right["P5"]
        with self.assertRaisesRegex(RuntimeError, "coverage"):
            task.raw_rows(left, right)

    def test_three_or_four_calls_and_original_fixed_replay(self):
        for fail in (False, True):
            model = Tiny().eval(); group = "cuda_fp32/parent"
            budget = task.Budget(time.monotonic()+60)
            for _ in range(3): budget.invoke(group, model, None)
            a,b = trace(),trace()
            if fail:
                b["output"] += 1
                b["target"] += 1  # Must survive successful fixed-query replay.
                b["candidate_indices"] = b["candidate_indices"].flip(1)
            def capture(m, image, invoke=None, fixed_indices=None, detailed=True):
                self.assertTrue(torch.equal(fixed_indices, a["candidate_indices"]))
                invoke(image)
                return a, {"fixed":True}
            with patch.object(observation, "capture_trace", side_effect=capture) as hook, \
                 patch.object(task, "observation_identity", return_value={}):
                result = task.compare_pair(a,b,model,None,budget,group)
            self.assertEqual(budget.counts[group], 4 if fail else 3)
            self.assertEqual(hook.call_count, int(fail))
            if fail:
                self.assertFalse(result["comparisons"]["target"]["raw_allclose"])
                self.assertTrue(result["fixed_candidate_replay_output"]["raw_allclose"])
                self.assertFalse(result["candidate_sensitivity_complete"])
                self.assertEqual(result["selection"]["changed_positions"], 600)

    def test_observation_failure_cleanup_is_checked(self):
        model = Tiny().eval()
        with patch.object(task, "observation_identity", side_effect=[{}, {"leftover":True}]), \
             patch.object(observation, "capture_trace", side_effect=RuntimeError("intentional")):
            with self.assertRaisesRegex(RuntimeError, "left state"):
                task.capture_checked(model,None,task.Budget(time.monotonic()+60),"cuda_fp32/parent")

    def test_flags_restore_on_exception(self):
        before = task.flags()
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with task.original_flags(before):
                self.assertFalse(task.flags()["matmul_tf32"])
                self.assertFalse(task.flags()["cudnn_tf32"])
                raise RuntimeError("intentional")
        self.assertEqual(task.flags(), before)

    def test_record_mismatch_has_priority(self):
        old = {"comparisons":{"target":{"max_abs":1.}}}
        new = deepcopy(old)
        self.assertTrue(task.original_b7_consistency(new,old)["exact_record_match"])
        new["comparisons"]["target"]["max_abs"] = 2.
        consistency = task.original_b7_consistency(new,old)
        self.assertFalse(consistency["exact_record_match"])
        self.assertIn("MISMATCH_FIRST",task.interpretation({"cuda_fp32/candidate":{"original_B7_consistency":consistency}}))
        partial = {m+"/"+r:{"status":"IN_PROGRESS"} for m in task.MODES for r in task.ROLES}
        self.assertIn("INCOMPLETE", task.interpretation(partial))
        self.assertEqual(task.parallel_table(partial)["cuda_fp32"]["target"]["parent"], {"measurement":"NOT_COLLECTED"})

    def test_file_identity_rejects_changed_or_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"tensor.pt"; path.write_bytes(b"original")
            record = dict(path=str(path), bytes=8, sha256=task.sha(path))
            task.verify_file(record)
            path.write_bytes(b"changed!")
            with self.assertRaisesRegex(RuntimeError,"differs"): task.verify_file(record)
            path.unlink()
            with self.assertRaisesRegex(RuntimeError,"Missing"): task.verify_file(record)

    def test_no_training_or_admission_calls_and_fixed_limits(self):
        tree = ast.parse((ROOT/"tools/compare_dpr_b7.py").read_text(encoding="utf-8"))
        forbidden = {"backward","optimizer_step","save_model","train","val","test","resume","bounded_updates","one_step"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                self.assertNotIn(name,forbidden)
        self.assertEqual((task.LIMIT,task.ATOL,task.RTOL),(300,2e-5,2e-4))

    def test_nonfinite_failure_can_be_written_without_hiding_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"diagnostic.json"
            task.write_json(path,{"finite":False,"raw_allclose":False,"max_abs":float("nan")})
            value=json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(value["finite"])
            self.assertFalse(value["raw_allclose"])
            self.assertEqual(value["max_abs"],{"nonfinite_float":"nan"})


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
