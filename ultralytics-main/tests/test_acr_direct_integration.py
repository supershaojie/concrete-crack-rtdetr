"""Optional LOCAL one-batch/API check. Never called by start-direct or by server preparation.

Set ACR_DIRECT_TEST_SOURCE and ACR_DIRECT_TEST_DATA to existing source weights/real data YAML.
ACR_DIRECT_TEST_REPORT optionally writes an exclusive JSON report outside tracked files.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from init_acr import initialize, sha256, branch_modules
from audit_acr import trainer_build, state_hashes, real_optimizer, compare_output
from train_acr import ACRDirectTrainer, install_preflight, build_locked_args, actual_args_check, DEFAULT_NAME
from ultralytics import RTDETR
from ultralytics.cfg import get_cfg
from ultralytics.models.rtdetr.val import RTDETRDataset
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import ModelEMA, init_seeds


def gradient_difference(left, right):
    """Report CUDA grid_sample backward variability; never loosen forward/loss tolerances."""
    rows = [{"key": key, "max_abs": float((left[key] - right[key]).abs().max()),
             "equal": torch.equal(left[key], right[key]),
             "within_tolerance": torch.allclose(left[key], right[key], atol=1e-6, rtol=1e-5)} for key in left]
    return {"max_abs": max(r["max_abs"] for r in rows), "bitwise_equal": all(r["equal"] for r in rows),
            "outside_tolerance_tensors": sum(not r["within_tolerance"] for r in rows),
            "largest": sorted(rows, key=lambda r: r["max_abs"], reverse=True)[:3]}


@unittest.skipUnless(os.environ.get("ACR_DIRECT_TEST_SOURCE") and os.environ.get("ACR_DIRECT_TEST_DATA"),
                     "Optional local source/data integration check; not part of start-direct")
class DirectIntegrationTests(unittest.TestCase):
    def test_clean_source_real_train_api_and_one_real_batch(self):
        torch.set_num_threads(4)
        source = Path(os.environ["ACR_DIRECT_TEST_SOURCE"]).resolve()
        data_path = Path(os.environ["ACR_DIRECT_TEST_DATA"]).resolve()
        report = {"scope": "Local one-batch comparison, not full server preflight", "formal_epochs_run": 0,
                  "real_batch_size": 2, "local_imgsz": 320,
                  "native_server_amp_check": "not_run", "training_setup_callbacks": "dataset-free fixture",
                  "torch": str(torch.__version__), "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                  "source_sha256": sha256(source), "data_sha256": sha256(data_path)}
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            root = Path(tmp)
            initialized = root / "clean.pt"
            init = initialize(source, initialized)
            report["initialization"] = {key: init[key] for key in
                                        ("status", "new_parameters", "constructor_common_states_and_rng_equal", "reloads")}
            report["initialization"]["exact_source_states"] = len(init["c2_source_mapping"])
            weights = RTDETR(str(initialized)).model
            native, expected_rng = trainer_build(weights.yaml, weights, 1)
            expected_states = state_hashes(native)
            recipe = YAML.load(ROOT / "ultralytics-main/tests/fixtures/c2_original_args.yaml")
            locked, rows = build_locked_args(recipe, initialized, DEFAULT_NAME)
            report["recipe_changes"] = [r["field"] for r in rows if not r["equal"]]

            class StopBeforeEpochs(Exception):
                pass

            class DatasetFreeDirect(ACRDirectTrainer):
                def __init__(self, overrides, _callbacks):
                    self.args = SimpleNamespace(**{k: v for k, v in overrides.items() if k != "session"})
                    self.data = {"nc": 1, "channels": 3}
                    self.callbacks = _callbacks
                    init_seeds(42, deterministic=True)

                def train(self):
                    actual_args_check(vars(self.args), locked)
                    assert state_hashes(self.model) == expected_states
                    assert torch.equal(torch.get_rng_state(), expected_rng)
                    self.start_epoch, self.amp = 0, True  # Fixture; native AMP is NOT claimed as tested here.
                    self.optimizer = real_optimizer(self.model)
                    self.ema = ModelEMA(self.model)
                    for event in ("on_pretrain_routine_start", "on_train_start"):
                        for callback in self.callbacks[event]:
                            callback(self)
                    raise StopBeforeEpochs()

            wrapper = RTDETR(str(initialized))
            install_preflight(wrapper, {"launch_mode": "direct", "stats_interval": 0, "target_args": locked,
                                       "runtime": {"git_commit": "local-fixture"}}, root)
            with patch("ultralytics.engine.model.checks.check_pip_update_available"):
                with self.assertRaises(StopBeforeEpochs):
                    wrapper.train(trainer=DatasetFreeDirect, **locked)
            direct = wrapper.model.eval()
            setup = json.loads((root / "training_setup.json").read_text())
            report["actual_train_api"] = {key: setup[key] for key in
                                          ("all_states_exact", "optimizer_parameters_exact", "optimizer_empty",
                                           "ema_updates", "ema_states_exact", "args_fields", "full_preflight")}
            report["actual_train_api"]["rng_equal"] = True
            data = YAML.load(data_path)
            data.update(nc=1, channels=3)
            dataset = RTDETRDataset(img_path=str(Path(data["path"]) / data["train"]), imgsz=320, batch_size=2,
                                    augment=True, hyp=get_cfg(overrides=locked), rect=False, cache=False, data=data)
            batch = dataset.collate_fn([dataset[0], dataset[1]])
            batch["img"] = batch["img"].float() / 255.0
            self.assertGreater(len(batch["bboxes"]), 0)
            report["real_gt_count"] = len(batch["bboxes"])
            report["comparisons"] = []
            for device, amp in [("cpu", False)] + ([("cuda", False), ("cuda", True)] if torch.cuda.is_available() else []):
                observed = []
                # CUDA control repeats the unchanged native path on this same batch to expose
                # the existing nondeterministic grid_sample backward independently of direct launch.
                for origin in ((native, direct, native) if device == "cuda" else (native, direct)):
                    model = deepcopy(origin).to(device).train()
                    model.nc = 1  # Normally attached by the native trainer's set_model_attributes().
                    local = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    targets = {"cls": local["cls"].long().flatten(), "bboxes": local["bboxes"],
                               "batch_idx": local["batch_idx"].long().flatten(),
                               "gt_groups": [(local["batch_idx"] == i).sum().item() for i in range(2)]}
                    if getattr(model, "criterion", None) is None:
                        model.criterion = model.init_criterion()
                    for module in branch_modules(model).values():
                        self.assertEqual(module.acr_stats_interval, 0)
                    torch.manual_seed(987)
                    with torch.autocast(device_type=device, enabled=amp):
                        predictions = model.predict(local["img"], batch=targets)
                        loss, items = model.loss(local, predictions)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                    self.assertTrue(all(m.acr_last_stats is None for m in branch_modules(model).values()))
                    self.assertEqual(predictions[4]["dn_num_split"][1], 300)
                    self.assertGreater(predictions[4]["dn_num_split"][0], 0)
                    optimizer = real_optimizer(model)
                    optimizer.step()  # One isolated local update per comparison model, never the clean checkpoint.
                    observed.append((tuple(p.detach().cpu() for p in predictions[:4]), loss.detach().cpu(),
                                     items.detach().cpu(), predictions[4]["dn_num_split"],
                                     {n: p.grad.detach().cpu() for n, p in model.named_parameters() if p.grad is not None}))
                    del model, predictions, loss, optimizer
                checks = compare_output(observed[0][:3], observed[1][:3])
                self.assertEqual(observed[0][3], observed[1][3])
                gradients = gradient_difference(observed[0][4], observed[1][4])
                if device == "cpu":
                    self.assertEqual(gradients["outside_tolerance_tensors"], 0)
                control = None
                if device == "cuda":
                    compare_output(observed[0][:3], observed[2][:3], "native_repeat_outputs")
                    control = gradient_difference(observed[0][4], observed[2][4])
                report["comparisons"].append({"device": device, "amp": amp, "dn_num_split": observed[0][3],
                    "loss": float(observed[0][1]), "outputs_max_abs": max(r["max_abs"] for r in checks),
                    "outputs_bitwise_equal": all(r["equal"] for r in checks),
                    "gradients_finite": True, "gradient_difference": gradients, "native_repeat_gradient_control": control,
                    "gradient_note": "CUDA grid_sample backward warns it is nondeterministic; exact whole-model gradient equality is not claimed.",
                    "atol": 1e-6, "rtol": 1e-5, "status": "passed"})
            self.assertEqual(sha256(initialized), init["output_sha256"])
            self.assertEqual(sha256(source), report["source_sha256"])
            report.update(status="passed", clean_initialization_unchanged=True, source_unchanged=True)
        if os.environ.get("ACR_DIRECT_TEST_REPORT"):
            with Path(os.environ["ACR_DIRECT_TEST_REPORT"]).open("x", encoding="utf-8") as file:
                json.dump(report, file, indent=2)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    unittest.main()
