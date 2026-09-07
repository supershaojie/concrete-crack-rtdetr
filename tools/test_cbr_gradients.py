"""Focused gradient-probe tests with synthetic models/data, never experimental results."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from cbr_rescue_common import ROOT, write_json, package, sha256
TEST_DIR = ROOT / "outputs/cbr_gradient_local_tests"
(TEST_DIR / "config").mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(TEST_DIR / "config")
os.environ["YOLO_AUTOINSTALL"] = "false"
sys.path.insert(0, str(ROOT / "ultralytics-main"))
import torch
from PIL import Image
from ultralytics.nn.modules.cbr import CrackBoundaryRefinement
from ultralytics.nn.tasks import RTDETRDetectionModel, load_checkpoint
from cbr_gradient_probe import (objects, mixed_graph_mode, capture_forward, loss_observer, state_record,
    probe_batch, gradient_read, stats_for_group, decide, DECISION_RULE)
from diagnose_cbr_gradients import choose_images, fixed_data, verify_fixed, check_model
from diagnose_cbr_rescue import SETTINGS

torch.set_num_threads(4)


def model_fixture(c20=True, device="cpu"):
    config = "rtdetr-resnet18-lite-cscef-cbr.yaml" if c20 else "rtdetr-resnet18-lite-cbr.yaml"
    torch.manual_seed(42)
    m = RTDETRDetectionModel(str(ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr" / config), nc=1, verbose=False)
    m.float().eval().to(device)
    m.nc = m.model[-1].nc  # native trainer normally supplies this metadata
    m.criterion = m.init_criterion()
    m.eval()
    head, roles, _ = objects(m)
    with torch.no_grad():
        head.cbr.offset_out.weight.normal_(0, .05)
        head.cbr.offset_out.bias.fill_(.1)
        if c20:
            roles["cscef_projection"].weight.normal_(0, .01)
    for p in m.parameters():
        p.requires_grad_(False)  # simulate stripped best.pt, must re-enable in memory
    return m


def batch_fixture(device="cpu", size=128, batch=2):
    return {"img": torch.rand(batch, 3, size, size, device=device),
            "batch_idx": torch.tensor([0., 0.], device=device),
            "cls": torch.tensor([[0.], [0.]], device=device),
            "bboxes": torch.tensor([[.5, .5, .2, .15], [.3, .2, .08, .3]], device=device),
            "im_file": [f"synthetic_{i}" for i in range(batch)]}


class GradientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=TEST_DIR)
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()

    def test_local_detach_only_conditions_keeps_cbr_and_boxes(self):
        torch.manual_seed(42)
        cbr = CrackBoundaryRefinement(16, 16).float()
        with torch.no_grad():
            cbr.offset_out.weight.normal_(0, .1)
        p3 = torch.randn(1, 16, 8, 8, requires_grad=True)
        query = torch.randn(1, 7, 16, requires_grad=True)
        boxes = torch.tensor([.5, .5, .2, .15]).repeat(1, 7, 1).requires_grad_()
        a = cbr(p3, query, boxes)
        b = cbr(p3.detach(), query.detach(), boxes)
        self.assertTrue(torch.equal(a, b))
        inputs = [p3, query, boxes, *cbr.parameters()]
        ga = gradient_read(a.square().sum(), inputs, True)
        gb = gradient_read(b.square().sum(), inputs, False)
        self.assertGreater(float(ga[0].norm()), 0)
        self.assertGreater(float(ga[1].norm()), 0)
        self.assertIsNone(gb[0]); self.assertIsNone(gb[1])
        self.assertGreater(float(gb[2].norm()), 0)
        self.assertTrue(all(p.grad is None for p in cbr.parameters()))
        for old, new in zip(ga[2:], gb[2:]):
            torch.testing.assert_close(old, new, atol=1e-7, rtol=1e-5)
        self.assertGreater(sum(float(g.norm()) for g in gb[3:] if g is not None), 0)

    def test_native_same_graph_DN_loss_gradients_and_state(self):
        prior_dn = None
        for c20 in (False, True):
            m = model_fixture(c20)
            before = state_record(m)
            with patch.object(torch.optim.Optimizer, "__init__", side_effect=AssertionError("optimizer forbidden")), \
                 patch.object(torch.Tensor, "backward", side_effect=AssertionError("use autograd.grad only")):
                with mixed_graph_mode(m):
                    with patch.object(m, "predict", wraps=m.predict) as predict:
                        result = probe_batch(m, batch_fixture(), 0)
                        self.assertEqual(predict.call_count, 1)
                    self.assertTrue(result["outputs_equal_bitwise"])
                    self.assertEqual(result["loss_A"], result["loss_B"])
                    self.assertEqual(result["DN"]["queries_per_image"], 200)
                    self.assertEqual(len(result["loss_A"]["hungarian"]), 4)
                    self.assertEqual(len(result["loss_A"]["classification_targets"]), 7)
                    self.assertEqual(len(result["loss_A"]["terms"]), 12)
                    self.assertTrue(all(result["nonzero_B_gradients"].values()))
                    self.assertTrue(result["box_path_jacobian"]["passed"])
                    self.assertEqual(result["repeated_A_gradient_reads"], 1)
                    self.assertEqual(result["gradient_groups"]["cscef_all"]["applicable"], c20)
                    self.assertGreater(result["gradient_groups"]["decoder_last"]["norm_condition"], 0)
                    self.assertEqual(state_record(m), before)
                    if prior_dn:
                        self.assertEqual(prior_dn, result["DN"]["initial_noisy_boxes"])
                    prior_dn = result["DN"]["initial_noisy_boxes"]
                    baseline = {k: v["repeat_A_noise_norm"] for k, v in result["gradient_groups"].items() if v["applicable"]}
                    later = probe_batch(m, batch_fixture(), 1, baseline)
                    self.assertEqual(later["repeated_A_gradient_reads"], 0)
                    self.assertEqual(later["gradient_groups"]["neck_p3"]["repeat_A_noise_norm"], baseline["neck_p3"])
            self.assertEqual(state_record(m), before)
            self.assertTrue(all(not p.requires_grad for p in m.parameters()))
            self.assertTrue(all(not mod.training for mod in m.modules()))

    def test_empty_GT_is_explicit_and_not_frozen_claim(self):
        m = model_fixture(False)
        batch = batch_fixture()
        batch.update(batch_idx=torch.zeros(0), cls=torch.zeros(0, 1), bboxes=torch.zeros(0, 4))
        with mixed_graph_mode(m):
            result = probe_batch(m, batch, 0)
        self.assertEqual(result["DN"]["queries_per_image"], 0)
        self.assertFalse(result["nonzero_B_gradients"]["cbr"])
        self.assertFalse(result["box_path_jacobian"]["applicable"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_repeat_noise_and_single_640_graph(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        m = model_fixture(True, "cuda")
        before = state_record(m)
        with mixed_graph_mode(m):
            result = probe_batch(m, batch_fixture("cuda", size=640, batch=1), 0)
        self.assertEqual(state_record(m), before)
        self.assertTrue(all(result["nonzero_B_gradients"].values()))
        self.assertTrue(result["loss_and_matching_equal"])
        write_json(TEST_DIR / "synthetic_cuda_640.json", result)
        print("Synthetic CUDA 640 gradient fixture passed; not server batch4 or trained weights.")
        del m
        torch.cuda.empty_cache()

    def test_hooks_and_modes_restored_after_exception(self):
        m = model_fixture()
        head, roles, _ = objects(m)
        hooks = {id(x): (len(x._forward_hooks), len(x._forward_pre_hooks)) for x in m.modules()}
        original = {name: getattr(m.criterion, name) for name in ("_get_loss", "_get_loss_class", "get_dn_match_indices")}
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with mixed_graph_mode(m), capture_forward(head, roles, {}), loss_observer(m.criterion, {}):
                raise RuntimeError("injected")
        for module in m.modules():
            self.assertEqual(hooks[id(module)], (len(module._forward_hooks), len(module._forward_pre_hooks)))
            self.assertFalse(module.training)
        for name, method in original.items():
            self.assertEqual(method, getattr(m.criterion, name))
        self.assertTrue(all(not p.requires_grad for p in m.parameters()))

    def test_forward_inequality_rejected_before_gradient_read(self):
        m = model_fixture(False)
        head, _, _ = objects(m)
        original, calls = head.cbr.forward, []
        def corrupt(*args, **kwargs):
            output, details = original(*args, **kwargs)
            calls.append(1)
            if len(calls) == 2:
                output = output + .001
                details = {**details, "after": output}
            return output, details
        with mixed_graph_mode(m), patch.object(head.cbr, "forward", side_effect=corrupt), \
             patch("cbr_gradient_probe.gradient_read", side_effect=AssertionError("must not read inequivalent gradients")):
            with self.assertRaisesRegex(RuntimeError, "diagnostics differ"):
                probe_batch(m, batch_fixture(), 0)

    def test_noise_floor_and_no_one_negative_cosine_decision(self):
        tensors = [torch.zeros(2)]
        tiny = stats_for_group([0], tensors, [torch.tensor([1., 1.])], [torch.tensor([1., 1.])])
        self.assertIsNone(tiny["cosine_condition_keep"])
        def statistics(negative=.2):
            return {"cosine_meaningful": True, "condition_over_keep": .4,
                    "negative_projection_over_keep": negative, "norm_keep": 1., "cosine_noise_floor": 1e-6}
        def report(negative):
            return {"gradient_groups": {role: statistics(negative) for role in DECISION_RULE["primary_roles"]},
                    "nonzero_B_gradients": {role: True for role in ("cbr", "box_head_last", "input_boxes")}}
        reports = {name: [report(.2) for _ in range(4)] for name in ("C19", "C20")}
        self.assertEqual(decide(reports)["category"], "没有明确依据")  # same in successful C19
        reports["C20"][0] = report(.5)
        self.assertEqual(decide(reports)["category"], "没有明确依据")
        reports["C20"] = [report(.5) for _ in range(4)]
        self.assertEqual(decide(reports)["category"], "有尝试依据")

    def test_selection_native_loader_hashes_and_four_batch_runner(self):
        (self.root / "images/train").mkdir(parents=True)
        (self.root / "labels/train").mkdir(parents=True)
        for i in range(24):
            Image.new("RGB", (50, 60), (i*9, 20, 50)).save(self.root / f"images/train/{i:02}.png")
            (self.root / f"labels/train/{i:02}.txt").write_text("0 .5 .5 .2 .15\n", encoding="utf-8")
        data = {"names": {0: "crack"}, "nc": 1, "channels": 3, "train": str(self.root / "images/train")}
        settings = {**SETTINGS, "imgsz": 128, "batch": 4, "device": "cpu", "split": "train"}
        out = self.root / "output"
        out.mkdir()
        dataset, fixed = fixed_data(data, out, settings)
        self.assertEqual(len(fixed["rows"]), 16)
        ordered, selected = choose_images(list(reversed(dataset.im_files)) + [str(self.root / f"unused{i}.png") for i in range(8)])
        self.assertEqual(len({row["image"] for row in selected}), 16)
        self.assertEqual(choose_images(ordered), (ordered, selected))
        verify_fixed(fixed)
        for criterion_state in ("missing", "none", "existing"):
            with self.subTest(criterion_state=criterion_state):
                m = model_fixture(False)
                if criterion_state == "missing":
                    del m.criterion
                elif criterion_state == "none":
                    m.criterion = None  # strip_optimizer stores this in real best.pt files
                else:
                    # A non-default matcher cost makes accidental reinitialization observable.
                    # Only this synthetic fixture changes; production loss configuration is untouched.
                    m.criterion.matcher.cost_gain["class"] = 1.75
                checkpoint = self.root / f"synthetic_{criterion_state}.pt"
                torch.save({"model": m, "train_args": {}}, checkpoint)
                before = sha256(checkpoint)
                case_out = out / criterion_state
                case_out.mkdir()
                loaded = {}

                def observe_load(*args, **kwargs):
                    # Exercise the real checkpoint loader, rather than return a prepared model.
                    model, ckpt = load_checkpoint(*args, **kwargs)
                    criterion = getattr(model, "criterion", None)
                    self.assertEqual(hasattr(model, "criterion"), criterion_state != "missing")
                    self.assertEqual(criterion is None, criterion_state != "existing")
                    loaded.update(model=model, criterion=criterion, state=state_record(model))
                    if criterion is not None:
                        loaded.update(gain=criterion.loss_gain, matcher=criterion.matcher,
                                      cost=criterion.matcher.cost_gain)
                    return model, ckpt

                with patch("ultralytics.nn.tasks.load_checkpoint", side_effect=observe_load) as loader, \
                     patch.object(RTDETRDetectionModel, "init_criterion", autospec=True,
                                  side_effect=RTDETRDetectionModel.init_criterion) as initialize, \
                     patch.object(torch.optim.Optimizer, "__init__", side_effect=AssertionError("optimizer forbidden")), \
                     patch.object(torch.Tensor, "backward", side_effect=AssertionError("use autograd.grad only")):
                    result = check_model("C19", checkpoint, dataset, fixed, data, settings, case_out)
                loader.assert_called_once()
                if criterion_state == "existing":
                    initialize.assert_not_called()
                    self.assertIs(loaded["model"].criterion, loaded["criterion"])
                    self.assertIs(loaded["model"].criterion.loss_gain, loaded["gain"])
                    self.assertIs(loaded["model"].criterion.matcher, loaded["matcher"])
                    self.assertIs(loaded["model"].criterion.matcher.cost_gain, loaded["cost"])
                    self.assertEqual(result["loss_configuration"]["matcher_cost"]["class"], 1.75)
                else:
                    initialize.assert_called_once_with(loaded["model"])
                    self.assertIsNotNone(loaded["model"].criterion)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(len(result["batches"]), 4)
                self.assertEqual([b["images"] for b in result["batches"]],
                                 [[r["image"] for r in fixed["rows"][i:i+4]] for i in range(0, 16, 4)])
                for batch in result["batches"]:
                    self.assertTrue(batch["outputs_equal_bitwise"])
                    self.assertEqual(batch["loss_A"], batch["loss_B"])
                    self.assertTrue(batch["loss_and_matching_equal"])
                    self.assertGreater(batch["DN"]["queries_per_image"], 0)
                    self.assertTrue(all(batch["nonzero_B_gradients"].values()))
                    self.assertTrue(batch["box_path_jacobian"]["passed"])
                    self.assertTrue(batch["parameters_buffers_grad_unchanged"])
                self.assertEqual(sha256(checkpoint), before)
                self.assertEqual(json.loads((case_out/'C19/state_before.json').read_text()),
                                 json.loads((case_out/'C19/state_after.json').read_text()))
                self.assertEqual(state_record(loaded["model"]), loaded["state"])
                self.assertTrue(all(not p.requires_grad for p in loaded["model"].parameters()))
                self.assertTrue(all(not mod.training for name, mod in loaded["model"].named_modules()
                                    if name != "criterion" and not name.startswith("criterion.")))
                verify_fixed(fixed)
                del m, loaded
        self.assertFalse(list(self.root.rglob("*.cache")))
        # Exercise reused packer on actual gradient reports, excluding synthetic checkpoint/images.
        (out / "console.log").write_text("Synthetic gradient tool verification only.\n")
        artifacts = [p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()]
        write_json(out / "summary.json", {"status": "completed", "fixture_only": True, "artifacts": ["summary.json", *artifacts]})
        package(out, self.root / "gradient_fixture.tar.gz")


if __name__ == "__main__":
    unittest.main(verbosity=2)
