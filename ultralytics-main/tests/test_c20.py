"""Combination risks: graph, mapped gradients, locked recipe and decoder default API."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from init_rtdetr_r18_lite_c20 import (
    BASE_CFG, C20_CFG, DEFAULT_OUTPUT, topology, remap_key, common_rows, verify_module, verify_protected,
)
from audit_rtdetr_r18_lite_c20 import compare_output
from train_rtdetr_r18_lite_c20 import build_locked_args, DEFAULT_NAME, actual_args_check
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.nn.modules.transformer import DeformableTransformerDecoder
from ultralytics.utils import YAML


class C20Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)

    def test_graph_rejects_lateral_as_p3_and_wrong_concat(self):
        graph = topology()
        self.assertEqual(graph["decoder_inputs"], [20, 23, 26])
        for change in ("p3", "concat", "cscef"):
            target = YAML.load(C20_CFG)
            if change == "p3": target["head"][-1][0][0] = 18
            if change == "concat": target["head"][11][0].reverse()
            if change == "cscef": target["head"][10][0].reverse()
            with self.assertRaises(RuntimeError): topology(target=target)

    def test_full_zero_model_dn_common_gradients_and_default_api(self):
        torch.manual_seed(42)
        base = RTDETRDetectionModel(str(BASE_CFG), nc=1, verbose=False).train()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        target = RTDETRDetectionModel(str(C20_CFG), nc=1, verbose=False).train()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(r["equal"] for r in common_rows(base.state_dict(), target.state_dict())))
        verify_module(target)
        batch = {"cls": torch.zeros(2, dtype=torch.long), "bboxes": torch.tensor([[.5,.5,.7,.1],[.5,.5,.1,.7]]),
                 "batch_idx": torch.tensor([0, 1]), "gt_groups": [1, 1]}
        x = torch.rand(2, 3, 128, 128)
        torch.manual_seed(8); a = base.predict(x, batch=batch)
        torch.manual_seed(8); b = target.predict(x, batch=batch)
        compare_output(a, b)
        self.assertGreater(b[0].shape[2], 300)
        a[0].sum().backward(); b[0].sum().backward()
        parameters = dict(target.named_parameters())
        for key, p in base.named_parameters():
            q = parameters[remap_key(key)]
            if p.grad is None: self.assertIsNone(q.grad, key)
            else: self.assertTrue(torch.equal(p.grad, q.grad), key)
        signature = inspect.signature(DeformableTransformerDecoder.forward)
        self.assertIs(signature.parameters["return_final_query"].default, False)
        head = base.model[-1].eval()
        feats, shapes = head._get_encoder_input([torch.rand(1,256,s,s) for s in (16,8,4)])
        embed, refs, _, _ = head._get_decoder_input(feats, shapes)
        inputs = (embed, refs, feats, shapes, head.dec_bbox_head, head.dec_score_head, head.query_pos_head)
        with torch.no_grad():
            default = head.decoder(*inputs)
            explicit = head.decoder(*inputs, return_final_query=True)
        self.assertEqual(len(default), 2)
        self.assertEqual(len(explicit), 3)
        compare_output(default, explicit[:2])
        self.assertEqual(list(explicit[2].shape), [1,300,256])

    def test_recipe_types_and_module_protection(self):
        verify_protected()
        recipe = YAML.load(ROOT / "ultralytics-main/tests/fixtures/c2_original_args.yaml")
        target, rows = build_locked_args(recipe, DEFAULT_OUTPUT, DEFAULT_NAME)
        self.assertEqual(len(rows), 109)
        self.assertEqual({r["field"] for r in rows if not r["equal"]}, {"model", "name", "save_dir"})
        actual_args_check(target, deepcopy(target))
        for key, value in (("amp", 1), ("epochs", 150), ("project", "runs/new"), ("lr0", .01), ("mosaic", 0)):
            bad = deepcopy(recipe); bad[key] = value
            with self.assertRaises(RuntimeError): build_locked_args(bad, DEFAULT_OUTPUT, DEFAULT_NAME)

    def test_c20_supervisor_records_bootstrap_failure(self):
        from c20_tmux_worker import supervise
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(supervise(folder, [sys.executable, "-c", "raise SystemExit(7)"]), 7)
            self.assertEqual(json.loads((Path(folder) / "process_exit_code.json").read_text())["exit_code"], 7)

    def test_package_requires_both_splits_and_excludes_weights(self):
        import hashlib
        import tarfile
        from c20_results import package
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run, launch, val, test = [root / n for n in ("run", "launch", "val", "test")]
            for p in (run / "weights", launch, val, test): p.mkdir(parents=True)
            (run / "weights/best.pt").write_bytes(b"synthetic package fixture")
            digest = hashlib.sha256(b"synthetic package fixture").hexdigest()
            for n in ("args.yaml", "results.csv"): (run / n).write_text("synthetic fixture")
            for n in ("initialization.json", "audit.json", "launch_plan.json", "train_args.yaml", "actual_train_args.yaml",
                      "preflight.json", "tmux.json", "exit_code.json", "process_exit_code.json"):
                (launch / n).write_text("{}")
            for split, p in (("val", val), ("test", test)):
                (p / "metrics_summary.json").write_text(json.dumps({"status": "completed", "split": split, "checkpoint_sha256": digest}))
                (p / "key_predictions_gt.jsonl.gz").write_bytes(b"synthetic fixture")
            output = root / "small.tar.gz"
            package(run, launch, val, test, output)
            with tarfile.open(output) as tar:
                names = tar.getnames()
                self.assertFalse(any(n.endswith((".pt", ".jpg", ".png")) for n in names))
                self.assertIn("source/tools/autodl_c20.sh", names)
                self.assertIn("evaluation/test/metrics_summary.json", names)
            self.assertLess(output.stat().st_size, 20 * 1024 * 1024)
            with self.assertRaises(RuntimeError): package(run, launch, val, test, output)
            (test / "key_predictions_gt.jsonl.gz").unlink()
            with self.assertRaises(RuntimeError): package(run, launch, val, test, root / "missing.tar.gz")


if __name__ == "__main__":
    unittest.main()
