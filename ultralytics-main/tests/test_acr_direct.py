"""Focused direct-launch regressions. Never execute a formal training epoch or full prepare."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import train_acr
from init_acr import ACR_CFG, branch_modules
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules.acr import CoverageRelationSelfAttention
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML


class DirectTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(4)

    def test_direct_plan_preserves_failed_prepare_and_rechecks_own_evidence(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            root = Path(tmp)
            old = root / "outputs/c22/launch"
            old.mkdir(parents=True)
            (old / "prepare.state").write_text("failed\n")
            (old / "audit.json").write_text('{"incomplete":')
            (old / "initialization.json").write_text("old failed report")
            before = {p.name: p.read_bytes() for p in old.iterdir()}
            data = root / "data.yaml"
            data.write_text("test fixture")
            recipe = ROOT / "ultralytics-main/tests/fixtures/c2_original_args.yaml"
            args = SimpleNamespace(c2_args=recipe, source=root / "source.pt", stats_interval=0,
                                   report_dir=old.with_name("launch_direct"), name=train_acr.DEFAULT_NAME)
            initialized = args.report_dir / "c22_acr_controlled_init.pt"
            target, rows = train_acr.build_locked_args(YAML.load(recipe), initialized, args.name)
            # Fixture paths only: production build_locked_args still verifies every original C2 value.
            target.update(data=str(data), project=str(root / "runs"), save_dir=str(root / "runs/new"))
            runtime = dict(git_status=[], git_commit="fixture", code_sha256={}, python="fixture",
                           torch_version="fixture", torch_cuda=None, gpu=None, ultralytics_file="fixture")

            def initialize(source, output):
                self.assertEqual(source, args.source)
                self.assertEqual(output, initialized)
                output.write_bytes(b"fresh fixture checkpoint; never a smoke state")
                return {"status": "passed", "source_sha256": train_acr.SOURCE_SHA256}

            with patch.object(train_acr, "ROOT", root), patch.object(train_acr, "runtime_info", return_value=runtime), \
                    patch.object(train_acr, "verify_protected"), \
                    patch.object(train_acr, "build_locked_args", return_value=(target, rows)), \
                    patch("init_acr.initialize", side_effect=initialize) as init, \
                    patch("acr_resources.ensure_resources", return_value={}), \
                    patch.object(train_acr, "prepare", side_effect=AssertionError("full prepare called")):
                plan, _ = train_acr.prepare_direct(args)
                train_acr.recheck(plan)
                init.assert_called_once()
                self.assertNotIn("audit_report", plan)
                self.assertEqual(plan["full_preflight"], "not_run")
                self.assertEqual(plan["stats_interval"], 0)
                self.assertEqual({p.name: p.read_bytes() for p in old.iterdir()}, before)
                self.assertFalse((args.report_dir / "prepare.state").exists())
                token = train_acr.claim_launch(plan)
                train_acr.verify_claim(plan, token)
                with self.assertRaises(FileExistsError):
                    train_acr.claim_launch(plan)
                state = train_acr.launch_status(args.report_dir, args.name)
                self.assertEqual(state["launch_mode"], "direct")
                self.assertEqual(state["full_preflight"], "not_run")
                self.assertTrue(state["lock_exists"])
                with self.assertRaises(RuntimeError):
                    train_acr.prepare_direct(args)
                initialized.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "initialized changed"):
                    train_acr.recheck(plan)

    def test_direct_cli_bypasses_marker_but_regular_start_keeps_it(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            report = Path(tmp)
            (report / "prepare.state").write_text("failed")
            argv = ["train_acr.py", "--execute", "--report-dir", tmp]
            with patch.object(sys, "argv", argv + ["--direct"]), \
                    patch.object(train_acr, "prepare_direct", return_value=({}, report / "launch_plan.json")) as direct, \
                    patch.object(train_acr, "prepare", side_effect=AssertionError("full prepare called")), \
                    patch.object(train_acr, "claim_launch", return_value="token"), \
                    patch.object(train_acr, "execute") as execute:
                train_acr.main()
                direct.assert_called_once()
                execute.assert_called_once()
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(RuntimeError, "Latest prepare"):
                train_acr.main()
            self.assertEqual((report / "prepare.state").read_text(), "failed")

    def test_default_callbacks_disable_inherited_statistics(self):
        for mode, interval in (("direct", 0), ("prepared", 0), ("prepared", 200)):
            with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
                callbacks = {}
                wrapper = SimpleNamespace(add_callback=lambda event, fn: callbacks.setdefault(event, []).append(fn))
                model = torch.nn.Sequential(CoverageRelationSelfAttention(32, 4))
                for module in branch_modules(model).values():
                    module.acr_stats_interval = 1
                    module.acr_last_stats = {"stale": True}
                trainer = SimpleNamespace(model=model, ema=SimpleNamespace(ema=deepcopy(model)))
                train_acr.install_preflight(wrapper, {"launch_mode": mode, "stats_interval": interval}, Path(tmp))
                callbacks["on_train_start"][0](trainer)
                for network in (trainer.model, trainer.ema.ema):
                    for module in branch_modules(network).values():
                        self.assertEqual(module.acr_stats_interval, interval)
                        self.assertIsNone(module.acr_last_stats)
                self.assertEqual(bool(callbacks.get("on_train_batch_end")), interval > 0)
                self.assertEqual((Path(tmp) / "allocation.jsonl").stat().st_size, 0)

    def test_stats_switch_preserves_nonzero_attention_loss_gradients_and_rng(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for amp in ([False, True] if device == "cuda" else [False]):
                base = CoverageRelationSelfAttention(32, 4, dropout=.2).to(device).train()
                torch.nn.init.normal_(base.acr_head.weight, std=.2)
                q = torch.randn(9, 2, 32, device=device)
                boxes = torch.rand(2, 9, 4, device=device)
                mask = torch.zeros(9, 9, dtype=torch.bool, device=device)
                mask[3:, :3] = True
                outputs = []
                for enabled in (False, True):
                    model = deepcopy(base)
                    model.acr_stats_interval = int(enabled)
                    query = q.clone().requires_grad_()
                    torch.manual_seed(91)
                    with patch.object(model, "_record_stats", wraps=model._record_stats) as stats:
                        with torch.autocast(device_type=device, enabled=amp):
                            out, attention = model(query, query, query, attn_mask=mask, refer_bbox=boxes,
                                                   num_queries=6, dn_meta={"dn_num_split": [3, 6]})
                            loss = out.float().square().mean()
                        loss.backward()
                        self.assertEqual(stats.call_count, int(enabled))
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue((attention[:, 3:, :3] == 0).all())
                    gradients = [query.grad] + [p.grad for p in model.parameters()]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.0001)
                    optimizer.step()
                    outputs.append([out, attention, loss, *gradients, *model.state_dict().values(),
                                    torch.get_rng_state(), *torch.cuda.get_rng_state_all()])
                for off, on in zip(*outputs):
                    self.assertTrue(torch.equal(off, on), f"Stats changed computation: {device=}, {amp=}")

    def test_actual_nc1_constructor_mapping_and_rng_match_native(self):
        from audit_acr import state_hashes
        weights = RTDETRDetectionModel(str(ACR_CFG), nc=80, verbose=False).eval()
        observed = []
        for cls in (RTDETRTrainer, train_acr.ACRDirectTrainer):
            trainer = cls.__new__(cls)
            trainer.data = {"nc": 1, "channels": 3}
            torch.manual_seed(42)
            model = trainer.get_model(cfg=deepcopy(weights.yaml), weights=weights, verbose=False)
            observed.append((state_hashes(model), torch.get_rng_state(), torch.cuda.get_rng_state_all()))
        self.assertEqual(observed[0][0], observed[1][0])
        self.assertTrue(torch.equal(observed[0][1], observed[1][1]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(observed[0][2], observed[1][2])))
        rows = trainer.acr_direct_mapping["states"]
        self.assertEqual(len(rows), 545)
        self.assertEqual(sum(r["origin"] == "native_fresh_nc1" for r in rows), 9)
        self.assertEqual(sum(r["origin"] == "clean_controlled_checkpoint" for r in rows), 536)


if __name__ == "__main__":
    unittest.main()
