"""Offline lifecycle fault tests; no training, network, dataset or tmux required."""
from __future__ import annotations
import argparse
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"ultralytics-main"))
import torch
from ultralytics.utils import YAML
import gpc_v1 as ops
from gpc_evaluate import corrected_predictions,EVAL,PROTOCOL


class OperationalTests(unittest.TestCase):
    def test_corrected_sort_mask_query_index_and_no_mutation(self):
        raw=torch.tensor([[[.1,.2,.05,.1,.6],[.4,.4,.1,.1,.0001],[.8,.8,.1,.1,.9]]])
        original=raw.clone()
        result=corrected_predictions(raw,640,.001)[0]
        self.assertEqual(result["conf"].tolist(),raw[0,[2,0],4].tolist())
        self.assertEqual(result["_all_queries"]["query_index"].tolist(),[2,0,1])
        self.assertTrue(torch.equal(raw,original))
        self.assertEqual(result["bboxes"].shape,(2,4))
        with self.assertRaises(FloatingPointError):corrected_predictions(raw*float("nan"),640,.001)

    def test_finite_backend_warmup(self):
        from ultralytics.nn.autobackend import AutoBackend
        backend=AutoBackend.__new__(AutoBackend)
        torch.nn.Module.__init__(backend)
        backend.device=torch.device("cpu");backend.fp16=False
        for name in ("pt","jit","onnx","engine","saved_model","pb","triton","nn_module"):
            setattr(backend,name,name=="triton")  # execute the real warmup code on CPU
        seen=[]
        backend.forward=lambda x:seen.append(x.clone())
        backend.warmup((1,3,32,64))
        self.assertEqual(len(seen),1)
        self.assertEqual(tuple(seen[0].shape),(1,3,32,64))
        self.assertTrue(bool(torch.isfinite(seen[0]).all()))
        self.assertEqual(int(torch.count_nonzero(seen[0])),0)

    def test_recipe_exact_alias_and_type_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            parent=YAML.load(ROOT/"docs/c19_lif_v1/resolved_formal_config.yaml")
            target=dict(parent,model=str(ops.INIT),data=str(ops.MAIN/"configs/crack_autodl.yaml"),
                        project=str(ops.RUN.parent),name=ops.NAME,save_dir=str(ops.RUN))
            YAML.save(folder/"recipe.yaml",target)
            parent["model"]=ops.ALIASES[1]
            YAML.save(folder/"args.yaml",parent)
            with patch.object(ops,"PARENT_RUN",folder),patch.object(ops,"RECIPE",folder/"recipe.yaml"):
                resolved,report=ops.recipe()
                self.assertEqual(resolved,target)
                self.assertEqual(report["exact_model_alias"],ops.ALIASES[1])
                parent["epochs"]=200.0
                YAML.save(folder/"args.yaml",parent)
                with self.assertRaisesRegex(RuntimeError,"differs"):ops.recipe()
                parent["epochs"]=200;parent["model"]="unknown/model.pt"
                YAML.save(folder/"args.yaml",parent)
                with self.assertRaisesRegex(RuntimeError,"differs"):ops.recipe()

    def test_functional_digest_lf_and_docs_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            for rel in ("ultralytics-main/ultralytics/example.py","tools/init_c19_lif_v1.py","tools/init_lif_down.py","tools/lif_down_topology.py","tools/c19_lif_v1_data.py"):
                path=folder/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"x=1\r\n")
            YAML.save(folder/"algorithm.yaml",dict(enabled=True))
            YAML.save(folder/"recipe.yaml",dict(batch=16))
            with patch.object(ops,"ROOT",folder),patch.object(ops,"ALGORITHM",folder/"algorithm.yaml"),patch.object(ops,"RECIPE",folder/"recipe.yaml"):
                before=ops.functional_digest()
                (folder/"README.md").write_text("documentation edit")
                (folder/"ultralytics-main/ultralytics/example.py").write_bytes(b"x=1\n")
                self.assertEqual(before,ops.functional_digest())
                (folder/"ultralytics-main/ultralytics/example.py").write_bytes(b"x=2\n")
                self.assertNotEqual(before,ops.functional_digest())

    def test_reuse_requires_artifacts_and_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);artifact=folder/"predictions.json";artifact.write_text("saved")
            locked=dict(best="best.pt",best_sha256="fixed",epoch_zero_based=4,identity={})
            key=ops.digest(dict(locked,split="test",protocol=PROTOCOL,settings=EVAL,evaluation_functional_sha256=ops.functional_digest()))
            report=dict(locked,status="SUCCESS",identity_key=key,artifacts={str(artifact):ops.sha(artifact)})
            ops.write(folder/"evaluation/test/latest.json",report)
            with patch.object(ops,"OUT",folder),patch.object(ops,"lock_best",return_value=locked),patch("gpc_evaluate.evaluate",side_effect=AssertionError("inference forbidden")):
                self.assertEqual(ops.evaluate_split("test",{}),report)
                artifact.write_text("changed")
                with self.assertRaises(KeyError):ops.evaluate_split("test",{})  # reuse refused; missing recipe cannot be fabricated

    def test_dispatch_retains_pane_and_real_pipeline_codes(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            prepared=dict(identity=dict(recipe={},algorithm={}))
            with patch.object(ops,"OUT",folder),patch.object(ops,"verified_prepared",return_value=prepared),patch.object(ops.shutil,"which",return_value="tmux"),patch.object(ops,"live_worker",return_value=False),patch.object(ops.subprocess,"run") as calls,patch.object(ops.subprocess,"check_output",return_value="@7"):
                ops.dispatch("test")
                commands=[call.args[0] for call in calls.call_args_list]
                self.assertTrue(any("remain-on-exit" in row for row in commands))
                script=next((folder/"workers").rglob("worker.sh")).read_text()
                self.assertIn('codes=("${PIPESTATUS[@]}")',script)
                self.assertIn('"${codes[0]}" "${codes[1]}"',script)
                self.assertIn("PYTHONPATH=",script)
                self.assertNotIn("nvidia-smi",script)
                self.assertNotIn("_worker pack",script)

    def test_status_process_start_token_not_pane_name(self):
        identity=dict(pid=123,start_token="first",cmd=str(ops.ROOT/"tools/gpc_v1.py")+" _worker start",cwd=str(ops.ROOT),run=str(ops.RUN))
        with patch.object(ops,"process_info",return_value=identity):self.assertTrue(ops.live_worker(dict(process=identity)))
        with patch.object(ops,"process_info",return_value=dict(identity,start_token="reused")):self.assertFalse(ops.live_worker(dict(process=identity)))
        with patch.object(ops,"process_info",return_value=None):self.assertFalse(ops.live_worker(dict(process=identity)))

    def test_completed_run_can_supplement_evaluation_only_fix(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);run=folder/"run";run.mkdir()
            (run/"training_finished.json").write_text("{}")
            old=dict(functional_sha256="old",training_source_sha256="core",recipe={})
            ops.write(folder/"prepared.json",dict(status="READY",identity=old))
            current=dict(old,functional_sha256="evaluation-fix")
            with patch.object(ops,"OUT",folder),patch.object(ops,"RUN",run),patch.object(ops,"verify_repository"),patch.object(ops,"recipe",return_value=({"data":"fixture"},{})),patch.object(ops,"data_identity",return_value={}),patch.object(ops,"current_identity",return_value=current):
                self.assertEqual(ops.verified_prepared(for_evaluation=True)["identity"],old)
                with self.assertRaises(RuntimeError):ops.verified_prepared()
                current["training_source_sha256"]="changed-loss"
                with self.assertRaises(RuntimeError):ops.verified_prepared(for_evaluation=True)

    def test_bounded_timeout_does_not_relabel_old_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            ops.write(folder/"preflight.json",dict(status="TECHNICAL_PASS",effective_updates=99))
            with patch.object(ops,"OUT",folder),patch.object(ops,"verified_prepared",return_value=dict(identity={})),patch.object(ops,"experiment_lock",return_value=nullcontext()),patch.object(ops.subprocess,"Popen") as popen,patch.object(ops.os,"killpg",create=True) as kill:
                popen.return_value.pid=12345
                popen.return_value.wait.side_effect=[subprocess.TimeoutExpired("owned fixture",900),0]
                with self.assertRaises(ops.Pending):ops.bounded("preflight")
                report=ops.read(folder/"preflight.json")
                self.assertEqual(report["status"],"PENDING")
                self.assertNotIn("effective_updates",report)
                self.assertEqual(len(list((folder/"history").glob("*.json"))),1)
                self.assertEqual(kill.call_args.args[0],12345)

    def test_pack_is_offline_and_marks_missing_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);output=folder/"outputs/gpc_v1";run=folder/"runs/fixture"
            output.mkdir(parents=True);run.mkdir(parents=True)
            (run/"results.csv").write_text("epoch,loss\n1,2\n")
            with patch.object(ops,"ROOT",folder),patch.object(ops,"OUT",output),patch.object(ops,"RUN",run),patch.object(ops,"git",return_value="source-fixture"),patch.object(ops,"functional_digest",return_value="fixture"),patch.object(ops,"evaluate_split",side_effect=AssertionError("no evaluation")):
                ops.pack()
                manifests=list((output/"packages").glob("*.manifest.json"))
                self.assertEqual(len(manifests),1)
                self.assertEqual(ops.read(manifests[0])["status"],"INCOMPLETE")
                archives=list((output/"packages").glob("*.tar.gz"))
                self.assertEqual(len(archives),1)
                ops.pack()
                self.assertEqual(len(list((output/"packages").glob("*.tar.gz"))),1)


def run(output=None):
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(OperationalTests))
    report=dict(status="PASS" if result.wasSuccessful() else "FAIL",tests=result.testsRun,
                failures=[dict(test=str(test),trace=trace) for test,trace in result.failures+result.errors],
                real_linux_tmux_execution="PENDING",real_server_dataset_and_initialization="PENDING")
    if output:ops.write(output,report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path)
    args=parser.parse_args();sys.exit(0 if run(args.output)["status"]=="PASS" else 1)
