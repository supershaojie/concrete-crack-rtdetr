"""Small lifecycle/export regression suite; synthetic fixtures, no training or dataset inference."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gzip
import json
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ncr_v1_common import EVAL, NCR, SOURCE_SHA, identity, read, recipe, sha, write
import torch
import numpy as np
from ncr_v1 import pack, require_training_success, shell_worker, status
from ncr_v1_results import NCRExportValidator, analyze, native_pairs, verify_eval
from ncr_v1_training import NCRTrainer, set_epoch
from c19_lif_v1_results import postprocess
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.models.utils.ncr import NCRConfig
from ultralytics.cfg import DEFAULT_CFG_DICT


class OperationsChecks(unittest.TestCase):
    def test_manifest_identity_survives_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/"init.pt";source.write_bytes(b"synthetic identity fixture")
            with patch("ncr_v1_common.paths",return_value=dict(source=source,data=Path(folder)/"data.yaml")), \
                 patch("ncr_v1_common.recipe",return_value=({"batch":16},[])), \
                 patch("ncr_v1_common.sha",return_value=SOURCE_SHA), \
                 patch("ncr_v1_common.source_identity",return_value=dict(head="fixture")), \
                 patch("ncr_v1_common.environment",return_value=dict(python="fixture")), \
                 patch("ncr_v1_common.data_identity",return_value=dict(config={"names":{0:"crack"}})):
                value=identity()
            write(Path(folder)/"manifest.json",value)
            self.assertEqual(value,read(Path(folder)/"manifest.json"))
            self.assertEqual(value["data"]["config"]["names"],{"0":"crack"})

    def test_full_recipe_and_logging(self):
        args, changes = recipe()
        self.assertEqual(len(args), 109)
        self.assertFalse(set(NCR) & set(args))
        self.assertFalse(set(args)-set(DEFAULT_CFG_DICT)-{"save_dir"})
        self.assertEqual((args["batch"], args["imgsz"], args["epochs"], args["warmup_epochs"]), (16,640,200,5))
        self.assertEqual((args["hsv_s"], args["mosaic"], args["mixup"], args["perspective"]), (.5,.8,.05,.0002))
        self.assertLessEqual({row["field"] for row in changes}, {"model", "name", "project", "save_dir", "data"})
        fake = NCRTrainer.__new__(NCRTrainer)
        with patch.object(RTDETRTrainerForTest, "get_validator", return_value="native"):
            self.assertEqual(NCRTrainer.get_validator(fake), "native")
        self.assertEqual(fake.loss_names, ("giou_loss", "cls_loss", "l1_loss", "ncr_loss"))

    def test_native_resume_epoch_and_ema(self):
        fake = SimpleNamespace(resume=True, args=SimpleNamespace(model="last.pt", close_mosaic=10), epochs=200,
            model=torch.nn.Linear(1,1), ema=SimpleNamespace(ema=torch.nn.Linear(1,1)), _load_checkpoint_state=lambda ckpt: None)
        from ultralytics.engine.trainer import BaseTrainer
        BaseTrainer.resume_training(fake, dict(epoch=18))
        self.assertEqual(fake.start_epoch, 19)
        fake.epoch = fake.start_epoch
        set_epoch(fake)
        self.assertEqual(fake.model.ncr_epoch, 19); self.assertEqual(fake.ema.ema.ncr_epoch, 19)
        self.assertEqual(NCRConfig().coefficient(fake.model.ncr_epoch), .25)

    def test_corrected_protocol_and_all_query_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); image = root/"images/val/sample.jpg"
            logits = torch.linspace(-12, 4, 300).roll(137).reshape(1,300,1)
            boxes = torch.tensor([.5,.5,.4,.4]).repeat(1,300,1)
            predictions = torch.cat([boxes, logits.sigmoid()], -1)
            raw = (predictions, (boxes[None], logits[None], boxes, logits, None))
            before = predictions.clone()
            validator = NCRExportValidator.__new__(NCRExportValidator)
            validator.args = SimpleNamespace(**EVAL)
            validator.data = dict(path=str(root))
            validator.device = torch.device("cpu"); validator.niou = 10; validator.iouv = torch.linspace(.5,.95,10)
            validator.output = root; validator.export_seen = set()
            validator.export_counts = dict(images=0, predictions=0, ground_truth=0)
            selected = validator.postprocess(raw)
            reference, _ = postprocess(raw, 640, .001)
            for key in ("bboxes", "conf", "cls"):
                torch.testing.assert_close(selected[0][key], reference[0][key], atol=0, rtol=0)
            torch.testing.assert_close(before, predictions, atol=0, rtol=0)
            batch = dict(img=torch.zeros(1,3,640,640), bboxes=torch.tensor([[.5,.5,.4,.4]]), cls=torch.zeros(1,1),
                         batch_idx=torch.zeros(1), ori_shape=[(100,200)], ratio_pad=[((6.4,3.2),(0,0))], im_file=[str(image)])
            def update_native(self, preds, batch):
                self._process_batch(preds[0], self._prepare_batch(0,batch))
                assert "_metric_tp" not in preds[0]
            with patch.object(RTDETRValidator, "update_metrics", update_native):
                validator.update_metrics(selected,batch)
            with gzip.open(root/"queries_gt.jsonl.gz", "rt", encoding="utf-8") as stream: row=json.loads(next(stream))
            self.assertEqual({p["query_index"] for p in row["predictions"]}, set(range(300)))
            self.assertEqual(row["input_size_hw"], [640,640])
            self.assertLess(len(row["metric_query_indices"]),300)
            self.assertEqual(len(row["metric_query_indices"]), len(row["metric_tp_iou50_to95"]))
            self.assertEqual(row["transform"]["kind"], "RTDETR stretch")
            self.assertEqual(row["ground_truth"][0]["normalized_cxcywh"], batch["bboxes"][0].tolist())
            for pred in row["predictions"]:
                self.assertEqual(pred["class_logits"], logits[0,pred["query_index"]].tolist())
            write(root/"metrics.json",dict(status="PASSED", export_sha256=sha(root/"queries_gt.jsonl.gz"),native_PR_conf=.5))
            offline=analyze(root)
            self.assertEqual(offline["status"], "PASSED")
            self.assertEqual(offline["mechanism"]["matches"], 1)
            self.assertEqual(sum(g["gt"] for g in offline["groups"].values()), 1)
            self.assertEqual(analyze(root), read(root/"analysis.json"))

    def test_offline_native_tp_matching(self):
        torch.manual_seed(42)
        validator = BaseValidator.__new__(BaseValidator); validator.iouv = torch.linspace(.5,.95,10)
        for shape in ((4,13),(0,13),(4,0)):
            iou = torch.rand(*shape)
            native = validator.match_predictions(torch.zeros(shape[1]), torch.zeros(shape[0]), iou).numpy()
            for k, threshold in enumerate(validator.iouv.tolist()):
                recreated = np.zeros(shape[1],dtype=bool)
                pairs=native_pairs(iou.numpy(),threshold)
                if len(pairs): recreated[pairs[:,1]]=True
                np.testing.assert_array_equal(recreated,native[:,k])

    def test_failed_process_and_tee_exit_codes(self):
        bash = shutil.which("bash") or ("C:/Program Files/Git/bin/bash.exe" if sys.platform == "win32" else None)
        if not bash or not Path(bash).exists(): self.skipTest("bash unavailable")
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for python_code, tee_fail, expected in ((7,False,7),(0,False,0),(0,True,9)):
                log=root/"console.log"; code=root/"exit.txt"
                script=shell_worker(["bash", "-c", f"echo synthetic; exit {python_code}"], root.as_posix(),
                                    Path(log.as_posix()), Path(code.as_posix()), Path((root/"own.lock").as_posix()))
                # Lock utility availability is separate from pipeline semantics on Windows Git Bash.
                script=script.replace("flock -n 9 || exit 91", "true")
                if tee_fail: script=script.replace("export YOLO_AUTOINSTALL", "tee() { cat >/dev/null; return 9; }\nexport YOLO_AUTOINSTALL")
                process=subprocess.run([bash,"-c",script],capture_output=True,text=True,timeout=20)
                self.assertEqual(process.returncode,expected,process.stdout+process.stderr)
                self.assertEqual(code.read_text().strip(),str(expected))
                self.assertIn("NCR process exit=",process.stdout)

    def test_partial_pack_idempotence_and_no_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            p=dict(output=Path(folder)/"evidence",run=Path(folder)/"absent_run")
            with patch("ncr_v1_results.RTDETR", side_effect=AssertionError("pack inferred")):
                first=pack(p); second=pack(p)
            self.assertEqual(first["status"],"PARTIAL")
            self.assertEqual(first["archive"],second["archive"])
            self.assertTrue(first["missing"])

    def test_successful_eval_missing_export_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as folder:
            expected=dict(split="val",best_sha256="same")
            with self.assertRaisesRegex(RuntimeError,"no automatic full re-inference"):
                verify_eval(dict(status="PASSED",identity=expected),expected,folder)

    def test_failed_training_cannot_finish(self):
        with tempfile.TemporaryDirectory() as folder:
            p=dict(output=Path(folder),run=Path(folder)/"run")
            write(p["output"]/"train_state.json",dict(status="FAILED"))
            with self.assertRaisesRegex(RuntimeError,"successfully ended"): require_training_success(p)

    def test_status_is_read_only_and_exit_beats_tmux_presence(self):
        with tempfile.TemporaryDirectory() as folder:
            p=dict(output=Path(folder)/"evidence",run=Path(folder)/"run")
            with redirect_stdout(io.StringIO()) as stream:
                status(p)
            self.assertFalse(p["output"].exists())
            self.assertIn("NOT_STARTED",stream.getvalue())
            write(p["output"]/"train_state.json",dict(status="SUCCEEDED"))
            (p["output"]/"train_exit_code.txt").write_text("7\n")
            before={str(f):sha(f) for f in p["output"].iterdir()}
            with patch("ncr_v1.session_state",return_value="1 7"), patch("ncr_v1.shutil.which",return_value="tmux"), redirect_stdout(io.StringIO()) as stream:
                status(p)
            self.assertIn("train FAILED",stream.getvalue())
            self.assertEqual(before,{str(f):sha(f) for f in p["output"].iterdir()})


from ultralytics.models.rtdetr.train import RTDETRTrainer as RTDETRTrainerForTest


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path); args=parser.parse_args()
    torch.set_num_threads(4)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OperationsChecks))
    if args.output:
        write(args.output,dict(status="PASSED" if result.wasSuccessful() else "FAILED",tests=result.testsRun,
              failures=[str(x) for x in result.failures],errors=[str(x) for x in result.errors],skipped=result.skipped,
              tmux_live="NOT_RUN; target Linux tmux unavailable in local Windows checks" if sys.platform=="win32" else "NOT_RUN; synthetic lifecycle suite only"))
    raise SystemExit(0 if result.wasSuccessful() else 1)
