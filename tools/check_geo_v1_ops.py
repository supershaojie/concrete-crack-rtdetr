"""Lifecycle regression checks with temporary evidence; never starts formal training/test."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from geo_v1_common import ROOT,OUT,read_json,write_json,digest,canonical_config
import geo_v1_common as common
import geo_v1_results as results
import geo_v1 as cli


class Operations(unittest.TestCase):
    def test_existing_quoted_path_never_downloads(self):
        from ultralytics.utils.downloads import attempt_download_asset
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"literal'quote.pt"; p.write_bytes(b"local file")
            with patch("ultralytics.utils.downloads.get_github_assets",side_effect=AssertionError("network")):
                self.assertEqual(Path(attempt_download_asset(str(p))),p)

    def test_json_class_identity_roundtrip_and_collision(self):
        c=dict(path=str(ROOT),train="images/train",val="images/val",test="images/test",names={0:"crack"})
        self.assertEqual(canonical_config(c),canonical_config(json.loads(json.dumps(c))))
        with self.assertRaises(RuntimeError): canonical_config(dict(c,names={0:"crack","0":"collision"}))

    def test_lock_recovery_and_incomplete_rejection(self):
        with tempfile.TemporaryDirectory() as d,patch.object(results,"OUT",Path(d)),patch.object(results,"COUNTS",dict(val=(2,1))):
            folder=Path(d)/"evaluations/001_val"; folder.mkdir(parents=True)
            predictions=folder/"all.jsonl.gz"; predictions.write_bytes(b"fixture export")
            curves=folder/"curves.json"; curves.write_text("{}")
            identity=dict(split="val",checkpoint_sha256="fixture",policy=results.POLICY)
            report=dict(status="PASS",identity=identity,images=2,Precision=0.,Recall=0.,F1=0.,AP50=0.,AP75=0.,mAP50_95=0.,
                native_best_F1_confidence=.2,AP_by_IoU=[0.]*10,artifacts=dict(predictions=results.file_info(predictions),curves=results.file_info(curves)))
            write_json(folder/"metrics.json",report)
            with patch.object(results,"evaluate",side_effect=AssertionError("inference forbidden")):
                restored=results.reuse_result("val",identity)
                self.assertEqual(restored["status"],"PASS")
                self.assertTrue((Path(d)/"val_lock.json").is_file())
                self.assertEqual(results.reuse_result("val",identity),restored)
            broken=deepcopy(report); broken["status"]="FAIL"
            self.assertFalse(results.valid_record(broken,identity))
            changed=deepcopy(identity); changed["checkpoint_sha256"]="different"
            self.assertFalse(results.valid_record(report,changed))
            predictions.write_bytes(b"tampered")
            self.assertFalse(results.valid_record(report,identity))
            with self.assertRaises(RuntimeError): results.reuse_result("val",identity)

    def test_offline_counts_include_empty_images(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"export.jsonl.gz"
            rows=[dict(boxes=[[0,0,1,1],[2,2,3,3]],scores=[.9,.8],classes=[0,0],gt_boxes=[[0,0,1,1]],gt_classes=[0]),
                dict(boxes=[],scores=[],classes=[],gt_boxes=[[0,0,1,1]],gt_classes=[0]),
                dict(boxes=[[0,0,1,1]],scores=[.7],classes=[0],gt_boxes=[],gt_classes=[])]
            with gzip.open(p,"wt",encoding="utf-8") as f:
                for row in rows: f.write(json.dumps(row)+"\n")
            value=results.counts_at_threshold(p,.5)
            self.assertEqual((value["TP"],value["FP"],value["FN"],value["images"]),(1,2,1,3))
            self.assertAlmostEqual(value["Precision"],1/3)

    def test_pack_is_read_only_and_marks_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"out"; out.mkdir(); run=Path(d)/"run"
            with patch.object(results,"OUT",out),patch.object(results,"RUN",run),patch.object(results,"git",return_value=""),\
                 patch.object(results,"code_identity",return_value={"fixture":True}),\
                 patch.object(results,"evaluate",side_effect=AssertionError("inference")),\
                 patch.object(common,"scan_data",side_effect=AssertionError("raw data scan")):
                result=results.pack(dict(fixture=True))
            self.assertEqual(result["status"],"INCOMPLETE")
            with tarfile.open(result["path"]) as a:
                manifest=json.load(a.extractfile("manifest.json"))
                self.assertNotIn("manifest.json",[r["path"] for r in manifest])
                self.assertFalse(any(n.endswith(".pt") for n in a.getnames()))
                self.assertIn(b"NOT included",a.extractfile("README.md").read())

    def test_preflight_cannot_accept_fallback_or_stale_binding(self):
        identity=dict(fixture=True)
        with self.assertRaises(RuntimeError): cli.verify_preflight(dict(status="PENDING",start_eligible=True),identity)
        checks={k:dict(status="PASS") for k in ("cpu","cuda_b16_amp","native_scale","mechanism","resume","new_process_val")}
        checks["cuda_b16_amp"].update(effective_updates=1,batch=16,imgsz=640,amp=True)
        good=dict(status="PASS",start_eligible=True,binding=identity,checks=checks,gpu=dict(micro_batches=16))
        cli.verify_preflight(good,identity)
        with self.assertRaises(RuntimeError): cli.verify_preflight(good,{"fixture":False})
        bad=deepcopy(good); bad["checks"]["native_scale"]["status"]="PENDING"
        with self.assertRaises(RuntimeError): cli.verify_preflight(bad,identity)

    def test_snapshot_binding_does_not_scan(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); data=out/"data.yaml"; manifest=out/"data_manifest.jsonl.gz"
            conf=dict(path=str(out),names={0:"crack"},train="images/train",val="images/val",test="images/test")
            common.YAML.save(data,conf); manifest.write_bytes(b"fixture saved manifest")
            record=dict(data=dict(config=canonical_config(conf),config_sha256=common.sha256(data)),manifest_sha256=common.sha256(manifest),directory_markers={})
            write_json(out/"data_snapshot.json",record)
            with patch.object(common,"OUT",out),patch.object(common,"directory_markers",return_value={}),\
                 patch.object(common,"scan_data",side_effect=AssertionError("raw data scan forbidden")):
                self.assertEqual(common.snapshot(data),record)

    def test_diagnostics_are_count_weighted(self):
        from ultralytics.models.rtdetr.geo_model import aggregate
        samples=[dict(M=1,images=1,gt_count=2,match_mean_sum=.3,ramp=1,original_losses=dict(x=2),finite=True),
                 dict(M=3,images=2,gt_count=3,match_mean_sum=0.,ramp=1,original_losses=dict(x=5),finite=True)]
        result=aggregate(samples)
        self.assertAlmostEqual(result["raw_geo"],.075)
        self.assertAlmostEqual(result["weighted"],.015)
        self.assertEqual(result["original_losses_image_weighted"]["x"],4.)


def help_checks():
    commands=[[sys.executable,str(ROOT/"tools/geo_v1.py"),"--help"]]
    for sub in ("prepare","preflight","status","start","resume","val","test","finish","pack","_preflight","_worker"):
        commands.append([sys.executable,str(ROOT/"tools/geo_v1.py"),sub,"--help"])
    for name in ("check_geo_v1.py","geo_v1_preflight.py","check_geo_v1_ops.py"):
        commands.append([sys.executable,str(ROOT/"tools"/name),"--help"])
    bash=shutil.which("bash") if os.name!="nt" else "C:/Program Files/Git/bin/bash.exe"
    if bash and Path(bash).is_file():
        for name in ("geo_v1.sh","sync_geo_v1.sh"):
            commands.append([bash,"tools/"+name,"--help"])
            commands.append([bash,"-n","tools/"+name])
    reports=[]
    for command in commands:
        result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=60)
        reports.append(dict(command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode: raise RuntimeError(json.dumps(reports[-1]))
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT/"operations_checks.json")
    parser.add_argument("--skip-help",action="store_true")
    args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Operations))
    report=dict(status="PASS" if result.wasSuccessful() else "FAIL",tests=result.testsRun,
        scope="Temporary evidence fixtures, no server training or full val/test")
    try:
        if not args.skip_help and result.wasSuccessful(): report["fresh_process_help"]=help_checks()
    except BaseException as error:
        report.update(status="FAIL",error=repr(error))
        raise
    finally: write_json(args.output,report)
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__=="__main__": main()
