"""CPU checks of conclusion-changing boundaries, optionally against the actual handoff."""
from __future__ import annotations
from copy import deepcopy
import gzip
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from PIL import Image

from benchmarks.visualization.chapter4.common import IDS,NAMES,ROOT,prepare,read_json,sha256,write_json
from benchmarks.visualization.chapter4.cache import ARCHIVED,canonical,ground_truth,read_cache
from benchmarks.visualization.chapter4.analysis import compare_predictions,correspondence,match,propose,target_correspondence
from benchmarks.visualization.chapter4.cam import gradcampp
from benchmarks.visualization.chapter4.geometry import geometry,inverse_boxes,inverse_cam,normalize_cam
from benchmarks.visualization.chapter4.render import matrix


class TestBoundaries(unittest.TestCase):
    def setUp(self):
        parent=ROOT/"outputs/ch4_unit_tests";parent.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(dir=parent)
        self.folder=Path(self.temporary.name).resolve()
        self.assertTrue(self.folder.is_relative_to(parent.resolve()))

    def tearDown(self):
        self.assertTrue(self.folder.is_relative_to((ROOT/"outputs/ch4_unit_tests").resolve()))
        self.temporary.cleanup()

    def test_integer_letterbox_separate_axis_inverse_and_padding(self):
        for width,height in [(333,1000),(1000,333),(4096,3072)]:
            info=geometry(width,height)
            boxes=np.array([[-10,20,width+12,height-3]],dtype=float)
            input_boxes=boxes.copy()
            input_boxes[:,[0,2]]=boxes[:,[0,2]]*info["gain_x"]+info["left"]
            input_boxes[:,[1,3]]=boxes[:,[1,3]]*info["gain_y"]+info["top"]
            np.testing.assert_allclose(inverse_boxes(input_boxes,info),boxes,atol=1e-9)
            self.assertEqual(info["left"]+info["right"]+info["resized_width"],640)
            self.assertEqual(info["top"]+info["bottom"]+info["resized_height"],640)
        info=geometry(333,1000)
        self.assertNotEqual(info["gain_x"],info["gain_y"])
        padding=np.ones((640,640),dtype=np.float32)
        padding[info["top"]:info["top"]+info["resized_height"],info["left"]:info["left"]+info["resized_width"]]=0
        np.testing.assert_array_equal(inverse_cam(padding,info),np.zeros((1000,333)))

    def test_stretch_landmark_alignment(self):
        info=geometry(333,1000,"stretch")
        cam=np.zeros((640,640),dtype=np.float32);cam[160:480,160:480]=1
        restored=inverse_cam(cam,info)
        self.assertEqual(restored.shape,(1000,333))
        self.assertGreater(restored[500,166],.99)
        self.assertEqual(float(restored[100,20]),0)
        self.assertEqual((info["left"],info["top"]),(0,0))

    def test_duplicate_detection_and_low_confidence_miss_diagnostics(self):
        gt=[dict(id=9,bbox=[0,0,10,10],category_id=1)]
        p=[dict(bbox=[0,0,10,10],score=.9),dict(bbox=[0,0,10,10],score=.8)]
        result=match(p,gt,.5,.25)
        self.assertEqual((result["tp"],result["fp"],result["fn"]),(1,1,0))
        result=match([dict(bbox=[0,0,10,10],score=.2)],gt,.75,.25)
        self.assertEqual(result["misses"][0]["reason"],"below_display_threshold")
        self.assertEqual(result["misses"][0]["best_score"],.2)

    def test_order_independent_smoke_with_ambiguous_duplicates(self):
        ref=[dict(category_id=1,bbox=[0,0,10,10],score=.5),dict(category_id=1,bbox=[.005,0,10,10],score=.5)]
        actual=[dict(category_id=1,bbox=[.005,0,10,10],score=.5),dict(category_id=1,bbox=[-.005,0,10,10],score=.5)]
        tol=dict(input_pixel_atol=.006,coordinate_rtol=0,score_atol=.00002)
        self.assertEqual(compare_predictions(ref,actual,tol)["detections"],2)
        with self.assertRaises(ValueError):compare_predictions(ref,actual[:1],tol)
        actual[1]["category_id"]=2
        with self.assertRaises(ValueError):compare_predictions(ref,actual,tol)

    def test_fixed_gt_correspondence_no_duplicate_candidate(self):
        gt=[dict(id=1,bbox=[0,0,10,10]),dict(id=2,bbox=[0,0,10,10])]
        mapped=correspondence([[0,0,10,10]],gt)
        self.assertEqual(len(mapped),1)
        self.assertEqual(mapped[0]["gt_id"],1)

    def test_explain_final_detection_before_nearer_low_score_anchor(self):
        gt=[dict(id=1,bbox=[0,0,10,10])]
        raw=[[0,0,10,10],[0,0,9,10]]
        final=[dict(category_id=1,bbox=raw[1],score=.8)]
        mapped=target_correspondence(raw,[.00001,.8],final,gt,.25)
        self.assertEqual(mapped[0]["candidate_index"],1)
        self.assertEqual(mapped[0]["association_tier"],"display_detection")
        mapped=target_correspondence(raw,[.00001,.2],[{**final[0],"score":.2}],gt,.25)
        self.assertEqual(mapped[0]["candidate_index"],1)
        self.assertEqual(mapped[0]["association_tier"],"formal_below_display_threshold")

    def test_run_lock_prevents_same_output_concurrent_mutations(self):
        from benchmarks.visualization.chapter4.common import run_lock
        with run_lock(self.folder):
            with self.assertRaises(RuntimeError):
                with run_lock(self.folder):pass
        with run_lock(self.folder):pass

    def test_proposals_remain_pending_and_reject_partial_pool(self):
        from benchmarks.visualization.chapter4.analysis import statistics
        with self.assertRaises(ValueError):propose({}, {}, {}, IDS[:4],self.folder,.25,self.folder)
        images={};predictions={mid:{} for mid in IDS}
        for i in range(7):
            name=f"crack_{i:06d}_UNIT_TEST_ONLY.jpg";key="images/test/"+name
            Image.new("RGB",(40,60),(50+i*20,60,90)).save(self.folder/name)
            gt=dict(id=i,category_id=1,bbox=[0,10,35,30])
            images[key]=dict(id=i,width=40,height=60,annotations=[gt])
            for mid in IDS:predictions[mid][key]=[]
            predictions["lcbr_detr"][key]=[dict(category_id=1,bbox=gt["bbox"],score=.9)]
        details=statistics(images,predictions,IDS,.25,self.folder/"stats")
        proposed,document=propose(images,predictions,details,IDS,self.folder,.25,self.folder/"candidate",count=5,candidates=7)
        self.assertEqual(proposed["status"],"proposed_pending_visual_review")
        self.assertEqual(len(proposed["cases"]),5)
        self.assertEqual(len(document["cases"]),7)
        self.assertEqual(len({c["source_family"] for c in proposed["cases"]}),5)
        self.assertTrue(all(c["target_gt_ids"] for c in proposed["cases"]))
        self.assertFalse((self.folder/"candidate/cases.reviewed.json").exists())

    def test_gradcampp_weighting_and_invalid_response(self):
        a=np.array([[[[1,2],[3,4]],[[4,3],[2,1]]]],dtype=np.float32)
        g=np.array([[[[.2,.1],[0,-.1]],[[0,.3],[.1,.1]]]],dtype=np.float32)
        response,metadata=gradcampp(a,g)
        expected=np.zeros((2,2))
        for c in range(2):
            coefficient=0
            for i in range(2):
                for j in range(2):
                    value=float(g[0,c,i,j])
                    if value!=0:coefficient+=max(value,0)*value**2/(2*value**2+float(a[0,c].sum())*value**3+1e-6)
            expected+=coefficient*a[0,c]
        np.testing.assert_allclose(response,np.maximum(expected,0),rtol=1e-6)
        self.assertEqual(metadata["status"],"passed")
        zero,meta=gradcampp(a,np.zeros_like(g))
        self.assertEqual(meta["status"],"zero_gradient_scale")
        with self.assertRaises(ValueError):normalize_cam(zero)
        with self.assertRaises(ValueError):gradcampp(a,np.full_like(g,np.nan))
        _,none=gradcampp(a,None)
        self.assertFalse(none["gradient_exists"])

    def test_ours_schema_native_class_and_unfiltered_queries(self):
        images={"images/test/a.jpg":dict(id=7,width=100,height=200)}
        model=dict(id="lcbr_detr",checkpoint=dict(sha256="a"*64))
        path=self.folder/"ours.jsonl.gz"
        row=dict(image="images/test/a.jpg",original_size_hw=[200,100],box_format="xyxy",coordinate_space="original_image_pixels",
                 predictions=[dict(class_id=0,bbox=[-10,0,30,50],score=.0001,used_for_metrics=False)])
        with gzip.open(path,"wt",encoding="utf-8") as f:f.write(json.dumps(row)+"\n")
        records,_=read_cache(path,model,images)
        self.assertEqual(records["images/test/a.jpg"][0]["category_id"],1)
        self.assertEqual(records["images/test/a.jpg"][0]["bbox"][0],-10)
        row["predictions"][0]["class_id"]=1
        with gzip.open(path,"wt",encoding="utf-8") as f:f.write(json.dumps(row)+"\n")
        with self.assertRaises(ValueError):read_cache(path,model,images)

    def test_c2_stem_join_xywh_and_quantization(self):
        images={"images/test/crack_a.jpg":dict(id=888,width=100,height=200)}
        model=dict(id="rtdetr_c2",checkpoint=dict(sha256="b"*64))
        path=self.folder/"c2.json.gz"
        row=dict(file_name="crack_a.jpg",image_id="crack_a",category_id=1,bbox=[1.111,2.222,3.333,4.444],score=.12345)
        with gzip.open(path,"wt",encoding="utf-8") as f:json.dump([row],f)
        records,meta=read_cache(path,model,images)
        np.testing.assert_allclose(records["images/test/crack_a.jpg"][0]["bbox"],[1.111,2.222,4.444,6.666])
        self.assertEqual(meta["source_metadata"]["precision"]["score_decimals"],5)

    def test_archive_traversal_rejected(self):
        from benchmarks.visualization.chapter4.common import safe_extract
        path=self.folder/"unsafe.zip"
        with zipfile.ZipFile(path,"w") as z:z.writestr("../escape","bad")
        with self.assertRaises(ValueError):safe_extract(path,self.folder/"unzip")

    def test_layout_all_models_same_case_order_and_pending_cam(self):
        # Explicit synthetic layout fixture, never presented as scientific results.
        images={};predictions={mid:{} for mid in IDS};cases=[]
        for i in range(5):
            name=f"UNIT_TEST_LAYOUT_ONLY_{i}.jpg";key="images/test/"+name
            Image.new("RGB",(90+i*10,130),(100,110,120)).save(self.folder/name)
            images[key]=dict(id=i,width=90+i*10,height=130,annotations=[])
            for mid in IDS:predictions[mid][key]=[]
            cases.append(dict(image=key,view_region=[0,0,90+i*10,130],display_threshold=.25))
        doc=dict(status="synthetic_unit_test_only",cases=cases)
        models=[dict(id=mid,model=name) for mid,name in zip(IDS,NAMES)]
        stem=self.folder/"UNIT_TEST_LAYOUT_ONLY"
        record=matrix(doc,images,predictions,models,self.folder,stem,config=dict(dpi=90),draft=True)
        self.assertEqual((record["rows"],record["columns"]),(9,5))
        with Image.open(stem.with_suffix(".png")) as raster:
            self.assertEqual(raster.size,(record["pixel_width"],record["pixel_height"]))
        self.assertEqual(record["row_labels"],["Ground truth"]+NAMES)
        svg=stem.with_suffix(".svg").read_text(encoding="utf-8")
        self.assertEqual(svg.count(">Faster R-CNN</text>"),1)
        self.assertEqual(svg.count(">ours</text>"),1)
        self.assertNotIn("rotate(",svg)
        self.assertLess(svg.count("<image "),svg.count("<use "))
        import xml.etree.ElementTree as ET
        ET.fromstring(svg)
        heat=matrix(doc,images,predictions,models,self.folder,self.folder/"UNIT_TEST_HEAT_LAYOUT_ONLY",
                    heatmap=True,config=dict(dpi=90),draft=True,formats=("svg",))
        self.assertEqual(len(heat["pending"]),40)
        self.assertEqual(heat["row_labels"],["Original"]+NAMES)
        zero=self.folder/"UNIT_TEST_ZERO_RESPONSE.npz"
        np.savez_compressed(zero,cam_s8=np.zeros((80,80),dtype=np.float32))
        meta=self.folder/"UNIT_TEST_ZERO_RESPONSE.json"
        write_json(meta,dict(geometry=geometry(90,130)))
        bad={IDS[0]:{cases[0]["image"]:dict(status="passed",npz=str(zero),metadata=str(meta))}}
        heat=matrix(doc,images,predictions,models,self.folder,self.folder/"UNIT_TEST_INVALID_CAM_ONLY",heatmap=True,
                    cam_index=bad,config=dict(dpi=90),draft=True,formats=("svg",))
        self.assertTrue(any(r["status"]=="render_failed" for r in heat["pending"]))

    def test_grouped_cam_reuses_successes_and_retries_only_failed_cell(self):
        from benchmarks.visualization.chapter4.cli import Pipeline
        from types import SimpleNamespace
        pipeline=object.__new__(Pipeline)
        pipeline.args=SimpleNamespace(offline=False);pipeline.out=self.folder;pipeline.protocol="UNIT_TEST_PROTOCOL_ONLY"
        pipeline.image_root=self.folder;pipeline.images={};cases=[]
        for i in range(5):
            name=f"UNIT_TEST_CACHE_ONLY_{i}.jpg";key="images/test/"+name
            Image.new("RGB",(20,20),(i*30,50,80)).save(self.folder/name)
            pipeline.images[key]=dict(file_name=key)
            cases.append(dict(image=key,target_gt_ids=[i],display_threshold=.25))
        pipeline.models=[dict(id=IDS[0],checkpoint=dict(sha256="a"*64)),dict(id=IDS[-1],checkpoint=dict(sha256="b"*64))]
        pipeline.selected_models=lambda:pipeline.models
        pipeline.cases=lambda:dict(status="unit_test_only",cases=cases)
        pipeline.has_verified_models=lambda report:True
        write_json(self.folder/"layer_protocol.json",dict(status="unit_test_fixture_only"))
        write_json(self.folder/"reports/smoke.json",dict(status="unit_test_fixture_only"))
        calls=[];failed_once=set()
        def fake_dispatch(model,action,**request):
            calls.append((model["id"],len(request["cases"])))
            result=dict(status="passed",images={})
            for case in request["cases"]:
                key=case["image"]
                if model["id"]==IDS[0] and key==cases[0]["image"] and key not in failed_once:
                    failed_once.add(key);result["images"][key]=dict(status="failed",error="UNIT TEST failure/recovery plumbing only");continue
                dest=Path(request["destinations"][key])/Path(key).stem;dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest.with_suffix(".npz"),cam_s8=np.ones((2,2),dtype=np.float32))
                write_json(dest.with_suffix(".json"),dict(status="passed",fixture_only=True))
                result["images"][key]=dict(status="passed",npz=str(dest.with_suffix(".npz")),metadata=str(dest.with_suffix(".json")))
            return result
        pipeline.dispatch=fake_dispatch
        pipeline.cam()
        self.assertEqual(calls,[(IDS[0],5),(IDS[-1],5)])
        pipeline.cam()
        self.assertEqual(calls[-1],(IDS[0],1))
        self.assertEqual(len(calls),3)
        self.assertEqual(len(list((self.folder/"reports/cam_failures").rglob("*.json"))),1)

    def test_pack_integrity_and_exclusive_size_bound(self):
        module=importlib.import_module("benchmarks.visualization.chapter4.pack")
        write_json(self.folder/"registry.resolved.json",dict(fixed_model_ids=IDS))
        write_json(self.folder/"reports/test.json",dict(status="unit_test_only",all_models=IDS))
        with self.assertRaises(ValueError):module.pack(self.folder)
        receipt=module.pack(self.folder,require_complete=False)
        self.assertLess(receipt["bytes"],100_000_000)
        with zipfile.ZipFile(receipt["path"]) as z:
            self.assertIsNone(z.testzip())
            manifest=json.loads(z.read("MANIFEST.json"))
            self.assertEqual(manifest["model_order"],IDS)
        with patch.object(module,"PACKAGE_LIMIT",1):
            with self.assertRaises(ValueError):module.pack(self.folder,self.folder/"too_big.zip",require_complete=False)
        self.assertFalse((self.folder/"too_big.zip").exists())


@unittest.skipUnless(os.environ.get("CH4_TEST_HANDOFF"),"Set CH4_TEST_HANDOFF for actual 960-file/four-cache checks")
class TestActualHandoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base,cls.integrity=prepare(os.environ["CH4_TEST_HANDOFF"],ROOT/"outputs/ch4_test_archive")
        cls.registry=read_json(cls.base/"model_registry.json")
        cls.images=ground_truth(cls.base)

    def test_actual_identity_and_corrected_runs(self):
        self.assertEqual(self.integrity["manifest_files_checked"],960)
        self.assertEqual([m["id"] for m in self.registry["models"]],IDS)
        self.assertEqual(self.registry["models"][0]["run_id"],"frcnn_coco_sgd_200_s42_b16_midaug_11")
        self.assertEqual(self.registry["models"][1]["run_id"],"v5_fullalign_aug_x13_02")
        self.assertEqual(self.registry["models"][4]["run_id"],"v13l_mid_x115_flash_05")
        self.assertEqual(sum(len(im["annotations"]) for im in self.images.values()),6663)
        self.assertEqual(self.registry["models"][-1]["checkpoint"]["sha256"],"24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa")

    def test_all_actual_archived_schemas_and_all_test_images(self):
        for model in self.registry["models"]:
            if model["id"] not in ARCHIVED:continue
            with self.subTest(model=model["id"]):
                records,metadata=read_cache(self.base/ARCHIVED[model["id"]],model,self.images,model["prediction_sha256"])
                self.assertEqual(len(records),864)
                self.assertEqual(set(records),set(self.images))
                self.assertTrue(all(p["category_id"]==1 for rows in records.values() for p in rows))
                if model["id"]=="rtdetr_c2":
                    self.assertEqual(sum(map(len,records.values())),259200)
                    self.assertGreater(min(p["score"] for rows in records.values() for p in rows),.001)
                # All data are real; no model identity/threshold substitution.
                first=next(iter(self.images))
                value=match(records[first],self.images[first]["annotations"],.5,.25)
                self.assertEqual(value["tp"]+value["fn"],len(self.images[first]["annotations"]))


if __name__=="__main__":unittest.main()
