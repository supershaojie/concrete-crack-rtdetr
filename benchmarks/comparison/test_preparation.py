"""Bounded correctness checks for data conversion and exported-prediction evaluation."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'evaluation'))
from common import digest, canonical
from dataset import parse_labels, audit_dataset
from convert_annotations import convert
from evaluate import evaluate, POLICY_SHA
from PIL import Image


class PreparationTests(unittest.TestCase):
    def test_data_conversion_and_family_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); output = root/'audit'; output.mkdir()
            for split in ('train','val','test'):
                (root/'images'/split).mkdir(parents=True); (root/'labels'/split).mkdir(parents=True)
                Image.new('RGB',(100,80)).save(root/'images'/split/(split+'.png'))
                (root/'labels'/split/(split+'.txt')).write_text('0 .5 .5 .2 .25\n' if split=='train' else '')
            # Include the image-list path form, with ./ relative to the list file.
            (root/'test.txt').write_text('./images/test/test.png\n')
            (root/'data.yaml').write_text('train: images/train\nval: images/val\ntest: test.txt\nnames: [crack]\n')
            (root/'split_manifest.csv').write_text('output_name,source_id,split\ntrain.png,family1,train\nval.png,family1,test\ntest.png,family2,val\n')
            d=audit_dataset(root/'data.yaml',root,root,output)
            self.assertEqual(d['status'],'AUDITED'); self.assertEqual(d['families']['status'],'CONFIRMED_OVERLAP')
            self.assertEqual(d['families']['split_mismatch_count'],2)
            self.assertEqual(d['exact_duplicates']['cross_split_groups'],1)
            count=convert(d,output/'coco'); self.assertEqual(count['train']['output_boxes'],1)
            train=json.loads((output/'coco/train.json').read_text())
            self.assertEqual(train['annotations'][0]['bbox'],[40.,30.,20.,20.]);self.assertEqual(train['annotations'][0]['category_id'],1)
            val=json.loads((output/'coco/val.json').read_text());self.assertEqual(len(val['images']),1);self.assertEqual(val['annotations'],[])
            self.assertEqual(d['splits']['test']['split_paths_sha256'],digest(b'images/test/test.png'))
            # A nonempty malformed label blocks conversion rather than silently dropping the bad box.
            (root/'labels/test/test.txt').write_text('0 .5 .5 -1 .2\n')
            bad=root/'bad';bad.mkdir();invalid=audit_dataset(root/'data.yaml',root,root,bad)
            with self.assertRaises(ValueError): convert(invalid,bad/'coco')
            (root/'labels/test/test.txt').unlink()
            missing=root/'missing';missing.mkdir();unknown=audit_dataset(root/'data.yaml',root,root,missing)
            self.assertEqual(unknown['status'],'INVALID')
            negative=root/'negative';negative.mkdir();legal=audit_dataset(root/'data.yaml',root,root,negative,missing_labels='negative')
            self.assertEqual(legal['status'],'AUDITED')

    def test_invalid_labels(self):
        for line in ['1 .5 .5 .1 .1','0 nan .5 .1 .1','0 .5 .5 0 .1','0 1.01 .5 .1 .1','0 .5 .5 .1']:
            self.assertTrue(parse_labels(line.encode(),'fixture.txt')[1])
        self.assertEqual(parse_labels(b'\n','negative.txt'),([],[]))

    def fixtures(self):
        gt={'info':{'split':'test','dataset_identity_sha256':'d'*64},'images':[{'id':1,'width':100,'height':80},{'id':2,'width':100,'height':80}],
            'annotations':[{'image_id':1,'category_id':1,'bbox':[10,10,20,20],'iscrowd':0}]}
        identity={'model':'fixture','model_code_sha':'a'*40,'checkpoint_sha256':'b'*64,'dataset_identity_sha256':'d'*64,
                  'evaluation_config_sha256':POLICY_SHA,'postprocessing':'fixture no NMS'}
        rows=[{'split':'test','image_id':i,'width':100,'height':80,'box_format':'xyxy','coordinate_space':'original_image_pixels','predictions':[]} for i in (1,2)]
        return gt,identity,rows

    def test_sorted_confidence_mask_and_empty_images(self):
        gt,identity,rows=self.fixtures()
        # Low score first: a mask computed before sorting would discard the true high-score detection.
        rows[0]['predictions']=[{'category_id':1,'score':0.0001,'bbox':[50,50,60,60]},
                                {'category_id':1,'score':0.9,'bbox':[10,10,30,30]}]
        out=evaluate(gt,rows,identity)
        self.assertEqual(out['metric_predictions'],1);self.assertEqual(out['empty_prediction_images'],1)
        self.assertAlmostEqual(out['recall'],1);self.assertAlmostEqual(out['AP50'],0.995)
        with self.assertRaises(ValueError):evaluate(gt,rows[:1],identity)
        with self.assertRaises(ValueError):evaluate(gt,rows+[rows[0]],identity)
        with self.assertRaises(ValueError):evaluate(gt,list(reversed(rows)),identity)
        rows[0]['predictions']=[];out=evaluate(gt,rows,identity)
        self.assertEqual(out['mAP50_95'],0);self.assertEqual(out['recall'],0)

    def test_greedy_matching_and_strict_threshold(self):
        gt,identity,rows=self.fixtures()
        rows[0]['predictions']=[{'category_id':1,'score':s,'bbox':[10,10,30,30]} for s in (.9,.8,.001)]
        out=evaluate(gt,rows,identity)
        self.assertEqual(out['metric_predictions'],2);self.assertLessEqual(out['recall'],1)
        from native_metrics import Matcher,box_iou
        import torch
        b=torch.tensor([[10,10,30,30],[10,10,30,30]],dtype=torch.float32)
        m=Matcher().match_predictions(torch.ones(2),torch.ones(1),box_iou(b[:1],b))
        self.assertTrue((m.sum(0)==1).all())
        bad=copy.deepcopy(rows);bad[0]['predictions'][0]['category_id']=0
        with self.assertRaises(ValueError):evaluate(gt,bad,identity)
        bad=copy.deepcopy(identity);bad['dataset_identity_sha256']='e'*64
        with self.assertRaises(ValueError):evaluate(gt,rows,bad)

    def test_reused_metric_bodies_are_unchanged(self):
        project=Path(__file__).resolve().parents[2]
        native=ast.parse((project/'benchmarks/comparison/evaluation/native_metrics.py').read_text(encoding='utf-8'))
        original=ast.parse((project/'ultralytics-main/ultralytics/utils/metrics.py').read_text(encoding='utf-8'))
        for name in ('smooth','compute_ap','ap_per_class','box_iou'):
            a=next(n for n in native.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b))
        orig=ast.parse((project/'ultralytics-main/ultralytics/engine/validator.py').read_text(encoding='utf-8'))
        a=next(n for c in native.body if isinstance(c,ast.ClassDef) and c.name=='Matcher' for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='match_predictions')
        b=next(n for c in orig.body if isinstance(c,ast.ClassDef) and c.name=='BaseValidator' for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='match_predictions')
        self.assertEqual(ast.dump(a),ast.dump(b))


if __name__=='__main__':
    unittest.main(verbosity=2)
