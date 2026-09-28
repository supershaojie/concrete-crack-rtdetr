"""Temporary-fixture checks for cached identity, no-inference packaging and exports."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import qcc_v1_common as c
import qcc_v1 as cli
from ultralytics.models.rtdetr.qcc_io import write_json
from ultralytics.models.rtdetr.qcc_val import QCCValidator
from ultralytics.utils import YAML


class Operations(unittest.TestCase):
    def test_class_normalization_conflict(self):
        self.assertEqual(c.data_identity_config(dict(names={0:'crack'})),c.data_identity_config(dict(names={'0':'crack'})))
        self.assertEqual(c.data_identity_config(dict(names=['crack'])),dict(names={'0':'crack'}))
        with self.assertRaises(RuntimeError): c.data_identity_config(dict(names={0:'crack','0':'other'}))

    def test_prepare_reuses_manifest_then_cached_only(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); out=root/'qcc'; old=root/'reference'; data_root=root/'dataset'
            out.mkdir(); old.mkdir()
            config=dict(path=str(data_root.resolve()),names={'0':'crack'},train='images/train',val='images/val',test='images/test')
            rows=[]
            for split in c.COUNTS:
                (data_root/f'images/{split}').mkdir(parents=True)
                (data_root/f'labels/{split}').mkdir(parents=True)
                rows.append(dict(split=split,image=f'images/{split}/one.jpg',label=f'labels/{split}/one.txt',boxes=1,size_hw=[2,2],image_sha256='fixture',label_sha256='fixture'))
            data=dict(config=config,config_sha256='previous-serialization',content_sha256=c.digest(rows),
                splits={s:dict(images=1,boxes=1,content_sha256=c.digest([r for r in rows if r['split']==s])) for s in c.COUNTS})
            source=old/'prepare.json'; write_json(source,dict(status='PASS',data=data,code={'commit':'fixture'}))
            with gzip.open(old/'data_manifest.jsonl.gz','wt',encoding='utf-8') as f:
                for row in rows: f.write(json.dumps(row)+'\n')
            path=root/'data.yaml'; YAML.save(path,config)
            with patch.object(c,'OUT',out),patch.object(c,'COUNTS',{s:(1,1) for s in c.COUNTS}),patch.object(c,'data_config',return_value=path),patch.object(c,'inventory',side_effect=AssertionError('unexpected raw inventory')):
                prepared=c.prepare_data(path,source)
                self.assertEqual(prepared,c.cached_data())
                self.assertEqual(prepared,c.prepare_data(path))
                self.assertIn('NOT rehashed',c.read_json(out/'data_identity.json')['reused_from']['note'])
                changed=dict(config,names={'0':'changed'}); YAML.save(path,changed)
                with self.assertRaises(RuntimeError): c.cached_data()

    def test_binding_no_inventory(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); source=root/'s'; init=root/'i'; source.write_text('s'); init.write_text('i')
            snapshot=root/'source_snapshot.tar.gz'; snapshot.write_bytes(b'fixture')
            data={'fixture':'fixed'}; args={'batch':16}
            write_json(root/'prepare.json',dict(status='PASS',data=data,args=args,code={'code':1},init_sha256=c.sha256(init),source_snapshot_sha256=c.sha256(snapshot)))
            with patch.multiple(c,OUT=root,SOURCE=source,INIT=init,SOURCE_SHA256=c.sha256(source)),patch.object(c,'code_identity',return_value={'code':1}),patch.object(c,'cached_data',return_value=data),patch.object(c,'data_config',return_value=root/'data.yaml'),patch.object(c,'recipe',return_value=(args,{})),patch.object(c,'inventory',side_effect=AssertionError('inventory forbidden')):
                self.assertEqual(c.binding()['data'],data)

    def test_status_pack_are_product_only(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); out=root/'evidence'; run=root/'run'; out.mkdir()
            (root/'source.txt').write_text('fixture')
            with patch.multiple(cli,OUT=out,RUN=run,ROOT=root),patch.object(cli,'git',return_value='source.txt'),patch.object(cli,'code_identity',return_value={'commit':'fixture'}),patch.object(cli,'has_tmux',return_value=False),patch.object(cli,'active_workers',return_value=[]),patch.object(cli,'inventory',side_effect=AssertionError('inventory')),patch.object(cli,'evaluate',side_effect=AssertionError('evaluate')),patch.object(QCCValidator,'__init__',side_effect=AssertionError('model evaluation')):
                self.assertEqual(cli.status()['training'],'NOT_COMPLETED')
                result=cli.pack()
                self.assertEqual(result['completeness']['status'],'INCOMPLETE')
                self.assertTrue(Path(result['path']).is_file())
                import tarfile
                with tarfile.open(result['path']) as f:
                    manifest=json.load(f.extractfile('manifest.json'))
                    self.assertFalse(any(x['path']=='manifest.json' for x in manifest))

    def test_missing_export_not_success(self):
        self.assertFalse(cli.export_complete({'status':'PASS','split':'val'}))
        with self.assertRaises(RuntimeError): cli.verify_preflight({'status':'PASS','start_eligible':True,'binding':{}}, {})

    def test_evaluation_reentry_reuses_without_predicting(self):
        import numpy as np
        import ultralytics.models.rtdetr.qcc_val as module
        from ultralytics.models.rtdetr.qcc_val import EVAL
        calls=[]
        class FakeValidator:
            def __init__(self,args,save_dir): self.actual_settings=args
            def __call__(self,model):
                calls.append(self.actual_settings['split'])
                p=self.export_path; p.parent.mkdir(parents=True,exist_ok=True)
                row=dict(query_indices=list(range(300)),gt_classes=[0],scores=[.5]*300,classes=[0]*300,
                    boxes_normalized_cxcywh=[[.5,.5,.1,.1]]*300,gt_boxes_normalized_cxcywh=[[.5,.5,.1,.1]])
                with gzip.open(p,'wt',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
                self.qcc_seen={'one'}
                self.qcc_metrics={'reported_workpoint':{'confidence':.5}}
                np.savez_compressed(p.parent/'evaluator_stats.npz',conf=np.array([.5]),tp=np.ones((1,10),dtype=bool),target_cls=np.array([0]))
                write_json(p.parent/'curves.json',{})
                write_json(Path(str(p)+'.complete.json'),dict(status='PASS',scope='full_split',predictions_sha256=c.sha256(p),images=1,gt=1,identity=self.export_identity))
                return {'metrics/mAP50-95(B)':1.0}
        with tempfile.TemporaryDirectory() as name:
            out=Path(name)/'out'; out.mkdir(); run=Path(name)/'run'; (run/'weights').mkdir(parents=True)
            (run/'weights/best.pt').write_bytes(b'fixture identity only; never load a model')
            write_json(out/'training_completed.json',{'status':'TRAINING_COMPLETED'})
            data={'content_sha256':'fixed'}
            write_json(out/'training_identity.json',dict(binding={'data':data}))
            with patch.multiple(cli,OUT=out,RUN=run,COUNTS={'val':(1,1),'test':(1,1)}),patch.object(cli,'evaluation_source',return_value={'eval_commit':'fixture'}),patch.object(cli,'active_workers',return_value=[]),patch.object(cli,'has_tmux',return_value=False),patch.object(cli,'finalize_weights'),patch.object(cli,'checkpoint_identity'),patch.object(cli,'data_config',return_value=out/'data.yaml'),patch.object(cli,'cached_data',return_value=data),patch.object(cli,'inventory',side_effect=AssertionError('inventory')),patch.object(module,'QCCValidator',FakeValidator):
                cli.evaluate('val'); cli.evaluate('test')
                self.assertTrue(cli.evaluate('val')['reused'])
                self.assertTrue(cli.evaluate('test')['reused'])
                self.assertEqual(calls,['val','test'])
                a=cli.offline_analysis(); self.assertEqual(a['status'],'PASS')
                self.assertEqual(cli.offline_analysis(),a)
                lock=cli.read_json(out/'val_lock.json'); Path(lock['predictions']['path']).unlink()
                with self.assertRaisesRegex(RuntimeError,'recover-export'): cli.evaluate('val')
                self.assertEqual(calls,['val','test'])

    def test_interrupted_export_never_complete(self):
        from ultralytics.models.rtdetr.val import RTDETRValidator
        with tempfile.TemporaryDirectory() as name:
            obj=QCCValidator.__new__(QCCValidator)
            obj.export_path=Path(name)/'queries.jsonl.gz'; obj.partial=Path(name)/'queries.jsonl.gz.partial'
            obj.qcc_stream=gzip.open(obj.partial,'wt',encoding='utf-8')
            obj.qcc_stream.write('{"partial":true}\n')
            with patch.object(RTDETRValidator,'__call__',side_effect=RuntimeError('fixture interruption')):
                with self.assertRaisesRegex(RuntimeError,'fixture interruption'): obj()
            self.assertFalse(obj.export_path.exists())
            self.assertFalse(Path(str(obj.export_path)+'.complete.json').exists())
            self.assertTrue(obj.partial.exists())


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Operations))
    write_json(c.OUT/'ops_checks.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        scope='temporary fixtures only: canonical classes, reused manifests, binding, read-only status/pack',
        errors=[str(x) for x in result.errors+result.failures]))
    raise SystemExit(0 if result.wasSuccessful() else 1)
