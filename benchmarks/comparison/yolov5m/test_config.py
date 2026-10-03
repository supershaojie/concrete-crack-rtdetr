"""Configuration rejection, frozen provenance, and CLI lifecycle boundaries."""
from pathlib import Path
import tempfile
import unittest
from config import resolve, read_yaml, freeze_config, snapshot
from support import canonical, digest, sha256, write_json

class ConfigTests(unittest.TestCase):
    def test_environment_identity_is_stable_after_vendored_imports(self):
        from support import environment
        before=environment()
        import pkg_resources
        import IPython
        after=environment()
        self.assertEqual(before,after)

    def test_native_defaults_and_candidate(self):
        from support import HERE
        from config import native_hyp
        config=resolve(read_yaml(HERE/'v5_ft_01.yaml'))
        self.assertEqual(config['freeze'],[0])
        self.assertEqual(config['evaluation']['nms_iou'],.7)
        self.assertEqual(native_hyp(config),read_yaml(HERE/'hyp.yaml'))
        self.assertEqual((config['box'],config['cls'],config['obj']),(.05,.3,.7))

    def test_reject_invalid_fields_types_ranges_and_combinations(self):
        cases=[{'batch_size':16},{'unknown':1},{'epochs':'200'},{'batch':-1},{'imgsz':639},
            {'lr0':float('nan')},{'lr0':True},{'amp':'true'},{'device':'0,1'},
            {'cutmix':.1},{'copy_paste':.1},{'resume':True},{'freeze':[1]},
            {'optimizer':'MuSGD'},{'optimizer':'SGD','momentum':0},
            {'close_mosaic':201},{'rect':True},{'mosaic':0,'mixup':.5},
            {'device':'cpu'},{'cache':'disk'},{'momentum':1},{'save_period':0}]
        for case in cases:
            with self.subTest(case=case),self.assertRaises(ValueError):resolve(case)

    def test_duplicate_yaml_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.yaml';p.write_text('lr0: .003\nlr0: .01\n')
            with self.assertRaisesRegex(ValueError,'Duplicate'):read_yaml(p)

    def test_source_edit_move_and_new_run_are_isolated(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'candidate.yaml';source.write_text('lr0: .003\nmosaic: .5\n')
            def freeze(name):
                run=root/name;run.mkdir()
                config=freeze_config(source,run)
                write_json(run/'frozen.json',{'config_sha256':digest(canonical(config)),
                    'config_checksums':{n:sha256(run/n) for n in ('user_config.yaml','resolved_config.yaml','train_hyp.yaml')}})
                return run
            old=freeze('old')
            source.write_text('lr0: .006\nmosaic: .2\noptimizer: AdamW\n')
            new=freeze('new')
            source.rename(root/'moved.yaml')
            self.assertEqual(snapshot(old)['lr0'],.003)
            self.assertEqual(snapshot(new)['lr0'],.006)
            self.assertEqual(snapshot(new)['optimizer'],'AdamW')
            (old/'resolved_config.yaml').write_text((old/'resolved_config.yaml').read_text().replace('lr0: 0.003','lr0: 0.004'))
            with self.assertRaisesRegex(ValueError,'snapshot changed'):snapshot(old)

if __name__=='__main__':unittest.main(verbosity=2)
