"""SCI contract and diagnostic lifecycle tests without detection training."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from sci_adapter import SCIAdapter,ROOT,VARIANTS,build,topology,torch
from sci_adapter_stats import statistics,install


class AdapterTests(unittest.TestCase):
    def test_identity_and_constructor_rng(self):
        torch.manual_seed(12);before=torch.get_rng_state().clone();m=SCIAdapter()
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in m.parameters()),17152)
        self.assertGreater(int(m.reduce.weight.count_nonzero()),0)
        for training in (False,True):
            m.train(training);x=torch.randn(2,256,5,7,requires_grad=True);y=m(x)
            self.assertTrue(torch.equal(x,y));self.assertNotEqual(x.data_ptr(),y.data_ptr())
            g=torch.autograd.grad(y.sum(),x)[0];self.assertTrue(torch.equal(g,torch.ones_like(x)))
        with self.assertRaises(ValueError):SCIAdapter(hidden=64)

    def test_proxy_cannot_enter_concat_or_decoder(self):
        for variant in VARIANTS:
            m=build(variant);topology(m,variant)
            m.yaml['head'][12][0]=[18,19]
            with self.assertRaises(RuntimeError):topology(m,variant)

    def test_shell_registry_matches_python(self):
        script=(ROOT/'tools/autodl_sci_adapter.sh').read_text()
        self.assertIn('|'.join(VARIANTS)+')',script)
        sync=(ROOT/'tools/sync_sci_adapter.sh').read_text()
        self.assertIn('SCI_ADAPTER_BRANCH=codex/rtdetr-sci-adapter\n',sync)

    def test_diagnostics_first_train_batch_each_epoch(self):
        callbacks={}
        model=SimpleNamespace(add_callback=lambda name,fn:callbacks.setdefault(name,[]).append(fn))
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            folder=Path(directory);install(model,folder)
            adapter=SCIAdapter();trainer=SimpleNamespace(model=adapter,epoch=0)
            for fn in callbacks['on_train_start']:fn(trainer)
            for epoch in range(2):
                trainer.epoch=epoch
                for fn in callbacks['on_train_epoch_start']:fn(trainer)
                x=torch.randn(2,256,4,6)
                adapter.eval();adapter(x)  # validation must not consume the observation
                adapter.train();adapter(x);adapter(x*2)
                for fn in callbacks['on_train_epoch_end']:fn(trainer)
            records=[json.loads(s) for s in (folder/'sci_adapter_stats.jsonl').read_text().splitlines()]
            self.assertEqual([r['epoch'] for r in records],[0,1])
            self.assertTrue(all(r['residual_to_Y4_RMS']==0 for r in records))
            self.assertEqual(len(records[0]['channel_std_shift']),256)
            with torch.no_grad():adapter.restore.bias.fill_(.1)
            y=adapter(x);report=statistics(adapter,x,y)
            self.assertGreater(report['channel_mean_shift_mean_abs'],.099)
            self.assertGreater(report['residual_mean_abs'],.099)
            self.assertLess(report['channel_std_shift_mean_abs'],1e-6)


if __name__=='__main__':unittest.main()
