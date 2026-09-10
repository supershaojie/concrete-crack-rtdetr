"""Graph interventions, semantic drift guards and unchanged C2 core files."""
import sys
from pathlib import Path
import unittest
from copy import deepcopy
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from triad_compat import *
from check_triad_compat import routing_interventions, feature_forward
from check_triad_compat_gradients import activate


class TriadTests(unittest.TestCase):
    def test_all_registry_graphs(self):
        for v in VARIANTS:topology(build(v),v)

    def test_cscef_late_no_pan_pollution(self):
        routing_interventions(build('triad_v1'),'triad_v1')

    def test_topdown_intervention_cannot_change_cscef_residual(self):
        m=build('triad_v1').eval();activate(m);roles=topology(m,'triad_v1')['roles']
        image=torch.rand(1,3,160,192)
        with torch.no_grad():
            _,before,_=feature_forward(m,image,roles)
            # Change AIFI output while fixing ONLY P3_base. Backbone refs are untouched naturally.
            hooks=[m.model[roles['aifi']].register_forward_hook(lambda m,a,o:o+torch.randn_like(o)*.3),
                m.model[roles['P3_base']].register_forward_hook(lambda m,a,o:before['P3_base'])]
            try:_,after,_=feature_forward(m,image,roles)
            finally:
                for h in hooks:h.remove()
        self.assertTrue(torch.equal(before['P3_ref'],after['P3_ref']))
        self.assertTrue(torch.equal(before['P4_ref'],after['P4_ref']))
        self.assertTrue(torch.equal(before['P3_enh'],after['P3_enh']))
        self.assertFalse(torch.equal(before['P4_base'],after['P4_base']))

    def test_reject_final_p3_as_cbr_ref(self):
        m=build('triad_v1');m.yaml['head'][-1][0][-1]=m.yaml['head'][-1][0][0]
        with self.assertRaises(RuntimeError):topology(m,'triad_v1')

    def test_c2_core_unchanged(self):
        files=['ultralytics-main/ultralytics/nn/modules/head.py','ultralytics-main/ultralytics/models/utils/loss.py',
            'ultralytics-main/ultralytics/models/utils/ops.py','ultralytics-main/ultralytics/models/rtdetr/train.py',
            'ultralytics-main/ultralytics/models/rtdetr/val.py','ultralytics-main/ultralytics/engine/trainer.py',
            'ultralytics-main/ultralytics/cfg/models/rt-detr/'+BASE_YAML]
        for path in files:self.assertEqual(git('diff',BASE_COMMIT,'--',path),'',path)


if __name__=='__main__':torch.set_num_threads(4);unittest.main()
