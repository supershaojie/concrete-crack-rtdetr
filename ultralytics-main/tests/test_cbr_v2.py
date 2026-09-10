"""Boundary geometry, final-layer routing, and live original-box derivatives."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from check_triad_compat_gradients import cbr_check, activate
from ultralytics.nn.modules import StableReferenceCBR, RTDETRDecoderCBRv2
from ultralytics.nn.modules.cbr import CrackBoundaryRefinement


class CBRTests(unittest.TestCase):
    def test_firewall_and_evidence(self):cbr_check()

    def test_original_kernel_only_rho_value_changes(self):
        old=CrackBoundaryRefinement(256,256);new=StableReferenceCBR(256,256)
        activate(old);new.load_state_dict(old.state_dict());old.rho=.075
        p=torch.randn(1,256,8,12);q=torch.randn(1,19,256);b=torch.rand(1,19,4)*.8+.1
        torch.testing.assert_close(old(p,q,b),new(p,q,b),atol=0,rtol=0)
        self.assertEqual(new.sampling_grid(b).shape,(1,19,4,3,3,2))
        self.assertEqual(sum(p.numel() for p in new.parameters()),45889)

    def test_decoder_fourth_ref_and_final_only(self):
        head=RTDETRDecoderCBRv2(nc=1).eval();activate(head)
        x=[torch.randn(1,256,h,w) for h,w in ((20,24),(10,12),(5,6),(20,24))]
        with torch.no_grad():
            result,detail=head.forward_with_diagnostics(x)
            core=[t.clone() for t in x];core[0].add_(.4)
            # Hold query/box fixed: a different core feature cannot enter the CBR evidence input.
            p=x[3];q=torch.randn(1,17,256);b=torch.rand(1,17,4)*.5+.25
            a=head.cbr(p,q,b,True)[1]['evidence'];b1=head.cbr(core[3],q,b,True)[1]['evidence']
            self.assertTrue(torch.equal(a,b1))
            self.assertEqual(result[0].shape,(1,300,5))
        with self.assertRaises(ValueError):head(x[:3])
        with self.assertRaises(ValueError):head([x[1],x[0],x[2],x[3]])


if __name__=='__main__':unittest.main()
