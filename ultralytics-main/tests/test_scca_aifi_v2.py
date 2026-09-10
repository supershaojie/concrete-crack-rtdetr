"""C24 forward identity with independent input derivatives."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from check_triad_compat_gradients import scca_check, activate
from ultralytics.nn.modules import GISCCAAIFI


class SCCATests(unittest.TestCase):
    def test_forward_pre_post_and_firewall(self):scca_check()

    def test_constant_and_single_token(self):
        m=GISCCAAIFI(256);activate(m)
        for n in (1,9):
            x=torch.ones(2,n,256);d,a=m.scca_channel(x,x)
            self.assertTrue(torch.isfinite(d).all())
            torch.testing.assert_close(a,torch.full_like(a,1/16))
        self.assertEqual(sum(p.numel() for n,p in m.named_parameters() if n.startswith('scca_')),65540)


if __name__=='__main__':unittest.main()
