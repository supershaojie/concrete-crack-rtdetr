"""Stable-reference CSCEF contract tests."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from check_triad_compat_gradients import cscef_check, activate
from ultralytics.nn.modules import DRCSCEFv6


class CSCEFTests(unittest.TestCase):
    def test_firewall_staging_and_batch(self):
        cscef_check()

    def test_small_boundaries_and_shape_validation(self):
        m=DRCSCEFv6(256,256,256);activate(m)
        for h,w in ((1,1),(1,7),(8,1),(4,6)):
            x=torch.randn(2,256,h,w);s=torch.randn(2,256,1,1)
            self.assertTrue(torch.isfinite(m([x,x,s])).all())
        with self.assertRaises(ValueError):m([x,s])
        with self.assertRaises(ValueError):DRCSCEFv6(128,256,256)


if __name__=='__main__':unittest.main()
