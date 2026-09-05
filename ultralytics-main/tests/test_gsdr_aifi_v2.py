# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Regression checks for C16 naming, RNG isolation, and CUDA attention arithmetic.

Run from the repository root with its ultralytics-main on PYTHONPATH.
These tests do not train or evaluate the crack dataset.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import AIFI, GSDRAIFIV2, SparseDeformableRelationV2
from ultralytics.nn.modules.gsdr_aifi import SparseDeformableRelation
from ultralytics.nn.tasks import RTDETRDetectionModel


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / 'ultralytics-main/ultralytics/cfg/models/rt-detr'
BASE = MODELS / 'rtdetr-resnet18-lite.yaml'
V2 = MODELS / 'rtdetr-resnet18-lite-gsdr-aifi-v2.yaml'


class GSDRV2RegressionTest(unittest.TestCase):
    def test_complete_model_construction_preserves_c2_states_and_rng(self):
        for nc in (1, 80):
            with self.subTest(nc=nc):
                torch.manual_seed(42)
                base = RTDETRDetectionModel(str(BASE), ch=3, nc=nc, verbose=False)
                expected_rng = torch.get_rng_state().clone()
                torch.manual_seed(42)
                candidate = RTDETRDetectionModel(str(V2), ch=3, nc=nc, verbose=False)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                common, actual = base.state_dict(), candidate.state_dict()
                self.assertEqual(len(common), 533)
                self.assertEqual(len(actual), 564)
                for key, value in common.items():
                    self.assertTrue(torch.equal(value, actual[key]), key)
                new_keys = set(actual) - set(common)
                self.assertEqual(len(new_keys), 31)
                self.assertTrue(all(k.startswith('model.9.sparse_relation.') for k in new_keys))
                del base, candidate

    def test_real_trainer_optimizer_routes_position_weights_as_weights(self):
        model = RTDETRDetectionModel(str(V2), ch=3, nc=1, verbose=False)
        trainer = RTDETRTrainer.__new__(RTDETRTrainer)
        trainer.args = SimpleNamespace(warmup_bias_lr=0.1)
        optimizer = trainer.build_optimizer(model, name='AdamW', lr=0.0005, momentum=0.937, decay=0.0001)
        groups = {g['param_group']: g for g in optimizer.param_groups}
        self.assertEqual({k: len(g['params']) for k, g in groups.items()},
                         {'weight': 132, 'bn': 82, 'bias': 143})
        membership = {id(p): name for name, g in groups.items() for p in g['params']}
        count = 0
        for name, parameter in model.named_parameters():
            if '.position_mlp.' in name and name.endswith('.weight'):
                self.assertEqual(membership[id(parameter)], 'weight', name)
                count += 1
        self.assertEqual(count, 8)
        self.assertEqual(groups['weight']['weight_decay'], 0.0001)
        self.assertEqual(groups['bias']['weight_decay'], 0.0)
        self.assertEqual(trainer.args.warmup_bias_lr, 0.1)

    def test_zero_branch_preserves_both_aifi_normalization_orders(self):
        for pre in (False, True):
            with self.subTest(normalize_before=pre):
                base = AIFI(32, 64, 4, dropout=0, normalize_before=pre).eval()
                candidate = GSDRAIFIV2(32, 64, 4, aux_dim=32, normalize_before=pre).eval()
                result = candidate.load_state_dict(base.state_dict(), strict=False)
                self.assertFalse(result.unexpected_keys)
                self.assertTrue(all(k.startswith('sparse_relation.') for k in result.missing_keys))
                with torch.no_grad():
                    x = torch.randn(2, 32, 5, 7)
                    self.assertTrue(torch.equal(base(x), candidate(x)))

    def test_fp32_relation_algebra_is_unchanged_after_name_mapping(self):
        original = SparseDeformableRelation(32, 32, 4, 4).eval()
        candidate = SparseDeformableRelationV2(32, 32, 4, 4).eval()
        with torch.no_grad():
            original.output_proj.weight.normal_(std=0.02)
            original.offset_out.weight.normal_(std=0.02)
            for mlp in original.relative_bias:
                mlp.fc2.weight.normal_(std=0.02)
        candidate.load_state_dict({k.replace('relative_bias.', 'position_mlp.'): v
                                   for k, v in original.state_dict().items()}, strict=True)
        for shape in ((5,7), (20,20), (1,5), (5,1), (1,1)):
            with self.subTest(shape=shape), torch.no_grad():
                x = torch.randn(2, 32, *shape)
                self.assertTrue(torch.equal(original(x), candidate(x)))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable: CUDA arithmetic is unverified')
    def test_cuda_amp_matmul_outputs_really_are_fp32(self):
        relation = SparseDeformableRelationV2(32, 32, 4, 4).cuda()
        observed = []
        original_matmul = torch.matmul

        def recorded(*args, **kwargs):
            output = original_matmul(*args, **kwargs)
            observed.append(output.dtype)
            return output

        with torch.no_grad():
            relation.output_proj.weight.normal_(std=0.02)
        x = torch.randn(2, 32, 5, 7, device='cuda', requires_grad=True)
        with patch('torch.matmul', side_effect=recorded):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y = relation(x)
        self.assertEqual(observed, [torch.float32, torch.float32])
        self.assertTrue(torch.isfinite(y).all().item())
        y.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertTrue(torch.isfinite(relation.output_proj.weight.grad).all().item())


if __name__ == '__main__':
    unittest.main(verbosity=2)
