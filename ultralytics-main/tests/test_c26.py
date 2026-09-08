"""C26 topology and backwards-compatible final-query contract, without training a model."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from init_c26 import build, verify_model, verify_sources, MODEL_DIR, MODELS
from init_scca import verify_model as verify_old
from ultralytics.nn.modules.transformer import DeformableTransformerDecoder
from ultralytics.utils import YAML


class Step(nn.Module):
    def forward(self, embed, *args, num_queries=None, dn_meta=None):
        assert num_queries == 7 and dn_meta == {"sentinel": True}
        return embed + 1


class TestC26(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)

    def test_only_two_yaml_substitutions(self):
        c19 = YAML.load(MODEL_DIR / MODELS["c19"])
        c24 = YAML.load(MODEL_DIR / MODELS["c24"])
        c26 = YAML.load(MODEL_DIR / MODELS["c26"])
        c19["head"][1][2] = "SCCAAIFI"
        c24["head"][-1][2] = "RTDETRDecoderCBR"
        self.assertEqual(c26, c19)
        self.assertEqual(c26, c24)

    def test_final_query_matches_last_emitted_layer_and_legacy_tuple(self):
        for training in (True, False):
            for eval_idx in (0, 1, 2):
                decoder = DeformableTransformerDecoder(4, Step(), 3, eval_idx).train(training)
                embed, boxes = torch.zeros(1, 7, 4), torch.zeros(1, 7, 4)
                heads = nn.ModuleList([nn.Identity() for _ in range(3)])
                args = (embed, boxes, torch.zeros(1, 1, 4), [[1, 1]], heads, heads, nn.Identity())
                # Existing positional num_queries/dn_meta remain in the same slots.
                legacy = decoder(*args, None, None, 7, {"sentinel": True})
                extended = decoder(*args, None, None, 7, {"sentinel": True}, return_final_query=True)
                self.assertEqual(len(legacy), 2)
                self.assertEqual(len(extended), 3)
                torch.testing.assert_close(legacy[0], extended[0], atol=0, rtol=0)
                torch.testing.assert_close(legacy[1], extended[1], atol=0, rtol=0)
                self.assertTrue(torch.equal(extended[2], extended[1][-1]))
                self.assertTrue(torch.equal(extended[2], torch.full_like(embed, 3 if training else eval_idx + 1)))

    def test_strict_c26_and_old_c24_checks_remain_distinct(self):
        model = build(nc=1)
        verify_sources()
        verify_model(model, zero=True)
        with self.assertRaises(RuntimeError):
            verify_old(model, "c24")
        model.model[-1].f = [20, 23, 26]
        with self.assertRaises(RuntimeError):
            verify_model(model)
        c24 = build("c24", nc=1)
        self.assertTrue(all(torch.equal(value, model.state_dict()[name]) for name, value in c24.state_dict().items()))
        verify_old(c24, "c24", zero=True)
        with self.assertRaises(RuntimeError):
            verify_model(c24)
        c19 = build("c19", nc=1)
        self.assertTrue(all(torch.equal(value, model.state_dict()[name]) for name, value in c19.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
