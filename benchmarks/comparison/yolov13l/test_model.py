"""Actual author model migration, protected tensors, arithmetic and preflight isolation."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from support import ROOT, SOURCE, ASSET, configure, read_json, recipe, native_recipe, write_json, sha256
configure(ROOT/'outputs/yolov13l-configurable-validation/model-runtime',SOURCE)
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from adapters import model_identity, transfer_report, tensor_digest, capture_rng
from backend import ArithmeticAudit, native_fp32
from configuration import resolve_config
from export import verify_arithmetic_receipt
from run import load_initial_model
from smoke import fixture
from verification import preflight_model_probe


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model,cls.record=load_initial_model()
        cls.original=YOLO(str(ASSET),task='detect').model.float()

    def test_measured_actual_L_graph_and_class_only_migration(self):
        measured=read_json(Path(__file__).parent/'initialization.lock.json')
        self.assertEqual(self.record['coco']['parameters_unfused'],measured['coco']['parameters_unfused'])
        self.assertEqual(self.record['crack']['parameters_unfused'],measured['crack']['parameters_unfused'])
        self.assertEqual(self.record['transferred_keys'],measured['transferred_keys'])
        self.assertEqual(self.record['skip_reasons'],measured['skip_reasons'])
        self.assertEqual(self.record['matching_tensor_sha256'],measured['matching_tensor_sha256'])
        self.assertEqual(model_identity(self.model,1)['scale'],'l')
        self.assertTrue(all(value != 0 for value in self.record['FullPAD_gate_values_after_transfer'].values()))

    def test_reset_pretrained_gate_and_foreign_scale_are_rejected(self):
        gate=self.model.model[12].gate
        original=gate.detach().clone()
        try:
            with torch.no_grad(): gate.zero_()
            with self.assertRaisesRegex(ValueError,'transferred exactly'):
                transfer_report(self.original,self.model)
        finally:
            with torch.no_grad(): gate.copy_(original)
        with patch.dict(self.model.yaml,scale='s'):
            with self.assertRaisesRegex(ValueError,'full graph'): model_identity(self.model,1)

    def test_class_output_mismatch_must_be_80_to_1_only(self):
        with patch('adapters.intersect_dicts',return_value={}):
            with self.assertRaisesRegex(ValueError,'Unexpected missing'):
                transfer_report(self.original,self.model)

    def test_native_fp32_executes_internal_attention_matmuls(self):
        probe=deepcopy(self.model).float().eval()
        with native_fp32(), torch.no_grad(), ArithmeticAudit(require_fp32=True) as audit:
            values=probe(torch.zeros(1,3,64,64))[0]
        report=audit.report()
        self.assertTrue(torch.isfinite(values).all())
        self.assertGreater(report['MACs_by_operator']['bmm_attention_and_hypergraph'],0)
        self.assertTrue(all(k.endswith('|torch.float32') for k in report['observed_dtypes']))
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_preflight_copy_preserves_all_weights_BN_and_RNG(self):
        output=ROOT/'outputs/yolov13l-configurable-validation'
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            _,run,manifest=fixture(Path(tmp))
            before=capture_rng()
            state={k:tensor_digest(v) for k,v in self.model.state_dict().items()}
            result=preflight_model_probe(self.model,{**recipe(),'train_attention_backend':'native'},run,manifest,device='cpu')
            after=capture_rng()
            self.assertEqual(before['python'],after['python'])
            self.assertEqual(before['numpy'][0],after['numpy'][0])
            np.testing.assert_array_equal(before['numpy'][1],after['numpy'][1])
            self.assertEqual(before['numpy'][2:],after['numpy'][2:])
            self.assertTrue(torch.equal(before['torch_cpu'],after['torch_cpu']))
            self.assertEqual(len(before['torch_cuda']),len(after['torch_cuda']))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(before['torch_cuda'],after['torch_cuda'])))
            self.assertEqual(state,{k:tensor_digest(v) for k,v in self.model.state_dict().items()})
            self.assertTrue(result['sample']['non_square_source'])
            self.assertEqual(result['cuda_amp_accuracy'],'NOT_RUN')
            self.assertEqual(result['formal_optimizer_updates'],0)

    def test_backend_and_epoch_limit_are_validated_before_model_creation(self):
        self.assertEqual(resolve_config({})['train_attention_backend'],'flash')
        for key,value in (('train_attention_backend','invalid'),('eval_attention_backend','flash'),('epochs',201)):
            with self.subTest(key=key),self.assertRaises(ValueError): resolve_config({key:value})
        native=vars(get_cfg(overrides=native_recipe()))
        self.assertNotIn('cutmix',native)
        self.assertNotIn('train_attention_backend',native)

    def test_precision_receipt_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            report={'required_fp32':True,'autocast':{'cpu':False,'cuda':False},
                    'parameter_dtype':'torch.float32','tf32':{'matmul':False,'cudnn':False}}
            path=run/'predictions/val_fp32_arithmetic.json'
            write_json(path,report)
            done={'attention_backend':'native','audit_sha256':sha256(path)}
            verify_arithmetic_receipt(run,'val',done)
            write_json(path,{**report,'parameter_dtype':'torch.float16'})
            with self.assertRaises(ValueError): verify_arithmetic_receipt(run,'val',done)


if __name__=='__main__': unittest.main(verbosity=2)
