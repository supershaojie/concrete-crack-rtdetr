"""Backend/schema/phase/freeze/official wheel regression; mocked calls are never CUDA evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from support import ROOT,HERE,SOURCE,configure,read_json,write_json,sha256,native_recipe
configure(ROOT/'outputs/yolov13l-flash-validation/backend-runtime')
import torch
from ultralytics.nn.modules import block
from ultralytics.models.yolo.detect import DetectionValidator
from adapters import CompleteValidator
from backend import attention_scope,native_fp32,resolve_train_backend,flash_forward_count,install_flash_observer,flash_evidence
from configuration import resolve_config,freeze_config,frozen_config,candidate,schema
from flash_install import select_wheel


class FlashTests(unittest.TestCase):
    def test_full_resolver_preset_differs_only_in_training_backend(self):
        cfg=resolve_config({})
        self.assertEqual(candidate(HERE/'configs/v13l_aug_x13_flash_01.yaml')[0],cfg)
        old=candidate(HERE/'configs/v13l_aug_x13_01.yaml')[0]
        self.assertEqual({k for k in cfg if cfg[k]!=old[k]},{'train_attention_backend'})
        self.assertEqual(cfg['train_attention_backend'],'flash');self.assertEqual(cfg['eval_attention_backend'],'native')
        self.assertEqual(schema(),read_json(HERE/'config_schema.json'))
        self.assertNotIn('train_attention_backend',native_recipe(cfg))
        for changes in ({'train_attention_backend':'sdpa'},{'train_attention_backend':'flash','amp':False},
                        {'deterministic':False},{'eval_attention_backend':'flash'}):
            with self.assertRaises(ValueError):resolve_config(changes)

    def test_strict_flash_failure_has_no_fallback_auto_records_cause(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'Strict Flash request failed; no fallback'):
                resolve_train_backend('flash')
            automatic=resolve_train_backend('auto')
            self.assertEqual(automatic['resolved'],'native');self.assertIn('CUDA',automatic['reason'])
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.get_device_capability',return_value=(8,9)),patch('backend.flash_identity',side_effect=ImportError('ABI mismatch')):
            with self.assertRaisesRegex(RuntimeError,'ABI mismatch'):resolve_train_backend('flash')
        self.assertFalse(block.USE_FLASH_ATTN)

    def test_native_fp32_restores_flash_flag_autocast_and_tf32_after_exception(self):
        original=(block.USE_FLASH_ATTN,torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        try:
            block.USE_FLASH_ATTN=True
            torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=True
            with self.assertRaisesRegex(ValueError,'validation failure'):
                with native_fp32():
                    self.assertFalse(block.USE_FLASH_ATTN)
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertFalse(torch.backends.cudnn.allow_tf32)
                    raise ValueError('validation failure')
            self.assertTrue(block.USE_FLASH_ATTN)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            block.USE_FLASH_ATTN,torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=original

    def test_real_validator_entry_scopes_native_and_restores_on_failure(self):
        previous=block.USE_FLASH_ATTN
        def failure(*args,**kwargs):
            self.assertFalse(block.USE_FLASH_ATTN)
            raise RuntimeError('validator interrupted')
        try:
            block.USE_FLASH_ATTN=True
            validator=CompleteValidator.__new__(CompleteValidator)
            with patch.object(DetectionValidator,'__call__',failure),self.assertRaisesRegex(RuntimeError,'validator interrupted'):
                validator(None)
            self.assertTrue(block.USE_FLASH_ATTN)
        finally:block.USE_FLASH_ATTN=previous

    def test_auto_resolution_is_frozen_and_backend_tamper_rejected(self):
        out=ROOT/'outputs/yolov13l-flash-validation';out.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            run=Path(tmp)/'auto_test'
            resolution={'requested':'auto','resolved':'native','reason':'UNIT_MOCK_NO_CUDA','deterministic':True,'flash_parity':'NOT_VERIFIED'}
            freeze_config(run,b'train_attention_backend: auto\n',{},run.name,require_clean=False,backend_resolution=resolution)
            with patch('backend.resolve_train_backend',side_effect=AssertionError('must not re-resolve')):
                self.assertEqual(frozen_config(run,check_code=False)['train_attention_backend'],'auto')
            write_json(run/'attention_backend_resolution.json',{**resolution,'resolved':'flash'})
            with self.assertRaisesRegex(ValueError,'Frozen run artifact'):frozen_config(run,check_code=False)

    def test_native_scope_cpu_math_remains_original_and_strict_flash_cpu_rejected(self):
        module=block.AAttn(64,2,area=4).float().eval()
        x=torch.randn(1,64,8,8,requires_grad=True)
        with native_fp32():
            y=module(x);y.square().mean().backward()
            self.assertTrue(torch.isfinite(y).all());self.assertTrue(torch.isfinite(x.grad).all())
        previous=block.USE_FLASH_ATTN
        try:
            block.USE_FLASH_ATTN=True
            with self.assertRaisesRegex(RuntimeError,'CPU attention tensor'):module(x.detach())
        finally:block.USE_FLASH_ATTN=previous

    def test_official_wheel_selection_uses_actual_abi_and_tags(self):
        target={'python_tag':'cp310','torch_tag':'2.2','cuda_tag':'cu12','abi':False,'platform':'linux_x86_64'}
        names=['flash_attn-2.7.3+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl',
               'flash_attn-2.7.3+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl',
               'flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl']
        release={'tag_name':'v2.7.3','assets':[{'name':name} for name in names]}
        self.assertEqual(select_wheel(release,target)['name'],names[0])
        self.assertEqual(select_wheel(release,{**target,'abi':True})['name'],names[1])
        self.assertIsNone(select_wheel(release,{**target,'python_tag':'cp39'}))

    def test_resume_identity_rejects_backend_change(self):
        from support import validate_checkpoint
        identity={'train_attention_backend':'flash','eval_attention_backend':'native'}
        checkpoint={'comparison_identity':{**identity,'train_attention_backend':'native'}}
        with self.assertRaisesRegex(ValueError,'identity differs'):
            validate_checkpoint(checkpoint,identity,resume=True)

    def test_readiness_rejects_recipe_environment_and_receipt_drift(self):
        from flash_checks import verify_readiness
        from support import canonical,digest,git,adapter_hash,LOCK
        out=ROOT/'outputs/yolov13l-flash-validation';out.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            report=Path(tmp)/'readiness.json';write_json(report,{'status':'UNIT_MOCK_ONLY'})
            cfg=resolve_config({})
            environment={'mock':'frozen'}
            pointer={'status':'PASSED_640_STARTUP_READINESS','code_sha':git('rev-parse','HEAD'),
                     'adapter_sha256':adapter_hash(),'patch_sha256':read_json(LOCK)['patch_sha256'],
                     'patched_source_sha256':read_json(LOCK)['patched_source_sha256'],
                     'config_sha256':digest(canonical(cfg)),'environment_identity_sha256':digest(canonical(environment)),
                     'data':str(Path(tmp).absolute()),'data_root':str(Path(tmp).absolute()),
                     'report':str(report),'report_sha256':sha256(report)}
            actual_read=read_json
            def snapshot_read(path):
                return actual_read(path) if Path(path)==LOCK else pointer
            with patch('flash_checks.read_json',side_effect=snapshot_read),patch('bootstrap.environment_identity',return_value=environment):
                verify_readiness(cfg,tmp,tmp)
                with self.assertRaises(ValueError):verify_readiness({**cfg,'lr0':.005},tmp,tmp)
                with patch('bootstrap.environment_identity',return_value={'mock':'changed'}),self.assertRaises(ValueError):
                    verify_readiness(cfg,tmp,tmp)
                report.write_bytes(b'tampered')
                with self.assertRaises(ValueError):verify_readiness(cfg,tmp,tmp)

    def test_observer_calls_supplied_function_and_records_dtype_without_tensors(self):
        # CPU fake function tests instrumentation plumbing only, NOT Flash execution.
        original=getattr(block,'flash_attn_func',None);flag=block.USE_FLASH_ATTN
        calls=[]
        def fake(q,k,v,*args,**kwargs):
            calls.append(kwargs);return q+k+v
        try:
            block.flash_attn_func=fake;block.USE_FLASH_ATTN=True
            install_flash_observer();before=flash_forward_count()
            x=torch.ones(1,2,1,32,dtype=torch.float16,requires_grad=True)
            y=block.flash_attn_func(x,x,x,dropout_p=0.0,causal=False,softmax_scale=32**-.5,deterministic=True)
            y.float().sum().backward()
            self.assertEqual(flash_forward_count(),before+1);self.assertTrue(calls[0]['deterministic'])
            self.assertFalse(flash_evidence()['tensor_references_retained'])
        finally:
            block.USE_FLASH_ATTN=flag
            if original is None:del block.flash_attn_func
            else:block.flash_attn_func=original


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
