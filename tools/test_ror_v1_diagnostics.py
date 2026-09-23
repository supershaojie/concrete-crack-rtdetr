"""Regression checks for diagnostic failure retention, state isolation and cleanup."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
import check_ror_v1 as checks
import ror_v1 as entry
import ror_v1_fusion_diagnostic as diagnostic
from c19_lif_v1_diagnostic import DiagnosticError, fusion_protocol
from init_c19_lif_v1 import build


class DiagnosticsTests(unittest.TestCase):
    def test_partial_checks_survive_integration_exception(self):
        def fail(source, device, report, persist):
            report.update(status='RUNNING', phase='fusion', cuda_direct_step_variation_check=False,
                          resume_e=7, fusion_report='bounded/fuse_diagnostic.json')
            persist()
            raise DiagnosticError(dict(status='FAILED_REAL_NUMERICAL_MISMATCH', key='lif_input'))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'source.pt'; source.touch(); report = {}
            with patch.object(checks, 'ROOT', root), patch.object(checks, 'runtime', return_value={}), \
                    patch.object(checks, 'math_checks', return_value={'status': 'PASS'}), \
                    patch.object(checks, 'criterion_fixture', return_value={'status': 'PASS'}), \
                    patch.object(checks, 'integration', side_effect=fail):
                with self.assertRaises(DiagnosticError): checks.run(source, 'cpu', report, root / 'result.json')
            saved = json.loads((root / 'result.json').read_text())
            self.assertEqual(saved, report)
            self.assertEqual(saved['math']['status'], 'PASS')
            self.assertEqual(saved['criterion']['status'], 'PASS')
            self.assertEqual(saved['integration']['resume_e'], 7)
            self.assertIs(saved['integration']['cuda_direct_step_variation_check'], False)
            self.assertEqual(saved['failure']['key'], 'lif_input')
            self.assertEqual(saved['integration']['status'], 'FAIL')

    def test_preflight_retains_both_cpu_and_failed_cuda(self):
        def run(source, device, report):
            report['integration'] = dict(cuda_direct_step_variation_check=False, phase='fusion')
            if device != 'cpu': raise RuntimeError('injected fusion failure')
            report['status'] = 'PASS'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(entry, 'checked_plan', return_value=({'source': 'unused', 'output': root}, {'identity': {}})), \
                    patch.object(entry, 'environment', return_value={}), patch.object(checks, 'run', side_effect=run):
                with self.assertRaisesRegex(RuntimeError, 'injected'): entry.preflight(None)
            saved = json.loads((root / 'preflight.json').read_text())
            self.assertEqual(saved['cpu_checks']['status'], 'PASS')
            self.assertEqual(saved['checks']['integration']['phase'], 'fusion')
            self.assertIs(saved['checks']['integration']['cuda_direct_step_variation_check'], False)

    def test_backend_restored_after_exception(self):
        before = diagnostic.backend()
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with diagnostic.isolated_backend(True):
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                raise RuntimeError('injected')
        self.assertEqual(diagnostic.backend(), before)

    def test_exact_failure_fixture_selection_and_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); target = root / 'outputs/ror_v1/checks/fusion_cuda_0'
            target.mkdir(parents=True)
            fixture = target / 'fixture.pt'
            payload = dict(precision='fp32', nc=1, yaml={}, image=torch.zeros(2, 3, 160, 160),
                           unfused={'w': torch.ones(1)}, fused={'w': torch.ones(1)}, rng={})
            torch.save(payload, fixture)
            prior = dict(precision='fp32', device='cuda', failure={'key': 'lif_input'}, runtime={}, precision_path={},
                         fixture={'sha256': diagnostic.sha256(fixture)})
            (target / 'fuse_diagnostic.json').write_text(json.dumps(prior))
            loaded = {}
            fake = SimpleNamespace(model=[SimpleNamespace()], criterion=SimpleNamespace(set_epoch=lambda e: None),
                                   load_state_dict=lambda value, strict: loaded.update(state=value, strict=strict))
            with patch.object(diagnostic, 'ROOT', root), patch.object(diagnostic, 'RTDETRDetectionModel', return_value=fake), \
                    patch.object(diagnostic, 'install', side_effect=lambda model, identity: model):
                model, image, info, state = diagnostic.load_fixture(Path('auto'), 'cuda:0')
                self.assertEqual(info['kind'], 'exact_failed_fixture')
                self.assertEqual(info['path'], str(fixture.resolve()))
                self.assertTrue(loaded['strict'])
                self.assertTrue(torch.equal(loaded['state']['w'], payload['unfused']['w']))
                with fixture.open('ab') as stream: stream.write(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'hash'): diagnostic.load_fixture(fixture, 'cuda:0')

    def test_real_fusion_isolation_and_compact_failure(self):
        torch.set_num_threads(4)
        model = build(nc=1).eval()
        before = diagnostic.state_hash(model.state_dict())
        fused, audit = diagnostic.checked_fuse(model)
        self.assertEqual(before, diagnostic.state_hash(model.state_dict()))
        self.assertTrue(audit['lif_bn_and_state_exact'])
        self.assertTrue(all(not m.training for m in fused.modules()))
        with patch.object(diagnostic, 'deepcopy', return_value=model):
            with self.assertRaisesRegex(RuntimeError, 'aliases'): diagnostic.checked_fuse(model)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'failure'
            failure = DiagnosticError(dict(status='FAILED_REAL_NUMERICAL_MISMATCH', key='lif_input', phase='natural_trace'))
            with patch('c19_lif_v1_probe.capture', side_effect=failure), \
                    patch.object(torch, 'save', side_effect=AssertionError('Must not save a giant fixture')):
                with self.assertRaises(DiagnosticError):
                    fusion_protocol(model, fused, torch.zeros(2, 3, 160, 160), target, 'cpu', save_failure_tensors=False)
            saved = json.loads((target / 'fuse_diagnostic.json').read_text())
            self.assertEqual(saved['failure']['key'], 'lif_input')
            self.assertEqual(saved['acceptance'], 'BLOCKED')
            self.assertFalse(saved['records']['failure_tensors_enabled'])
            self.assertEqual(list(target.glob('*.pt')), [])


if __name__ == '__main__': unittest.main()
