"""Targeted regression checks. Tiny CUDA fixtures are NOT server capacity proof."""
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lbc_v1_training import LBCTrainer, epoch_end
from lbc_v1_checks import bare_trainer
from lbc_v1_diagnostics import configure_capacity_scaler, UpdateDiagnostics, require_valid_attempts
from lbc_v1_reporting import write_json, finalize_json
from ultralytics.models.rtdetr.lbc import LBCHead, regions, score_loss
from ultralytics.utils.torch_utils import autocast, TORCH_2_4
import torch
from torch import nn


def strict_read(path):
    def reject(value):
        raise AssertionError('Nonstandard JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


class SmallModel(nn.Module):
    """Real LBC head/loss; synthetic backbone and labels, no RT-DETR capacity claim."""
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.branch2a = nn.Module()
        block.branch2a.conv = nn.Conv2d(3, 128, 3, stride=2, padding=1, bias=False)
        stage = nn.Module()
        stage.blocks = nn.ModuleList([block])
        self.model = nn.ModuleList([nn.Identity() for _ in range(5)] + [stage])
        self.lbc_head = LBCHead()

    def forward(self, batch):
        feature = self.model[5].blocks[0].branch2a.conv(batch['img'])
        pairs, stats, _ = regions(batch, feature.shape[-2:])
        auxiliary, detail = score_loss(self.lbc_head(feature), pairs)
        self.lbc_last = dict(stats, **detail, feature_hw=list(feature.shape[-2:]),
                             weighted=float(auxiliary.detach())*.05)
        loss = feature.float().square().mean() + .05*auxiliary
        return loss, loss.detach().reshape(1)


def small_batch():
    return dict(img=torch.rand(2, 3, 32, 32, device='cuda'),
                bboxes=torch.tensor([[.5, .5, .3, .3]]*2, device='cuda'),
                cls=torch.zeros(2, 1, device='cuda'), batch_idx=torch.arange(2, device='cuda'))


def trainer_fixture(diagnostic=True):
    torch.manual_seed(724)
    trainer = bare_trainer(SmallModel().cuda())
    trainer.amp = True
    trainer.scaler = torch.amp.GradScaler('cuda', enabled=True) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=True)
    if diagnostic:
        trainer.scaler_configuration = configure_capacity_scaler(trainer)
        trainer.lbc_update_diagnostics = UpdateDiagnostics()
    return trainer


def window(trainer, overflow=False):
    batch = small_batch()
    for _ in range(4):
        trainer._lbc_ni += 1
        with autocast(True):
            loss, _ = trainer.model(batch)
        trainer.scaler.scale(loss).backward()
    if overflow:
        # Deliberate test fault only; production never injects or repairs gradients.
        trainer.model.lbc_head.prototype.grad[0] = float('inf')
    trainer.optimizer_step()


class ReportingChecks(unittest.TestCase):
    def test_nonfinite_json_and_epoch_callback(self):
        value = dict(status='FAILED', error='original calculation failed',
                     nested=[float('nan'), float('inf'), float('-inf')],
                     **{'a/b~c': float('inf')})
        with tempfile.TemporaryDirectory() as directory:
            for name in ('bounded.json', 'bounded_progress.json'):
                path = Path(directory)/name
                write_json(path, value)
                saved = strict_read(path)
                self.assertEqual(saved['status'], 'FAILED')
                self.assertEqual(saved['error'], value['error'])
                self.assertEqual(saved['nonfinite_fields'], [
                    dict(path='/nested/0', value='NaN'), dict(path='/nested/1', value='+Inf'),
                    dict(path='/nested/2', value='-Inf'), dict(path='/a~1b~0c', value='+Inf')])
            trainer = SimpleNamespace(epoch=20, lbc_epoch_rows=[dict(pairs=1, raw_pair=float('nan'))],
                lbc_effective_updates=0, lbc_overflow_skips=1, lbc_output=directory,
                lbc_clip=dict(original=float('inf'), head=float('-inf')))
            epoch_end(trainer)
            saved = strict_read(Path(directory)/'epochs.jsonl')
            self.assertEqual(saved['clip_last']['head'], {'__nonfinite_float__': '-Inf'})
            self.assertEqual(saved['effective_updates'], 0)
        self.assertTrue(math.isnan(value['nested'][0]))
        self.assertEqual(value['nested'][1:], [float('inf'), float('-inf')])
        self.assertTrue(math.isinf(trainer.lbc_clip['original']))

    def test_original_exception_survives_report_io_failure(self):
        original = RuntimeError('original computation sentinel')
        with patch('lbc_v1_reporting.write_json', side_effect=OSError('write sentinel')):
            with patch('sys.stderr', new_callable=io.StringIO) as output:
                try:
                    try:
                        raise original
                    finally:
                        finalize_json('bounded.json', dict(status='FAILED', clip=float('inf')))
                except RuntimeError as caught:
                    self.assertIs(caught, original)
                else:
                    self.fail('Computation failure hidden')
                self.assertIn('write sentinel', output.getvalue())
            with self.assertRaisesRegex(OSError, 'write sentinel'):
                finalize_json('bounded.json', dict(status='PASSED'))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required; never substitute CPU for AMP proof')
class CudaChecks(unittest.TestCase):
    def test_native_overflow_skip_then_finite_updates(self):
        trainer = trainer_fixture()
        before = deepcopy(trainer.model.state_dict())
        window(trainer, overflow=True)
        row = trainer.lbc_update_diagnostics.attempts[-1]
        self.assertEqual((row['scale_before'], row['scale_after']), (128., 64.))
        self.assertEqual((trainer.lbc_effective_updates, trainer.lbc_overflow_skips), (0, 1))
        self.assertTrue(all(torch.equal(v, trainer.model.state_dict()[n]) for n, v in before.items()))
        self.assertTrue(row['original_gradients']['finite'])
        self.assertFalse(row['head_gradients']['finite'])
        self.assertIn('lbc_head.prototype', row['head_gradients']['nonfinite_parameters'])
        for _ in range(2):
            window(trainer)
            row = trainer.lbc_update_diagnostics.attempts[-1]
            self.assertTrue(row['effective_update'] and row['head_updated'] and row['backbone_updated'])
            self.assertTrue(row['original_gradients']['finite'] and row['head_gradients']['finite'])
            self.assertTrue(row['parameters_finite'])
            self.assertEqual(row['accumulated_microbatches'], 4)
        self.assertEqual((trainer.lbc_effective_updates, trainer.lbc_overflow_skips), (2, 1))
        require_valid_attempts(trainer.lbc_update_diagnostics.attempts)
        rejected = deepcopy(trainer.lbc_update_diagnostics.attempts[-1:])
        rejected[0]['clip_pre_norms']['head'] = float('inf')
        with self.assertRaisesRegex(RuntimeError, 'finite gradients/norms'):
            require_valid_attempts(rejected)
        EVIDENCE['native_cuda_updates'] = trainer.lbc_update_diagnostics.attempts

    def test_observation_does_not_change_updates(self):
        observed = trainer_fixture()
        plain = trainer_fixture(diagnostic=False)
        plain.scaler.load_state_dict(deepcopy(observed.scaler.state_dict()))
        for trainer in (observed, plain):
            torch.manual_seed(901)
            window(trainer)
        self.assertTrue(all(torch.equal(v, plain.model.state_dict()[n])
                            for n, v in observed.model.state_dict().items()))
        self.assertEqual(observed.scaler.state_dict(), plain.scaler.state_dict())

    def test_default_and_native_checkpoint_scaler_restoration(self):
        formal = trainer_fixture(diagnostic=False)
        original = deepcopy(formal.scaler.state_dict())
        diagnostic = trainer_fixture()
        self.assertEqual(original['scale'], 65536.)
        self.assertEqual(diagnostic.scaler_configuration['native_default_init_scale'], 65536.)
        self.assertEqual(diagnostic.scaler.get_scale(), 128.)
        self.assertEqual(formal.scaler.state_dict(), original)
        window(diagnostic, overflow=True)
        window(diagnostic)
        saved = deepcopy(diagnostic.scaler.state_dict())
        self.assertEqual(saved['scale'], 64.)
        self.assertEqual(saved['_growth_tracker'], 1)
        # Exercise the actual inherited checkpoint restoration used by native resume.
        formal._load_checkpoint_state(dict(scaler=saved, optimizer=None, ema=None))
        self.assertEqual(formal.scaler.state_dict(), saved)
        self.assertFalse(hasattr(formal, 'lbc_update_diagnostics'))
        formal.resume = True
        with self.assertRaisesRegex(RuntimeError, 'fresh isolated trainer'):
            configure_capacity_scaler(formal)
        self.assertEqual(formal.scaler.state_dict(), saved)
        self.assertEqual(torch.cuda.amp.GradScaler(enabled=True).get_scale(), 65536.)
        # Cover the legacy constructor used on the pinned torch 2.1 server too.
        legacy = trainer_fixture(diagnostic=False)
        legacy.scaler = torch.cuda.amp.GradScaler(enabled=True)
        configure_capacity_scaler(legacy)
        self.assertEqual(legacy.scaler.get_scale(), 128.)
        EVIDENCE['scaler_isolation'] = dict(default=original, diagnostic=diagnostic.scaler_configuration,
                                           checkpoint_restored_exactly=saved)

    def test_bounded_persistent_overflow_preserves_failure(self):
        from lbc_v1_preflight import bounded_check

        class OverflowFixture(LBCTrainer):
            def __init__(self, overrides):
                fixture = trainer_fixture(diagnostic=False)
                self.__dict__.update(fixture.__dict__)
                self.train_loader = [small_batch(), small_batch()]
                self.lbc_optimizer_groups = []
                self.lbc_model_audit = dict(scope='synthetic regression fixture')
                self.lbc_epoch_rows = []
                for parameter in self.model.parameters():
                    parameter.register_hook(lambda grad: torch.full_like(grad, float('inf')))

            def _setup_train(self):
                pass

            def preprocess_batch(self, batch):
                self._lbc_ni += 1
                return batch

        with tempfile.TemporaryDirectory() as directory:
            with patch('lbc_v1_preflight.LBCTrainer', OverflowFixture), patch(
                    'lbc_v1_preflight.visualize', return_value=dict(counts=dict(pairs=2, gt=2))):
                with self.assertRaisesRegex(RuntimeError, 'Capacity check incomplete'):
                    bounded_check(dict(batch=2, imgsz=32, amp=True), directory, server=True)
            report = strict_read(Path(directory)/'bounded.json')
            self.assertEqual(report, strict_read(Path(directory)/'bounded_progress.json'))
            self.assertEqual(report['status'], 'FAILED')
            self.assertEqual((report['microbatches'], report['effective_updates'], report['overflow_skips']), (16, 0, 4))
            self.assertEqual(len(report['update_attempts']), 4)
            self.assertIn('Capacity check incomplete', report['error'])
            self.assertIn('RuntimeError', report['traceback'])
            self.assertFalse(report['head_updated'] or report['backbone_updated'])
            self.assertTrue(report['nonfinite_fields'])
            self.assertEqual(report['scaler_configuration']['reinitializations'], 1)
            EVIDENCE['bounded_failure_fixture'] = dict(status=report['status'], error=report['error'],
                microbatches=report['microbatches'], effective_updates=report['effective_updates'],
                overflow_skips=report['overflow_skips'], attempts=report['update_attempts'])


EVIDENCE = {}
if __name__ == '__main__':
    torch.set_num_threads(4)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if len(sys.argv) == 2:
        from datetime import datetime, timezone
        import hashlib
        sources = ('lbc_v1.py', 'lbc_v1_training.py', 'lbc_v1_preflight.py',
                   'lbc_v1_precision.py', 'lbc_v1_reporting.py', 'lbc_v1_diagnostics.py',
                   'lbc_v1_preflight_checks.py')
        fingerprints = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                        for name in sources}
        write_json(sys.argv[1], dict(status='PASSED' if result.wasSuccessful() else 'FAILED',
            created=datetime.now(timezone.utc).isoformat(), tests_run=result.testsRun,
            skipped=len(result.skipped), failures=[text for _, text in result.failures+result.errors],
            server_validation='服务器待验证', formal_training='NOT_RUN',
            scope='Targeted synthetic regression fixtures; no RTX4090/B16/640 capacity result',
            source_sha256_lf=fingerprints,
            runtime=dict(python=sys.version, torch=str(torch.__version__),
                         gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            evidence=EVIDENCE))
    sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
