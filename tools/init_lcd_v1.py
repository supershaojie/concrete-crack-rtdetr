"""Original strict pair audit first; fixed LCD construction and native nc adaptation second."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
from pathlib import Path

import torch
import init_c19_lif_v1 as mother
from init_c19_lif_v1 import ROOT, MODEL_DIR, SOURCE_SHA256, require, sha256, write_json
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import BlocksLCD, LCD, LIFDown, RTDETRDecoderCBR
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load

BASE = 'a0459d6a652cb702699087c88fa39a3e4c4087ec'
BRANCH = 'exp-rtdetr-r18-lite-lcd-v1'
MODEL = MODEL_DIR / 'rtdetr-resnet18-lite-cbr-lif-lcd.yaml'
PREFIX = 'model.5.lcd.'
NEW_KEYS = {PREFIX + k for k in ('P.weight', 'norm.weight', 'norm.bias', 'shared_dw.weight', 'G.weight', 'O.weight')}
RESEARCH = dict(version='LCD_v1', channels=128, rank=32, seed=424002, layer=5,
                ln_eps=1e-6, ln_affine=True, conv_bias=False, shared_kernel=True,
                dilations=[1, 2], padding=[1, 2], output_init='zeros', extra_loss=None)


def runtime():
    info = mother.runtime()
    from ultralytics.models.utils.loss import RTDETRDetectionLoss
    info.update(criterion=inspect.getfile(RTDETRDetectionLoss), research=RESEARCH)
    return info


def verify_model(model, zero=False):
    mother.source_contract()
    expected = YAML.load(MODEL)
    original=deepcopy(YAML.load(MODEL_DIR/mother.CONFIGS['pair']))
    original['backbone'][5][2]='BlocksLCD'
    require(all(expected[k]==original[k] for k in ('backbone','head','scales')),'Only mother layer 5 may change')
    require(type(model) is RTDETRDetectionModel, 'Native model type changed')
    require(all(model.yaml[k] == expected[k] for k in ('backbone', 'head', 'scales')), 'LCD topology changed')
    require(type(model.model[5]) is BlocksLCD and sum(isinstance(m, LCD) for m in model.modules()) == 1, 'One S3 LCD required')
    require(type(model.model[20]) is LIFDown and type(model.model[26]) is RTDETRDecoderCBR, 'Original modules required')
    require(model.model[17].f == 5 and model.model[26].f == [19, 22, 25] and 5 in model.save, 'Route drift')
    require(len(model.model[26].decoder.layers) == 3 and model.model[26].num_queries == 300, 'Decoder changed')
    head=model.model[26];lif=model.model[20]
    require(head.hidden_dim==256 and head.decoder.eval_idx==2 and head.cbr.rho==head.cbr.normal_fraction==.10,'CBR contract changed')
    require(sum(p.numel() for p in head.cbr.parameters())==45889,'CBR parameter count changed')
    require(hasattr(lif,'bn') and lif.forward.__func__ is LIFDown.forward,'LIF fusion/forward protection changed')
    added = {k for k in model.state_dict() if k.startswith(PREFIX)}
    require(added == NEW_KEYS, 'LCD state inventory differs')
    require(sum(p.numel() for p in model.model[5].lcd.parameters()) == 9568, 'LCD count differs')
    if not model.is_fused() and model.model[-1].nc == 1:
        require(sum(p.numel() for p in model.parameters()) == 20159333, 'Total count differs')
    if zero:
        require(torch.count_nonzero(model.model[5].lcd.O.weight) == 0, 'LCD O must start at zero')
    return True


def build(nc=80):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL), nc=nc, verbose=False)


def controlled(source):
    # Do not relax or bypass the original strict type/semantic audits.
    _, pair80, original = mother.controlled_models(source)
    lcd80 = build()
    fresh_pair = mother.build()
    require(set(lcd80.state_dict()) - set(pair80.state_dict()) == NEW_KEYS, 'Unexpected added keys')
    require(all(torch.equal(v, lcd80.state_dict()[k]) for k, v in fresh_pair.state_dict().items()), 'LCD disturbed public RNG')
    lcd80.load_state_dict({**lcd80.state_dict(), **pair80.state_dict()}, strict=True)
    # Both native rebuilds see the same CPU RNG; the audit never changes the caller's stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        pair1, pair_report = mother.build_training_model(pair80.yaml, pair80, dict(nc=1, channels=3))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        lcd1 = mother.native_rebuild(lcd80.yaml, lcd80, nc=1)
    before, after = pair1.state_dict(), lcd1.state_dict()
    require(set(after) - set(before) == NEW_KEYS and set(before) <= set(after), 'Public state keys lost')
    require(all(torch.equal(v, after[k]) for k, v in before.items()), 'Controlled nc1 common values differ')
    require(all(torch.equal(lcd80.state_dict()[k], after[k]) for k in NEW_KEYS), 'Native adaptation lost LCD')
    verify_model(lcd1, zero=True)
    report = dict(base=BASE, source_sha256=SOURCE_SHA256, mother_audit=original,
                  mother_nc1_audit=pair_report, COMMON=sorted(before), NEW_TRAINABLE=sorted(NEW_KEYS),
                  NEW_BUFFER=[], MISSING=[], UNEXPECTED=[], public_equal=True,
                  ALLOWED_CLASS_ADAPTATION=pair_report['ALLOWED_CLASS_ADAPTATION'],
                  new_parameters=9568, parameters=20159333, final_nc=1, research=RESEARCH)
    return pair1, lcd1, report


def tensor_digest(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(str((v.shape, v.dtype)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initialize(source, output):
    output = Path(output)
    require(not output.exists(), 'Existing LCD init preserved: ' + str(output))
    _, model, report = controlled(source)
    model.eval(); model.args = {**DEFAULT_CFG_DICT, 'model': str(MODEL), 'task': 'detect'}
    model.task = 'detect'; model.pt_path = str(output.resolve())
    report['tensor_sha256'] = tensor_digest(model)
    checkpoint = dict(epoch=-1, model=model, ema=None, optimizer=None, scaler=None, updates=None,
                      best_fitness=None, train_args=model.args, lcd_initialization=report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f: torch.save(checkpoint, f)
    restored = torch_load(output, map_location='cpu')['model']
    require(tensor_digest(restored) == report['tensor_sha256'], 'Serialized initial state changed')
    # This is the actual native Trainer.get_model path used by the formal trainer.
    rebuilt = mother.native_rebuild(restored.yaml, restored, nc=1)
    verify_model(rebuilt, zero=True)
    require(tensor_digest(rebuilt) == report['tensor_sha256'], 'Formal trainer rebuild differs')
    report.update(status='PASSED', init_sha256=sha256(output), output=str(output), runtime=runtime())
    return report
