"""Strict public-source initialization and native nc=1 Trainer reconstruction for PBI-v1."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ultralytics-main'))
import ultralytics
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import Conv, LIFDown, PBI, PBIConv, RTDETRDecoder, RTDETRDecoderCBR
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from init_lif_down import SOURCE_SHA256, C2_COMMIT, require, sha256, write_json
from init_c19_lif_v1 import source_contract

MODEL_DIR = ROOT / 'ultralytics-main/ultralytics/cfg/models/rt-detr'
BASE_COMMIT = 'a0459d6a652cb702699087c88fa39a3e4c4087ec'
VARIANTS = {
    'cbr_lif_pbi_v1': ('rtdetr-resnet18-lite-cbr-lif-pbi-v1.yaml',
                       'rtdetr-resnet18-lite-cbr-lif-down.yaml',
                       'cbr_lif_pbi_v1_rtdetr_r18_lite_e200_b16_onlineaug'),
    'pbi_v1': ('rtdetr-resnet18-lite-pbi-v1.yaml', 'rtdetr-resnet18-lite.yaml',
               'pbi_v1_rtdetr_r18_lite_e200_b16_onlineaug'),
}
PBI_KEYS = tuple(f'model.17.pbi.{name}.weight' for name in ('W1', 'W2', 'Wo'))
PARAM_COUNTS = {'pbi_v1': (20107348, 19902292), 'cbr_lif_pbi_v1': (20174341, 19969541)}


def is_added(name):
    return name in PBI_KEYS


def tensor_sha256(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def runtime():
    require(Path(ultralytics.__file__).resolve().is_relative_to(ROOT / 'ultralytics-main'), 'Wrong ultralytics import')
    import ultralytics.nn.modules.pbi as module
    require(Path(module.__file__).resolve() == ROOT / 'ultralytics-main/ultralytics/nn/modules/pbi.py', 'Wrong PBI import')
    return dict(worktree=str(ROOT), python=sys.version, executable=sys.executable, torch=str(torch.__version__),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                ultralytics=ultralytics.__file__, pbi_module=module.__file__, module_hashes=source_contract(),
                commit=subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                                'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())


def build(variant='cbr_lif_pbi_v1', nc=80, baseline=False):
    require(variant in VARIANTS, f'Unknown variant {variant}')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[variant][int(baseline)]), nc=nc, verbose=False)


def verify_model(model, variant='cbr_lif_pbi_v1', zero=False):
    require(variant in VARIANTS, 'Unknown variant')
    source_contract()
    parent = YAML.load(MODEL_DIR / VARIANTS[variant][1])
    expected = deepcopy(parent)
    require(len(parent['backbone']) + len(parent['head']) == 27, 'Parent node count changed')
    require(parent['head'][9] == [5, 1, 'Conv', [256, 1, 1, 'None', 1, 1, False]], 'Parent P3 projection contract changed')
    expected['head'][9] = [5, 1, 'PBIConv', [256, 1, 1, 'None', 1, 1, False, 32]]
    require(type(model) is RTDETRDetectionModel, 'Native detection model required')
    require(all(model.yaml[k] == expected[k] for k in ('backbone', 'head', 'scales')), 'Only original node 17 may change')
    require(len(model.model) == 27 and model.model[17].f == 5, 'Node references changed')
    m = model.model[17]
    require(type(m) is PBIConv and type(m.act) is torch.nn.Identity, 'Original projection act changed')
    require(m.conv.in_channels == 128 and m.conv.out_channels == 256 and m.conv.kernel_size == (1,1)
            and m.conv.stride == (1,1) and m.conv.padding == (0,0), 'Projection geometry changed')
    require(sum(isinstance(v, PBI) for v in model.modules()) == 1, 'PBI must occur exactly once')
    require(set(m.pbi.state_dict()) == {'W1.weight', 'W2.weight', 'Wo.weight'} and not list(m.pbi.buffers()), 'Unexpected PBI state')
    require(sum(p.numel() for p in m.pbi.parameters()) == 24576, 'Wrong PBI parameter delta')
    for proj, shape in ((m.pbi.W1, (32,256,1,1)), (m.pbi.W2, (32,256,1,1)), (m.pbi.Wo, (256,32,1,1))):
        require(tuple(proj.weight.shape) == shape and proj.bias is None and proj.stride == (1,1)
                and proj.padding == (0,0) and proj.groups == 1, 'PBI projection contract changed')
    require(len({p.untyped_storage().data_ptr() for p in m.pbi.parameters()}) == 3, 'PBI parameters share storage')
    head = model.model[-1]
    require(head.f == [19,22,25] and head.hidden_dim == 256 and head.num_queries == 300
            and len(head.decoder.layers) == 3 and head.decoder.eval_idx == 2, 'Decoder contract changed')
    pair = variant == 'cbr_lif_pbi_v1'
    require(type(model.model[20]) is (LIFDown if pair else Conv), 'Original downsample changed')
    require(type(head) is (RTDETRDecoderCBR if pair else RTDETRDecoder), 'Original decoder changed')
    if pair:
        require(head.cbr.rho == head.cbr.normal_fraction == .10, 'Original CBR ratio changed')
    if head.nc == 1:
        fused = not hasattr(m, 'bn')
        require(sum(p.numel() for p in model.parameters()) == PARAM_COUNTS[variant][int(fused)], 'Parameter count changed')
    if zero:
        require(torch.count_nonzero(m.pbi.Wo.weight).item() == 0, 'PBI Wo must initialize zero')
        require(all(torch.count_nonzero(w.weight).item() > 0 for w in (m.pbi.W1,m.pbi.W2)), 'PBI input projections must be nonzero')
        require(not torch.equal(m.pbi.W1.weight,m.pbi.W2.weight), 'PBI W1/W2 must differ')
    return dict(node=17, source=5, channels=256, latent_channels=32, nodes=27, decoder=26)


def controlled_models(source, variant='cbr_lif_pbi_v1'):
    source = Path(source)
    require(source.is_file() and sha256(source) == SOURCE_SHA256, 'Unified untrained source SHA256 mismatch')
    checkpoint = torch_load(source, map_location='cpu')
    require(checkpoint.get('epoch') == -1 and all(checkpoint.get(k) is None for k in
            ('ema','optimizer','scaler','updates','train_metrics','train_results','best_fitness')), 'Source has trained state')
    original = deepcopy(checkpoint['model']).float()
    c2 = build('pbi_v1', baseline=True)
    source_state, c2_state = original.state_dict(), c2.state_dict()
    require(original.model[-1].nc == 80 and len(source_state) == 533, 'Source nc80 inventory changed')
    require(original.yaml['backbone'] == c2.yaml['backbone'] and original.yaml['head'] == c2.yaml['head'], 'Source semantics changed')
    require(set(source_state) == set(c2_state) and all(v.shape == c2_state[k].shape for k,v in source_state.items()), 'Source inventory/shape mismatch')
    parent, target = build(variant,baseline=True), build(variant)
    before, fresh = parent.state_dict(), target.state_dict()
    require(set(fresh)-set(before) == set(PBI_KEYS) and not set(before)-set(fresh), 'Unexpected new/missing states')
    require(all(torch.equal(v,fresh[k]) for k,v in before.items()), 'PBI constructor disturbed public RNG')
    require(set(PBI_KEYS) <= dict(target.named_parameters()).keys(), 'PBI states must be trainable parameters')
    # The original CBR and LIF constructor parameters are retained exactly. All C2
    # source tensors are then mapped by identical keys with an exhaustive audit.
    parent.load_state_dict({**before, **source_state}, strict=True)
    target.load_state_dict({**fresh, **parent.state_dict()}, strict=True)
    public = parent.state_dict()
    require(all(torch.equal(v,target.state_dict()[k]) for k,v in public.items()), 'Public tensor copy mismatch')
    verify_model(target,variant,zero=True)
    other = build('pbi_v1' if variant == 'cbr_lif_pbi_v1' else 'cbr_lif_pbi_v1')
    require(all(torch.equal(fresh[k],other.state_dict()[k]) for k in PBI_KEYS), 'Variant PBI initialization differs')
    require(all(dict(target.named_parameters())[k].untyped_storage().data_ptr() !=
                dict(other.named_parameters())[k].untyped_storage().data_ptr() for k in PBI_KEYS), 'Variants share PBI storage')
    report = dict(variant=variant, source=str(source.resolve()), source_sha256=SOURCE_SHA256,
                  source_nc=80, target_nc=80, seed=42, base_commit=BASE_COMMIT, c2_commit=C2_COMMIT,
                  COMMON=[dict(source=k,target=k,shape=list(v.shape),equal=True) for k,v in public.items()],
                  NEW_TRAINABLE=list(PBI_KEYS), NEW_BUFFER=[], MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[],
                  ALLOWED_CLASS_ADAPTATION=[], new_parameters=24576, public_constructor_equal=True,
                  original_parent_added_states=sorted(set(public)-set(source_state)),
                  new_initial_values={k:dict(shape=list(fresh[k].shape),sha256=tensor_sha256(fresh[k]),
                                            nonzero=int(torch.count_nonzero(fresh[k]))) for k in PBI_KEYS},
                  no_parameter_storage_sharing=True, source_storage='Original half source exactly promoted to FP32')
    report['variant_new_state_exact'] = True
    report['variant_no_storage_sharing'] = True
    return parent,target,report


def native_rebuild(cfg, weights, nc=1, channels=3):
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.data = dict(nc=nc, channels=channels)
    return RTDETRTrainer.get_model(trainer,cfg=deepcopy(cfg),weights=weights,verbose=False)


def build_training_model(cfg, weights, data, variant='cbr_lif_pbi_v1'):
    # Replay the identical native construction RNG for the parent, then let the
    # candidate consume the one normal native construction sequence.
    with torch.random.fork_rng(devices=[]):
        baseline = native_rebuild(str(MODEL_DIR/VARIANTS[variant][1]),weights,data['nc'],data['channels'])
    target = native_rebuild(cfg,weights,data['nc'],data['channels'])
    verify_model(target,variant)
    before, after, public = weights.state_dict(), target.state_dict(), baseline.state_dict()
    allowed = {'model.26.denoising_class_embed.weight','model.26.enc_score_head.weight','model.26.enc_score_head.bias'}
    allowed |= {f'model.26.dec_score_head.{i}.{s}' for i in range(3) for s in ('weight','bias')}
    require(set(before) == set(after), 'Trainer state keys changed')
    changed = {k for k in before if before[k].shape != after[k].shape}
    require(changed == (allowed if weights.model[-1].nc != data['nc'] else set()), 'Unexpected class adaptation')
    require(all(torch.equal(v,after[k]) for k,v in before.items() if k not in changed), 'Trainer lost initialized state')
    require(set(after)-set(public) == set(PBI_KEYS), 'Trainer public key gap')
    require(all(torch.equal(v,after[k]) for k,v in public.items()), 'Native nc1 parent/candidate adaptation differs')
    report = dict(variant=variant, COMMON=sorted(public), NEW_TRAINABLE=list(PBI_KEYS), NEW_BUFFER=[],
                  MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[], native_get_model=True,
                  parent_nc1_public_exact=True, parent_nc1_public_states=len(public),
                  loaded_exact=len(before)-len(changed), parent_initial_values_preserved=True,
                  ALLOWED_CLASS_ADAPTATION=[dict(name=k,source_shape=list(before[k].shape),target_shape=list(after[k].shape),
                                                rule='Native Trainer RNG; exact adapted parent value') for k in sorted(changed)])
    return target, report


def training_reconstruction(initialized, expected, variant):
    """Exercise native checkpoint setup and actual Model.train dispatch, stopping before optimization."""
    expected_state = {k:tensor_sha256(v) for k,v in expected.state_dict().items()}
    observed = {}

    class StopBeforeTraining(Exception):
        pass

    class AuditTrainer(RTDETRTrainer):
        # Native get_model is inherited. Only the costly data/optimizer setup and
        # training loop are omitted in this explicitly bounded dispatch audit.
        def __init__(self, overrides, _callbacks):
            self.args = SimpleNamespace(**{k:v for k,v in overrides.items() if k != 'session'})
            self.data = dict(nc=1,channels=3)
            torch.random.default_generator.manual_seed(42)

        def train(self):
            RTDETRTrainer.setup_model(self)  # native normal path: model was set by Model.train
            actual = {k:tensor_sha256(v) for k,v in self.model.state_dict().items()}
            require(actual == expected_state, 'Actual Model.train dispatch reconstruction differs')
            verify_model(self.model,variant,zero=True)
            observed.update(actual_model_train_dispatch=True,public_and_added_values_exact=True)
            raise StopBeforeTraining()

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        trainer = RTDETRTrainer.__new__(RTDETRTrainer)
        trainer.data = dict(nc=1,channels=3)
        trainer.args = SimpleNamespace(pretrained=True)
        trainer.model = str(Path(initialized).resolve())
        checkpoint = RTDETRTrainer.setup_model(trainer)
        require(checkpoint['epoch'] == -1, 'Native checkpoint setup loaded trained state')
        require({k:tensor_sha256(v) for k,v in trainer.model.state_dict().items()} == expected_state,
                'Native setup_model checkpoint reconstruction differs')
        observed['native_setup_model'] = True
        args = YAML.load(ROOT/'docs/pbi/parent_args.yaml')
        args.update(model=str(Path(initialized).resolve()),name='pbi_reconstruction_audit',
                    save_dir=str(ROOT/'outputs/pbi/reconstruction_not_created'))
        with patch('ultralytics.engine.model.checks.check_pip_update_available'):
            try:
                RTDETR(str(initialized)).train(trainer=AuditTrainer,**args)
            except StopBeforeTraining:
                pass
    require(observed.get('actual_model_train_dispatch'), 'Model.train dispatch audit did not run')
    return dict(status='PASSED',native_get_model=True,optimizer_steps=0,target_nc=1,
                states_exact=len(expected_state),dataset_loading='NOT_RUN',**observed)


def initialize(source, output, variant='cbr_lif_pbi_v1', verify_existing=False):
    output = Path(output)
    require(verify_existing or not output.exists(), f'Existing init preserved: {output}; use --verify-existing to audit')
    _, target, report = controlled_models(source, variant)
    target.eval()
    target.args = {**DEFAULT_CFG_DICT, 'model':str(MODEL_DIR/VARIANTS[variant][0]), 'task':'detect'}
    target.task, target.pt_path = 'detect', str(output.resolve())
    report['runtime'] = runtime()
    if output.exists():
        saved = torch_load(output,map_location='cpu')
        require(saved.get('epoch') == -1 and saved.get('optimizer') is None and saved.get('ema') is None,
                'Existing file is not an untrained initialization')
        require(saved.get('pbi_provenance',{}).get('variant') == variant
                and saved['pbi_provenance'].get('source_sha256') == SOURCE_SHA256, 'Existing initialization provenance mismatch')
    else:
        checkpoint = dict(epoch=-1,best_fitness=None,model=deepcopy(target).float(),ema=None,updates=None,
                          optimizer=None,scaler=None,train_args=target.args,train_metrics=None,train_results=None,
                          date=datetime.now(timezone.utc).isoformat(),version=ultralytics.__version__,
                          license='AGPL-3.0',pbi_provenance=report)
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('xb') as stream:
            torch.save(checkpoint,stream)
    restored = RTDETR(str(output)).model
    require(set(restored.state_dict()) == set(target.state_dict()) and
            all(torch.equal(v,restored.state_dict()[k]) for k,v in target.state_dict().items()), 'Saved initialization differs')
    verify_model(restored,variant,zero=True)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        rebuilt, rebuild = build_training_model(restored.yaml, restored, dict(nc=1,channels=3), variant)
    dispatch = training_reconstruction(output,rebuilt,variant)
    from train_pbi import code_identity
    report.update(output=str(output.resolve()),output_sha256=sha256(output),reload_exact=True,
                  native_nc1_rebuild=rebuild,status='PASSED',report_kind='controlled_initialization_audit',
                  contract_version='pbi_acceptance_v1',code_identity=code_identity(),training_reconstruction=dispatch)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant',nargs='?',default='cbr_lif_pbi_v1',choices=VARIANTS)
    for name in ('source','output','report'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--verify-existing',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.report,initialize(args.source,args.output,args.variant,args.verify_existing))
