"""C2 public init -> original C19 + original LIF; strict semantic/value audit."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import torch
from init_lif_down import (ROOT, MODEL_DIR, SOURCE_SHA256, C2_COMMIT, require,
                           sha256, write_json, runtime as lif_runtime)
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import LIFDown, RTDETRDecoderCBR, Conv
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from lif_down_topology import locate

BASE_COMMIT = '0e95bbade3558b0d2b77c5531483c60810391d88'
C19_COMMIT = '025997e3c51eaf6933534308a95da6ebf97bff53'
VARIANT = 'c19_lif_v1'
VARIANTS = {VARIANT: ('rtdetr-resnet18-lite-cbr-lif-down.yaml',
                     'rtdetr-resnet18-lite.yaml',
                     'c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug')}
CONFIGS = dict(C2=VARIANTS[VARIANT][1], LIF='rtdetr-resnet18-lite-lif-down.yaml',
               C19='rtdetr-resnet18-lite-cbr.yaml', pair=VARIANTS[VARIANT][0])
MODULE_HASHES = dict(lif_down='26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7',
                     cbr='d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787')


def runtime():
    info = lif_runtime()
    from ultralytics.nn.modules import cbr
    info['cbr_module'] = cbr.__file__
    require(Path(cbr.__file__).resolve() == ROOT/'ultralytics-main/ultralytics/nn/modules/cbr.py', 'Wrong CBR import')
    info['module_hashes'] = source_contract()
    return info


def source_contract():
    rows = {}
    for module, expected in MODULE_HASHES.items():
        p = ROOT/f'ultralytics-main/ultralytics/nn/modules/{module}.py'
        raw = p.read_bytes()
        rows[module] = dict(raw_sha256=sha256(p), lf_sha256=hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest())
        require(rows[module]['lf_sha256'] == expected, f'Original {module} source changed')
    return rows


def topology():
    return locate(YAML.load(MODEL_DIR/CONFIGS['C2']))


def new_keys():
    t = topology()
    lif = {f"model.{t['p3_to_p4']['downsample']}.{n}.weight" for n in ('B_proj','P','U_mix','U_dw','O_proj')}
    return lif, f"model.{t['decoder']}.cbr."


def is_added(key):
    lif, cbr = new_keys()
    return key in lif or key.startswith(cbr)


def build(kind='pair', nc=80):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR/CONFIGS[kind]), nc=nc, verbose=False)


def verify_model(model, variant=VARIANT, zero=False):
    require(variant == VARIANT, 'Pair-only entry')
    source_contract()
    base = YAML.load(MODEL_DIR/CONFIGS['C2']); expected = deepcopy(base); t = locate(base)
    expected['head'][t['p3_to_p4']['downsample']-len(base['backbone'])][2] = 'LIFDown'
    expected['head'][t['decoder']-len(base['backbone'])][2] = 'RTDETRDecoderCBR'
    require(type(model) is RTDETRDetectionModel, 'Native detection model required')
    require(all(model.yaml[k] == expected[k] for k in ('backbone','head','scales')), 'Only the two original nodes may differ from C2')
    lif, head = model.model[t['p3_to_p4']['downsample']], model.model[t['decoder']]
    require(type(lif) is LIFDown and sum(isinstance(m,LIFDown) for m in model.modules()) == 1, 'Exactly one original LIF')
    require(type(model.model[t['p4_to_p5']['downsample']]) is Conv, 'Second downsample changed')
    require(type(head) is RTDETRDecoderCBR and head.f == [t['p3_to_p4']['input'],t['p3_to_p4']['output'],t['p4_to_p5']['output']], 'Final Neck P3/P4/P5 only')
    require(head.hidden_dim == 256 and head.num_queries == 300 and len(head.decoder.layers) == 3, 'Decoder dimensions changed')
    require(head.decoder.eval_idx == 2 and head.cbr.rho == head.cbr.normal_fraction == .10, 'Original CBR contract changed')
    require(sum(p.numel() for p in head.cbr.parameters()) == 45889 and len(list(head.cbr.parameters())) == 14, 'CBR parameters changed')
    require(sum(p.numel() for n,p in model.named_parameters() if is_added(n)) == 66993, 'New parameter delta changed')
    require(hasattr(lif,'bn') and lif.forward.__func__ is LIFDown.forward, 'LIF pre-BN residual/fuse protection lost')
    if head.nc == 1:
        # Evaluation may call this on an already fused model; count only unfused here.
        if not model.is_fused(): require(sum(p.numel() for p in model.parameters()) == 20149765, 'nc1 unfused count changed')
    if zero:
        require(torch.count_nonzero(lif.O_proj.weight) == 0, 'LIF O must start zero')
        require(all(torch.count_nonzero(v) == 0 for v in head.cbr.offset_out.parameters()), 'CBR output must start zero')
        require(all(torch.count_nonzero(v.weight) > 0 for v in (lif.B_proj,lif.P,lif.U_mix,lif.U_dw)), 'LIF upstream initialization changed')
    return t


def controlled_models(source, variant=VARIANT):
    require(variant == VARIANT, 'Pair-only entry')
    require(Path(source).is_file() and sha256(source) == SOURCE_SHA256, 'Unified init hash mismatch')
    checkpoint = torch_load(source,map_location='cpu')
    require(checkpoint.get('epoch') == -1 and all(checkpoint.get(k) is None for k in
            ('ema','optimizer','scaler','updates','train_metrics','train_results','best_fitness')), 'Source contains trained state')
    original = deepcopy(checkpoint['model']).float()
    require(original.model[-1].nc == 80, 'Source nc must be 80')
    base, target = build('C2'), build()
    public, fresh = base.state_dict(), target.state_dict(); before = original.state_dict()
    require(len(public) == 533 and set(before) == set(public), 'C2 state inventory mismatch')
    require(original.yaml['backbone'] == base.yaml['backbone'] and original.yaml['head'] == base.yaml['head'], 'Source semantics differ from C2')
    require(all(v.shape == public[k].shape for k,v in before.items()), 'Source shape mismatch')
    require(all(torch.equal(v,fresh[k]) for k,v in public.items()), 'Pair constructor disturbed C2 RNG/state')
    extra = set(fresh)-set(public); parameters = dict(target.named_parameters())
    require(len(extra) == 19 and extra == {k for k in fresh if is_added(k)} and extra <= parameters.keys(), 'Unexpected new state/buffers')
    parents = {}
    for kind in ('LIF','C19'):
        parent = build(kind); state = parent.state_dict()
        added = set(state)-set(public)
        require(all(torch.equal(state[k],fresh[k]) for k in added), kind+' parent initialization mismatch')
        require(all(torch.equal(v,state[k]) for k,v in public.items()), kind+' common constructor mismatch')
        parents[kind] = {k:dict(shape=list(state[k].shape),equal=True,sha256=hashlib.sha256(state[k].numpy().tobytes()).hexdigest()) for k in sorted(added)}
    base.load_state_dict(before,strict=True)
    target.load_state_dict({**fresh,**before},strict=True)
    verify_model(target,zero=True)
    require(all(torch.equal(v,target.state_dict()[k]) for k,v in before.items()), 'Common values changed')
    report = dict(source=str(Path(source).resolve()),source_sha256=SOURCE_SHA256,source_nc=80,target_nc=80,seed=42,
                  c2_commit=C2_COMMIT,lif_commit=BASE_COMMIT,c19_commit=C19_COMMIT,variant=variant,
                  COMMON=[dict(source=k,target=k,shape=list(v.shape),equal=True) for k,v in before.items()],
                  NEW_TRAINABLE=sorted(extra),NEW_BUFFER=[],ALLOWED_CLASS_ADAPTATION=[],MISSING=[],UNEXPECTED=[],SHAPE_MISMATCH=[],
                  parent_initial_values=parents,new_parameters=66993,public_constructor_equal=True)
    return base,target,report


def native_rebuild(cfg, weights, nc=1, channels=3):
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.data = dict(nc=nc,channels=channels)
    return RTDETRTrainer.get_model(trainer,cfg=deepcopy(cfg),weights=weights,verbose=False)


def build_training_model(cfg, weights, data, variant=VARIANT):
    # The real native get_model performs construction/loading. The parallel C2 check
    # is in an isolated RNG scope, so classification adaptation follows native RNG.
    with torch.random.fork_rng(devices=[]):
        baseline = native_rebuild(str(MODEL_DIR/CONFIGS['C2']),weights,data['nc'],data['channels'])
    target = native_rebuild(cfg,weights,data['nc'],data['channels'])
    verify_model(target,variant,zero=True)
    before,after = weights.state_dict(),target.state_dict()
    head=topology()['decoder']; prefix=f'model.{head}.'
    allowed = {prefix+'denoising_class_embed.weight',prefix+'enc_score_head.weight',prefix+'enc_score_head.bias'}
    allowed |= {prefix+f'dec_score_head.{i}.{s}' for i in range(3) for s in ('weight','bias')}
    changed = {k for k in before if before[k].shape != after[k].shape}
    require(set(before) == set(after) and changed == (allowed if weights.model[-1].nc != data['nc'] else set()), 'Unexpected Trainer adaptation')
    require(all(torch.equal(v,after[k]) for k,v in before.items() if k not in changed), 'Native Trainer reload lost initialized tensor')
    require(all(torch.equal(v,after[k]) for k,v in baseline.state_dict().items()), 'Native nc1 C2 adaptation mismatch')
    report=dict(COMMON=[k for k in before if k not in changed and not is_added(k)],
                NEW_TRAINABLE=[k for k in before if is_added(k)],NEW_BUFFER=[],MISSING=[],UNEXPECTED=[],SHAPE_MISMATCH=[],
                ALLOWED_CLASS_ADAPTATION=[dict(name=k,source_shape=list(before[k].shape),target_shape=list(after[k].shape),
                    rule='Native RTDETRTrainer constructor at original RNG; exact C2 nc1 value') for k in sorted(changed)],
                loaded_exact=len(before)-len(changed),c2_nc1_public_exact=True,c2_nc1_public_states=len(baseline.state_dict()),
                parent_initial_values_preserved=True,native_get_model=True)
    return target,report


def initialize(source, output, variant=VARIANT):
    output=Path(output); require(not output.exists(),f'Existing init preserved: {output}')
    _,target,report=controlled_models(source,variant)
    target.eval();target.args={**DEFAULT_CFG_DICT,'model':str(MODEL_DIR/CONFIGS['pair']),'task':'detect'}
    target.task,target.pt_path='detect',str(output.resolve())
    checkpoint=dict(epoch=-1,best_fitness=None,model=deepcopy(target).float(),ema=None,updates=None,optimizer=None,scaler=None,
                    train_args=target.args,train_metrics=None,train_results=None,date=datetime.now(timezone.utc).isoformat(),
                    c19_lif_v1_provenance=report)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream: torch.save(checkpoint,stream)
    restored=RTDETR(str(output)).model
    require(all(torch.equal(v,restored.state_dict()[k]) for k,v in target.state_dict().items()),'Serialized init differs')
    report.update(output=str(output.resolve()),output_sha256=sha256(output),reload_exact=True,status='passed',runtime=runtime())
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output','report'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    write_json(args.report,initialize(args.source,args.output))
