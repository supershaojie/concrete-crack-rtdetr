"""Fixed v2 registry and C2 public-state mapping through the original C17 insertion."""
from __future__ import annotations

import inspect
from triad_compat import (ROOT, BASE_COMMIT, SOURCE_SHA256, MODEL_DIR, BASE_YAML, require, sha256,
                          write_json, git, rows, sources, common_key, torch, YAML, ultralytics,
                          RTDETRDetectionModel, deepcopy, sys)
from ultralytics.nn.modules import CSCEFv51, CSCEFv52Compat, SCCAAIFI, RTDETRDecoderCBR, CrackBoundaryRefinement

PARENT_COMMIT = "05e6de8b582f4a5f6c11cd50f8bc0406dacd7d0b"
VARIANTS = {key: dict(yaml=f"rtdetr-resnet18-lite-{suffix}.yaml", cscef=True, scca=sc, cbr=cb,
    parameters=20082772+26912+65540*sc+45889*cb, name=f"mincompat_{key}_rtdetr_r18_lite_e200_b16_onlineaug",
    log_dir=f"outputs/mincompat_v2/{key}") for key,suffix,sc,cb in (
        ("cscef_v52_compat", "cscef-v52-compat", False, False),
        ("cscef_v52_scca_compat", "cscef-v52-scca-compat", True, False),
        ("triad_mincompat_v2", "triad-mincompat-v2", True, True))}
REPLAY_YAML = "rtdetr-resnet18-lite-c25-replay.yaml"  # deliberately excluded from formal launch registry


def runtime():
    from triad_compat import runtime as original_runtime
    info = original_runtime()
    info['modules'] = {c.__name__: inspect.getfile(c) for c in (CSCEFv52Compat, SCCAAIFI, RTDETRDecoderCBR)}
    return info


def expected_config(variant):
    cfg = YAML.load(MODEL_DIR / 'rtdetr-resnet18-lite-cscef-v51.yaml')
    graph = rows(cfg)
    if variant != 'c25_replay': graph[18][2] = 'CSCEFv52Compat'
    if variant == 'c25_replay' or VARIANTS[variant]['scca']: graph[9][2] = 'SCCAAIFI'
    if variant != 'c25_replay' and VARIANTS[variant]['cbr']: graph[27][2] = 'RTDETRDecoderCBR'
    return cfg


def build(variant='triad_mincompat_v2', nc=1):
    file = BASE_YAML if variant == 'c2' else REPLAY_YAML if variant == 'c25_replay' else VARIANTS[variant]['yaml']
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / file), nc=nc, verbose=False)


def topology(model, variant):
    expected = expected_config(variant)
    require(all(model.yaml[k] == expected[k] for k in ('backbone', 'head', 'scales')), 'Original C17 forward graph drift')
    g, bg = rows(model.yaml), rows(YAML.load(MODEL_DIR / BASE_YAML))
    mapping = {i: i if i < 18 else i+1 for i in range(len(bg))}
    inverse = {v: k for k,v in mapping.items()}
    for b,t in mapping.items():
        module = {'SCCAAIFI':'AIFI', 'RTDETRDecoderCBR':'RTDETRDecoder'}.get(g[t][2], g[t][2])
        require([g[t][1], module, g[t][3]] == bg[b][1:], 'Public operation drift')
        mapped = [17 if j == 18 else inverse[j] if j >= 0 else j for j in sources(g[t],t)]
        require(mapped == sources(bg[b],b), 'Public forward edge drift')
    require(sources(g[18],18) == [17,16] and sources(g[19],19) == [16,18], 'CSCEF/Concat edge changed')
    cs, head = model.model[18], model.model[-1]
    cls = CSCEFv51 if variant == 'c25_replay' else CSCEFv52Compat
    require(type(cs) is cls, 'Parser replaced CSCEF class')
    if variant != 'c25_replay': require(cs.semantic_grad_scale == .25, 'Fixed semantic gradient scale changed')
    scca = variant == 'c25_replay' or VARIANTS[variant]['scca']
    from ultralytics.nn.modules import AIFI, RTDETRDecoder
    require(type(model.model[9]) is (SCCAAIFI if scca else AIFI), 'AIFI class changed')
    cbr = variant != 'c25_replay' and VARIANTS[variant]['cbr']
    require(type(head) is (RTDETRDecoderCBR if cbr else RTDETRDecoder), 'Decoder class changed')
    require(head.f == [20,23,26] and head.nl == 3, 'Decoder must read final neck P3/P4/P5 only')
    require((head.num_queries, head.num_decoder_layers, head.decoder.eval_idx) == (300,3,2), 'Decoder shape drift')
    require((head.num_denoising, head.label_noise_ratio, head.box_noise_scale) == (100,.5,1.), 'DN drift')
    if cbr: require(type(head.cbr) is CrackBoundaryRefinement and head.cbr.rho == head.cbr.normal_fraction == .1, 'Original CBR preset drift')
    if head.nc == 1:
        count = 20175224 if variant == 'c25_replay' else VARIANTS[variant]['parameters']
        require(sum(p.numel() for p in model.parameters()) == count, 'Parameter count drift')
    return dict(roles=dict(aifi=9,Y5=10,Y4=15,semantic=16,P3_ref=17,P4_ref=12,CSCEF=18,
                          P3_enh=18,P3_base=20,P4_base=23,P5_base=26,decoder=27),
                common_layer_mapping=mapping,decoder_inputs=head.f,semantic_grad_scale=.25 if cls is CSCEFv52Compat else 1.,
                nodes=[dict(index=i,module=r[2],sources=sources(r,i)) for i,r in enumerate(g)])


def new_keys(model):
    prefixes = []
    for i,m in enumerate(model.model):
        if isinstance(m,CSCEFv51): prefixes.append(f'model.{i}.')
        if isinstance(m,SCCAAIFI): prefixes.append(f'model.{i}.scca_')
        if isinstance(m,RTDETRDecoderCBR): prefixes.append(f'model.{i}.cbr.')
    return {k for k in model.state_dict() if any(k.startswith(p) for p in prefixes)}


def verify_zero(model):
    for m in model.modules():
        ps = []
        if isinstance(m,CSCEFv51): ps = [m.output_projection.weight]
        if isinstance(m,SCCAAIFI): ps = [m.scca_o.weight]
        if isinstance(m,CrackBoundaryRefinement): ps = [m.offset_out.weight,m.offset_out.bias]
        require(all(not torch.count_nonzero(p) for p in ps), 'Expected original zero innovation output heads')
