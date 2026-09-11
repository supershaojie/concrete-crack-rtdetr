"""Fixed SCI presets, original graph protection and C2 public-state correspondence."""
from __future__ import annotations
import inspect
from triad_compat import (ROOT, BASE_COMMIT, SOURCE_SHA256, MODEL_DIR, BASE_YAML, require, sha256,
                          write_json, git, rows, sources, common_key, torch, YAML, ultralytics,
                          RTDETRDetectionModel, deepcopy, sys)
from ultralytics.nn.modules import (CSCEFv51, SCCAAIFI, RTDETRDecoderCBR, CrackBoundaryRefinement,
                                    SCIAdapter, AIFI, RTDETRDecoder)

PARENT_COMMIT = '33535ab2a5d9acee4e62d34ae382df8b98b9bbde'
C17 = '0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139'
C24 = 'f6e9dfda765046ae7691302cf5ec89d3f76cec5d'
C19 = '025997e3c51eaf6933534308a95da6ebf97bff53'
C25 = 'ac32e223a509981ee9e61be66a352f2b80c5bfd1'
VARIANTS = {key: dict(yaml=f'rtdetr-resnet18-lite-{suffix}.yaml',cscef=True,scca=sc,cbr=cb,
    parameters=20082772+26912+17152+65540*sc+45889*cb,
    name=f'sci_{key}_rtdetr_r18_lite_e200_b16_onlineaug',log_dir=f'outputs/sci_adapter/{key}')
    for key,suffix,sc,cb in [('cscef_v51_sci_control','cscef-v51-sci-control',False,False),
        ('scca_sci_cscef_v51','scca-sci-cscef-v51',True,False),
        ('scca_sci_cscef_v51_cbr','scca-sci-cscef-v51-cbr',True,True)]}
LEGACY = {'c2':BASE_YAML,'c17':'rtdetr-resnet18-lite-cscef-v51.yaml',
          'c24':'rtdetr-resnet18-lite-scca.yaml','c25':'rtdetr-resnet18-lite-c25-replay.yaml',
          'c19':'rtdetr-resnet18-lite-cbr.yaml'}


def runtime():
    require(__import__('pathlib').Path(ultralytics.__file__).resolve().is_relative_to(ROOT/'ultralytics-main'),
            'Wrong ultralytics path')
    return dict(executable=sys.executable,python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,ultralytics=ultralytics.__file__,
        modules={c.__name__:inspect.getfile(c) for c in (SCIAdapter,CSCEFv51,SCCAAIFI,RTDETRDecoderCBR)},
        commit=git('rev-parse','HEAD'),dirty=bool(git('status','--porcelain')))


def build(variant='scca_sci_cscef_v51',nc=1):
    file=LEGACY[variant] if variant in LEGACY else VARIANTS[variant]['yaml']
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR/file),nc=nc,verbose=False)


def expected_config(variant):
    cfg=YAML.load(MODEL_DIR/LEGACY['c17']);g=rows(cfg)
    for i,r in enumerate(g):
        if i>=18:
            f=r[0];r[0]=[j+1 if j>=18 else j for j in f] if isinstance(f,list) else f+1 if f>=18 else f
    g.insert(18,[16,1,'SCIAdapter',[32]]);g[19][0]=[17,18]
    if VARIANTS[variant]['scca']:g[9][2]='SCCAAIFI'
    if VARIANTS[variant]['cbr']:g[28][2]='RTDETRDecoderCBR'
    cfg['backbone'],cfg['head']=g[:8],g[8:]
    return cfg


def topology(model,variant):
    expected=expected_config(variant)
    require(all(model.yaml[k]==expected[k] for k in ('backbone','head','scales')),'SCI graph drift')
    g,bg=rows(model.yaml),rows(YAML.load(MODEL_DIR/BASE_YAML))
    mapping={i:i if i<18 else i+2 for i in range(len(bg))};inverse={v:k for k,v in mapping.items()}
    for b,t in mapping.items():
        module={'SCCAAIFI':'AIFI','RTDETRDecoderCBR':'RTDETRDecoder'}.get(g[t][2],g[t][2])
        require([g[t][1],module,g[t][3]]==bg[b][1:],'Public operation drift')
        mapped=[17 if j==19 else inverse[j] if j>=0 else j for j in sources(g[t],t)]
        require(mapped==sources(bg[b],b),'Original main edge drift')
    require(sources(g[18],18)==[16] and sources(g[19],19)==[17,18] and sources(g[20],20)==[16,19],
            'SCI must only feed the original CSCEF semantic slot')
    require([i for i,r in enumerate(g) if 18 in sources(r,i)]==[19],'SCI leaked into main path')
    require(type(model.model[18]) is SCIAdapter and type(model.model[19]) is CSCEFv51,'Wrong SCI/CSCEF')
    require(type(model.model[9]) is (SCCAAIFI if VARIANTS[variant]['scca'] else AIFI),'Wrong AIFI')
    head=model.model[-1];cbr=VARIANTS[variant]['cbr']
    require(type(head) is (RTDETRDecoderCBR if cbr else RTDETRDecoder),'Wrong decoder')
    require(head.f==[21,24,27] and head.nl==3,'Original final P3/P4/P5 required')
    require((head.num_queries,head.num_decoder_layers,head.decoder.eval_idx)==(300,3,2),'Decoder drift')
    require((head.num_denoising,head.label_noise_ratio,head.box_noise_scale)==(100,.5,1.),'DN drift')
    if cbr:require(type(head.cbr) is CrackBoundaryRefinement and head.cbr.rho==head.cbr.normal_fraction==.1,'Original CBR drift')
    require(sum(p.numel() for p in model.model[18].parameters())==17152,'SCI count drift')
    if head.nc==1:require(sum(p.numel() for p in model.parameters())==VARIANTS[variant]['parameters'],'Count drift')
    return dict(roles=dict(aifi=9,Y5=10,Y4=15,semantic=16,SCI=18,compat=18,P3_ref=17,P4_ref=12,CSCEF=19,
                          P3_enh=19,P3_base=21,P4_base=24,P5_base=27,decoder=28),
        common_layer_mapping=mapping,decoder_inputs=head.f,semantic_identity_jacobian=1.,
        nodes=[dict(index=i,module=r[2],sources=sources(r,i)) for i,r in enumerate(g)])


def new_keys(model):
    prefixes=[]
    for i,m in enumerate(model.model):
        if isinstance(m,(CSCEFv51,SCIAdapter)):prefixes.append(f'model.{i}.')
        if isinstance(m,SCCAAIFI):prefixes.append(f'model.{i}.scca_')
        if isinstance(m,RTDETRDecoderCBR):prefixes.append(f'model.{i}.cbr.')
    return {k for k in model.state_dict() if any(k.startswith(p) for p in prefixes)}


def verify_zero(model):
    for m in model.modules():
        ps=[]
        if isinstance(m,SCIAdapter):ps=[m.restore.weight,m.restore.bias]
        if isinstance(m,CSCEFv51):ps=[m.output_projection.weight]
        if isinstance(m,SCCAAIFI):ps=[m.scca_o.weight]
        if isinstance(m,CrackBoundaryRefinement):ps=[m.offset_out.weight,m.offset_out.bias]
        require(all(not torch.count_nonzero(p) for p in ps),'Expected zero output projections')
