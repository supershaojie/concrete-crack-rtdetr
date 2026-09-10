"""Variant registry, semantic graph validation, exact common-state correspondence."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from copy import deepcopy
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
sys.path.insert(0, str(ROOT / "ultralytics-main"))
import ultralytics
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML

BASE_COMMIT = "67c3078e54a657fd96d65fee657a75fbb1dae0d6"
SOURCE_SHA256 = "fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e"
MODEL_DIR = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr"
BASE_YAML = "rtdetr-resnet18-lite.yaml"
_CONFIGS = {
    "cscef_v6": ("cscef-v6", True, False, False),
    "scca_v2": ("scca-v2", False, True, False),
    "cbr_v2": ("cbr-v2", False, False, True),
    "cscef_v6_scca_v2": ("cscef-v6-scca-v2", True, True, False),
    "cscef_v6_cbr_v2": ("cscef-v6-cbr-v2", True, False, True),
    "scca_v2_cbr_v2": ("scca-v2-cbr-v2", False, True, True),
    "triad_v1": ("triad-compat-v1", True, True, True),
}
VARIANTS = {key: dict(yaml=f"rtdetr-resnet18-lite-{suffix}.yaml", cscef=cs, scca=sc, cbr=cb,
    parameters=20082772 + 26912*cs + 65540*sc + 45889*cb,
    name=f"triad_{key}_rtdetr_r18_lite_e200_b16_onlineaug",
    log_dir=f"outputs/triad_compat/{key}", evaluation_dirs={split:f"outputs/triad_compat/{key}/evaluation_{split}" for split in ('val','test')},
    package_dir=f"downloads/triad_compat/{key}") for key,(suffix,cs,sc,cb) in _CONFIGS.items()}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str)+"\n", encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT, text=True).strip()


def runtime():
    import inspect
    from ultralytics.nn.modules import DRCSCEFv6, GISCCAAIFI, StableReferenceCBR
    require(Path(ultralytics.__file__).resolve().is_relative_to(ROOT / "ultralytics-main"), "Wrong ultralytics import")
    return dict(executable=sys.executable, python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None, ultralytics=ultralytics.__file__,
        modules={c.__name__:inspect.getfile(c) for c in (DRCSCEFv6, GISCCAAIFI, StableReferenceCBR)},
        commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")))


def rows(cfg):
    return cfg["backbone"] + cfg["head"]


def sources(row, index):
    f=row[0]
    return [index+j if j<0 else j for j in (f if isinstance(f,list) else [f])]


def baseline_roles(cfg):
    """Find references by backbone stage and its actual original projection edge."""
    graph=rows(cfg)
    stages={r[3][3]:i for i,r in enumerate(cfg["backbone"]) if r[2]=="Blocks"}
    refs={}
    for stage,role in ((3,"P3_ref"),(4,"P4_ref"),(5,"P5_projection")):
        matches=[i for i,r in enumerate(graph) if i>=len(cfg["backbone"]) and r[2]=="Conv"
                 and sources(r,i)==[stages[stage]] and r[3][1:3]==[1,1] and r[3][-1] is False]
        require(len(matches)==1, f"Ambiguous {role} projection")
        refs[role]=matches[0]
    dec=[i for i,r in enumerate(graph) if r[2]=="RTDETRDecoder"]
    aifi=[i for i,r in enumerate(graph) if r[2]=="AIFI"]
    require(len(dec)==len(aifi)==1, "Expected one native decoder/AIFI")
    refs.update(decoder=dec[0], aifi=aifi[0])
    for role,index in zip(("P3_base","P4_base","P5_base"),sources(graph[dec[0]],dec[0])):
        refs[role]=index
        require(graph[index][2]=="RepC3", f"Unexpected {role} producer")
    return refs


def expected_config(variant):
    cfg=YAML.load(MODEL_DIR / BASE_YAML)
    roles=baseline_roles(cfg); graph=rows(cfg); v=VARIANTS[variant]
    if v["scca"]:
        graph[roles["aifi"]][2]="GISCCAAIFI"
    dec=graph.pop()
    if v["cscef"]:
        p3_enh=len(graph)
        graph.append([[roles[k] for k in ("P3_base","P3_ref","P4_ref")],1,"DRCSCEFv6",[32,8,1e-6]])
        dec[0][0]=p3_enh
    if v["cbr"]:
        dec[2]="RTDETRDecoderCBRv2"
        dec[0].append(roles["P3_ref"])
    graph.append(dec)
    cfg["head"]=graph[len(cfg["backbone"]):]
    return cfg


def build(variant="triad_v1", nc=1):
    file=BASE_YAML if variant=="c2" else VARIANTS[variant]["yaml"]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR/file), nc=nc, verbose=False)


def topology(model, variant):
    base=YAML.load(MODEL_DIR/BASE_YAML); expected=expected_config(variant)
    require(all(model.yaml[k]==expected[k] for k in ("backbone","head","scales")), "Variant graph drift")
    g=rows(model.yaml); bg=rows(base); roles=baseline_roles(base)
    public=[i for i,r in enumerate(g) if r[2]!="DRCSCEFv6"]
    require(len(public)==len(bg), "Unexpected extra/missing main modules")
    mapping=dict(enumerate(public)); inv={v:k for k,v in mapping.items()}
    side=[i for i,r in enumerate(g) if r[2]=="DRCSCEFv6"]
    def canonical_source(index):
        return inv[sources(g[index],index)[0]] if index in side else (inv[index] if index>=0 else index)
    for b,t in mapping.items():
        m={"GISCCAAIFI":"AIFI","RTDETRDecoderCBRv2":"RTDETRDecoder"}.get(g[t][2],g[t][2])
        require(m==bg[b][2] and g[t][1]==bg[b][1] and g[t][3]==bg[b][3], "Public operation drift")
        fs=sources(g[t],t)
        if g[t][2]=="RTDETRDecoderCBRv2": fs=fs[:3]
        require([canonical_source(j) for j in fs]==sources(bg[b],b), "Public forward edge drift")
    observed={k:mapping[v] for k,v in roles.items()}
    observed["P3_enh"]=side[0] if side else observed["P3_base"]
    head=model.model[observed["decoder"]]
    require(head.num_queries==300 and head.nl==3 and head.num_decoder_layers==3 and head.decoder.eval_idx==2,
            "Native decoder dimensions changed")
    require((head.num_denoising,head.label_noise_ratio,head.box_noise_scale)==(100,.5,1.), "DN changed")
    if VARIANTS[variant]["cbr"]:
        require(head.cbr.rho==.075 and head.cbr.normal_fraction==.10, "CBR preset changed")
    if head.nc==1:
        require(sum(p.numel() for p in model.parameters())==VARIANTS[variant]["parameters"], "Parameter count drift")
    return dict(roles=observed, common_layer_mapping=mapping, decoder_inputs=head.f,
        nodes=[dict(index=i, module=r[2], sources=sources(r,i)) for i,r in enumerate(g)])


def common_key(key, mapping):
    prefix,index,suffix=key.split(".",2)
    require(prefix=="model", "Unexpected public state prefix")
    return f"model.{mapping[int(index)]}.{suffix}"


def new_keys(model):
    from ultralytics.nn.modules import DRCSCEFv6, GISCCAAIFI, RTDETRDecoderCBRv2
    prefixes=[]
    for i,m in enumerate(model.model):
        if isinstance(m,DRCSCEFv6): prefixes.append(f"model.{i}.")
        if isinstance(m,GISCCAAIFI): prefixes.append(f"model.{i}.scca_")
        if isinstance(m,RTDETRDecoderCBRv2): prefixes.append(f"model.{i}.cbr.")
    return {k for k in model.state_dict() if any(k.startswith(p) for p in prefixes)}


def verify_zero(model):
    from ultralytics.nn.modules import DRCSCEFv6, GISCCAAIFI, StableReferenceCBR
    for m in model.modules():
        ps=[]
        if isinstance(m,DRCSCEFv6): ps=[m.output_projection.weight]
        if isinstance(m,GISCCAAIFI): ps=[m.scca_o.weight]
        if isinstance(m,StableReferenceCBR): ps=[m.offset_out.weight,m.offset_out.bias]
        require(all(not torch.count_nonzero(p) for p in ps), "Expected zero innovation output head")
