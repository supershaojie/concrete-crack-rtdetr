"""CEA identity, strict initialization and recipe/data audit. No process/GPU gates."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
import torch
import ultralytics
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr.cea import CEAConfig, CEADetectionModel, CEADetectionLoss
from ultralytics.models.rtdetr.cea_trainer import strict_load
from ultralytics.nn.modules.lif_down import LIFDown
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from init_c19_lif_v1 import build, is_added, topology, SOURCE_SHA256
from c19_lif_v1_data import dataset_inventory

BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
REMOTE = "https://github.com/supershaojie/concrete-crack-rtdetr.git"
BRANCH = "exp-rtdetr-r18-lite-cea-v1"
NAME = "cea_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug"
MAIN = Path("/root/autodl-tmp/projects/Crack_RTDETR")
RUN = ROOT / "runs/c_series" / NAME
OUT = ROOT / "outputs/cea_v1"
INIT = ROOT / "weights/cea_v1_controlled_init.pt"
MODEL = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml"
EXPECTED_COUNTS = {"train": (6048, 45573), "val": (1728, 12840), "test": (864, 6663)}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, timeout=60).strip()


def functional_files():
    files = set((ROOT / "ultralytics-main/ultralytics").rglob("*.py"))
    files.update((ROOT / "ultralytics-main/ultralytics/cfg").rglob("*.yaml"))
    files.update((ROOT / "tools").glob("*cea_v1*"))
    for name in ("init_c19_lif_v1.py", "init_lif_down.py", "lif_down_topology.py", "c19_lif_v1_data.py"):
        files.add(ROOT / "tools" / name)
    files.update(ROOT / "docs/cea_v1" / n for n in ("algorithm_config.yaml", "resolved_formal_config.yaml"))
    return sorted(p for p in files if p.is_file() and p.suffix in (".py", ".yaml", ".sh"))


def functional_identity():
    rows = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in functional_files()}
    evidence = ROOT / "docs/cea_v1/mother_evidence.json"
    rows[evidence.relative_to(ROOT).as_posix()] = digest(read_json(evidence))
    return dict(sha256=digest(rows), files=rows, newline_policy="CRLF normalized to LF; prose excluded")


def runtime():
    require(Path(ultralytics.__file__).resolve().is_relative_to(ROOT / "ultralytics-main"), "Wrong ultralytics import")
    cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                torch=str(torch.__version__), cuda=torch.version.cuda, cuda_available=cuda_available,
                gpu=torch.cuda.get_device_name(0) if cuda_available else None,
                ultralytics=ultralytics.__version__, ultralytics_file=ultralytics.__file__,
                model=CEADetectionModel.__module__ + ".CEADetectionModel",
                criterion=CEADetectionLoss.__module__ + ".CEADetectionLoss",
                commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"), remote=git("remote", "get-url", "origin"))


def algorithm():
    raw = YAML.load(ROOT / "docs/cea_v1/algorithm_config.yaml")
    expected = asdict(CEAConfig())
    require(raw == expected and all(type(raw[k]) is type(v) for k, v in expected.items()),
            "Formal CEA v1 configuration/field types differ from fixed specification")
    return raw


def recipe(main=MAIN, archive=None):
    mother = YAML.load(ROOT / "docs/c19_lif_v1/resolved_formal_config.yaml")
    archive = Path(archive or main / "runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml")
    require(archive.is_file(), f"PENDING: actual mother args missing: {archive}")
    actual = YAML.load(archive)
    normalized = deepcopy(actual)
    old_model = "/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1/weights/c19_lif_v1_controlled_init.pt"
    alias = "/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix/weights/c19_lif_v1_controlled_init.pt"
    aliases = []
    if normalized.get("model") == alias:
        normalized["model"] = old_model
        aliases.append(dict(field="model", original=alias, normalized=old_model, reason="documented exact historical alias"))
    differences = {k: dict(archived=normalized.get(k), expected=mother.get(k),
                            archived_type=type(normalized.get(k)).__name__, expected_type=type(mother.get(k)).__name__)
                   for k in set(mother) | set(normalized)
                   if k not in mother or k not in normalized or type(normalized[k]) is not type(mother[k]) or normalized[k] != mother[k]}
    require(not differences, f"Mother recipe mismatch (all fields/types compared): {differences}")
    expected = YAML.load(ROOT / "docs/cea_v1/resolved_formal_config.yaml")
    permitted = {"model", "project", "name", "save_dir"}
    changed = {k for k in mother if type(mother[k]) is not type(expected[k]) or mother[k] != expected[k]}
    require(set(expected) == set(mother) and changed == permitted, "CEA recipe changes outside explicit path/identity whitelist")
    effective = deepcopy(expected)
    effective.update(model=str(INIT), project=str(RUN.parent), name=RUN.name, save_dir=str(RUN))
    return effective, dict(archive=str(archive), archive_sha256=sha256(archive), exact_model_aliases=aliases,
                           changes={k: dict(mother=mother[k], cea=expected[k], reason="experiment identity/path") for k in sorted(changed)},
                           compared_fields=len(mother), type_comparison=True)


def audit_data(data_path, cache_path=OUT / "data_audit.json"):
    resolved = check_det_dataset(str(data_path), autodownload=False)
    require(resolved["nc"] == 1 and resolved["names"] == {0: "crack"}, "Expected nc=1 crack")
    folder = Path(resolved["path"]).resolve()
    for split in EXPECTED_COUNTS:
        require(Path(resolved[split]).resolve() == folder / "images" / split, "Unexpected dataset split mapping")
    files = sorted(p for part in ("images", "labels") for p in (folder / part).rglob("*")
                   if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".txt"))
    inventory_stats = [(p.relative_to(folder).as_posix(), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    stat_digest = digest(inventory_stats)
    if Path(cache_path).is_file():
        previous = read_json(cache_path)
        if previous.get("stat_sha256") == stat_digest and previous.get("data_yaml_sha256") == sha256(data_path):
            return previous, "REUSED"
    inventory = dataset_inventory(folder)  # original path/label hash contract, no image decoding
    mother_inventory = read_json(ROOT / "docs/cea_v1/mother_evidence.json")["data_inventory"]
    require(inventory == mother_inventory, "Dataset path/label fingerprint differs from verified mother archive")
    for split, counts in EXPECTED_COUNTS.items():
        require((inventory[split]["images"], inventory[split]["boxes"]) == counts,
                f"Dataset count mismatch in {split}: {inventory[split]}; preserve dataset and investigate")
    row = dict(dataset_root=str(folder), data_yaml=str(Path(data_path).resolve()), data_yaml_sha256=sha256(data_path),
               stat_sha256=stat_digest, split_inventory=inventory,
               fingerprint=digest(dict(inventory=inventory, file_metadata=stat_digest)),
               image_identity="relative paths, size, mtime_ns; label contents SHA256; no image-content hash claim")
    write_json(cache_path, row)
    return row, "AUDITED"


def initialize(source):
    """Same controlled mapping as init_c19_lif_v1, then explicit audited native nc1 adaptation.

    The mother's helper hard-codes the old CBR file hash. Reuse its build/topology/key
    functions, and audit actual tensors here because this branch adds a context API.
    """
    source = Path(source)
    require(source.is_file() and sha256(source) == SOURCE_SHA256, "Public untrained source SHA256 mismatch/missing")
    if INIT.exists():
        record = read_json(OUT / "initialization.json")
        require(record["source_sha256"] == SOURCE_SHA256 and record["output_sha256"] == sha256(INIT), "Existing init identity unknown/changed")
        require(record["functional_sha256"] == functional_identity()["sha256"], "Existing init generated by different functional source; preserve and inspect")
        return record
    ckpt = torch_load(source, map_location="cpu")
    require(ckpt.get("epoch") == -1 and all(ckpt.get(k) is None for k in
            ("ema", "optimizer", "scaler", "updates", "train_metrics", "train_results", "best_fitness")), "Public source contains trained state")
    original = ckpt["model"].float()
    base, target = build("C2"), build()
    before, public, fresh = original.state_dict(), base.state_dict(), target.state_dict()
    require(original.model[-1].nc == 80 and set(before) == set(public) and len(public) == 533, "Public source nc/key inventory mismatch")
    require(original.yaml["backbone"] == base.yaml["backbone"] and original.yaml["head"] == base.yaml["head"], "Public source topology mismatch")
    require(all(before[k].shape == public[k].shape and torch.equal(public[k], fresh[k]) for k in public), "Public mapping/RNG mismatch")
    extra = set(fresh) - set(public)
    require(len(extra) == 19 and extra == {k for k in fresh if is_added(k)}, "Original LIF/CBR state keys changed")
    for kind in ("LIF", "C19"):
        parent = build(kind).state_dict()
        require(all(torch.equal(v, fresh[k]) for k, v in parent.items()), f"{kind} original initialization mismatch")
    target.load_state_dict({**fresh, **before}, strict=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        nc1 = CEADetectionModel(str(MODEL), nc=1, verbose=False, cea_config=algorithm())
        adaptation = strict_load(nc1, target)
    # Native mother's nc=80 -> nc=1 get_model uses the same constructor seed.
    from init_c19_lif_v1 import native_rebuild
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        native = native_rebuild(str(MODEL), target, nc=1)
    require(all(torch.equal(v, native.state_dict()[k]) for k, v in nc1.state_dict().items()), "Controlled nc1 differs from native mapping")
    require(sum(p.numel() for p in nc1.parameters()) == 20149765, "Unfused parameter count changed")
    require(all(torch.count_nonzero(v) == 0 for v in nc1.model[-1].cbr.offset_out.parameters()), "CBR zero initialization changed")
    require(all(torch.count_nonzero(m.O_proj.weight) == 0 for m in nc1.modules() if isinstance(m, LIFDown)), "LIF zero initialization changed")
    nc1.nc = 1
    nc1.args = {**DEFAULT_CFG_DICT, "model": str(MODEL), "task": "detect"}
    nc1.task = "detect"; nc1.pt_path = str(INIT); nc1.eval()
    provenance = dict(source=str(source), source_sha256=SOURCE_SHA256, mother=BASE,
                      functional_sha256=functional_identity()["sha256"], new_cea_parameters=0, adaptation=adaptation,
                      native_nc1_exact=True, source_nc=80, saved_nc=1, seed=42,
                      common=[dict(name=k, shape=list(v.shape)) for k, v in before.items()],
                      original_lif_cbr_keys=sorted(extra), parameter_count=20149765)
    INIT.parent.mkdir(parents=True, exist_ok=True)
    with INIT.open("xb") as stream:
        torch.save(dict(epoch=-1, best_fitness=None, model=nc1, ema=None, optimizer=None, scaler=None, updates=None,
                        train_args=nc1.args, cea_initialization=provenance), stream)
    reload = torch_load(INIT, map_location="cpu")["model"]
    require(all(torch.equal(v, reload.state_dict()[k]) for k, v in nc1.state_dict().items()), "Initialization serialization changed values")
    provenance.update(output=str(INIT), output_sha256=sha256(INIT), reload_exact=True, status="PASS")
    write_json(OUT / "initialization.json", provenance)
    return provenance


def current_identity(prepared):
    data, _ = audit_data(prepared["recipe"]["data"])
    env = runtime()
    return dict(functional_sha256=functional_identity()["sha256"], algorithm=algorithm(), recipe=prepared["recipe"],
                init_sha256=sha256(INIT), data_fingerprint=data["fingerprint"], run=str(RUN), out=str(OUT),
                environment={k: env[k] for k in ("python", "executable", "torch", "cuda", "gpu", "ultralytics")})


def prepared_identity():
    prepared = read_json(OUT / "prepare.json")
    require(prepared["status"] == "PREPARED", "prepare is not successful")
    identity = current_identity(prepared)
    require(identity == prepared["identity"], "Code/config/init/data/run changed since prepare; previous preflight invalid")
    return prepared, identity
