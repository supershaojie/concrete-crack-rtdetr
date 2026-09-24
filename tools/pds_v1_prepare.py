"""Prepare fixed PDS F inputs and invalidate gates only on material identity changes."""
from __future__ import annotations
from copy import deepcopy
import shutil
import torch
from pds_v1_common import *
from c19_lif_v1_data import dataset_inventory
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import ASSETS
from ultralytics.utils.patches import torch_load


def verify_delivery():
    check_clean()
    info = read_json(OUT / "delivery.json")
    require(info["sha"] == git("rev-parse","HEAD") and info["branch"] == BRANCH, "Run official PDS sync FULL_SHA first")
    require(Path(info["worktree"]).resolve()==ROOT and Path(info["main"]).resolve()==MAIN,"Delivery paths differ")
    require(info["source_fingerprint"]==source_fingerprint(),"Delivery source files changed")
    require(git("remote","get-url","origin").rstrip("/").removesuffix(".git") ==
            "https://github.com/supershaojie/concrete-crack-rtdetr", "Wrong repository origin")
    require((ROOT/".git").is_file(), "PDS must use linked worktree")
    require(Path(git("rev-parse","--path-format=absolute","--git-common-dir")).resolve() ==
            Path(git("rev-parse","--path-format=absolute","--git-common-dir",cwd=MAIN)).resolve(),
            "Worktree not linked to selected main")
    return info


def record_delivery(sha):
    require(len(sha)==40 and all(c in "0123456789abcdef" for c in sha),"Full lowercase SHA required")
    require(git("rev-parse","HEAD")==sha,"Synced HEAD differs")
    check_clean()
    archive_file(OUT/"delivery.json")
    record = dict(sha=sha,branch=BRANCH,base=BASE,worktree=str(ROOT),main=str(MAIN),
                  source_fingerprint=source_fingerprint(),time=utc())
    write_json(OUT/"delivery.json",record)
    return record


def resources():
    import cv2
    from ultralytics import YOLO
    rows=[]
    for name,dest,candidates in (
        ("bus.jpg",ASSETS/"bus.jpg",[MAIN/"ultralytics-main/ultralytics/assets/bus.jpg",MAIN/"bus.jpg"]),
        ("yolo26n.pt",ROOT/"yolo26n.pt",[MAIN/"yolo26n.pt",MAIN/"weights/yolo26n.pt"])):
        if not dest.exists():
            src=next((p for p in candidates if p.is_file()),None)
            require(src is not None,f"Missing native AMP resource {name}; supply a verified main-repository copy")
            dest.parent.mkdir(parents=True,exist_ok=True)
            with src.open("rb") as reader, dest.open("xb") as writer:
                shutil.copyfileobj(reader,writer)
            require(sha256(src)==sha256(dest),"AMP resource copy mismatch")
            origin=str(src)
        else:
            origin="existing worktree asset"
        rows.append(dict(name=name,path=str(dest),origin=origin,sha256=sha256(dest)))
    require(cv2.imread(str(ASSETS/"bus.jpg")) is not None,"Native check_amp bus.jpg unreadable")
    YOLO(str(ROOT/"yolo26n.pt"))  # fail before dispatch if checkpoint can't be imported
    return rows


def data_identity(config):
    data = check_det_dataset(str(config),autodownload=False)
    expected=YAML.load(ROOT/"docs/c19_lif_v1/c2_data.yaml")
    actual=YAML.load(config)
    expected["path"]=str(MAIN/"datasets/crack_det")
    require(Path(actual["path"]).resolve()==Path(expected["path"]).resolve(),"Physical dataset root changed")
    require({k:v for k,v in actual.items() if k!="path"}=={k:v for k,v in expected.items() if k!="path"},
            "Physical splits/classes changed")
    inventory=dataset_inventory(Path(data["path"]))
    historical=read_json(ROOT/"docs/c19_lif_v1/summary.json")["dataset"]
    require(inventory==historical,"Historical split path/label fingerprint differs; no data regeneration")
    # Track image replacements as well without decoding validation/test data.
    images=[]
    for split in ("train","val","test"):
        for p in sorted((Path(data["path"])/"images"/split).rglob("*")):
            if p.is_file():
                s=p.stat()
                images.append((p.relative_to(data["path"]).as_posix(),s.st_size,s.st_mtime_ns))
    return dict(config_sha256=sha256(config),inventory=inventory,image_file_metadata=digest(images))


def prepare():
    delivery=verify_delivery()
    require(SOURCE.is_file() and sha256(SOURCE)==SOURCE_SHA256,"Unified source missing/hash differs")
    args=recipe()
    # Resolve all 109 inherited fields; no new runtime default may silently enter the recipe.
    parsed=vars(get_cfg(overrides=args))
    require(set(parsed)==set(args) and all(parsed[k]==v for k,v in args.items()),"Parsed recipe fields changed")
    assets=resources()
    data=data_identity(args["data"])
    _,native,report=controlled_models(SOURCE)
    if INIT.exists():
        ckpt=torch_load(INIT,map_location="cpu")
        require(ckpt["epoch"]==-1 and ckpt.get("optimizer") is None and ckpt.get("ema") is None,"Initial file is trained")
        require(all(torch.equal(v,ckpt["model"].state_dict()[k]) for k,v in native.state_dict().items()),
                "Existing initialization values differ")
        require(set(native.state_dict())==set(ckpt["model"].state_dict()),"Existing init inventory differs")
    else:
        native.eval()
        native.args={**DEFAULT_CFG_DICT,"model":str(INIT),"task":"detect"}
        native.task="detect"
        INIT.parent.mkdir(parents=True,exist_ok=True)
        with INIT.open("xb") as f:
            torch.save(dict(epoch=-1,model=native,ema=None,optimizer=None,scaler=None,updates=None,
                            best_fitness=None,train_args=native.args,pds_definition=CONFIG,
                            pds_source=report),f)
    # Actual nc80 -> nc1 get_model reconstruction, native audit first.
    from pds_v1_trainer import PDSTrainer
    trainer=PDSTrainer.__new__(PDSTrainer)
    trainer.data=dict(nc=1,channels=3)
    model=trainer.get_model(cfg=native.yaml,weights=native,verbose=False)
    write_json(OUT/"initialization.json",dict(source=report,rebuild=trainer.pds_rebuild_audit,
                pds_config=CONFIG,init_sha256=sha256(INIT),note="nc80 native checkpoint; head attached inside actual nc1 get_model"))
    YAML.save(OUT/"train_args.yaml",args)
    write_json(OUT/"pds_config.json",CONFIG)
    write_json(OUT/"dataset_inventory.json",data)
    write_json(OUT/"amp_resources.json",assets)
    info=environment()
    identity=dict(source_fingerprint=source_fingerprint(),source_sha256=sha256(SOURCE),
                  init_sha256=sha256(INIT),recipe_sha256=sha256(OUT/"train_args.yaml"),
                  pds_definition=digest(CONFIG),data=data,assets={r["name"]:r["sha256"] for r in assets},
                  environment={k:info.get(k) for k in ("python","torch","cuda","gpu","gpu_total_bytes","cudnn","ultralytics")})
    archive_file(OUT/"prepared.json")
    result=dict(status="PREPARED",time=utc(),delivery=delivery,identity=identity,runtime=info,
                historical_cross_split_augmented_source_leakage="not resolved by PDS")
    write_json(OUT/"prepared.json",result)
    (OUT/"source_from_base.patch").write_bytes(subprocess.check_output(["git","diff","--binary",BASE,"HEAD"],cwd=ROOT))
    return result


def current_identity():
    prepared=read_json(OUT/"prepared.json")
    verify_delivery()
    info=environment()
    current=dict(source_fingerprint=source_fingerprint(),source_sha256=sha256(SOURCE),
                 init_sha256=sha256(INIT),recipe_sha256=sha256(OUT/"train_args.yaml"),
                 pds_definition=digest(CONFIG),data=data_identity(recipe()["data"]),
                 assets={r["name"]:sha256(r["path"]) for r in read_json(OUT/"amp_resources.json")},
                 environment={k:info.get(k) for k in ("python","torch","cuda","gpu","gpu_total_bytes","cudnn","ultralytics")})
    require(current==prepared["identity"],"Source/config/init/data/assets/environment changed; run prepare/preflight")
    return current


def require_gate():
    identity=current_identity()
    report=read_json(OUT/"preflight.json")
    require(report["status"]=="PASS" and report["identity"]==identity,
            "No valid PDS B16/640 preflight for this identity; run preflight")
    require(report["capacity"]["effective_updates"]>=1,"Preflight has no effective optimizer update")
    return report
