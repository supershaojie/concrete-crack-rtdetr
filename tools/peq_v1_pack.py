"""Archive existing PEQ evidence only. LIGHT excludes datasets, .pt and prediction streams."""
from __future__ import annotations
from collections import deque
import io
import tarfile
from peq_v1_common import *
from peq_v1_runtime import status


def verify_archive(path):
    with tarfile.open(path,"r:gz") as archive:
        members=archive.getmembers()
        require(len({m.name for m in members})==len(members),"Duplicate archive members")
        manifest=json.load(archive.extractfile("MANIFEST.json"))
        require({m.name for m in members}==set(manifest)|{"MANIFEST.json"},"Archive manifest coverage differs")
        for member in members:
            require(member.isfile() and not member.name.startswith("/") and ".." not in Path(member.name).parts,"Unsafe archive member")
            if member.name=="MANIFEST.json":
                continue
            data=archive.extractfile(member).read()
            require(len(data)==manifest[member.name]["bytes"] and hashlib.sha256(data).hexdigest()==manifest[member.name]["sha256"],
                    f"Archive member differs: {member.name}")
    return dict(status="PASS",members=len(members))


def archive_files(destination, files, extras):
    destination=Path(destination)
    require(not destination.exists(),"Existing archive preserved")
    manifest={name:dict(bytes=path.stat().st_size,sha256=sha256(path)) for name,path in files.items()}
    manifest.update({name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in extras.items()})
    extras={**extras,"MANIFEST.json":json_bytes(manifest)}
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+".partial")
    with temporary.open("xb") as stream,tarfile.open(fileobj=stream,mode="w:gz") as archive:
        for name,path in sorted(files.items()):
            require(not path.is_symlink(),"Symlinks are excluded from evidence archives")
            archive.add(path,arcname=name,recursive=False)
        for name,data in sorted(extras.items()):
            info=tarfile.TarInfo(name);info.size=len(data)
            archive.addfile(info,io.BytesIO(data))
    verification=verify_archive(temporary)
    require(not destination.exists(),"Archive appeared while packing")
    temporary.rename(destination)
    result=dict(**identity(destination),verification=verification)
    write_json(Path(str(destination)+".verification.json"),result)
    Path(str(destination)+".sha256").write_text(result["sha256"]+"  "+destination.name+"\n",encoding="utf-8")
    return result


def pack(include_predictions=False):
    current=status()
    require(not current.get("active") and current.get("phase") not in ("DISPATCHED","SETTING_UP","RUNNING"),
            "Wait for the current PEQ dispatch to finish before making a consistent evidence archive")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    commit=git("rev-parse","HEAD")
    destination=OUT/"packages"/f"peq_v1_LIGHT_{stamp}_{commit[:12]}.tar.gz"
    files={};omitted=[];predictions={}
    for folder,prefix in ((OUT,"outputs"),(RUN,"training"),(ROOT/"docs/peq_v1","docs")):
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative=path.relative_to(folder)
            if "packages" in relative.parts or "__pycache__" in relative.parts or path.suffix==".tmp":
                continue
            name=prefix+"/"+relative.as_posix()
            if path.name=="predictions_gt.jsonl.gz":
                omitted.append(identity(path))
                if "evaluation_val" in relative.parts or "evaluation_train" in relative.parts:
                    predictions[name]=path
                    metrics=path.parent/"metrics.json"
                    if metrics.is_file():
                        predictions[prefix+"/"+metrics.relative_to(folder).as_posix()]=metrics
                continue
            if path.suffix==".pt" or path.suffix==".log" or path.suffix.lower() in (".jpg",".jpeg",".bmp"):
                omitted.append(identity(path))
                continue
            if path.suffix in (".json",".jsonl",".yaml",".yml",".md",".csv",".txt",".patch",".png",".sh") or path.name.endswith(".tar.gz"):
                files[name]=path
            else:
                omitted.append(identity(path))
    # Commit-addressed source archive, plus a real diff from the specified mother.
    snapshot=OUT/f"source_{commit}.tar.gz"
    if not snapshot.exists():
        subprocess.run(["git","archive","--format=tar.gz","--output="+str(snapshot),"HEAD"],cwd=ROOT,check=True)
    files["source/source_snapshot.tar.gz"]=snapshot
    patch=subprocess.check_output(["git","diff","--binary",BASE,"HEAD"],cwd=ROOT)
    weights={name:identity(path) for name,path in dict(public_source=SOURCE,controlled_init=INIT,best=RUN/"weights/best.pt",last=RUN/"weights/last.pt").items()}
    omitted.extend(value for value in weights.values() if value["status"]=="PRESENT")
    logs={}
    for path in sorted(OUT.glob("console_*.log")):
        with path.open(encoding="utf-8",errors="replace") as stream:
            logs[path.name]=list(deque(stream,maxlen=100))
    summary=dict(created=utc(),commit=commit,base=BASE,state=current,weights=weights,omitted_files=omitted,
                 source=source_identity()["sha256"],environment=environment(),
                 evidence={name:("PRESENT" if (OUT/path).exists() else "NOT_RUN/PENDING") for name,path in
                           dict(initialization="initialization.json",preflight="preflight.json",
                                training="mechanism_epochs.jsonl",val="evaluation_val/metrics.json",test="evaluation_test/metrics.json").items()},
                 limitations="No dataset or large checkpoints. No automatic training/evaluation. All mechanism JSONL retained.")
    extras={"PACKAGE.json":json_bytes(summary),"source/from_mother.patch":patch,
            "console_tail.json":json_bytes(logs),
            "README.txt":("PEQ v1 LIGHT: existing evidence only. Inspect PACKAGE.json for NOT_RUN/PENDING and omissions.\n"
                          "Formal score is PEQ. Raw score is a same-final-box diagnostic.\n").encode()}
    result=dict(status="PACKED",light=archive_files(destination,files,extras))
    if include_predictions:
        error_path=OUT/"packages"/f"peq_v1_ERROR_ANALYSIS_{stamp}_{commit[:12]}.tar.gz"
        result["error_analysis"]=archive_files(error_path,predictions,{"README.txt":
            b"Existing train/val prediction streams and their metadata only. No test streams and no reruns.\n",
            "IDENTITY.json":json_bytes(dict(commit=commit,source=summary["source"],files=len(predictions)))})
    write_json(OUT/"last_package.json",result)
    return result
