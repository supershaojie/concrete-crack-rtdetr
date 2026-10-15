"""Single review ZIP with strict decimal-100MB bound and per-entry integrity."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import zipfile
from PIL import Image

from .common import PACKAGE_LIMIT, ROOT, read_json, sha256, write_json


def require_review_materials(output):
    registry=read_json(output/"registry.resolved.json")
    mids=registry["fixed_model_ids"]
    required=["candidates/candidates.json","statistics/per_image.json","statistics/per_image.csv",
              "reports/prediction_sources.json","cam_index.json"]
    for stem in ("detection_comparison","heatmap_comparison"):
        required.extend(f"figures/{stem}.{suffix}" for suffix in ("png","pdf","svg","layout.json","dimensions.json"))
    missing=[name for name in required if not (output/name).is_file()]
    if missing:raise ValueError("Review package is incomplete; missing "+str(missing))
    for stem in ("detection_comparison","heatmap_comparison"):
        dimensions=read_json(output/"figures"/(stem+".dimensions.json"))
        with Image.open(output/"figures"/(stem+".png")) as im:
            if im.size!=(dimensions["pixel_width"],dimensions["pixel_height"]):raise ValueError("Raster dimensions differ from actual pixels")
    detection=read_json(output/"figures/detection_comparison.layout.json")
    heatmap=read_json(output/"figures/heatmap_comparison.layout.json")
    active=detection["column_images"]
    if not 4<=len(active)<=6 or active!=heatmap["column_images"] or detection["rows"]!=len(mids)+1 or heatmap["rows"]!=len(mids)+1:
        raise ValueError("Review matrices disagree on sample/model identities")
    stats=read_json(output/"statistics/per_image.json")
    if stats["model_order"]!=mids or len(stats["images"])!=864:
        raise ValueError("Review statistics must cover every active model and all 864 images")
    if set(read_json(output/"reports/prediction_sources.json")["models"])!=set(mids):
        raise ValueError("Review prediction provenance is incomplete")
    pending={(r["model_id"],r["image"]) for r in heatmap.get("pending",[])}
    index=read_json(output/"cam_index.json")
    for key in active:
        if not (output/"selected/originals"/Path(key).name).is_file():raise ValueError("Missing selected original "+key)
        annotations=read_json(output/"selected/annotations"/(Path(key).stem+".json"))
        if set(annotations["model_predictions"])!=set(mids):raise ValueError("Missing selected model predictions")
        if not (output/"selected/full_panels/ground_truth"/(Path(key).stem+".jpg")).is_file():raise ValueError("Missing selected GT full panel")
        for mid in mids:
            if not (output/"selected/full_panels"/mid/(Path(key).stem+".jpg")).is_file():raise ValueError("Missing selected detection full panel: "+mid)
            entry=index.get(mid,{}).get(key,{})
            if entry.get("status")=="passed":
                if not Path(entry["npz"]).is_file() or not Path(entry["metadata"]).is_file() or read_json(entry["metadata"]).get("npz_sha256")!=sha256(entry["npz"]):
                    raise ValueError("Selected native CAM evidence missing or altered")
                if not (output/"selected/heatmaps"/mid/(Path(key).stem+".jpg")).is_file():raise ValueError("Missing selected full-view heatmap")
            elif (mid,key) not in pending:raise ValueError("Unaccounted missing CAM cell "+mid+" "+key)
    return dict(status="complete_review_materials",scientific_figure_status=heatmap["status"],
                selected_images=active,model_ids=mids,cam_cells_pending=len(pending))


def pack(output, destination=None, require_complete=True):
    output=Path(output).resolve()
    completeness=require_review_materials(output) if require_complete else {"status":"explicit_unit_test_fixture_only"}
    destination=Path(destination or output/"chapter4_review.zip").resolve()
    index=read_json(output/"cam_index.json") if (output/"cam_index.json").exists() else {}
    layout=output/"figures/detection_comparison.layout.json"
    active=set(read_json(layout)["column_images"]) if layout.exists() else None
    active_stems={Path(key).stem for key in active} if active else None
    candidates_path=output/"candidates/candidates.json"
    candidates=read_json(candidates_path).get("cases",[]) if candidates_path.exists() else []
    candidate_stems={Path(c["image"]).stem for c in candidates}
    allowed_previews=candidate_stems|(active_stems or set())
    contact_names=set()
    for group in {c["advantage_type"] for c in candidates}:
        count=sum(c["advantage_type"]==group for c in candidates)
        contact_names.update(f"contact_{group}_{i+1}" for i in range((count+7)//8))
    paths=[];omitted_stale=[]
    for directory in ("figures","statistics","candidates","previews","selected","cases","reports","logs"):
        folder=output/directory
        if folder.exists():
            for p in folder.rglob("*"):
                if not p.is_file() or ".partial" in p.name:continue
                relative=p.relative_to(output)
                if directory=="selected" and active_stems is not None and p.stem not in active_stems:continue
                if directory=="previews" and candidates and p.stem not in allowed_previews|contact_names:continue
                if directory=="candidates" and p.parent.name=="predictions" and candidates and p.stem not in allowed_previews:continue
                if relative.parts[:2]==("selected","heatmaps") and active is not None:
                    mid=relative.parts[2]
                    key=next((k for k in active if Path(k).stem==p.stem),None)
                    if index.get(mid,{}).get(key,{}).get("status")!="passed":
                        omitted_stale.append(dict(server_path=str(p),reason="stale prior overlay; current CAM is pending/failed; failure remains in cam_index/layout"))
                        continue
                paths.append(p)
    for filename in ("run.json","config.resolved.json","registry.resolved.json","cam_index.json","progress.json","layer_protocol.json"):
        if (output/filename).is_file():paths.append(output/filename)
    # Include only selected-case native CAMs; validation activations/maps remain on server.
    for model in index.values():
        for image,entry in model.items():
            if active is not None and image not in active:continue
            for key in ("npz","metadata","failure_file"):
                if entry.get(key) and Path(entry[key]).is_file():paths.append(Path(entry[key]))
    paths=sorted(set(paths))
    if any(p.suffix.lower() in (".pt",".pth",".onnx",".ckpt",".safetensors") for p in paths):
        raise ValueError("Refusing weights in review ZIP")
    for p in paths:
        if not p.resolve().is_relative_to(output):raise ValueError("Review evidence escaped run directory")
    categories=defaultdict(int)
    for p in paths:categories[p.relative_to(output).parts[0]]+=p.stat().st_size
    write_json(output/"pack_sizes.json",dict(uncompressed_bytes_by_category=categories,limit_exclusive=PACKAGE_LIMIT))
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+".partial")
    # Preserve final figures and native CAMs exactly. Only regenerable preview pixels are reduced.
    for maximum,quality in ((1600,85),(1200,80),(800,70),(600,60)):
        inventory=[]
        with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=7) as archive:
            for p in paths:
                relative=p.relative_to(output).as_posix()
                data=p.read_bytes();policy="original evidence bytes"
                if relative.startswith("logs/") and len(data)>1_000_000:
                    # Keep first/last 0.5MB with a visible middle omission; the full log stays on server.
                    data=data[:500_000]+b"\n[package-only middle-log omission; full log path in run manifest]\n"+data[-500_000:]
                    policy="first+last 500000 log bytes; original path/SHA retained"
                if maximum and relative.startswith(("previews/","selected/full_panels/","selected/heatmaps/")) and p.suffix.lower() in (".png",".jpg",".jpeg"):
                    with Image.open(p) as im:
                        im=im.convert("RGB");im.thumbnail((maximum,maximum),Image.Resampling.LANCZOS)
                        buffer=io.BytesIO();im.save(buffer,format="JPEG",quality=quality,optimize=True)
                        data=buffer.getvalue()
                    relative=str(Path(relative).with_suffix(".jpg")).replace("\\","/")
                    policy=f"regenerable preview derivative; max_dimension={maximum}; JPEG quality={quality}"
                archive.writestr(relative,data)
                inventory.append(dict(path=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                                      server_path=str(p),server_sha256=sha256(p),package_policy=policy))
            docs=ROOT/"docs/chapter4_visualization.md"
            if docs.is_file():
                raw=docs.read_bytes();archive.writestr("REPRODUCE.md",raw)
                inventory.append(dict(path="REPRODUCE.md",bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
            manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),limit_exclusive=PACKAGE_LIMIT,completeness=completeness,
                          entries=inventory,model_order=read_json(output/"registry.resolved.json")["fixed_model_ids"],
                          stale_prior_overlays_excluded=omitted_stale,
                          excluded="weights/full dataset/environments/full-test caches/full-resolution intermediate activations remain on server",
                          preview_derivative_max_dimension=maximum,uncompressed_bytes_by_category=categories)
            import json
            archive.writestr("MANIFEST.json",json.dumps(manifest,ensure_ascii=False,indent=2).encode())
        if temporary.stat().st_size<PACKAGE_LIMIT:break
    if temporary.stat().st_size>=PACKAGE_LIMIT:
        temporary.unlink()
        raise ValueError("Complete package exceeds 100000000 bytes even after preview compression; see pack_sizes.json. No models/cases/figures/failures removed.")
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip():raise ValueError("ZIP CRC failed")
        for item in inventory:
            raw=archive.read(item["path"])
            if len(raw)!=item["bytes"] or hashlib.sha256(raw).hexdigest()!=item["sha256"]:
                raise ValueError("ZIP member digest failed: "+item["path"])
    temporary.replace(destination)
    receipt=dict(path=str(destination),bytes=destination.stat().st_size,sha256=sha256(destination),
                 limit_exclusive=PACKAGE_LIMIT,strictly_under_limit=destination.stat().st_size<PACKAGE_LIMIT,
                 verified_entries=len(inventory),preview_derivative_max_dimension=maximum)
    write_json(output/"pack_receipt.json",receipt)
    print(receipt,flush=True)
    return receipt
