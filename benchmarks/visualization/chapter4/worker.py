"""One pinned native framework per process. No inference API wraps the CAM graph."""
from __future__ import annotations
import argparse
from contextlib import ExitStack, nullcontext
import importlib
import os
from pathlib import Path
import inspect
import re
import sys
import traceback
from datetime import datetime, timezone

from .common import (ROOT, fingerprint, git_head, path_value, progress, read_json,
                     sha256, write_json)
from .geometry import geometry, inverse_boxes
from .analysis import compare_predictions, target_correspondence
from .cam import METHOD_VERSION, gradcampp


def check_source(model, base):
    work=Path(path_value(model,"work_dir"))
    head=git_head(work)
    if model.get("code_commit") and head!=model["code_commit"]:
        raise ValueError(f"Pinned HEAD differs for {model['id']}: {head}")
    mid=model["id"]; pairs=[]
    if mid in ("rtdetr_c2","lcbr_detr"):
        source=base/"source_snapshots"/("c2" if mid=="rtdetr_c2" else "lcbr")/"ultralytics-main"
        vendor=Path(path_value(model,"vendor"))
        pairs=[(p,vendor/p.relative_to(source)) for p in source.rglob("*") if p.is_file() and p.suffix in (".py",".yaml")]
    elif mid=="fasterrcnn_r50_fpn":
        source=base/"source_snapshots/fasterrcnn/implementation"
        directory=Path(path_value(model,"restore_script")).parent
        pairs=[(p,directory/p.name) for p in source.glob("*.py")]
    elif mid=="yolov5m":
        source=base/"source_snapshots/yolov5m/metadata/upstream"
        vendor=Path(path_value(model,"vendor"))
        if (vendor/".git").exists() and git_head(vendor)!=model["upstream_commit"]:
            raise ValueError("YOLOv5 upstream commit differs")
        pairs=[(p,vendor/p.relative_to(source)) for p in source.rglob("*") if p.is_file() and p.suffix in (".py",".yaml")]
    else:
        source=base/"source_snapshots"/mid
        implementation=source/"implementation"
        if mid=="yolov13l":
            implementation=source/"benchmarks/comparison/yolov13l"
        directory=Path(path_value(model,"export_script")).parent
        pairs=[(implementation/p,directory/p) for p in ("export.py","adapters.py","support.py","upstream.lock.json")]
        if mid=="yolov13l":
            pairs.append((implementation/"backend.py",directory/"backend.py"))
    checked=[]
    for archived,actual in pairs:
        if not actual.is_file() or actual.read_bytes().replace(b"\r\n",b"\n")!=archived.read_bytes().replace(b"\r\n",b"\n"):
            raise ValueError("Pinned source content differs: "+str(actual))
        checked.append(dict(path=str(actual),sha256=sha256(actual)))
    return dict(pinned_commit=model.get("code_commit"),current_formal_head=head,
                project_head_pending=model.get("code_commit") is None,checked_source_files=checked)


class Native:
    def __init__(self, request):
        self.request=request;self.spec=request["model"];self.mid=self.spec["id"]
        self.output=Path(request["output"]);self.base=Path(request["handoff"])
        self.source=check_source(self.spec,self.base)
        self.weight=Path(path_value(self.spec,"formal_test_weight"))
        if sha256(self.weight)!=self.spec["checkpoint"]["sha256"]:
            raise ValueError("Critical formal checkpoint SHA mismatch: "+self.mid)
        vendor=path_value(self.spec,"vendor")
        if vendor:sys.path.insert(0,vendor)
        import torch
        import numpy as np
        import cv2
        self.torch,self.np,self.cv2=torch,np,cv2
        self.device=torch.device(request["device"])
        self.context_factory=nullcontext
        if self.mid=="fasterrcnn_r50_fpn":
            self._faster()
        elif self.mid=="yolov5m":
            from models.experimental import attempt_load
            from utils.general import init_seeds
            init_seeds(42,deterministic=True)
            saved=torch.load(self.weight,map_location="cpu",weights_only=False)
            self.restoration=self.checkpoint_metadata(saved)
            if self.restoration["selected_field"]!="model" or self.restoration["stored_dtype"]!="torch.float16":
                raise ValueError("YOLOv5 saved selected FP16 EMA/model contract differs")
            self.model=attempt_load(str(self.weight),device=self.device,fuse=True).float().eval()
            self.restoration["parameter_origin"]="saved FP16 selected EMA -> FP32; original FP32 EMA not recoverable"
        else:
            self._ultralytics()
        self.model.to(self.device).float().eval()
        for parameter in self.model.parameters():parameter.requires_grad_(True)
        if any(p.dtype!=torch.float32 for p in self.model.parameters()):
            raise ValueError("Non-FP32 parameter after formal restoration")
        self.features={};self.handles=[];self.layer_names={}
        self.install_layers()
        import torchvision
        if self.mid=="fasterrcnn_r50_fpn":
            vision_root=Path(torchvision.__file__).parent
            checked={}
            for relative,expected in self.spec["environment"].get("installed_source_sha256",{}).items():
                actual=sha256(vision_root/relative)
                if actual!=expected:raise ValueError("Formal torchvision source differs: "+relative)
                checked[relative]=actual
            extension=self.spec["environment"].get("torchvision_extension_path")
            extension_sha=self.spec["environment"].get("torchvision_extension_sha256")
            if extension and extension_sha:
                actual=sha256(vision_root/Path(extension).name)
                if actual!=extension_sha:raise ValueError("Formal torchvision compiled operator SHA differs")
                checked[Path(extension).name]=actual
            self.source["torchvision_installed_source_sha256"]=checked
        expected_env=self.spec.get("environment",{})
        actual_versions=dict(torch=torch.__version__,torchvision=torchvision.__version__)
        for package,actual in actual_versions.items():
            expected=expected_env.get("versions",{}).get(package) or expected_env.get(package)
            if isinstance(expected,str) and "历史" not in expected:
                version=re.match(r"[0-9]+\.[0-9]+\.[0-9]+(?:\+[A-Za-z0-9.]+)?",expected)
                if version and version.group()!=actual:raise ValueError(f"Pinned {package} version differs: {actual} vs {expected}")
            expected_path=path_value(self.spec,package+"_import") or expected_env.get(package+"_file")
            actual_module=torch if package=="torch" else torchvision
            if expected_path and Path(actual_module.__file__).resolve()!=Path(expected_path).resolve():
                raise ValueError("Unexpected "+package+" import path: "+actual_module.__file__)
        framework_file=getattr(sys.modules.get("ultralytics"),"__file__",None)
        if self.mid=="yolov5m":framework_file=sys.modules["models.yolo"].__file__
        if self.mid=="fasterrcnn_r50_fpn":framework_file=inspect.getfile(type(self.model))
        self.identity=dict(model_id=self.mid,run_id=self.spec["run_id"],weight_sha256=sha256(self.weight),
                           registry_record_fingerprint=fingerprint(self.spec),
                           device=str(self.device),cuda_device_name=torch.cuda.get_device_name(self.device) if self.device.type=="cuda" else None,
                           cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                           restoration=self.restoration,source=self.source,visualization_head=git_head(ROOT),
                           imports=dict(python=sys.executable,torch=torch.__file__,torchvision=torchvision.__file__,
                                        framework=framework_file,model_class_source=inspect.getfile(type(self.model))),
                           versions=actual_versions,
                           arithmetic=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                           cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                           fp32_precision=torch.get_float32_matmul_precision()),
                           layer_policy="fixed equal-weight stride8/16/32 classification-connected detection features v1",
                           layer_names=self.layer_names,cam_method=METHOD_VERSION)
        self.parameter_versions={n:p._version for n,p in self.model.named_parameters()}
        self.buffers={n:b.detach().clone() for n,b in self.model.named_buffers() if "running_" in n or "num_batches_tracked" in n}

    def checkpoint_metadata(self,saved):
        key="ema" if saved.get("ema") is not None else "model"
        value=saved.get(key)
        if hasattr(value,"parameters"):
            dtype=str(next(value.parameters()).dtype)
        elif isinstance(value,dict):
            dtype=sorted({str(v.dtype) for v in value.values() if hasattr(v,"dtype")})
        else:raise ValueError("No recoverable native model/EMA field")
        return dict(selected_field=key,stored_dtype=dtype,epoch=saved.get("epoch"),
                    checkpoint_keys=sorted(saved),formal_checkpoint_field_evidence="actual torch.load; not guessed")

    def _faster(self):
        # Build the exact torchvision final architecture without downloading/reloading COCO
        # initialization or writing the original run's weight cache. Strict selected state
        # restores every tensor; torchvision COCO_V1 FrozenBN epsilon is 0, not its default.
        torch=self.torch
        directory=Path(path_value(self.spec,"restore_script")).parent
        sys.path.insert(0,str(directory))
        from configuration import frozen_config
        from torchvision.models import resnet50
        from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
        from torchvision.models.detection import FasterRCNN
        from torchvision.ops.misc import FrozenBatchNorm2d
        cfg=frozen_config(Path(path_value(self.spec,"run_dir")))
        backbone=_resnet_fpn_extractor(resnet50(weights=None,norm_layer=FrozenBatchNorm2d),cfg["trainable_backbone_layers"])
        kwargs={k:cfg[k] for k in ("box_score_thresh","box_nms_thresh","box_detections_per_img","rpn_nms_thresh",
                 "rpn_pre_nms_top_n_train","rpn_post_nms_top_n_train","rpn_pre_nms_top_n_test","rpn_post_nms_top_n_test")}
        self.model=FasterRCNN(backbone,num_classes=2,min_size=cfg["model_transform_min_size"],
                              max_size=cfg["model_transform_max_size"],**kwargs)
        for module in self.model.modules():
            if isinstance(module,FrozenBatchNorm2d):module.eps=0.
        saved=torch.load(self.weight,map_location="cpu",weights_only=False)
        self.model.load_state_dict(saved["model"],strict=True)
        self.restoration=dict(selected_field="model",stored_dtype="torch.float32",epoch=saved.get("epoch"),
                              construction="native ResNet50 FrozenBN eps0 + native FPN/FasterRCNN; strict best['model']; no initialization download")
        if not all(v.dtype==torch.float32 for v in saved["model"].values() if v.is_floating_point()):
            raise ValueError("Faster checkpoint is not saved FP32")
        torch.manual_seed(42);torch.backends.cudnn.benchmark=False

    def _ultralytics(self):
        torch=self.torch
        vendor=Path(path_value(self.spec,"vendor"))
        if self.mid not in ("rtdetr_c2","lcbr_detr"):
            # Reuse the pinned entry's complete vendor/patch digest guard and isolated
            # framework setup, including the formal v13 backend observer and settings.
            directory=Path(path_value(self.spec,"export_script")).parent
            sys.path.insert(0,str(directory))
            support=importlib.import_module("support")
            self.source["native_source_verification"]=support.configure(self.output/"runtime"/self.mid/"native",vendor)
        import ultralytics
        if Path(ultralytics.__file__).resolve()!=(vendor/"ultralytics/__init__.py").resolve():
            raise ValueError("Unexpected ultralytics import: "+ultralytics.__file__)
        from ultralytics import YOLO,RTDETR
        saved=torch.load(self.weight,map_location="cpu",weights_only=False)
        self.restoration=self.checkpoint_metadata(saved)
        rt=self.mid in ("rtdetr_c2","lcbr_detr")
        native=(RTDETR if rt else YOLO)(str(self.weight))
        self.model=native.model.float()
        if not rt:
            state=saved["comparison_training_state"]["ema"]
            if any(v.is_floating_point() and v.dtype!=torch.float32 for v in state.values()):
                raise ValueError("EMA sidecar not FP32")
            self.model.load_state_dict(state,strict=True)
            self.restoration.update(compatibility_module_field=self.restoration["selected_field"],
                                    compatibility_module_stored_dtype=self.restoration["stored_dtype"],
                                    selected_field="comparison_training_state.ema",stored_dtype="torch.float32",
                                    strict_state_dict=True,parameter_origin="formally selected stored FP32 EMA sidecar")
            from ultralytics.utils.torch_utils import init_seeds
            init_seeds(42,deterministic=True)
        elif self.mid=="rtdetr_c2" and (saved.get("ema") is not None or self.restoration["selected_field"]!="model"):
            raise ValueError("C2 archived model-field contract differs")
        head=self.model.model[-1]
        if head.nc!=1:raise ValueError("Native crack head is not nc=1")
        if rt and (head.num_queries!=300 or head.num_decoder_layers!=3):
            raise ValueError("RT regular query/decoder contract differs")
        if self.mid=="lcbr_detr":
            if type(self.model.model[9]).__name__!="AIFI" or type(self.model.model[20]).__name__!="LIFDown" or type(head).__name__!="RTDETRDecoderCBR" or head.f!=[19,22,25]:
                raise ValueError("Ours original mother topology differs")
        if self.mid=="yolov13l":
            sys.path.insert(0,str(Path(path_value(self.spec,"arithmetic_backend")).parent))
            self.backend=importlib.import_module("backend")
            self.context_factory=self.backend.native_fp32
        if self.mid=="yolo26m":
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.set_float32_matmul_precision("highest")
            if not head.end2end or not hasattr(head,"one2one_cv3"):
                raise ValueError("YOLO26 requires native one-to-one branch")
        # AutoBackend uses native fuse at formal prediction time. Preserve its exact graph.
        # Native AutoBackend moves FP32 weights to the requested device BEFORE fusion;
        # CPU-fusing then copying to CUDA can introduce avoidable rounding differences.
        self.model.to(self.device).float().eval()
        self.model.fuse(verbose=False)
        if self.mid=="lcbr_detr":
            lif=self.model.model[20]
            if not hasattr(lif,"bn") or lif.forward.__func__ is not type(lif).forward:
                raise ValueError("Original LIF BN/forward fusion protection lost")
            self.restoration["lif_bn_and_native_forward_preserved"]=True

    def capture(self,name,value):
        if not hasattr(value,"ndim") or value.ndim!=4 or value.shape[0]!=1:
            raise ValueError("Target feature is not batch=1 BCHW: "+name)
        self.features[name]=value

    def install_layers(self):
        if self.mid=="fasterrcnn_r50_fpn":
            # ROIAlign may route a thin crack to P2. Capturing P3's output-only 3x3
            # would then be disconnected, since P2 consumes the top-down INNER P3.
            # Capture the fused lateral+top-down tensor before each output conv;
            # it feeds both that level and the following finer FPN level unchanged.
            for stride,index in zip((8,16,32),(1,2,3)):
                name=f"backbone.fpn.layer_blocks.{index}:input"
                self.layer_names[name]=stride
                self.handles.append(self.model.backbone.fpn.layer_blocks[index].register_forward_pre_hook(
                    lambda m,args,name=name:self.capture(name,args[0])))
            self.roi={}
            self.handles.append(self.model.roi_heads.register_forward_pre_hook(lambda m,args:self.roi.update(proposals=args[1])))
            self.handles.append(self.model.roi_heads.box_predictor.register_forward_hook(lambda m,args,out:self.roi.update(logits=out[0],regression=out[1])))
            return
        head=self.model.model[-1]
        for i,stride in enumerate((8,16,32)):
            if self.mid in ("rtdetr_c2","lcbr_detr"):
                module=head.input_proj[i];name=f"model.{len(self.model.model)-1}.input_proj.{i}"
                hook=lambda m,args,out,name=name:self.capture(name,out)
                self.handles.append(module.register_forward_hook(hook))
            else:
                if self.mid=="yolov5m":
                    module=head.m[i];suffix=f"m.{i}:input"
                else:
                    branch="one2one_cv3" if self.mid=="yolo26m" else "cv3"
                    module=getattr(head,branch)[i][-1];suffix=f"{branch}.{i}.last:input"
                name=f"model.{len(self.model.model)-1}.{suffix}"
                hook=lambda m,args,name=name:self.capture(name,args[0])
                self.handles.append(module.register_forward_pre_hook(hook))
            self.layer_names[name]=stride

    def preprocess(self,image):
        cv2,np,torch=self.cv2,self.np,self.torch
        path=Path(self.request["image_root"])/Path(image["file_name"]).name
        bgr=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2]!=(image["height"],image["width"]):
            raise ValueError("Decoded image identity/dimensions differ: "+str(path))
        info=geometry(image["width"],image["height"],"stretch" if self.mid in ("rtdetr_c2","lcbr_detr") else "letterbox")
        resized=cv2.resize(bgr,(info["resized_width"],info["resized_height"]),interpolation=cv2.INTER_LINEAR)
        padded=cv2.copyMakeBorder(resized,info["top"],info["bottom"],info["left"],info["right"],cv2.BORDER_CONSTANT,value=(114,114,114))
        x=torch.from_numpy(np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1))).to(self.device).float()[None]/255
        info.update(input_shape=list(x.shape),input_dtype=str(x.dtype),channel_order="RGB",value_scale="uint8 / 255",
                    normalization="native ImageNet mean/std inside GeneralizedRCNNTransform" if self.mid=="fasterrcnn_r50_fpn" else "0..1; no ImageNet normalization")
        return x,info

    def forward(self,image):
        torch=self.torch
        self.features={}
        x,info=self.preprocess(image)
        x.requires_grad_(True)
        with self.context_factory():
            output=self.model([x[0]]) if self.mid=="fasterrcnn_r50_fpn" else self.model(x)
        if self.mid=="fasterrcnn_r50_fpn":
            from torchvision.ops.boxes import clip_boxes_to_image
            result=output[0]
            boxes=inverse_boxes(result["boxes"],info);scores=result["scores"]
            if not (result["labels"]==1).all():raise ValueError("Faster nonforeground detection")
            raw_boxes=self.model.roi_heads.box_coder.decode(self.roi["regression"],self.roi["proposals"])[:,1,:]
            raw_boxes=inverse_boxes(clip_boxes_to_image(raw_boxes,(640,640)),info)
            raw_scores=self.roi["logits"].softmax(-1)[:,1]
            raw_ids=list(range(len(raw_scores)));target="ROI foreground softmax probability"
        else:
            decoded=output[0] if isinstance(output,(list,tuple)) else output
            from ultralytics.utils import ops
            if self.mid in ("rtdetr_c2","lcbr_detr"):
                values=decoded[0]
                raw_boxes=ops.xywh2xyxy(values[:,:4]*640)
                raw_boxes=inverse_boxes(raw_boxes,info);raw_scores=values[:,4]
                raw_ids=list(range(len(values)));target="ordinary query native sigmoid crack score; refined final boxes" if self.mid=="lcbr_detr" else "ordinary query native sigmoid crack score"
                order=raw_scores.argsort(descending=True)
                # Preserve the historical C2 sorted-row/unsorted-mask semantics exactly.
                if self.mid=="rtdetr_c2":order=order[raw_scores>.001]
                boxes,scores=raw_boxes[order],raw_scores[order]
                if self.mid=="lcbr_detr":boxes,scores=raw_boxes[order],raw_scores[order]  # export all 300 queries
            elif self.mid=="yolo26m":
                if decoded.ndim!=3 or decoded.shape[-1]!=6:raise ValueError("Not native one-to-one output")
                rows=decoded[0]
                if not (rows[:,5]==0).all():raise ValueError("Noncrack one-to-one result")
                # Recover the real one-to-one anchor indices with the native top-k operator.
                branch=output[1]["one2one"]
                head=self.model.model[-1]
                _,_,anchor_indices=head.get_topk_index(branch["scores"].sigmoid().permute(0,2,1),head.max_det)
                raw_boxes=inverse_boxes(rows[:,:4],info);raw_scores=rows[:,4]
                raw_ids=anchor_indices[0,:,0].detach().cpu().tolist();target="native one-to-one sigmoid crack score; raw anchor index via native top-k"
                keep=raw_scores>.001;boxes,scores=raw_boxes[keep],raw_scores[keep]
            else:
                if self.mid=="yolov5m":
                    from utils.general import xywh2xyxy,non_max_suppression
                    values=decoded[0]
                    raw_boxes=inverse_boxes(xywh2xyxy(values[:,:4]),info)
                    raw_scores=values[:,4]*values[:,5];target="native objectness × crack class sigmoid probability"
                    rows=non_max_suppression(decoded.detach().clone(),conf_thres=.001,iou_thres=.7,
                                             agnostic=False,multi_label=False,max_det=300)[0]
                else:
                    values=decoded[0].T
                    raw_boxes=inverse_boxes(ops.xywh2xyxy(values[:,:4]),info)
                    raw_scores=values[:,4];target="native sigmoid crack probability"
                    rows=ops.non_max_suppression(decoded.detach().clone(),.001,.7,agnostic=False,
                                                max_det=300,classes=None,max_time_img=float("inf"))[0]
                if not (rows[:,5]==0).all():raise ValueError("Noncrack native NMS result")
                boxes,scores=inverse_boxes(rows[:,:4],info),rows[:,4]
                raw_ids=list(range(len(raw_scores)))
        predictions=[dict(category_id=1,bbox=b,score=float(s)) for b,s in zip(boxes.detach().cpu().tolist(),scores.detach().cpu().tolist())]
        return predictions,dict(boxes=raw_boxes,scores=raw_scores,ids=raw_ids,definition=target,geometry=info)

    def unchanged(self):
        if any(p._version!=self.parameter_versions[n] for n,p in self.model.named_parameters()):
            raise ValueError("Parameters changed during visualization")
        for name,buffer in self.model.named_buffers():
            if name in self.buffers and not self.torch.equal(buffer,self.buffers[name]):
                raise ValueError("BN buffer changed during eval/CAM")

    def cam(self,image,case,destination):
        torch,np=self.torch,self.np
        before,raw=self.forward(image)
        annotations=[g for g in image["annotations"] if g["id"] in case["target_gt_ids"]]
        if len(annotations)!=len(case["target_gt_ids"]) or not annotations:raise ValueError("CAM target GT set differs")
        mapped=target_correspondence(raw["boxes"].detach().cpu().numpy(),raw["scores"].detach().cpu().numpy(),before,
                                    annotations,case["display_threshold"],self.request["config"]["cam"]["minimum_correspondence_iou"])
        if len(mapped)!=len(annotations):raise ValueError("No reasonable one-to-one candidate for every fixed GT instance")
        # Same arithmetic mean of the selected native class detection scores in every model.
        indices=[row["candidate_index"] for row in mapped]
        scalar=raw["scores"][indices].mean()
        scalar_value=float(scalar.detach())
        tensors=list(self.features.values())
        if not scalar.requires_grad:raise ValueError("Detached class score; no CAM fallback")
        grads=torch.autograd.grad(scalar,tensors,allow_unused=True)
        arrays,layer_info={},[]
        for (name,value),gradient in zip(self.features.items(),grads):
            stride=self.layer_names[name]
            if tuple(value.shape[-2:])!=(640//stride,640//stride):
                raise ValueError("Target stride/shape differs from validation protocol: "+name)
            response,stats=gradcampp(value.detach().cpu().numpy(),None if gradient is None else gradient.detach().cpu().numpy())
            arrays[f"cam_s{stride}"]=response
            stats.update(activation_shape=list(value.shape),activation_dtype=str(value.dtype),gradient_dtype=str(gradient.dtype) if gradient is not None else None)
            layer_info.append(dict(name=name,stride=stride,**stats))
        if not any(row["status"]=="passed" for row in layer_info):
            raise ValueError("All three fixed scales have zero/near-zero/unused CAM gradients or responses")
        for row in mapped:
            i=row["candidate_index"]
            row.update(raw_candidate_or_roi_or_query_index=raw["ids"][i],score=float(raw["scores"][i].detach()),
                       box=raw["boxes"][i].detach().cpu().tolist(),formed_display_detection=any(
                           p["score"]>=case["display_threshold"] and abs(p["score"]-float(raw["scores"][i].detach()))<1e-6
                           and max(abs(a-b) for a,b in zip(p["bbox"],raw["boxes"][i].detach().cpu().tolist()))<.02 for p in before))
        destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(destination.with_suffix(".npz"),**arrays)
        del scalar,grads,tensors,raw
        with torch.no_grad():after,_=self.forward(image)
        consistency=compare_predictions(before,after,self.request["config"]["smoke_tolerance"],max(image["width"],image["height"])/640)
        self.unchanged()
        metadata=dict(status="passed",**self.identity,image=image["file_name"],image_sha256=sha256(Path(self.request["image_root"])/Path(image["file_name"]).name),
                      geometry=self.preprocess(image)[1],target=dict(definition="arithmetic mean of fixed GT-associated native crack detection scores",
                      instances=mapped,scalar_value=scalar_value,display_threshold_at_compute=case["display_threshold"],
                      gt_pixel_mask=False,localization_loss=False),layers=layer_info,
                      normalization="per-scale positive min-max, equal mean of all 3 scales (unused=zero), full-image min-max; bilinear inverse",
                      no_optimizer_step=True,bn_unchanged=True,parameters_unchanged=True,pre_post_detection_consistency=consistency,
                      protocol_fingerprint=self.request["protocol_fingerprint"],case_fingerprint=fingerprint(case))
        write_json(destination.with_suffix(".json"),metadata)
        return dict(status="passed",npz=str(destination.with_suffix(".npz")),metadata=str(destination.with_suffix(".json")))


def formal_export(request):
    """Call the actual ours evaluation entry in a new directory; redirect dataset cache IO."""
    model=request["model"]
    if model["id"]!="lcbr_detr":raise ValueError("Original-entry export currently applies to ours only")
    source=check_source(model,Path(request["handoff"]))
    weight=Path(path_value(model,"formal_test_weight"))
    if sha256(weight)!=model["checkpoint"]["sha256"]:raise ValueError("Ours weight SHA differs")
    sys.path.insert(0,path_value(model,"vendor"))
    entry=Path(path_value(model,"export_script"))
    sys.path.insert(0,str(entry.parent))
    module=importlib.import_module(entry.stem)
    # The formal entry's arithmetic/validator/streaming postprocess remain intact. Its
    # label/npy cache IO is redirected so no shared dataset or original run is written.
    import ultralytics.data.dataset as dataset
    import inspect
    cache=Path(request["output"])/"runtime/lcbr_detr/export_dataset_cache"
    cache.mkdir(parents=True,exist_ok=True)
    def redirect(function):
        signature=inspect.signature(function)
        def isolated(*args,**kwargs):
            bound=signature.bind(*args,**kwargs)
            old=bound.arguments["path"]
            bound.arguments["path"]=cache/(fingerprint(str(old))+".cache")
            return function(*bound.args,**bound.kwargs)
        return isolated
    dataset.load_dataset_cache_file=redirect(dataset.load_dataset_cache_file)
    dataset.save_dataset_cache_file=redirect(dataset.save_dataset_cache_file)
    from ultralytics.data.base import BaseDataset
    original_load=BaseDataset.load_image
    def isolated_image(self,i,*args,**kwargs):
        old=self.npy_files[i]
        self.npy_files[i]=cache/(fingerprint(str(old))+".npy")
        try:return original_load(self,i,*args,**kwargs)
        finally:self.npy_files[i]=old
    BaseDataset.load_image=isolated_image
    destination=Path(request["destination"])
    data=Path(path_value(model,"run_data_config"))
    val=Path(path_value(model,"val_metrics"))
    if not val.is_file():
        val=destination/"val/metrics.json"
        if not val.is_file():module.evaluate(weight,data,"val",val.parent,device=request["device"].replace("cuda:",""))
    test=destination/"test"
    if not (test/"metrics.json").is_file():
        module.evaluate(weight,data,"test",test,device=request["device"].replace("cuda:",""),val_report=val)
    receipt=read_json(test/"metrics.json")
    predictions=test/"predictions_gt.jsonl.gz"
    if receipt.get("status")!="completed" or receipt.get("images")!=864 or receipt.get("checkpoint_sha256")!=sha256(weight) or receipt.get("predictions_gt_sha256")!=sha256(predictions):
        raise ValueError("Original-entry re-export did not complete with verified identity")
    result=dict(status="passed",kind="new_visualization_reexport",prediction_file=str(predictions),sha256=sha256(predictions),
                source=source,entry=str(entry),entry_sha256=sha256(entry),formal_original_unchanged=True,
                dataset_cache_io="redirected to visualization runtime",metrics_receipt=str(test/"metrics.json"))
    write_json(request["response"],result)
    return result


def execute(request):
    if request["action"]=="formal-export":return formal_export(request)
    if request["model"].get("adapter_entry"):
        model=request["model"]
        if git_head(path_value(model,"work_dir"))!=model["code_commit"] or sha256(path_value(model,"formal_test_weight"))!=model["checkpoint"]["sha256"]:
            raise ValueError("Extension formal code/weight identity differs")
        vendor=path_value(model,"vendor")
        if vendor:sys.path.insert(0,vendor)
        name,factory=model["adapter_entry"].split(":")
        native=getattr(importlib.import_module(name),factory)(request)
        for attribute in ("identity","torch","forward","cam","unchanged","mid"):
            if not hasattr(native,attribute):raise ValueError("Extension factory lacks native worker contract: "+attribute)
        native.identity["registry_record_fingerprint"]=fingerprint(model)
    else:native=Native(request)
    torch=native.torch
    action=request["action"];output=Path(request["response"])
    report=dict(identity=native.identity,status="passed",action=action,images={})
    if action=="preflight":
        report["model_restored"]=True
        report["gradient_validated"]=False
    elif action=="cam":
        images={im["file_name"]:im for im in request["images"]}
        for cell_index,case in enumerate(progress(request["cases"],desc=native.mid+" CAM",total=len(request["cases"]))):
            key=case["image"]
            try:
                with ExitStack() as stack:
                    stack.enter_context(torch.enable_grad())
                    if native.mid=="yolov13l":
                        stack.enter_context(native.backend.native_fp32())
                        audit=stack.enter_context(native.backend.ArithmeticAudit(require_fp32=True))
                    result=native.cam(images[key],case,Path(request.get("destinations",{}).get(key,request["destination"]))/Path(key).stem)
                    if native.mid=="yolov13l":
                        metadata=read_json(result["metadata"]);metadata["arithmetic_audit"]=audit.report();write_json(result["metadata"],metadata)
            except Exception as error:
                result=dict(status="failed",error=repr(error),traceback=traceback.format_exc(),image=key,
                            protocol_fingerprint=request["protocol_fingerprint"])
                write_json(Path(request["destination"])/(Path(key).stem+".failed.json"),result)
                report["status"]="failed"
            report["images"][key]=result
            write_json(output,report)
            if request.get("progress_context"):
                state=request["progress_context"];done=state["completed"]+cell_index+1
                write_json(Path(request["output"])/"progress.json",dict(stage="cam",completed=done,total=state["total"],
                    percent=round(100*done/state["total"],2),model=native.mid,image=key,status=result["status"],utc=datetime.now(timezone.utc).isoformat()))
    else:
        # Independent native direct graph is used for parity, with no_grad chosen only here.
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if native.mid=="yolov13l":
                audit=stack.enter_context(native.backend.ArithmeticAudit(require_fp32=True))
            for image in progress(request["images"],desc=native.mid+" "+action,total=len(request["images"])):
                actual,_=native.forward(image)
                key=image["file_name"]
                if action=="smoke":
                    report["images"][key]=compare_predictions(request["reference"][key],actual,request["config"]["smoke_tolerance"],
                        max(image["width"],image["height"])/640,rounded_c2=native.mid=="rtdetr_c2")
                else:report["images"][key]=actual
            if native.mid=="yolov13l":report["arithmetic_audit"]=audit.report()
    native.unchanged()
    write_json(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request",type=Path,required=True)
    args=parser.parse_args();request=read_json(args.request)
    try:execute(request)
    except Exception as error:
        write_json(request["response"],dict(status="failed",error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=="__main__":main()
