"""Optional offline geometry survey of at most 64 parent val images; never opens images/test."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from ncr_v1_common import read, require, sha, write
import torch
from ncr_v1_results import native_pairs
from ultralytics.models.utils.ncr import distribution, ncr_terms
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xyxy2xywh


def diagnose(folder, output):
    report = read(folder/"metrics.json")
    require(report["split"] == "val" and report["status"] == "completed" and report["export_complete"], "Completed parent val export required")
    stream = folder/"predictions_gt.jsonl.gz"
    require(sha(stream) == report["predictions_gt_sha256"], "Parent export hash differs")
    gate_rows, errors, ids, matched = [], [], [], 0
    raw_sum = 0.
    with gzip.open(stream, "rt", encoding="utf-8") as data:
        for index, line in enumerate(data):
            if index == 64: break
            row=json.loads(line); ids.append(row["image"])
            pred=[p for p in row["predictions"] if p["used_for_metrics"]]
            pb=torch.tensor([p["bbox"] for p in pred]).reshape(-1,4)
            gb=torch.tensor([p["bbox"] for p in row["ground_truth"]]).reshape(-1,4)
            pairs=native_pairs(box_iou(gb,pb).numpy(),.5)
            if not len(pairs): continue
            h,w=row["original_size_hw"]; normalization=torch.tensor([w,h,w,h])
            b=xyxy2xywh(pb[pairs[:,1]])/normalization; g=xyxy2xywh(gb[pairs[:,0]])/normalization
            raw,gate,u,_=ncr_terms(b,g,(640,640))
            raw_sum += float(raw)*len(pairs); matched += len(pairs)
            gate_rows.append(gate); errors.append(u.abs())
    gate=torch.cat(gate_rows) if gate_rows else torch.empty(0,2)
    error=torch.cat(errors) if errors else torch.empty(0,2)
    result=dict(status="PASSED", scope="offline first at most 64 parent val images; no inference or test",
                parent_best_sha256=report["checkpoint_sha256"], export_sha256=sha(stream), image_ids=ids,
                images=len(ids), matches=matched, matching="native IoU50 detection matching recomputed in archived original-pixel coordinates",
                active_fraction_xy=(gate>0).float().mean(0).tolist() if matched else [0.,0.],
                nested_and_offset_fraction_xy=((gate>0)&(error>1e-7)).float().mean(0).tolist() if matched else [0.,0.],
                h_xy=[distribution(gate[:,i]) for i in (0,1)], relative_center_on_active_axes=distribution(error[gate>0]),
                raw_geometry_average=raw_sum/max(matched,1),
                limitations=["Parent archive lacks original query indices/logits; this is not training-Hungarian positive incidence.",
                            "Recomputed FP32 coordinate transforms can differ at matching boundaries.",
                            "Descriptive nesting opportunity only; no NCR efficacy or AP improvement evidence."])
    write(output,result)
    print(json.dumps({k:v for k,v in result.items() if k != "image_ids"},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-val",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); torch.set_num_threads(4); diagnose(args.parent_val,args.output)
