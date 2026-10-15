"""Display-point diagnostics and diversity-aware proposals; no formal AP rewriting."""
from __future__ import annotations
import csv
import math
from pathlib import Path
import re
import numpy as np

from .common import fingerprint, progress, write_json


def iou_matrix(a, b):
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                              np.maximum(a[:, None, :2], b[None, :, :2])).prod(2)
    aa = np.maximum(a[:, 2:]-a[:, :2], 0).prod(1)
    bb = np.maximum(b[:, 2:]-b[:, :2], 0).prod(1)
    union = aa[:, None] + bb[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match(predictions, annotations, threshold, display_threshold):
    """Score-ordered, same-class, one-to-one detection matching, stable on ties."""
    overlaps = iou_matrix([p["bbox"] for p in predictions], [g["bbox"] for g in annotations])
    visible = sorted((i for i, p in enumerate(predictions) if p["score"] >= display_threshold),
                     key=lambda i: (-predictions[i]["score"], i))
    used, matches, false_positives = set(), [], []
    for index in visible:
        remaining = [j for j in range(len(annotations)) if j not in used]
        gt_index = max(remaining, key=lambda j: overlaps[index, j]) if remaining else None
        if gt_index is not None and overlaps[index, gt_index] >= threshold:
            used.add(gt_index)
            matches.append(dict(prediction_index=index, gt_index=gt_index,
                                gt_id=annotations[gt_index]["id"], iou=float(overlaps[index, gt_index]),
                                score=predictions[index]["score"]))
        else:
            false_positives.append(index)
    misses = []
    for j, gt in enumerate(annotations):
        if j in used:
            continue
        candidates = sorted(range(len(predictions)), key=lambda i: (-overlaps[i, j], -predictions[i]["score"], i))
        low = [i for i in candidates if overlaps[i, j] >= threshold and predictions[i]["score"] < display_threshold]
        shown = [i for i in visible if overlaps[i, j] >= threshold]
        reason = "duplicate_assignment" if shown else "below_display_threshold" if low else "localization_or_no_candidate"
        best = candidates[0] if candidates else None
        misses.append(dict(gt_id=gt["id"], gt_index=j, reason=reason,
                           best_prediction_index=best, best_iou=float(overlaps[best, j]) if best is not None else 0,
                           best_score=predictions[best]["score"] if best is not None else None,
                           below_threshold_candidate_indices=low))
    fp_details=[]
    for i in false_positives:
        j=int(np.argmax(overlaps[i])) if len(annotations) else None
        overlap=float(overlaps[i,j]) if j is not None else 0
        reason="duplicate_assignment" if overlap>=threshold else "localization_insufficient" if overlap>=.1 else "no_GT_overlap_background_candidate"
        fp_details.append(dict(prediction_index=i,score=predictions[i]["score"],best_gt_id=annotations[j]["id"] if j is not None else None,
                               best_iou=overlap,reason=reason))
    return dict(tp=len(matches), fp=len(false_positives), fn=len(misses), matches=matches,
                false_positive_prediction_indices=false_positives, false_positive_details=fp_details,misses=misses)


def statistics(images, predictions, model_ids, threshold, output):
    rows, details = [], {}
    for key, image in progress(images.items(), total=len(images), desc="per-image statistics"):
        details[key] = {}
        for mid in model_ids:
            result = {str(t): match(predictions[mid][key], image["annotations"], t, threshold) for t in (.5, .75)}
            details[key][mid] = result
            for t, value in result.items():
                rows.append(dict(image=key, image_id=image["id"], model_id=mid, iou=t,
                                 display_threshold=threshold, tp=value["tp"], fp=value["fp"], fn=value["fn"],
                                 gt_count=len(image["annotations"]), raw_prediction_count=len(predictions[mid][key]),
                                 mean_matched_iou=float(np.mean([m["iou"] for m in value["matches"]])) if value["matches"] else 0))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "per_image.json", {"policy": "display-point diagnostics; original AP unchanged",
               "model_order": model_ids, "display_threshold": threshold, "images": details})
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return details


def correspondence(boxes, annotations, minimum_iou=.1):
    """GT determines instance identity only; no threshold or pixel mask enters the score."""
    overlaps = iou_matrix(boxes, [g["bbox"] for g in annotations])
    edges = sorted(((float(overlaps[i, j]), i, j) for i in range(len(boxes)) for j in range(len(annotations))
                    if overlaps[i, j] >= minimum_iou), key=lambda e: (-e[0], e[2], e[1]))
    used_p, used_g, selected = set(), set(), []
    for overlap, i, j in edges:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            selected.append(dict(candidate_index=i, gt_id=annotations[j]["id"], iou=overlap))
    return sorted(selected, key=lambda row: row["gt_id"])


def target_correspondence(raw_boxes,raw_scores,final_predictions,annotations,display_threshold,minimum_iou=.1):
    """Prefer real final detections; do not explain an arbitrary low-score near-GT anchor.

    One common protocol: visible final detections, below-display formal detections,
    then raw candidates for missed instances. Each stage uses geometric one-to-one
    correspondence. The raw score, never a detached final JSON value, is differentiated.
    """
    raw_boxes=np.asarray(raw_boxes,dtype=float).reshape(-1,4)
    raw_scores=np.asarray(raw_scores,dtype=float)
    selected=[];used=set();remaining=list(annotations)
    for visible,tier in ((True,"display_detection"),(False,"formal_below_display_threshold")):
        finals=[p for p in final_predictions if (p["score"]>=display_threshold)==visible]
        mapped=correspondence([p["bbox"] for p in finals],remaining,minimum_iou)
        for row in mapped:
            final=finals[row["candidate_index"]]
            errors=np.max(np.abs(raw_boxes-np.asarray(final["bbox"])),axis=1)
            limit=.002+1e-6*max(map(abs,final["bbox"]))
            possible=[i for i in np.flatnonzero((errors<=limit)&(np.abs(raw_scores-final["score"])<=1e-6)) if i not in used]
            if not possible:raise ValueError("Native final detection could not be mapped back to a differentiable candidate")
            index=min(possible,key=lambda i:(errors[i],i));used.add(index)
            selected.append({**row,"candidate_index":int(index),"association_tier":tier,
                             "final_score":final["score"],"final_bbox":final["bbox"]})
        explained={r["gt_id"] for r in selected}
        remaining=[g for g in annotations if g["id"] not in explained]
    available=[i for i in range(len(raw_boxes)) if i not in used]
    for row in correspondence(raw_boxes[available],remaining,minimum_iou):
        selected.append({**row,"candidate_index":available[row["candidate_index"]],"association_tier":"raw_candidate_no_final_detection"})
    return sorted(selected,key=lambda r:r["gt_id"])


def compare_predictions(reference, actual, tolerance, scale=1, rounded_c2=False):
    """Full bipartite tolerance matching handles ordering, duplicate boxes and score ties."""
    if len(reference) != len(actual):
        raise ValueError(f"Cache/forward count differs: {len(reference)} vs {len(actual)}")
    n = len(reference)
    coord_atol = tolerance["input_pixel_atol"] * scale + (.00101 if rounded_c2 else 0)
    score_atol = tolerance["score_atol"] + (.00000501 if rounded_c2 else 0)
    edges = []
    for p in reference:
        valid = []
        for j, q in enumerate(actual):
            error = max(abs(x-y) for x, y in zip(p["bbox"], q["bbox"]))
            allowed = coord_atol + tolerance["coordinate_rtol"] * max(map(abs, p["bbox"]))
            if p["category_id"] == q["category_id"] and error <= allowed and abs(p["score"]-q["score"]) <= score_atol:
                valid.append((error, j))
        edges.append([j for _, j in sorted(valid)])
    assigned = {}
    def assign(i, seen):
        for j in edges[i]:
            if j not in seen:
                seen.add(j)
                if j not in assigned or assign(assigned[j], seen):
                    assigned[j] = i
                    return True
        return False
    for i in sorted(range(n), key=lambda k: len(edges[k])):
        if not assign(i, set()):
            # Report nearest observed errors, never automatically relax tolerances.
            nearest = sorted((max(abs(x-y) for x, y in zip(reference[i]["bbox"], q["bbox"])),
                              abs(reference[i]["score"]-q["score"])) for q in actual)[:3]
            raise ValueError(f"No class/coordinate/score bijection; reference {i}, nearest errors {nearest}, "
                             f"coord_atol={coord_atol}, score_atol={score_atol}")
    pairs = [(i, j) for j, i in assigned.items()]
    return dict(status="passed", detections=n, matching="full bipartite; order-independent",
                max_coordinate_error=max((abs(x-y) for i,j in pairs for x,y in zip(reference[i]["bbox"], actual[j]["bbox"])), default=0),
                max_score_error=max((abs(reference[i]["score"]-actual[j]["score"]) for i,j in pairs), default=0),
                coordinate_atol=coord_atol, score_atol=score_atol, tolerance=tolerance,
                c2_serialization="xywh three decimals; score five decimals" if rounded_c2 else None)


def image_descriptor(path):
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        small = np.asarray(ImageOps.fit(image.convert("RGB"), (24, 24)), dtype=float)/255
    gray = small.mean(2)
    return dict(brightness=float(gray.mean()), global_contrast=float(gray.std()),
                texture_energy=float(np.abs(np.diff(gray, axis=0)).mean()+np.abs(np.diff(gray, axis=1)).mean())), small.flatten()


def propose(images, predictions, details, model_ids, image_root, threshold, output, count=5, candidates=40):
    if "lcbr_detr" not in model_ids or not all(mid in model_ids for mid in ("rtdetr_c2", "yolo26m", "yolov13l")):
        raise ValueError("All eight models, including strong comparisons, are required for selection")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pool = []
    for key, data in details.items():
        ours = data["lcbr_detr"]["0.5"]
        if not ours["tp"]:
            continue
        competitors = [mid for mid in model_ids if mid != "lcbr_detr"]
        strong = ["rtdetr_c2", "yolo26m", "yolov13l"]
        own_error = ours["fp"]+ours["fn"]
        delta = {mid: data[mid]["0.5"]["fp"]+data[mid]["0.5"]["fn"]-own_error for mid in competitors}
        recall = sum(data[mid]["0.5"]["fn"]-ours["fn"] for mid in strong)
        fp = sum(data[mid]["0.5"]["fp"]-ours["fp"] for mid in strong)
        location = sum(data["lcbr_detr"]["0.75"]["tp"]-data[mid]["0.75"]["tp"] for mid in strong)
        if max(delta[mid] for mid in strong) <= 0 and location <= 0:
            continue
        group = max([(recall, "fewer_misses"), (fp, "fewer_false_positives"),
                     (location, "better_localization")])[1]
        quality = ours["tp"] / max(1, ours["tp"]+ours["fp"]+ours["fn"])
        if quality<.5:
            continue
        # Clear margins avoid selecting a serialization/threshold boundary as a gain.
        robust=[]
        annotations=images[key]["annotations"]
        for own_match in ours["matches"]:
            gt=annotations[own_match["gt_index"]]
            if own_match["iou"]<.55 or own_match["score"]<threshold+.05:continue
            for mid in strong:
                rows=predictions[mid][key]
                overlap=iou_matrix([p["bbox"] for p in rows],[gt["bbox"]])[:,0]
                visible=[i for i,p in enumerate(rows) if p["score"]>=threshold]
                best_visible=max((float(overlap[i]) for i in visible),default=0)
                best_corresponding_score=max((p["score"] for i,p in enumerate(rows) if overlap[i]>=.5),default=0)
                clear_localization=best_visible>=.1 and own_match["iou"]-best_visible>=.05
                clear_miss=best_visible<.5 and best_corresponding_score<=max(0,threshold-.05)
                if clear_localization or clear_miss:
                    robust.append(dict(gt_id=gt["id"],comparison_model=mid,ours_iou=own_match["iou"],
                                       best_visible_comparison_iou=best_visible,best_geometrically_matching_comparison_score=best_corresponding_score,
                                       kind="localization_margin" if clear_localization else "clear_display_miss"))
        background_gain=[mid for mid in strong if delta[mid]>0 and any(r["reason"]=="no_GT_overlap_background_candidate" and r["score"]>=threshold+.05
                          for r in data[mid]["0.5"]["false_positive_details"])]
        if not robust and not background_gain:continue
        rank = 2*sum(delta[mid] for mid in strong)+sum(delta.values())/max(1,len(competitors))+location+3*quality
        image = images[key]
        path = Path(image_root)/Path(key).name
        descriptor, vector = image_descriptor(path) if path.is_file() else ({"image_available": False}, None)
        target_ids = list(dict.fromkeys(r["gt_id"] for r in robust))[:3]
        if not target_ids:target_ids=[r["gt_id"] for r in sorted(ours["matches"],key=lambda r:-r["iou"])[:3]]
        # Source family prevents selecting augmented copies of one physical photograph.
        family = re.match(r"crack_\d+", Path(key).stem)
        pool.append(dict(image=key, image_id=image["id"], target_gt_ids=target_ids,
                         advantage_type=group, rank_score=rank, quality=quality,
                         clear_advantage_evidence=robust,background_false_positive_candidate_models=background_gain,
                         target_gt_selection="up to 3 clear-advantage GTs; otherwise up to 3 highest-IoU ours TPs; fixed before CAM",
                         comparison_error_delta=delta, comparison_statistics=data,
                         observable_image_descriptors=descriptor,
                         scene_status="requires visual confirmation; descriptors are not semantic scene labels",
                         source_family=family.group() if family else Path(key).stem,
                         view_region=[0,0,image["width"],image["height"]], display_threshold=threshold,
                         reason=f"IoU .50 error delta vs C2/v26/v13={tuple(delta[mid] for mid in strong)}; "
                                f"IoU .75 TP advantage sum={location}; ours TP/FP/FN={ours['tp']}/{ours['fp']}/{ours['fn']}",
                         _vector=vector))
    pool.sort(key=lambda row: (-row["rank_score"], row["image"]))
    rank_min=min((r["rank_score"] for r in pool),default=0)
    rank_span=max((r["rank_score"] for r in pool),default=1)-rank_min
    chosen, families = [], set()
    while pool and len(chosen) < min(candidates, len(images)):
        def diversity_value(row):
            penalty = 0 if row["source_family"] not in families else 1000
            group_bonus = .45 if row["advantage_type"] not in {r["advantage_type"] for r in chosen} else 0
            distances = [float(np.mean((row["_vector"]-r["_vector"])**2)) for r in chosen
                         if row["_vector"] is not None and r["_vector"] is not None]
            normalized_rank=(row["rank_score"]-rank_min)/max(1,rank_span)
            return 2*normalized_rank + group_bonus + 8*min(distances, default=0)-penalty
        best = max(pool, key=diversity_value)
        pool.remove(best)
        if best["source_family"] in families:
            continue
        families.add(best["source_family"])
        chosen.append(best)
    for i, row in enumerate(chosen):
        row.pop("_vector")
        row["order"] = i+1
        path=Path(image_root)/Path(row["image"]).name
        from .common import sha256
        row["image_identity"] = {"file_name":row["image"], "gt_image_id":row["image_id"],
                                 "bytes":path.stat().st_size if path.is_file() else None,
                                 "sha256":sha256(path) if path.is_file() else None}
    if len(chosen) < count:
        raise ValueError("Too few distinct qualifying source photographs; no arbitrary fill-in")
    candidates_doc = dict(status="proposed_pending_visual_review", model_order=model_ids,
                          display_threshold=threshold, selection_protocol="quality + strong/all-model deltas + scene proxy diversity v1",
                          warning="Automatic diagnostics are not a representative-scene claim", cases=chosen)
    write_json(output/"candidates.json", candidates_doc)
    result = {**candidates_doc, "cases": chosen[:count]}
    result["fingerprint"] = fingerprint(result)
    write_json(output/"cases.proposed.json", result)
    # Keep the complete unfiltered predictions for all candidate previews.
    for case in chosen:
        write_json(output/"predictions"/(Path(case["image"]).stem+".json"),
                   {"case":case, "ground_truth":images[case["image"]]["annotations"],
                    "models":{mid: predictions[mid][case["image"]] for mid in model_ids}})
    return result, candidates_doc
