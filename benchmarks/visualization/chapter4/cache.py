"""Lossless cache adapters. All internal boxes are original-pixel xyxy/category 1."""
from __future__ import annotations
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path, PurePosixPath

from .common import read_json, sha256

ARCHIVED = {"fasterrcnn_r50_fpn": "evidence/fasterrcnn/run/predictions/test.jsonl.gz",
            "yolov5m": "evidence/yolov5m/public_fp32/predictions/test.jsonl.gz",
            "yolov13l": "evidence/yolov13l/run/predictions/test.jsonl.gz",
            "rtdetr_c2": "evidence/rtdetr_c2/test_run/predictions.json.gz"}


def canonical(value):
    return "images/test/" + PurePosixPath(str(value).replace("\\", "/")).name


def ground_truth(base):
    gt = read_json(Path(base) / "data/test_gt.json")
    images, by_id = {}, {}
    for image in gt["images"]:
        key = canonical(image["file_name"])
        if key in images or image["id"] in by_id:
            raise ValueError("Duplicate GT image identity")
        images[key] = {**image, "file_name": key, "annotations": []}
        by_id[image["id"]] = key
    for annotation in gt["annotations"]:
        x, y, w, h = annotation["bbox"]
        if annotation["category_id"] != 1 or min(w, h) <= 0:
            raise ValueError("Unexpected public GT category/area")
        images[by_id[annotation["image_id"]]]["annotations"].append(
            {**annotation, "bbox": [x, y, x+w, y+h]})
    return images


def _prediction(row, index, xywh=False, native_zero=False):
    category = row.get("class_id") if native_zero else row.get("category_id")
    if category != (0 if native_zero else 1):
        raise ValueError("Unexpected native/public crack category")
    box = list(row["bbox"])
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError("Nonfinite/malformed box")
    if xywh:
        box[2] += box[0]
        box[3] += box[1]
    score = float(row["score"])
    if not 0 <= score <= 1 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Invalid score/box area; padding boxes are allowed")
    return dict(bbox=box, score=score, category_id=1, prediction_index=index, raw=row)


def read_cache(path, model, images, expected_sha=None):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        text = stream.read()
    c2 = model["id"] == "rtdetr_c2" and text.lstrip().startswith("[")
    digest = __import__("hashlib").sha256(text.encode()).hexdigest() if c2 else sha256(path)
    if expected_sha and digest != expected_sha:
        raise ValueError("Prediction SHA differs for " + model["id"])
    records, meta = {}, {}
    if c2:
        grouped = defaultdict(list)
        for row in json.loads(text):
            # C2 stem image_id intentionally ignored; filename is the audited join key.
            grouped[canonical(row["file_name"])].append(row)
        source_rows = [{"image": key, "predictions": rows} for key, rows in grouped.items()]
        meta = {"precision": {"bbox_xywh_decimals": 3, "score_decimals": 5},
                "sha256_uncompressed": digest, "id_join": "file_name; never numeric GT image_id"}
    else:
        source_rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "metadata":
                if meta or source_rows:
                    raise ValueError("Duplicate/misplaced metadata row")
                meta = row
                weight = row.get("identity", {}).get("checkpoint_sha256")
                if weight and weight != model["checkpoint"]["sha256"]:
                    raise ValueError("Prediction checkpoint identity differs")
                identity=row.get("identity",{})
                if model.get("test_gt_sha256") and identity.get("gt_sha256")!=model["test_gt_sha256"]:
                    raise ValueError("Formal cache GT reference differs (Faster keeps its own metadata SHA)")
                if identity.get("split") not in (None,"test"):
                    raise ValueError("Refusing a non-test prediction cache")
            else:
                source_rows.append(row)
    for row in source_rows:
        key = canonical(row["image"])
        if key not in images or key in records:
            raise ValueError("Unknown/duplicate prediction filename: " + key)
        image = images[key]
        if not c2:
            if row.get("box_format") != "xyxy" or row.get("coordinate_space") != "original_image_pixels":
                raise ValueError("Refusing second inverse/unknown coordinate space")
            if "image_id" in row and row["image_id"] != image["id"]:
                raise ValueError("Public JSONL image_id differs from filename join")
            if (row.get("width", image["width"]), row.get("height", image["height"])) != (image["width"], image["height"]):
                raise ValueError("Image dimensions differ")
            if row.get("original_size_hw", [image["height"], image["width"]]) != [image["height"], image["width"]]:
                raise ValueError("Native stream dimensions differ")
        predictions = [_prediction(p, i, xywh=c2, native_zero="class_id" in p)
                       for i, p in enumerate(row["predictions"])]
        if len(predictions) > 300:
            raise ValueError("Native max_det exceeded")
        if "ground_truth" in row:
            from .analysis import compare_predictions
            native_gt=[{**_prediction({**p,"score":1},i,native_zero=True),"score":1} for i,p in enumerate(row["ground_truth"])]
            common_gt=[dict(category_id=1,score=1,bbox=g["bbox"]) for g in image["annotations"]]
            compare_predictions(common_gt,native_gt,dict(input_pixel_atol=.002,coordinate_rtol=0,score_atol=0))
        records[key] = predictions
    if set(records) != set(images):
        raise ValueError(f"Incomplete {model['id']} cache: {len(records)}/{len(images)}")
    return records, {"path": str(path), "sha256": sha256(path), "source_metadata": meta,
                     "images": len(records), "native_class": 1 if model["id"].startswith("faster") else 0,
                     "public_class": 1, "internal_boxes": "original pixels xyxy; no clipping/NMS"}
