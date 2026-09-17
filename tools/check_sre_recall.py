"""Small semantic checks for fixed-label recall and original validator matching."""
import argparse
import json
from pathlib import Path
import tempfile
import gzip
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from eval_sre import fixed_precision_recall, original_matcher, postprocess, write_new
from pack_sre_light import package, verify_archive


def capture_api_smoke(folder):
    """Two synthetic images, actual validator update_metrics/capture; no detector inference."""
    import eval_sre as evaluation
    folder = Path(folder)
    identity = dict(images=2, boxes=2, split_paths_sha256="synthetic", label_inventory_sha256="synthetic")
    (folder / "docs/sre").mkdir(parents=True)
    (folder / "docs/sre/parent_dataset_inventory.json").write_text(json.dumps(dict(val=identity)))
    weights, data = folder / "synthetic.pt", folder / "data.yaml"
    weights.write_bytes(b"synthetic no-model-load API fixture")
    data.write_text("names: [crack]\n")
    class SyntheticModel:
        def __init__(self, path):
            self.model = torch.nn.Linear(1, 1)
        def val(self, validator, **settings):
            v = validator(args=settings, save_dir=folder / "evaluation/plots")
            v.device, v.training = torch.device("cpu"), False
            v.data = dict(path=str(folder), val="synthetic")
            v.dataloader = SimpleNamespace(dataset=object())
            v.init_metrics(SimpleNamespace(names={0: "crack"}))
            raw = torch.tensor([[[.5,.5,.5,.5,.9],[.5,.5,.5,.5,.8]],
                                [[.5,.5,.5,.5,-1],[.5,.5,.5,.5,-1]]])
            batch = dict(img=torch.zeros(2,3,640,640), batch_idx=torch.tensor([0,1]),
                         cls=torch.tensor([[0.],[0.]]), bboxes=torch.tensor([[.5,.5,.5,.5],[.5,.5,.5,.5]]),
                         ori_shape=[(640,640),(640,640)], ratio_pad=[None,None],
                         im_file=[str(folder / "image1.jpg"), str(folder / "image2.jpg")])
            v.update_metrics(v.postprocess(raw), batch)
            v.metrics.process(plot=False)
            v.metrics.confusion_matrix = v.confusion_matrix
            return v.metrics
    with patch.object(evaluation, "ROOT", folder), patch.object(evaluation, "RTDETR", SyntheticModel), \
         patch.object(evaluation, "check_det_dataset", return_value=dict(nc=1)), \
         patch.object(evaluation, "runtime", return_value=dict(commit="synthetic-no-inference")), \
         patch.object(evaluation, "matcher_identity", return_value={}), \
         patch.object(evaluation, "split_identity", return_value=identity), \
         patch.object(evaluation, "verify_architecture", return_value=dict(architecture="synthetic")):
        report = evaluation.evaluate(weights, data, "val", folder / "evaluation", "parent_cbr_lif", "cpu")
    with gzip.open(folder / "evaluation/predictions_gt.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    actual = original_matcher()._process_batch(
        dict(bboxes=torch.tensor([[160.,160.,480.,480.],[160.,160.,480.,480.]]),cls=torch.zeros(2)),
        dict(bboxes=torch.tensor([[160.,160.,480.,480.]]),cls=torch.zeros(1)))["tp"]
    assert np.array_equal(np.asarray([p["tp_iou"] for p in rows[0]["predictions"]]), actual)
    assert rows[0]["image"] == "image1.jpg" and rows[1]["image"] == "image2.jpg"
    assert rows[1]["predictions"] == [] and len(rows[1]["ground_truth"]) == 1
    assert report["ground_truth"] == 2 and report["metric_predictions"] == 2 and report["status"] == "completed"
    assert np.asarray(report["ap_by_class"]).shape == (1,10) and len(report["curves"]) == 4
    return report


def checks():
    passed = []
    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        passed.append(name)
    tied = fixed_precision_recall([.9, .8, .8], [1, 1, 0], 2)
    check(tied["R_at_P"] == .5 and tied["threshold"] == .9, "ties pooled; impossible internal tie point rejected")
    none = fixed_precision_recall([.9, .8], [0, 1], 2)
    check(none["status"] == "NOT_ACHIEVED" and none["R_at_P"] is none["threshold"] is None, "no achieved point is null")
    empty = fixed_precision_recall([], [], 2)
    check(empty["status"] == "NOT_ACHIEVED" and empty["precision"] is None, "empty prediction never a valid point")
    best = fixed_precision_recall([.95, .9, .85], [1, 0, 0], 2, .3)
    check(best["threshold"] == .95 and best["precision"] == 1, "maximum recall ties choose highest precision")
    pooled = fixed_precision_recall([.8, .9, .8], [0, 1, 1], 2, .6)
    check(pooled["R_at_P"] == 1 and pooled["TP"] == 2 and pooled["FP"] == 1, "whole dataset original score groups")
    matcher = original_matcher()
    gt = dict(bboxes=torch.tensor([[0., 0., 10., 10.]]), cls=torch.tensor([0.]))
    pred = dict(bboxes=torch.tensor([[0., 0., 10., 10.], [.1, .1, 9.9, 9.9]]), cls=torch.tensor([0., 0.]))
    flags = matcher._process_batch(pred, gt)["tp"]
    check(flags[:,0].sum() == 1, "actual validator: two predictions cannot duplicate one GT")
    dup_gt = dict(bboxes=torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]]), cls=torch.tensor([0., 0.]))
    one_pred = dict(bboxes=pred["bboxes"][:1], cls=pred["cls"][:1])
    check(matcher._process_batch(one_pred, dup_gt)["tp"][:,0].sum() == 1, "actual validator: one prediction cannot match two GT")
    wrong_cls = dict(bboxes=one_pred["bboxes"], cls=torch.tensor([1.]))
    check(not matcher._process_batch(wrong_cls, gt)["tp"].any(), "actual validator class-aware match")
    # An unsorted low/high pair catches the original unaligned mask bug.
    raw = torch.tensor([[[.5,.5,.2,.2,.0001], [.5,.5,.2,.2,.9]]])
    selected, affected = postprocess(raw, 640, .001)
    check(affected == 1 and len(selected[0]["conf"]) == 1 and selected[0]["conf"][0] > .8,
          "original corrected_sorted_conf_mask_v1 filters sorted scores")
    try:
        fixed_precision_recall([.9,.8], [1,1], 1)
    except RuntimeError:
        passed.append("impossible TP count rejected")
    else:
        raise AssertionError("too many TP accepted")
    with tempfile.TemporaryDirectory(prefix="sre-light-check-") as temporary:
        root = Path(temporary)
        capture_api_smoke(root / "capture")
        passed.append("two-image synthetic API smoke: actual validator capture, image identity, empty image and full metric JSON")
        metadata = root / "metadata"
        metadata.mkdir()
        (metadata / "training_state.json").write_text('{"status":"NOT_STARTED"}')
        (metadata / "fake.pt").write_bytes(b"must never be packaged")
        (metadata / "predictions_gt.jsonl.gz").write_bytes(b"large evidence provenance only")
        report = package(metadata, root / "light.tar.gz")
        manifest = verify_archive(root / "light.tar.gz")
        check(report["bytes"] < 20*1024*1024 and len(manifest["omitted"]) == 2,
              "light archive bounded and independently hash verified; weights/raw predictions excluded")
        check(not any(k.endswith(".pt") or "predictions_gt" in k for k in manifest["files"]),
              "excluded evidence recorded only in manifest")
    return dict(status="PASSED", checks=passed, count=len(passed), training="NOT_STARTED", final_test="NOT_RUN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = checks()
    if args.output:
        write_new(args.output, report)
    print(json.dumps(report, indent=2))
