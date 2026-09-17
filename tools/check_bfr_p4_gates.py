#!/usr/bin/env python
"""CPU-only negative gate and evaluation-protocol unit diagnostics.

All preflight fixtures are explicitly synthetic and exist only in memory. This
tool never writes a preflight report, launches training, opens a checkpoint,
loads images, or performs val/test inference. Its output cannot authorize start.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import torch

from bfr_p4_lifecycle import ROOT, VARIANTS, recipe, require_preflight, sha256
from eval_bfr_p4 import EVAL, POLICY, corrected_postprocess
from ultralytics.utils import YAML


# Independent literal transcription of the user-supplied Appendix A contract.
# Comparing only parent_args.yaml with appendix_args.yaml would miss a shared
# accidental change in both files.
EXPECTED_PARENT = dict(
    task="detect", mode="train",
    model="/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix/weights/c19_lif_v1_controlled_init.pt",
    data="/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml",
    epochs=200, time=None, patience=50, batch=16, imgsz=640, save=True, save_period=-1,
    cache=False, device="0", workers=8, project="/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series",
    name="c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug", exist_ok=False, pretrained=True,
    optimizer="AdamW", verbose=True, seed=42, deterministic=True, single_cls=False,
    rect=False, cos_lr=True, close_mosaic=10, resume=False, amp=True, fraction=1.0,
    profile=False, freeze=None, multi_scale=0.0, compile=False, overlap_mask=True,
    mask_ratio=4, dropout=0.0, val=True, split="val", save_json=False, conf=None,
    iou=0.7, max_det=300, half=False, dnn=False, plots=True, end2end=None, source=None,
    vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False,
    classes=None, retina_masks=False, embed=None, show=False, save_frames=False,
    save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True,
    show_boxes=True, line_width=None, format="torchscript", keras=False, optimize=False,
    int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False,
    lr0=0.0005, lrf=0.01, momentum=0.937, weight_decay=0.0001, warmup_epochs=5,
    warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0,
    kobj=1.0, rle=1.0, angle=1.0, nbs=64, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
    degrees=5, translate=0.1, scale=0.4, shear=1.5, perspective=0.0002, flipud=0.2,
    fliplr=0.5, bgr=0.0, mosaic=0.8, mixup=0.05, cutmix=0, copy_paste=0,
    copy_paste_mode="flip", auto_augment=None, erasing=0, cfg=None, tracker="botsort.yaml",
    save_dir="/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug",
)


def same_typed_mapping(actual, expected):
    assert set(actual) == set(expected), (set(actual) - set(expected), set(expected) - set(actual))
    for key, value in expected.items():
        assert type(actual[key]) is type(value) and actual[key] == value, (key, actual[key], value)


def recipe_checks():
    assert len(EXPECTED_PARENT) == 109
    parent = YAML.load(ROOT / "docs/bfr_p4/parent_args.yaml")
    appendix = YAML.load(ROOT / "docs/bfr_p4/appendix_args.yaml")
    same_typed_mapping(parent, EXPECTED_PARENT)
    same_typed_mapping(appendix, EXPECTED_PARENT)
    identities = {"model", "data", "name", "project", "save_dir"}
    reports = {}
    for variant in VARIANTS:
        initialized, data = ROOT / "synthetic_not_created/init.pt", ROOT / "synthetic_not_created/data.yaml"
        args, differences = recipe(variant, initialized, data)
        assert len(args) == len(differences) == 109
        assert set(args) == set(EXPECTED_PARENT)
        unchanged = {key: value for key, value in args.items() if key not in identities}
        same_typed_mapping(unchanged, {key: value for key, value in EXPECTED_PARENT.items() if key not in identities})
        assert args["model"] == str(initialized.resolve()) and args["data"] == str(data.resolve())
        assert args["name"] == variant + "_rtdetr_r18_lite_e200_b16_onlineaug"
        assert Path(args["save_dir"]).name == args["name"]
        assert Path(args["save_dir"]).parent == Path(args["project"])
        changed = {row["field"] for row in differences if row["changed"]}
        assert changed <= identities and {"model", "name", "save_dir"} <= changed
        for row in differences:
            assert row["parent"] == parent[row["field"]] and row["target"] == args[row["field"]]
            assert row["changed"] == (row["parent"] != row["target"])
        reports[variant] = {"fields": 109, "all_nonidentity_fields_exact_with_types": True,
                            "changed_identity_fields": sorted(changed)}
    return reports


def expect_rejection(function, label):
    try:
        function()
    except (RuntimeError, ValueError, KeyError, TypeError) as error:
        return {"case": label, "rejected": True, "exception": type(error).__name__, "reason": str(error)}
    raise AssertionError("Unsafe acceptance of negative fixture: " + label)


def gate_checks():
    # Only a tiny explicitly synthetic referenced-evidence file is written.
    # No synthetic preflight/status document is ever persisted. The temporary
    # directory is scoped to this workspace and removed when the check ends.
    temporary_root = (ROOT / "outputs").resolve()
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bfr-gates-synthetic-", dir=temporary_root) as directory:
        folder = Path(directory).resolve()
        assert folder.parent == temporary_root and folder.name.startswith("bfr-gates-synthetic-")
        evidence = folder / "synthetic_hash_reference_only.json"
        evidence.write_text(json.dumps({"fixture_kind": "synthetic_hash_reference_only",
                                        "warning": "Not a preflight or engineering result; no training performed"}), encoding="utf-8")
        result = _gate_checks(evidence)
    assert not folder.exists()
    result["synthetic_evidence_directory_removed"] = True
    return result


def _gate_checks(evidence):
    # Deliberately non-real identities. They are never passed to start/identity(),
    # written to disk, or represented as a genuine successful preflight.
    identity = {
        "variant": "cbr_lif_bfr_p4_v1", "code": {"commit": "synthetic_commit_not_a_real_sha",
            "files_lf_sha256": {"synthetic_code.py": "synthetic_hash"}, "dirty": False, "untracked_code": []},
        "source_sha256": "synthetic_source_hash", "initialization_sha256": "synthetic_init_hash",
        "initialization_audit": {"synthetic": True, "untrained": True, "new_state_matches": True},
        "data": {"root": "synthetic_server_data_root", "config_sha256": "synthetic_data_hash",
            "inventory": {"train": {"images": 6048, "split_paths_sha256": "synthetic_split_hash", "labels_sha256": "synthetic_label_hash"}}},
        "recipe": {"epochs": 200, "batch": 16, "amp": True, "lr0": .0005},
    }
    fixture = {"status": "PASSED", "scope": "server_B16_640_native_AMP", "identity": deepcopy(identity),
        "capacity": {"batch": 16, "imgsz": 640, "amp": True, "effective_optimizer_updates": 2, "gradient_startup": "PASSED"},
        "native_resume": {"status": "PASSED"}, "learned_lifecycle": {"status": "PASSED"},
        "engineering": {"status": "PASSED", "report": str(evidence), "sha256": sha256(evidence)},
        "mathematics": {"status": "PASSED"}}
    require_preflight(fixture, identity)
    rejected = []
    for status in ("PENDING", "FAILED", "RUNNING", "PASSED_LOCAL", "", None):
        changed = deepcopy(fixture)
        changed["status"] = status
        rejected.append(expect_rejection(lambda: require_preflight(changed, identity), "report_status_" + str(status)))
    for scope in ("local", "CPU", "server_B2_160_native_AMP", None):
        changed = deepcopy(fixture)
        changed["scope"] = scope
        rejected.append(expect_rejection(lambda: require_preflight(changed, identity), "wrong_scope_" + str(scope)))
    for section in fixture:
        changed = deepcopy(fixture)
        del changed[section]
        rejected.append(expect_rejection(lambda: require_preflight(changed, identity), "missing_section_" + section))
    for section in ("native_resume", "learned_lifecycle", "engineering", "mathematics"):
        for status in ("PENDING", "FAILED", "RUNNING", None):
            changed = deepcopy(fixture)
            changed[section]["status"] = status
            rejected.append(expect_rejection(lambda: require_preflight(changed, identity), section + "_" + str(status)))
    for key, bad in (("batch", 8), ("imgsz", 320), ("amp", False), ("amp", 1),
                     ("effective_optimizer_updates", 0), ("effective_optimizer_updates", 1),
                     ("gradient_startup", "PENDING")):
        changed = deepcopy(fixture)
        changed["capacity"][key] = bad
        rejected.append(expect_rejection(lambda: require_preflight(changed, identity), "capacity_" + key + "_" + str(bad)))
    for key in fixture["capacity"]:
        changed = deepcopy(fixture)
        del changed["capacity"][key]
        rejected.append(expect_rejection(lambda: require_preflight(changed, identity), "missing_capacity_" + key))
    mutations = [
        (("variant",), "bfr_p4_v1"), (("code", "commit"), "different_commit"),
        (("code", "files_lf_sha256", "synthetic_code.py"), "different_code_hash"),
        (("code", "dirty"), True), (("code", "untracked_code"), ["untracked_module.py"]),
        (("source_sha256",), "different_source"), (("initialization_sha256",), "different_init"),
        (("initialization_audit", "new_state_matches"), False),
        (("data", "config_sha256"), "different_data_yaml"), (("data", "root"), "different_local_path"),
        (("data", "inventory", "train", "images"), 6047),
        (("data", "inventory", "train", "split_paths_sha256"), "different_split"),
        (("data", "inventory", "train", "labels_sha256"), "different_labels"),
        (("recipe", "epochs"), 150), (("recipe", "batch"), 8),
        (("recipe", "amp"), False), (("recipe", "lr0"), .01),
    ]
    for keys, value in mutations:
        current = deepcopy(identity)
        nested = current
        for key in keys[:-1]:
            nested = nested[key]
        nested[keys[-1]] = value
        rejected.append(expect_rejection(lambda: require_preflight(fixture, current), "identity_" + ".".join(keys)))
    for key in ("report", "sha256"):
        changed = deepcopy(fixture)
        del changed["engineering"][key]
        rejected.append(expect_rejection(lambda: require_preflight(changed, identity), "missing_evidence_" + key))
    changed = deepcopy(fixture)
    changed["engineering"]["report"] = str(evidence.with_name("nonexistent_synthetic_evidence.json"))
    rejected.append(expect_rejection(lambda: require_preflight(changed, identity), "missing_referenced_evidence_file"))
    changed = deepcopy(fixture)
    changed["engineering"]["sha256"] = "synthetic_wrong_evidence_hash"
    rejected.append(expect_rejection(lambda: require_preflight(changed, identity), "wrong_referenced_evidence_hash"))
    evidence.write_text('{"fixture_kind":"synthetic_mutated_evidence"}', encoding="utf-8")
    rejected.append(expect_rejection(lambda: require_preflight(fixture, identity), "mutated_referenced_evidence_file"))
    evidence.unlink()
    rejected.append(expect_rejection(lambda: require_preflight(fixture, identity), "deleted_referenced_evidence_file"))
    return {"scope": "Synthetic in-memory API negative fixtures; does not authorize training",
            "positive_api_fixture_accepted": True, "negative_cases": len(rejected), "rejections": rejected}


def evaluation_checks():
    assert POLICY == "corrected_sorted_conf_mask_v1"
    same_typed_mapping(EVAL, dict(imgsz=640, batch=16, workers=0, half=False, conf=.001,
                                  iou=.7, max_det=300, augment=False, rect=False, seed=42))
    # Low confidence comes first, so masking sorted boxes with UNSORTED scores
    # drops the highest prediction and admits an excluded row. Rows 1 and 4
    # overlap exactly; both must survive because this protocol adds no NMS.
    sample = torch.tensor([[.1, .2, .1, .2, .0005], [.5, .5, .2, .2, .9],
                           [.7, .8, .2, .4, .2], [.3, .3, .1, .1, .001],
                           [.5, .5, .2, .2, .6]], dtype=torch.float32)
    empty_sample = sample.clone()
    empty_sample[:, 4] = .0001
    predictions = torch.stack((sample, empty_sample))
    results = corrected_postprocess(predictions, 640, .001)
    ordered_ids = (1, 4, 2)
    expected_boxes = torch.tensor([[(float(sample[i, 0]) - float(sample[i, 2]) / 2) * 640,
                                    (float(sample[i, 1]) - float(sample[i, 3]) / 2) * 640,
                                    (float(sample[i, 0]) + float(sample[i, 2]) / 2) * 640,
                                    (float(sample[i, 1]) + float(sample[i, 3]) / 2) * 640] for i in ordered_ids])
    torch.testing.assert_close(results[0]["bboxes"], expected_boxes, atol=5e-5, rtol=1e-6)
    torch.testing.assert_close(results[0]["conf"], torch.tensor([.9, .6, .2]), atol=0, rtol=0)
    assert torch.equal(results[0]["cls"], torch.zeros(3))
    assert results[1]["bboxes"].shape == (0, 4) and results[1]["conf"].shape == (0,)
    assert torch.equal(results[0]["bboxes"][0], results[0]["bboxes"][1])
    tuple_results = corrected_postprocess((predictions, "ignored_auxiliary"), 640, .001)
    assert all(torch.equal(results[b][key], tuple_results[b][key]) for b in range(2) for key in results[b])
    multiclass = torch.tensor([[[.5, .5, .2, .2, .1, .8], [.3, .3, .1, .1, .7, .2]]])
    classes = corrected_postprocess(multiclass, 640, .001)[0]
    assert torch.equal(classes["cls"], torch.tensor([1., 0.]))
    rejections = []
    for name, value in (("NaN", float("nan")), ("Inf", float("inf"))):
        invalid = predictions.clone()
        invalid[0, 0, 0] = value
        rejections.append(expect_rejection(lambda: corrected_postprocess(invalid, 640, .001), "nonfinite_" + name))
    return {"policy": POLICY, "unsorted_input_retained_original_row_ids": list(ordered_ids),
            "threshold_equality_excluded": True, "overlapping_rows_both_retained_no_nms": True,
            "empty_sample": True, "batch_and_tuple_container": True, "class_argmax": True,
            "nonfinite_rejections": rejections, "inference_performed": False}


def run_checks():
    return {"status": "PASSED_DIAGNOSTICS", "device": "cpu", "formal_training": "NOT_STARTED",
            "final_test": "NOT_RUN", "synthetic_fixtures_persisted": False,
            "scope": "Pure function/API tests only; this is not a server preflight and cannot open the start gate",
            "recipe": recipe_checks(), "gate": gate_checks(), "evaluation_protocol": evaluation_checks()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/bfr_p4/gates_cpu.json")
    args = parser.parse_args()
    result = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "negative_cases": result["gate"]["negative_cases"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
