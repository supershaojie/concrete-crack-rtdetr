"""Bounded local delivery regression; no training loop, no formal split evaluation."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import subprocess
import tempfile

import torch
from init_scca import ROOT, MODEL_DIR, C17_COMMIT, build, initialize, require, runtime, sha256, write_json
from check_scca import rebuild
from train_scca import optimizer_audit, recipe
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.patches import torch_load

C24_COMMIT = "f6e9dfda765046ae7691302cf5ec89d3f76cec5d"


def check(source, output, tiny_data=None):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(runtime=runtime(), formal_training_started=False, full_server_preflight="NOT_RUN",
                  scope="Fresh nc=1 model counts; controlled initialization/reload; native get_model/optimizer. Optional <=4-image split pipeline only.")
    unchanged = {}
    for revision, names in ((C17_COMMIT, ("cscef_v51.py", "cscef_v5.py")), (C24_COMMIT, ("scca_aifi.py",))):
        for name in names:
            path = "ultralytics-main/ultralytics/nn/modules/" + name
            expected = subprocess.check_output(["git", "show", revision + ":" + path], cwd=ROOT)
            require((ROOT / path).read_bytes().replace(b"\r\n", b"\n") == expected.replace(b"\r\n", b"\n"), f"Original module changed: {path}")
            unchanged[path] = revision
    yaml_path = "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cscef-v51.yaml"
    original = subprocess.check_output(["git", "show", C17_COMMIT + ":" + yaml_path], cwd=ROOT).decode().replace("\r\n", "\n")
    combo = (MODEL_DIR / "rtdetr-resnet18-lite-cscef-scca.yaml").read_text(encoding="utf-8")
    require(combo == original.replace("1, AIFI, [1024, 8]", "1, SCCAAIFI, [1024, 8]"), "C25 YAML differs beyond layer 9 replacement")
    report["original_sources_unchanged"] = unchanged
    report["yaml_only_layer9_replacement"] = True
    counts = {}
    for label, variant, baseline in (("C2", "c24", True), ("C17", "c25", True), ("C24", "c24", False), ("C25", "c25", False)):
        model = build(variant, nc=1, baseline=baseline)
        counts[label] = sum(p.numel() for p in model.parameters())
        del model
    require(counts == dict(C2=20082772, C17=20109684, C24=20148312, C25=20175224), "Unexpected parameter counts")
    report["parameters_nc1_unfused"] = counts
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        folder = Path(temporary)
        fresh = folder / "fresh_init.pt"
        report["initialization"] = initialize(source, fresh, "c25")
        original_model = RTDETR(str(fresh)).model
        model, report["native_nc1_loading"] = rebuild(original_model, "c25")
        require(report["native_nc1_loading"]["loaded_exact"] == 536, "Expected 536/545 states exactly loaded")
        report["native_nc1_loading"]["total_states"] = len(model.state_dict())
        report["connections"] = {str(i): dict(module=type(model.model[i]).__name__, inputs=model.model[i].f) for i in (9, 18, 19, 27)}
        head = model.model[-1]
        report["decoder"] = dict(layers=head.decoder.num_layers, queries=head.num_queries, denoising=head.num_denoising)
        trainer = object.__new__(RTDETRTrainer)
        optimizer = trainer.build_optimizer(model, name="AdamW", lr=.0005, momentum=.937, decay=.0001)
        report["optimizer_occurrences"] = optimizer_audit(model, optimizer, "c25")
        args, rows = recipe(ROOT / "docs/scca/c2_args.yaml", "c25", ROOT / "weights/c25_scca_controlled_init.pt")
        report["recipe"] = dict(fields=len(args), changed=[r["field"] for r in rows if r["changed"]], source="local archive; actual server args checked by start-direct")
        write_json(output / "parameter_diff.json", rows)
        if tiny_data:
            from ultralytics.data.utils import check_det_dataset
            from ultralytics.data.base import BaseDataset
            from scca_results import evaluate
            data = check_det_dataset(str(tiny_data), autodownload=False)
            dataset = object.__new__(BaseDataset)
            dataset.prefix, dataset.fraction = "", 1.0
            for split in ("val", "test"):
                require(0 < len(dataset.get_img_files(data[split])) <= 4, "Only <=4-image local pipeline splits permitted")
            checkpoint = torch_load(fresh, map_location="cpu")
            checkpoint["model"] = model.float().eval()
            weight = folder / "disposable_nc1.pt"
            torch.save(checkpoint, weight)
            report["tiny_pipeline"] = {}
            for split in ("val", "test"):
                row = evaluate(weight, tiny_data, split, output / ("tiny_" + split), device="cpu", batch=1,
                               val_report=output / "tiny_val/metrics.json" if split == "test" else None)
                report["tiny_pipeline"][split] = dict(status=row["status"], images=row["images"],
                    export=row["prediction_export"], ap_array_shape=[len(row["ap_by_class"]), len(row["ap_by_class"][0])],
                    note="Untrained disposable initialization, <=4 images; pipeline verification only, no research metric claim.")
        del model, original_model, optimizer
        gc.collect()
    report.update(status="passed", disposable_checkpoints_deleted=True)
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiny-data", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    check(args.source, args.output, args.tiny_data)
