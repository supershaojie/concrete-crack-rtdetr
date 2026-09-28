"""Local one-batch FP32 inference check; explicitly not B16 training capacity."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import traceback

from arg_v2_common import OUT, MAIN, ROOT, data_config, write_json, runtime, sha256
import torch
from ultralytics.nn.tasks import RTDETRDetectionModel, load_checkpoint
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules import LIFDown
from ultralytics.models.rtdetr.arg_v2_model import ARGv2DetectionModel
from ultralytics.models.rtdetr.arg_v2_val import ARGv2Validator, EVAL


def run(checkpoint, device, output, data=None):
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status="FAIL", runtime=runtime(), checkpoint_sha256=sha256(checkpoint),
                  scope="local FP32 B1/640 real val; not formal B16 capacity or full val/test")
    try:
        model, _ = load_checkpoint(str(checkpoint), device=torch.device(device), fuse=False)
        assert type(model) is ARGv2DetectionModel
        model.eval().float()
        mother = deepcopy(model); mother.__class__ = RTDETRDetectionModel
        import cv2
        path = sorted((MAIN / "datasets/crack_det/images/val").glob("*.jpg"))[0]
        im = cv2.imread(str(path))
        image = torch.from_numpy(cv2.resize(im, (640, 640))[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().to(device) / 255
        with torch.no_grad():
            before = model(image)[0]
            assert torch.equal(before, mother(image)[0])
        model.fuse(verbose=False); mother.fuse(verbose=False)
        assert isinstance(model.model[20], LIFDown) and hasattr(model.model[20], "bn")
        with torch.no_grad():
            after = model(image)[0]
            assert torch.equal(after, mother(image)[0])
            torch.testing.assert_close(after, before, atol=2e-5, rtol=2e-4)
        report["mother_parity"] = dict(before_fuse_exact=True, after_fuse_exact=True, lif_bn_preserved=True,
            fusion_max_abs=float((after - before).abs().max()), fusion_atol=2e-5, fusion_rtol=2e-4)
        del model, mother, before, after
        warmup = []
        original = AutoBackend.warmup
        def observed(backend, imgsz=(1, 3, 640, 640)):
            handle = backend.model.register_forward_pre_hook(lambda m, a: warmup.append(dict(shape=list(a[0].shape),
                finite=bool(torch.isfinite(a[0]).all()), zero=bool(torch.count_nonzero(a[0]) == 0))))
            try:
                return original(backend, imgsz)
            finally:
                handle.remove()
        AutoBackend.warmup = observed
        validator = ARGv2Validator(args=dict(EVAL, batch=1, model=str(checkpoint), data=str(data or data_config()), split="val",
            device=device, plots=False), save_dir=output / "real_val")
        validator.one_batch = True
        validator.export_path = output / "val_predictions_gt.jsonl.gz"
        validator.export_identity = dict(diagnostic=True, checkpoint_sha256=sha256(checkpoint), split="val")
        try:
            metrics = validator(model=str(checkpoint))
        finally:
            AutoBackend.warmup = original
        assert len(validator.arg_seen) == 1
        if device != "cpu":
            assert warmup and all(row["finite"] and row["zero"] for row in warmup)
        report.update(status="PASS", real_val_images=1, warmup=warmup, actual_settings=validator.actual_settings,
                      metrics=metrics, predictions_sha256=sha256(validator.export_path))
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc()); raise
    finally:
        write_json(output / "lifecycle.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=OUT / "local_checks/diagnostic_resume.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUT / "local_lifecycle")
    parser.add_argument("--data", type=Path, help="Optional local data YAML alias; same dataset identity")
    args = parser.parse_args(); torch.set_num_threads(4)
    run(args.checkpoint, args.device, args.output, args.data)
