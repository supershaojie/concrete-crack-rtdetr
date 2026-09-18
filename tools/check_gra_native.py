"""One-batch learned-state native epoch-half validation and AutoBackend warmup smoke."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
from pathlib import Path
from types import SimpleNamespace
import traceback

import torch

from init_gra import ROOT, VARIANTS, require, runtime, sha256, tensor_summary, verify_model, write_json
from train_gra import read_json, source_manifest
from c19_lif_v1_data import real_batch
from c19_lif_v1_diagnostic import rng_state, restore_rng
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.patches import torch_load


def gra_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.model[18].state_dict().items()}


def assert_gra_state(model, reference, label):
    current = gra_state(model)
    require(set(current) == set(reference) and all(torch.equal(value, current[key]) for key, value in reference.items()),
            label + ": GRA learned state changed")
    require(int(torch.count_nonzero(current["offset.weight"])) > 0, label + ": offset head reset")
    return {key: tensor_summary(value) for key, value in current.items()}


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)


def native_smoke(variant, checkpoint, batch, output):
    output.mkdir(parents=True, exist_ok=False)
    loaded = torch_load(checkpoint, map_location="cpu")
    require(loaded.get("epoch", -1) >= 0 and loaded.get("optimizer") and loaded.get("ema"),
            "Disposable checkpoint must contain a real update, optimizer and EMA")
    model = deepcopy(loaded["ema"]).eval().cuda().float()
    model.nc = 1
    verify_model(model, variant, zero=False)
    before = gra_state(model)
    require(int(torch.count_nonzero(before["offset.weight"])) > 0, "EMA offset must already be learned")
    quantized = {key: value.half().float() for key, value in before.items()}
    report = dict(status="FAILED", checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
                  checkpoint_selection="actual learned EMA", epoch=loaded["epoch"], ema_updates=loaded["updates"],
                  input_shape=list(batch["img"].shape), original_gra={key: tensor_summary(value) for key, value in before.items()})
    sample = {key: value.detach().cpu().clone() for key, value in batch.items()}
    sample["img"] = (sample["img"] * 255).round().to(torch.uint8)

    class OneBatch:
        dataset = list(range(len(sample["img"])))

        def __len__(self):
            return 1

        def __iter__(self):
            yield deepcopy(sample)

    observed = dict(gra_calls=0)

    class FiniteValidator(RTDETRValidator):
        # Only aggregation is replaced; BaseValidator.__call__, preprocess,
        # model inference, native RT-DETR detection loss and postprocess all run.
        def init_metrics(self, actual):
            observed.update(model_dtype=str(next(actual.parameters()).dtype), half=self.args.half,
                            bn_count=sum(isinstance(m, torch.nn.BatchNorm2d) for m in actual.modules()))
            require(actual is model and self.training and self.args.half and
                    next(actual.parameters()).dtype == torch.float16, "Native training-time EMA half path not taken")

        def update_metrics(self, predictions, data):
            require(data["img"].dtype == torch.float16 and len(predictions) == len(sample["img"]), "Native validator input/output mismatch")
            require(all(bool(torch.isfinite(value).all()) for value in tensors(predictions)), "Nonfinite native epoch-half prediction")
            observed.update(images=len(predictions), input_shape=list(data["img"].shape), input_dtype=str(data["img"].dtype),
                            input_device=str(data["img"].device), prediction_counts=[len(p["conf"]) for p in predictions],
                            gt_count=len(data["cls"]), predictions_finite=True)

        def gather_stats(self):
            pass

        def get_stats(self):
            return {}

        def finalize_metrics(self):
            pass

        def print_results(self):
            pass

    validator = FiniteValidator(dataloader=OneBatch(), save_dir=output / "epoch_half",
                               args=dict(imgsz=160, plots=False, save_json=False, save_txt=False, workers=0, task="detect"))
    trainer = SimpleNamespace(device=torch.device("cuda"), data=dict(nc=1, channels=3, names={0: "crack"}), amp=True,
                              ema=SimpleNamespace(ema=model), model=model, args=SimpleNamespace(compile=False),
                              loss_items=torch.zeros(3, device="cuda"), stopper=SimpleNamespace(possible_stop=False),
                              epoch=0, epochs=200, world_size=1,
                              label_loss_items=lambda loss, prefix: {prefix + "/" + str(i): float(v) for i, v in enumerate(loss)})
    bn_before = sum(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())

    def gra_hook(module, inputs, value):
        observed["gra_calls"] += 1
        require(bool(torch.isfinite(value).all()), "Native half GRA output nonfinite")

    hook = model.model[18].register_forward_hook(gra_hook)
    try:
        losses = validator(trainer)
    finally:
        hook.remove()
    require(bool(torch.isfinite(validator.loss).all()) and next(model.parameters()).dtype == torch.float32,
            "Native epoch validation loss/FP32 restoration failed")
    require(observed["gra_calls"] == 1 and observed["bn_count"] == bn_before, "Epoch validation bypassed GRA or unexpectedly fused")
    if variant == "cbr_lif_gra_v1":
        require(hasattr(model.model[20], "bn"), "Original LIF BN lost")
    report["native_epoch_half"] = dict(status="PASSED", **observed, loss=losses, metrics="NOT_EVALUATED", fused=False,
                                      restored_dtype="torch.float32", gra_same_quantization_exact=True,
                                      gra_state=assert_gra_state(model, quantized, "native epoch half/float"))

    # Real backend construction performs fusion in FP32, then half conversion.
    before_backend = gra_state(model)
    backend = AutoBackend(model=deepcopy(model), device=torch.device("cuda"), fp16=True, verbose=False)
    expected_half = {key: value.half() for key, value in before_backend.items()}
    require(next(backend.model.parameters()).dtype == torch.float16, "AutoBackend did not select half")
    require(not hasattr(backend.model.model[15], "bn"), "AutoBackend did not fuse the original Conv-BN path")
    report["backend_construct"] = dict(status="PASSED", fuse_input_dtype="torch.float32", model_dtype="torch.float16",
                                       gra_state=assert_gra_state(backend.model, expected_half, "AutoBackend fuse/half"))
    warmup_inputs, warmup_outputs, calls = [], [], []

    def before_forward(module, inputs):
        value = inputs[0]
        record = dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                      finite=bool(torch.isfinite(value).all()), nonzero=int(torch.count_nonzero(value)))
        warmup_inputs.append(record)
        require(record["finite"] and record["nonzero"] == 0, "AutoBackend warmup must use finite zeros")

    def after_forward(module, inputs, value):
        rows = [dict(shape=list(t.shape), dtype=str(t.dtype), device=str(t.device), finite=bool(torch.isfinite(t).all()))
                for t in tensors(value)]
        warmup_outputs.append(rows)
        require(rows and all(r["finite"] for r in rows), "AutoBackend warmup output nonfinite")

    hooks = [backend.model.register_forward_pre_hook(before_forward), backend.model.register_forward_hook(after_forward),
             backend.model.model[18].register_forward_hook(lambda *_: calls.append("gra"))]
    try:
        backend.warmup(imgsz=(1, 3, 160, 160))
    finally:
        for hook in hooks:
            hook.remove()
    require(len(warmup_inputs) == len(warmup_outputs) == len(calls) == 1, "Expected one native PyTorch backend warmup call")
    report["backend_warmup"] = dict(status="PASSED", input=warmup_inputs, outputs=warmup_outputs, gra_calls=len(calls),
                                    state_preserved=True)
    assert_gra_state(backend.model, expected_half, "warmup")
    with torch.no_grad():
        prediction = backend(batch["img"][:1].cuda())[0]
    require(tuple(prediction.shape) == (1, 300, 5) and bool(torch.isfinite(prediction).all()), "AutoBackend real-image half forward failed")
    report["backend_real_image"] = dict(status="PASSED", shape=list(prediction.shape), dtype=str(prediction.dtype),
                                        device=str(prediction.device), finite=True,
                                        gra_state=assert_gra_state(backend.model, expected_half, "backend real image"))
    report["status"] = "PASSED"
    write_json(output / "native.json", report)
    del backend, model, loaded, validator, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", type=Path, required=True, help="Completed check_gra output directory")
    parser.add_argument("--dataset", type=Path, required=True, help="Canonical dataset root containing images/train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Existing native smoke evidence protected")
    args.output.mkdir(parents=True, exist_ok=False)
    state = rng_state()
    report = dict(status="RUNNING", formal_training="NOT_STARTED", full_val="NOT_RUN", final_test="NOT_RUN",
                  scope="two true train images, one batch/variant, native epoch-half loss and AutoBackend fuse/half/warmup",
                  runtime=runtime(), variants={})
    try:
        require(torch.cuda.is_available(), "PENDING: native CUDA half smoke requires CUDA")
        torch.set_num_threads(4)
        checks = read_json(args.checks / "checks.json")
        require(checks.get("status") == "PASSED_LOCAL_CHECKS_SERVER_PENDING", "Completed local structural/learned checks required")
        report["checks_sha256"] = sha256(args.checks / "checks.json")
        batch, report["samples"] = real_batch(args.dataset, size=160, count=2)
        require(report["samples"] == checks["samples"], "Native smoke must use the same real training samples")
        for variant in VARIANTS:
            prior = checks["variants"][variant]["cuda_amp"]
            require(prior["status"] == "PASSED" and prior["effective_updates"] >= 2, "Passed learned CUDA AMP state required")
            checkpoint = args.checks / variant / "cuda_amp/learned_disposable.pt"
            require(sha256(checkpoint) == prior["lifecycle"]["checkpoint_sha256"], "Learned checkpoint identity changed")
            report["variants"][variant] = native_smoke(variant, checkpoint, batch, args.output / variant)
            write_json(args.output / "native_smoke.json", report)
        report["status"] = "PASSED"
    except BaseException as error:
        report.update(status="PENDING" if "PENDING:" in str(error) else "FAILED", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        restore_rng(state)
        report["source_manifest"] = source_manifest()
        files = [Path(__file__), ROOT / "tools/c19_lif_v1_precision.py", ROOT / "tools/c19_lif_v1_data.py",
                 ROOT / "ultralytics-main/ultralytics/engine/validator.py", ROOT / "ultralytics-main/ultralytics/nn/autobackend.py"]
        report["direct_source_hashes"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in files}
        write_json(args.output / "native_smoke.json", report)
        print(report["status"] + " native smoke: " + str(args.output / "native_smoke.json"))


if __name__ == "__main__":
    main()
