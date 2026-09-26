"""Bounded independent references and lifecycle tests; never a formal AP result."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import io
from pathlib import Path
import tempfile
import traceback

from tcr_v1_core import *
from ultralytics.nn.modules.tcr import TCR
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import ModelEMA


def close(a, b, atol=2e-6, rtol=2e-5):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return float((a.detach().float() - b.detach().float()).abs().max()) if a.numel() else 0.0


def direct_reference(z):
    """Independent five-point coordinate reads; no production shift/contrast helper."""
    b, c, h, w = z.shape
    out = torch.zeros(b, 8 * c, h, w, device=z.device)
    directions = [((1, 0), (0, 1)), ((1, 1), (-1, 1)), ((0, 1), (-1, 0)), ((-1, 1), (-1, -1))]
    for d, ((tx, ty), (nx, ny)) in enumerate(directions):
        for side in (1, 2):
            for y in range(4, h - 4):
                for x in range(4, w - 4):
                    center = sum(z[:, :, y + r * ty, x + r * tx].float() for r in range(-2, 3)) / 5
                    plus = sum(z[:, :, y + r * ty + side * ny, x + r * tx + side * nx].float() for r in range(-2, 3)) / 5
                    minus = sum(z[:, :, y + r * ty - side * ny, x + r * tx - side * nx].float() for r in range(-2, 3)) / 5
                    a, bdiff = center - plus, center - minus
                    result = torch.where((a > 0) & (bdiff > 0), torch.minimum(a, bdiff),
                                         torch.where((a < 0) & (bdiff < 0), torch.maximum(a, bdiff), 0))
                    j = (2 * d + side - 1) * c
                    out[:, j:j + c, y, x] = result
    return out


def algorithm(device):
    a = torch.tensor([3., -3., 3., 0., 2.], device=device)
    b = torch.tensor([2., -2., -2., 4., 0.], device=device)
    close(TCR.signed_agreement(a, b), a.new_tensor([2., -2., 0., 0., 0.]), 0, 0)
    close(TCR.signed_agreement(a, b), TCR.signed_agreement(b, a), 0, 0)
    close(TCR.signed_agreement(-a, -b), -TCR.signed_agreement(a, b), 0, 0)
    errors = []
    for shape in ((2, 3, 13, 17), (1, 2, 8, 16), (2, 1, 17, 7), (1, 1, 5, 3)):
        z = torch.randn(*shape, device=device)
        errors.append(close(TCR.responses(z), direct_reference(z)))
    yy, xx = torch.meshgrid(torch.arange(25, device=device), torch.arange(29, device=device), indexing="ij")
    plane = (2. * xx - 3. * yy + 7)[None, None]
    close(TCR.responses(plane), torch.zeros(1, 8, 25, 29, device=device), 5e-6, 0)
    close(TCR.responses(torch.ones_like(plane)), torch.zeros(1, 8, 25, 29, device=device), 0, 0)
    examples = {}
    for i, (nx, ny) in enumerate(TCR.normals):
        distance = (xx - 14) * nx + (yy - 12) * ny
        line = (distance == 0).float()[None, None]
        response = TCR.responses(line)
        require(float(response[0, 2 * i, 12, 14]) > .99, "Bright aligned center not positive")
        close(TCR.responses(-line), -response, 0, 0)
        step = (distance >= 0).float()[None, None]
        close(TCR.responses(step)[:, 2 * i:2 * i + 2], torch.zeros(1, 2, 25, 29, device=device), 1e-7, 0)
        examples[str(i)] = dict(center_bright=float(response[0, 2 * i, 12, 14]), center_dark=-float(response[0, 2 * i, 12, 14]), aligned_step_zero=True)
    for name, sample in dict(thick=(abs(yy - 12) <= 4), curved=(abs(yy - (12 + (xx - 14).square() // 20)) <= 1),
                             fork=((yy == 12) | (yy - 12 == abs(xx - 14)))).items():
        r = TCR.responses(sample.float()[None, None])
        close(r, direct_reference(sample.float()[None, None]))
        require(torch.isfinite(r).all(), "Nonfinite geometry")
        examples[name] = dict(rms=float(r.square().mean().sqrt()), interpretation="diagnostic only; no universal suppression claim")
    m = TCR().to(device)
    x = torch.randn(2, 256, 13, 17, device=device, requires_grad=True)
    close(m(x), x, 0, 0)
    m(x).square().mean().backward()
    identity_input=x.detach().clone().requires_grad_(True)
    identity_input.square().mean().backward()
    close(x.grad,identity_input.grad,0,0)
    require(m.O.weight.grad.norm() > 0 and m.P.weight.grad.norm() == 0, "Zero-init first gradient contract")
    first = dict(O=float(m.O.weight.grad.norm()), P=float(m.P.weight.grad.norm()))
    with torch.no_grad(): m.O.weight -= .1 * m.O.weight.grad
    m.zero_grad(); x.grad = None
    m(x).square().mean().backward()
    require(all(torch.isfinite(v).all() and v.norm() > 0 for v in (m.P.weight.grad, m.O.weight.grad, x.grad)), "Post-update gradient flow")
    small = torch.randn(1, 256, 8, 19, device=device)
    close(m(small), small, 0, 0)
    m.enabled = False
    close(m(x), x, 0, 0)
    return dict(status="PASS", max_reference_error=max(errors), examples=examples, first_gradient=first,
                later_gradient=dict(P=float(m.P.weight.grad.norm()), O=float(m.O.weight.grad.norm()), input=float(x.grad.norm())))


def batch(device, empty=False, size=(160, 192)):
    return dict(img=torch.rand(2, 3, *size, device=device),
                cls=torch.zeros((0 if empty else 2, 1), device=device),
                bboxes=torch.empty(0, 4, device=device) if empty else torch.tensor([[.5,.5,.4,.2],[.4,.6,.2,.3]], device=device),
                batch_idx=torch.empty(0, device=device) if empty else torch.tensor([0,1], device=device))


def model_checks(device, source=None, folder=None):
    mother = parent.build(nc=1).to(device)
    target = build().to(device)
    mother.nc = target.nc = 1  # Native Trainer.set_model_attributes normally supplies this.
    inventory = verify_model(target, zero=True)
    require(all(torch.equal(v, target.state_dict()[k]) for k, v in mother.state_dict().items()), "Common initial state")
    sample = batch(device)
    mother.eval(); target.eval()
    with torch.no_grad():
        output_error = close(mother(sample["img"])[0], target(sample["img"])[0], 0, 0)
    mother.train(); target.train()
    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    loss0, items0 = mother(sample); loss0.backward()
    torch.random.set_rng_state(state)
    if cuda_state is not None: torch.cuda.set_rng_state_all(cuda_state)
    loss1, items1 = target(sample); loss1.backward()
    close(loss0, loss1, 0, 0); close(items0, items1, 0, 0)
    gradients = dict(target.named_parameters())
    errors = []
    for name, parameter in mother.named_parameters():
        got = gradients[name].grad
        require((parameter.grad is None) == (got is None), f"Gradient presence: {name}")
        if got is not None:
            # The additional zero-gradient edge may reorder shared DAG reductions.
            # Check both elementwise tolerance and a tighter whole-tensor norm ratio.
            relative = float((parameter.grad-got).float().norm()/(parameter.grad.float().norm()+1e-12))
            difference = float((parameter.grad-got).float().norm())
            bound = 2e-6 * parameter.numel() ** .5 + 2e-5 * float(parameter.grad.float().norm())
            require(difference <= bound, f"Common gradient L2 {name}: {difference} > {bound}; relative {relative}")
            maximum = float((parameter.grad-got).abs().max())
            require(maximum <= 2e-5 + 2e-5*float(parameter.grad.abs().max()), f"Common gradient max-scale error: {name}, max_diff={maximum}, reference_max={float(parameter.grad.abs().max())}, L2_relative={relative}")
            errors.append(maximum)
    require(target.model[17].tcr.O.weight.grad.norm() > 0, "Native loss cannot reach O")
    with torch.no_grad(): target.model[17].tcr.O.weight.normal_(0, .01)
    target.zero_grad()
    loss2, _ = target(sample); loss2.backward()
    require(target.model[17].tcr.P.weight.grad.norm() > 0, "Native loss cannot reach P after O activation")
    new_grads = {n: float(p.grad.norm()) for n, p in target.named_parameters() if n in NEW}
    target.zero_grad()
    empty_loss, _ = target(batch(device, empty=True)); empty_loss.backward()
    require(torch.isfinite(empty_loss), "Empty-GT native loss")
    target.zero_grad(); target.eval()
    with torch.no_grad():
        enabled = target(sample["img"])[0]
        target.model[17].tcr.enabled = False
        disabled = target(sample["img"])[0]
        target.model[17].tcr.enabled = True
    require((enabled - disabled).abs().max() > 0, "Nonzero O has no full-model effect")
    fused = deepcopy(target).fuse(verbose=False)
    verify_model(fused)
    require(not hasattr(fused.model[17], "bn"), "ConvTCR not fused")
    from c19_lif_v1_diagnostic import fusion_protocol
    from c19_lif_v1_cutoff import fusion_accepted
    diagnostic = Path(tempfile.mkdtemp(prefix="fusion_",dir=folder)) / "evidence"
    fusion = fusion_protocol(target,fused,sample["img"],diagnostic,torch.device(device).type,"fp32")
    require(fusion_accepted(fusion,torch.device(device).type,"fp32"), f"Mother candidate-aware fusion gate failed: {diagnostic}")
    with torch.no_grad(): fused_output = fused(sample["img"])[0]
    fused_error = dict(status=fusion["status"],selection=fusion["selection"]["kind"],evidence=str(diagnostic))
    ema = ModelEMA(target); ema.update(target)
    require(ema.ema.model[17].tcr.enabled and torch.count_nonzero(ema.ema.model[17].tcr.O.weight), "EMA omitted TCR")
    stream = io.BytesIO(); torch.save(target, stream); stream.seek(0)
    restored = torch_load(stream, map_location=device)
    with torch.no_grad(): close(enabled, restored(sample["img"])[0], 0, 0)
    backend = AutoBackend(deepcopy(target), device=torch.device(device), fuse=True, verbose=False)
    backend_fusion = fusion_protocol(target,backend.model,sample["img"],diagnostic.parent/"backend",torch.device(device).type,"fp32")
    require(fusion_accepted(backend_fusion,torch.device(device).type,"fp32"),"AutoBackend candidate-aware fusion mismatch")
    with torch.no_grad(): backend_error = close(backend.model(sample["img"])[0], backend(sample["img"])[0], 0, 0)
    amp_report = dict(status="NOT_RUN", reason="CPU path")
    if str(device).startswith("cuda"):
        del mother, fused, ema, restored, backend
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        target.train(); target.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            amp_loss, amp_items = target(sample)
        amp_loss.backward()
        require(torch.isfinite(amp_loss) and all(torch.isfinite(p.grad).all() for p in target.parameters() if p.grad is not None), "AMP nonfinite")
        amp_report = dict(status="PASS", scope="B2 160x192 interface only", loss=float(amp_loss),
                          items=amp_items.tolist(), peak_allocated_bytes=torch.cuda.max_memory_allocated())
        amp_report["native_half_epoch_validator"] = half_validator(target, folder)
    init_report = dict(status="PENDING", reason="Source not supplied")
    if source:
        _, controlled_model, audit = controlled(source)
        rebuilt, native = native_rebuild(controlled_model.yaml, controlled_model, dict(nc=1, channels=3))
        init_report = dict(status="PASS", controlled=audit, native=native)
        del controlled_model, rebuilt
    criterion = target.init_criterion()
    return dict(status="PASS", inventory=inventory, output_error=output_error, loss=float(loss1.detach()),
                loss_items=items1.tolist(), common_gradient_max_abs=max(errors),
                common_gradient_tolerance="L2 <= 2e-6*sqrt(n)+2e-5*reference_L2; maxabs <= 2e-5+2e-5*reference_maxabs. DAG/grid_sample floating reduction order is not bitwise guaranteed.", new_gradients=new_grads,
                empty_gt_loss=float(empty_loss.detach()), nonzero_effect=float((enabled-disabled).abs().max()),
                fused_error=fused_error, backend_error=backend_error, backend_fusion=backend_fusion["status"], ema=True, save_load=True, amp=amp_report,
                initialization=init_report, criterion=type(criterion).__name__, loss_gain=criterion.loss_gain,
                matcher=type(criterion.matcher).__name__, matcher_cost=criterion.matcher.cost_gain)


def half_validator(target, folder):
    from types import SimpleNamespace
    from ultralytics.models.rtdetr.val import RTDETRValidator
    sample=batch("cpu",size=(160,160))
    sample["img"]=(sample["img"]*255).to(torch.uint8)
    model=deepcopy(target).eval()
    model.model[17].tcr.capture=True
    class OneBatch:
        dataset=[0,1]
        def __len__(self): return 1
        def __iter__(self): yield deepcopy(sample)
    observed={}
    class FiniteValidator(RTDETRValidator):
        def init_metrics(self,m):
            require(self.training and self.args.half and next(m.parameters()).dtype==torch.float16,"Native half precision not exercised")
        def update_metrics(self,preds,data):
            require(all(torch.isfinite(v).all() for p in preds for v in p.values()),"Nonfinite half predictions")
            observed.update(model_dtype=str(next(model.parameters()).dtype),input_dtype=str(data["img"].dtype),tcr=model.model[17].tcr.last_stats)
        def gather_stats(self): pass
        def get_stats(self): return {}
        def finalize_metrics(self): pass
        def print_results(self): pass
    validator=FiniteValidator(dataloader=OneBatch(),save_dir=Path(folder)/"half_epoch",args=dict(imgsz=160,plots=False,workers=0,task="detect"))
    trainer=SimpleNamespace(device=torch.device("cuda:0"),data=dict(nc=1,names={0:"crack"}),amp=True,
                            ema=SimpleNamespace(ema=model),model=model,args=SimpleNamespace(compile=False),
                            loss_items=torch.zeros(3,device="cuda"),stopper=SimpleNamespace(possible_stop=False),
                            epoch=0,epochs=200,world_size=1,label_loss_items=lambda loss,prefix:{str(i):float(v) for i,v in enumerate(loss)})
    validator(trainer)
    require(torch.isfinite(validator.loss).all() and next(model.parameters()).dtype==torch.float32,"Native half validation/restoration failed")
    require(observed["tcr"]["input_dtype"]=="torch.float16" and observed["tcr"]["response_dtype"]=="torch.float32", "TCR local precision contract")
    return dict(status="PASS",metrics="NOT_EVALUATED",loss=validator.loss.tolist(),**observed)


def lifecycle(folder):
    """Actual Trainer setup, optimizer/EMA save and native resume on synthetic data."""
    import cv2
    from ultralytics import RTDETR
    from tcr_v1_train import TCRTrainer
    folder = Path(folder)
    for split in ("train", "val"):
        for i in range(4):
            p = folder / "dataset/images" / split / f"{i}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            im = np.random.default_rng(i).integers(0, 256, (160, 192, 3), dtype=np.uint8)
            require(cv2.imwrite(str(p), im), "Cannot write fixture")
            label = folder / "dataset/labels" / split / f"{i}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("0 0.5 0.5 0.4 0.2\n")
    data = folder / "dataset.yaml"
    YAML.save(data, dict(path=str(folder / "dataset"), train="images/train", val="images/val", names={0:"crack"}))
    model = build()
    with torch.no_grad(): model.model[17].tcr.O.weight.normal_(0, .01)
    model.args = {"task":"detect", "model":str(MODEL)}
    initial = folder / "fixture.pt"
    torch.save(dict(model=model, train_args=model.args), initial)
    args = dict(model=str(initial), data=str(data), device="cpu", batch=2, imgsz=160, workers=0,
                epochs=2, amp=False, plots=False, optimizer="AdamW", lr0=.0005, project=str(folder),
                name="native", save=True, seed=42, deterministic=True, val=True)
    # Fixture deliberately contains nonzero O, so audit the native resume/equality path.
    trainer = TCRTrainer(overrides=args)
    trainer.fixture_allow_nonzero = True
    trainer._setup_train()
    groups = optimizer_audit(trainer.model, trainer.optimizer)
    trainer.model.train(); trainer.epoch = 0
    sample = trainer.preprocess_batch(next(iter(trainer.train_loader)))
    loss, _ = trainer.model(sample); loss.backward(); trainer.optimizer_step()
    trainer.best_fitness = trainer.fitness = .1
    trainer.metrics = {}; trainer.save_model()
    checkpoint = torch_load(trainer.last, map_location="cpu")
    expected = deepcopy(checkpoint["ema"]).float().state_dict()
    resume = TCRTrainer(overrides={**args, "resume":str(trainer.last)})
    resume._setup_train()
    require(resume.start_epoch == 1 and resume.ema.updates == checkpoint["updates"], "Epoch/EMA resume failed")
    require(resume.optimizer.state_dict()["state"] and resume.scaler.state_dict() == checkpoint["scaler"], "Optimizer/scaler resume failed")
    require(all(torch.equal(v, resume.model.state_dict()[k]) for k,v in expected.items()), "Resume parameters changed")
    verify_model(resume.model)
    resume.epoch = 1
    resume.loss_items = torch.zeros(3)
    metrics = resume.validator(trainer=resume)
    # Native each-epoch validator uses EMA and original precision policy.
    require(resume.ema.ema.model[17].tcr.enabled, "Epoch validator disabled TCR")
    api = RTDETR(str(initial))
    predictions = api.predict(source=[np.zeros((160,192,3), dtype=np.uint8)], imgsz=160, device="cpu", verbose=False)
    require(len(predictions) == 1 and api.predictor.model.model.model[17].tcr.enabled, "Predict omitted TCR")
    # Use the real standalone validator path with nonzero weights on synthetic data.
    from tcr_v1_eval import CorrectedValidator
    val = RTDETR(str(initial)).val(validator=CorrectedValidator, data=str(data), imgsz=160, batch=2,
                                  workers=0, device="cpu", half=False, plots=False, project=str(folder), name="standalone")
    return dict(status="PASS", data="synthetic; metrics are not research results", native_setup=True,
                optimizer_parameter_count=sum(len(g["parameters"]) for g in groups), save=True, resume=True,
                epoch_validator=True, independent_validator=True, predict=True)


def export_check(folder):
    from ultralytics import RTDETR
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = build().eval()
    with torch.no_grad(): target.model[17].tcr.O.weight.normal_(0, .01)
    target.args = {"task":"detect", "model":str(MODEL)}
    file = folder / "export_fixture.pt"
    torch.save(dict(model=target, train_args=target.args), file)
    api = RTDETR(str(file))
    output = api.export(format="torchscript", imgsz=160, batch=1, device="cpu", half=False, optimize=False)
    jit = torch.jit.load(str(output)).eval()
    eager = deepcopy(target).fuse(verbose=False).eval()
    eager.model[-1].export = True
    x = torch.randn(1,3,160,160)
    with torch.no_grad():
        error = close(jit(x), eager(x), 3e-5, 3e-4)
        eager.model[17].tcr.enabled = False
        require((jit(x)-eager(x)).abs().max() > 0, "Export silently removed TCR")
    try:
        api.export(format="onnx", imgsz=160, device="cpu")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Unvalidated backend not rejected")
    return dict(status="PASS", backend="static FP32 TorchScript", nonzero_effect=True, max_abs_error=error)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lifecycle", action="store_true")
    p.add_argument("--export", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    report = dict(status="FAILED", scope="local interface checks, not B16/640 server capacity", runtime=runtime(), source=source_identity())
    try:
        with isolated_rng():
            from ultralytics.utils.torch_utils import init_seeds
            init_seeds(42,deterministic=True)
            report["algorithm"] = algorithm(args.device)
            write_json(args.output, report)
            report["model"] = model_checks(args.device, args.source, args.output.parent)
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            write_json(args.output, report)
            if args.lifecycle:
                report["lifecycle"] = lifecycle(args.output.parent / "lifecycle_fixture")
            if args.export:
                report["export"] = export_check(args.output.parent / "export_fixture")
            report["status"] = "PASS"
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(args.output, report)
        print(json.dumps(dict(status=report["status"], report=str(args.output))))


if __name__ == "__main__":
    main()
