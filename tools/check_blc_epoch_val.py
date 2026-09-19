"""One real val batch through native half EMA validator; not final/full evaluation."""
from __future__ import annotations
from copy import deepcopy
from types import SimpleNamespace
import torch
from blc_common import *
from blc_preflight import OneBatchLoader
from ultralytics.utils.torch_utils import ModelEMA, init_seeds


def check():
    variant = "cbr_lif_blc_v1"
    p = paths(variant)
    folder = p["evidence"]/("local-half-epoch-val-"+stamp())
    report = dict(status="FAILED", scope="B2/160, one real val batch, explicit nonzero test copy; not server capacity",
                  code=code_identity(), runtime=runtime())
    init_seeds(42, deterministic=True)
    args, _ = recipe(variant)
    args.update(imgsz=160, batch=2, workers=0, plots=False, project=str(folder), name="probe", save_dir=str(folder/"probe"))
    trainer = BLCTrainer(overrides=args)
    weights = torch_load(p["init"], map_location="cpu")["model"].float()
    trainer.model = trainer.get_model(weights.yaml, weights).cuda()
    trainer.set_model_attributes()
    with torch.no_grad():
        trainer.model.model[5].blc.Wo.weight.normal_(0, .01)
    trainer.test_loader = trainer.get_dataloader(trainer.data["val"], batch_size=2, rank=-1, mode="val")
    trainer.validator = trainer.get_validator()
    trainer.validator.dataloader = OneBatchLoader(trainer.test_loader)
    trainer.ema = ModelEMA(trainer.model)
    trainer.amp, trainer.epoch, trainer.world_size = True, 0, 1
    trainer.stopper = SimpleNamespace(possible_stop=False)
    trainer.loss_items = torch.zeros(3, device="cuda")
    calls = []
    hook = trainer.ema.ema.model[5].blc.register_forward_hook(
        lambda m, a, y: calls.append(dict(dtype=str(y.dtype), finite=bool(torch.isfinite(y).all()),
                                          increment=float((y-a[0]).float().norm()))))
    try:
        metrics = trainer.validator(trainer=trainer)
        require(trainer.validator.args.half and calls and all(row["finite"] and row["increment"] > 0 for row in calls),
                "Native half EMA lost the BLC branch")
        require(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()), "Nonfinite epoch-val metrics")
        report.update(status="PASSED", half=True, calls=calls, metrics=metrics, evaluated_images=2,
                      note="Temporary untrained/nonzero copy; metrics are only a finite-path diagnostic")
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        hook.remove()
        write_json(folder/"report.json", report)
        print(folder/"report.json", report["status"], flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    check()
