"""Small CPU serialization/guard fixture; not a CUDA capacity or real-resume pass."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
from rdm_common import ROOT, paths, recipe, torch_load, training_rebuild, require, write_json, verify_model, code_identity
from preflight_rdm import same_saved, optimizer_inventory
from rdm import preflight_failures
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import ModelEMA


def fixture(destination):
    torch.set_num_threads(4)
    variant = "cbr_lif_rdm_v1"
    p = paths(variant)
    report = dict(status="RUNNING", phase="load", scope="CPU native save/load/state-restore fixture only; server real resume PENDING", code=code_identity())
    folder = None
    try:
        write_json(destination, report)
        initial = torch_load(p["init"], map_location="cpu")["model"].float()
        model, _ = training_rebuild(initial.yaml, initial, dict(nc=1, channels=3), variant)
        args = recipe(variant)[0]
        t = RTDETRTrainer.__new__(RTDETRTrainer)
        t.model, t.args, t.epoch = model, SimpleNamespace(**args), 0
        t.optimizer = RTDETRTrainer.build_optimizer(t, model, name="AdamW", lr=.0005, momentum=.937, decay=.0001)
        report["optimizer"] = optimizer_inventory(t)
        # Exercise nonzero branch state only; no full-model training or dataset loader.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(9)
            x = torch.randn(1, 128, 8, 8)
            model.model[5].rdm(x).square().mean().backward()
            t.optimizer.step(); t.optimizer.zero_grad()
        require(torch.count_nonzero(model.model[5].rdm.Wo.weight) > 0, "Fixture did not update Wo")
        t.ema = ModelEMA(model)
        t.scaler = torch.cuda.amp.GradScaler(enabled=False)
        t.metrics, t.fitness, t.best_fitness, t.save_period = {}, 0., 0., -1
        with tempfile.TemporaryDirectory(prefix="rdm-v1-cpu-fixture-", dir=p["output"]) as temporary:
            folder = Path(temporary)
            t.wdir, t.last, t.best, t.csv = folder, folder/"last.pt", folder/"best.pt", folder/"results.csv"
            report["phase"] = "native_save"
            write_json(destination, report)
            t.save_model()
            saved = torch_load(t.last, map_location="cpu")
            require(saved["model"] is None and saved["ema"] is not None, "Native selected state changed")
            loaded, _ = load_checkpoint(t.last, device="cpu", fuse=False)
            require(all(same_saved(v, loaded.state_dict()[k]) for k,v in saved["ema"].state_dict().items()), "Native reload lost half-save state")
            model2, _ = training_rebuild(loaded.yaml, loaded, dict(nc=1, channels=3), variant, fresh=False)
            r = RTDETRTrainer.__new__(RTDETRTrainer)
            r.model, r.args, r.epochs, r.resume = model2, SimpleNamespace(**args), 200, True
            r.optimizer = RTDETRTrainer.build_optimizer(r, model2, name="AdamW", lr=.0005, momentum=.937, decay=.0001)
            r.ema, r.scaler = ModelEMA(model2), torch.cuda.amp.GradScaler(enabled=False)
            r.resume_training(saved)
            require(same_saved(saved["optimizer"], r.optimizer.state_dict()), "Native optimizer restore differs")
            require(r.start_epoch == 1 and r.ema.updates == saved["updates"], "Native epoch/EMA restore differs")
            report["native_save_reload"] = dict(fields=sorted(saved), bytes=t.last.stat().st_size, same_quantization_exact=True, optimizer_state_restore=True)
            report["phase"] = "cpu_backend"
            write_json(destination, report)
            backend = AutoBackend(str(t.last), device=torch.device("cpu"), fp16=False, fuse=True, verbose=False)
            verify_model(backend.model, variant)
            calls = []
            handle = backend.model.model[5].rdm.register_forward_hook(lambda m,a,o: calls.append(bool(torch.isfinite(o).all())))
            with torch.no_grad():
                prediction = backend(torch.zeros(1, 3, 160, 160))[0]
            handle.remove()
            require(calls == [True] and prediction.shape == (1, 300, 5) and torch.isfinite(prediction).all(), "Nonzero CPU backend fixture failed")
            report["cpu_backend"] = dict(shape=list(prediction.shape), rdm_calls=1, finite=True, imgsz=160)
        current = {"variant": variant, "code": "fixture"}
        good = dict(status="PASSED", identity=current, training_batches=6, temporary_files_cleaned=True)
        for key in ("start", "save_reload", "resume", "half_ema_val", "backend_fp32", "backend_half"):
            good[key] = dict(status="PASSED", effective_updates=2 if key=="start" else 1)
        good["start"]["batches"] = 4
        good["half_ema_val"]["batches"] = 1
        require(not preflight_failures(good, current), "Guard rejects valid fixture")
        faults = {}
        for key in ("start", "save_reload", "resume", "half_ema_val", "backend_fp32", "backend_half"):
            bad = deepcopy(good); bad[key]["status"] = "NOT_RUN"
            faults[key] = bool(preflight_failures(bad, current))
        faults["stale_identity"] = bool(preflight_failures(good, dict(current, code="changed")))
        bad = deepcopy(good); bad["training_batches"] = 17
        faults["budget"] = bool(preflight_failures(bad, current))
        bad = deepcopy(good); bad["start"]["effective_updates"] = 1
        faults["effective_updates"] = bool(preflight_failures(bad, current))
        require(all(faults.values()), "Guard accepted incomplete/stale evidence")
        report.update(status="PASSED", guard_faults_rejected=faults)
    except BaseException as error:
        report.update(status="FAILED", error=repr(error))
        raise
    finally:
        report["temporary_files_cleaned"] = folder is None or not folder.exists()
        write_json(destination, report)
    return report


if __name__ == "__main__":
    print(json.dumps(fixture(paths("cbr_lif_rdm_v1")["output"] / "ops.json"), indent=2))
