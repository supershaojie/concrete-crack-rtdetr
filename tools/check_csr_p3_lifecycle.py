"""CPU lifecycle contract checks; mocked training loop, no dataset/training/evaluation run."""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from init_csr_p3 import ROOT, SOURCE_SHA256, build, require, sha256, write_json, runtime
import train_csr_p3 as lifecycle
from csr_p3_results import postprocess
import pack_csr_p3 as packer
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import ModelEMA


def mock_native_path(folder, resumed=False, fail_final=False):
    """Use real RTDETR(checkpoint).train and native get_model; stub only setup/loop/final_eval."""
    variant = "cbr_lif_csr_p3_v1"
    p = dict(run=folder / "run", launch=folder / "launch", init=folder / "initial.pt")
    p["launch"].mkdir(parents=True)
    attempt = p["launch"] / "attempt"
    attempt.mkdir()
    model = build(variant, nc=1)
    if resumed:
        with torch.no_grad():
            model.model[17].csr.offset_pw.bias.fill_(0.07)
    saved_optimizer = torch.optim.AdamW(model.parameters())
    if resumed:
        model.model[17].csr.offset_pw.bias.grad = torch.ones_like(model.model[17].csr.offset_pw.bias)
        saved_optimizer.step()
        saved_optimizer.zero_grad(set_to_none=True)
    saved_scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    learned = {k: v.detach().clone() for k, v in model.state_dict().items() if ".csr." in k}
    arguments, _ = lifecycle.recipe(variant, p["init"], folder / "data.yaml")
    arguments.update(save_dir=str(p["run"]), project=str(folder), name="run")
    model.args = dict(arguments)
    checkpoint = dict(model=model, train_args=arguments, epoch=11 if resumed else -1,
                      optimizer=saved_optimizer.state_dict() if resumed else None,
                      scaler=saved_scaler.state_dict() if resumed else None,
                      ema=deepcopy(model) if resumed else None, updates=7, best_fitness=0.37)
    torch.save(checkpoint, p["init"])
    write_json(attempt / "preflight.json", {"status": "MOCKED_GATE"})
    write_json(attempt / "dispatch.json", dict(mode="resume" if resumed else "start", checkpoint=str(p["init"]),
               checkpoint_sha256=sha256(p["init"]), preflight_sha256=sha256(attempt / "preflight.json")))
    write_json(p["launch"] / "plan.json", dict(args=arguments))
    write_json(p["launch"] / "training_state.json", dict(status="DISPATCHED", attempt=str(attempt), completed_epochs=0))
    observed = {}
    def fake_init(self, overrides=None, _callbacks=None, **kwargs):
        actual = {k: v for k, v in overrides.items() if k != "session"}
        if actual.get("resume"):
            actual["resume"] = actual["model"] = str(p["init"])
        self.args = SimpleNamespace(**actual)
        self.callbacks = _callbacks
        self.data = dict(nc=1, channels=3)
        self.save_dir = p["run"]
        self.best, self.last = p["run"] / "weights/best.pt", p["run"] / "weights/last.pt"
        self.start_epoch = 12 if resumed else 0
        self.resume = resumed
        self.epochs = 200
        self.amp = True
        self.validator = SimpleNamespace(metrics={})
        self.model = str(p["init"])
    def fake_train(self):
        if resumed:
            loaded, _ = load_checkpoint(p["init"])
            self.model = self.get_model(cfg=loaded.yaml, weights=loaded, verbose=False)
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        if resumed:
            self.ema = ModelEMA(self.model)
            self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
            self.resume_training(torch.load(p["init"], map_location="cpu", weights_only=False))
            require(self.ema.updates == 7 and self.start_epoch == 12 and self.best_fitness == 0.37, "Native resume metadata lost")
            require(self.scaler.state_dict() == checkpoint["scaler"], "Native scaler resume lost state")
            require(len(self.optimizer.state) == len(saved_optimizer.state) == 1, "Native optimizer state lost")
            restored_state = next(iter(self.optimizer.state.values()))
            prior_state = next(iter(saved_optimizer.state.values()))
            require(all(torch.equal(v, restored_state[k]) for k, v in prior_state.items()), "Native AdamW momentum state lost")
            require(all(torch.equal(v, self.ema.ema.state_dict()[k]) for k, v in learned.items()), "Native EMA resume lost learned CSR")
            observed["native_resume_state"] = dict(optimizer=True, ema=True, scaler=True, scaler_enabled=bool(checkpoint["scaler"]))
        observed["epoch_absent_at_start"] = not hasattr(self, "epoch")
        self.run_callbacks("on_train_start")
        observed["csr_preserved"] = all(torch.equal(v, self.model.state_dict()[k]) for k, v in learned.items())
        require(observed["csr_preserved"], "Native model.train Trainer rebuild lost trained CSR")
        self.epoch = 199
        self.run_callbacks("on_fit_epoch_end")
        self.best.parent.mkdir(parents=True, exist_ok=True)
        self.model.args = dict(vars(self.args))
        torch.save(dict(model=deepcopy(self.model), train_args=vars(self.args)), self.best)
        self.final_eval()
    def fake_final(self):
        if fail_final:
            raise RuntimeError("intentional final_eval failure after epoch 200")
    with patch.object(lifecycle, "paths", return_value=p), patch.object(lifecycle, "require_clean"), \
         patch.object(lifecycle, "check_preflight"), patch.object(lifecycle, "native_amp_context", return_value=nullcontext()), \
         patch.object(RTDETRTrainer, "__init__", fake_init), \
         patch.object(RTDETRTrainer, "train", fake_train), patch.object(RTDETRTrainer, "final_eval", fake_final), \
         patch("ultralytics.utils.checks.check_pip_update_available", return_value=None):
        if fail_final:
            try:
                lifecycle.worker(variant, attempt)
            except RuntimeError as error:
                require("intentional final_eval" in str(error), "Wrong exception")
            else:
                raise AssertionError("Expected final_eval failure")
        else:
            lifecycle.worker(variant, attempt)
    state = lifecycle.load_json(p["launch"] / "training_state.json")
    require(state["completed_epochs"] == 200, "Completed epochs lost")
    require(state["status"] == ("FAILED" if fail_final else "COMPLETED_200"), "Wrong lifecycle classification")
    require(state["final_eval"] == ("FAILED" if fail_final else "PASSED"), "Final eval state lost")
    require(state["exit_code"] == (1 if fail_final else 0), "Wrong actual exit status")
    require(observed["epoch_absent_at_start"], "Did not test actual on_train_start epoch timing")
    return dict(resume=resumed, native_model_train=True, real_native_get_model=True, training_loop="MOCKED",
                epoch_absent_at_start=True, csr_preserved=True, completed_epochs=state["completed_epochs"],
                status=state["status"], final_eval=state["final_eval"], exit_code=state["exit_code"],
                native_resume_state=observed.get("native_resume_state"))


def protocol_check():
    values = torch.tensor([[[.5, .5, .2, .2, .1], [.5, .5, .2, .2, .9], [.5, .5, .2, .2, .6]]])
    before = values.clone()
    output = postprocess(values, 640, .5)[0]
    require(torch.equal(output["conf"], torch.tensor([.9, .6])), "Confidence mask is not applied after sorting")
    require(torch.equal(values, before), "Evaluation mutated raw model output")
    return dict(status="PASSED", sorted_confidences=output["conf"].tolist(), input_unchanged=True)


def gate_check(folder):
    folder.mkdir(parents=True)
    recipe_path = folder / "recipe.yaml"
    recipe_path.write_text("amp: true\n", encoding="utf-8")
    record = dict(variant="cbr_lif_csr_p3_v1", source=str(folder / "source.pt"),
                  args=dict(model=str(folder / "init.pt"), data=str(folder / "data.yaml")),
                  recipe_path=str(recipe_path), recipe_sha256=sha256(recipe_path),
                  parent_recipe_sha256=sha256(lifecycle.PARENT_ARGS))
    bound = dict(variant=record["variant"], commit="fixed_commit", source_sha256=SOURCE_SHA256,
                 init_sha256="fixed_init", data_inventory={"train": "fixed_manifest"}, recipe_sha256=sha256(recipe_path))
    good = dict(status="PASSED", mode="server", identity=bound,
                server_capacity=dict(status="PASSED", batch=16, imgsz=640, amp=True, effective_updates=2))
    report_path = folder / "preflight.json"
    rejected = []
    with patch("preflight_csr_p3.identity", return_value=bound):
        write_json(report_path, good)
        lifecycle.check_preflight(record, report_path)
        cases = {"pending": ("status", "PENDING"), "local_only": ("mode", "local")}
        for name, (key, value) in cases.items():
            candidate = deepcopy(good)
            candidate[key] = value
            write_json(report_path, candidate)
            try:
                lifecycle.check_preflight(record, report_path)
            except RuntimeError:
                rejected.append(name)
        for key, value in (("batch", 8), ("imgsz", 320), ("amp", False), ("effective_updates", 1)):
            candidate = deepcopy(good)
            candidate["server_capacity"][key] = value
            write_json(report_path, candidate)
            try:
                lifecycle.check_preflight(record, report_path)
            except RuntimeError:
                rejected.append(key)
        for key in ("commit", "variant", "init_sha256", "source_sha256", "data_inventory", "recipe_sha256"):
            candidate = deepcopy(good)
            candidate["identity"][key] = "changed"
            write_json(report_path, candidate)
            try:
                lifecycle.check_preflight(record, report_path)
            except RuntimeError:
                rejected.append(key)
    require(len(rejected) == 12, "A preflight gate accepted mismatched evidence")
    return dict(status="PASSED", accepted_exact_server_report=True, rejected=rejected)


def amp_resource_check(folder):
    folder.mkdir(parents=True)
    bus, weights = folder / "bus.jpg", folder / "local_amp.pt"
    bus.write_bytes(b"bus resource fixture")
    weights.write_bytes(b"weights resource fixture")
    resources = {"bus": dict(path=str(bus), sha256=sha256(bus)),
                 "weights": dict(path=str(weights), sha256=sha256(weights))}
    report = dict(server_capacity=dict(amp_resources=resources))
    with patch("ultralytics.utils.ASSETS", folder), patch("preflight_csr_p3.enforced_amp_check", return_value=True) as enforce:
        with lifecycle.native_amp_context(report):
            from ultralytics.engine.trainer import check_amp
            require(check_amp(torch.nn.Identity()) is True, "Native enforced AMP bridge was not used")
        require(enforce.call_count == 1, "Native AMP was bypassed")
        weights.write_bytes(b"changed")
        try:
            lifecycle.native_amp_context(report)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Changed AMP resource accepted")
    return dict(status="PASSED", native_checker_routing=True, changed_resource_rejected=True,
                actual_native_amp="Not run by this mocked test; covered separately by server preflight")


def package_check(folder):
    p = dict(run=folder / "run", launch=folder / "launch")
    p["launch"].mkdir(parents=True)
    (p["launch"] / "console.log").write_text("a" * 80000 + "END_EVIDENCE", encoding="utf-8")
    write_json(p["launch"] / "initialization.json", {"status": "PENDING", "reason": "test fixture"})
    write_json(p["launch"] / "training_state.json", dict(status="NOT_STARTED"))
    output = folder / "light.tar.gz"
    with patch.object(packer, "paths", return_value=p), patch.object(packer, "runtime", return_value={"commit": "MOCK"}), \
         patch.object(packer.subprocess, "check_output", return_value=b""):
        report = packer.package("cbr_lif_csr_p3_v1", output)
    manifest = packer.verify_archive(output)
    tail = manifest["lifecycle/console.log.tail.txt"]
    require(tail["bytes"] == 65536 and not report["evidence_complete"], "Log tail/absence evidence lost")
    require(report["bytes"] < 20*1024*1024, "Light archive exceeds limit")
    weight = folder / "do_not_pack.pt"
    weight.write_bytes(b"forbidden checkpoint fixture")
    with patch.object(packer, "paths", return_value=p):
        try:
            packer.package("cbr_lif_csr_p3_v1", folder / "bad.tar.gz", [weight])
        except RuntimeError:
            pass
        else:
            raise AssertionError("Direct evidence bypassed package allowlist")
    return dict(status="PASSED", log_tail_bytes=tail["bytes"], archive_verified=True, evidence_complete=False,
                direct_weight_evidence_rejected=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    (ROOT / "outputs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="csr_p3_lifecycle_", dir=ROOT / "outputs") as tmp:
        folder = Path(tmp)
        result = dict(status="PASSED", runtime=runtime(), scope="CPU integration with mocked loop, never formal training/test",
                      start=mock_native_path(folder / "start"), resume=mock_native_path(folder / "resume", resumed=True),
                      final_eval_failure=mock_native_path(folder / "failure", fail_final=True),
                      evaluation_protocol=protocol_check(), start_gate=gate_check(folder / "gate"),
                      amp_resource_guard=amp_resource_check(folder / "amp_resources"),
                      lightweight_package=package_check(folder / "pack"),
                      source_hashes={name: sha256(ROOT / name) for name in
                          ("tools/train_csr_p3.py", "tools/csr_p3_results.py", "tools/pack_csr_p3.py", "tools/check_csr_p3_lifecycle.py")},
                      formal_training="NOT_STARTED", test="NOT_RUN")
    write_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
