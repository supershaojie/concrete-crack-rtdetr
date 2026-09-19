"""Small failure-path fixtures, without model training or extra CUDA forwards."""
from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import patch

import torch
import blc_lifecycle as life
import blc_preflight as probe
import blc_server as server
from blc_common import ROOT, require, sha256, stamp, write_json
from blc_probe_io import temporary_probe


class Add(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
    def forward(self, x):
        return x + self.weight


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])
        self.model[5].blc = Add()
    def forward(self, x):
        return (self.model[5].blc(x),)


class Backend(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = Tiny()
    def forward(self, x):
        return self.model(x)


def check():
    result = dict(status="RUNNING", scope="isolated failure/budget/cleanup fixtures; zero training batches")
    folder = ROOT / "outputs/blc_v1" / ("light-faults-" + stamp())
    folder.mkdir(parents=True)
    try:
        instances = []
        def load(*args, **kwargs):
            net = Tiny(); instances.append(net); return net, {}
        rows = {}
        tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        mismatch = dict(allclose=False, finite=True, max_abs=1., relative_L2=1., atol=2e-5, rtol=2e-4)
        with patch.object(life, "load_checkpoint", load), patch.object(life, "AutoBackend", Backend), \
             patch.object(life, "verify_model"), patch.object(life, "difference", return_value=mismatch):
            try:
                life.file_precision_paths("fixture.pt", torch.ones(1, 1, 2, 2), "cbr_lif_blc_v1", rows)
            except RuntimeError as error:
                require(str(error) == "Same-path AutoBackend output differs", "Wrong mismatch rejection")
            else:
                raise AssertionError("Mismatch accepted")
        require(rows["FP32"]["status"] == "FAILED" and rows["FP32"]["autobackend"]["max_abs"] == 1., "Failed row lost")
        require(rows["FP32"]["exception_stage"] == "same_path_assertion", "Exception stage lost")
        require(all(rows[m]["status"] == "NOT_RUN" for m in ("AMP", "half")), "Unexecuted modes misreported")
        require(all(not m._forward_hooks for net in instances for m in net.modules()), "Hook leaked")
        require(tf32 == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32), "TF32 flags leaked")
        result["failed_mode_preserved"] = rows
        result["hooks_and_tf32_restored"] = True

        for failure in (None, RuntimeError, KeyboardInterrupt):
            evidence = {}
            try:
                with temporary_probe(folder, "cleanup-fixture", evidence) as temporary:
                    (temporary / "native-last.pt").write_bytes(b"owned fixture checkpoint")
                    if failure:
                        raise failure("injected")
            except (RuntimeError, KeyboardInterrupt):
                require(failure is not None, "Unexpected cleanup error")
            require(evidence["temporary_artifacts"][0]["cleaned"], "Temporary checkpoint leaked")
            require(evidence["temporary_artifacts"][0]["checkpoints"][0]["sha256"], "Checkpoint hash lost")
        result["temporary_cleanup_success_failure_interrupt"] = True

        t = SimpleNamespace(blc_seen=0, blc_limit=16, blc_batches=[], blc_steps=[], blc_effective=0,
                            loss=torch.tensor(1.), blc_gt=[1], accumulate=1, amp=True)
        for index in range(16):
            probe.budget_start(t)
            try:
                probe.budget_end(t)
            except probe.BudgetComplete:
                require(index == 15, "Wrong budget cutoff")
        try:
            probe.budget_start(t)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Seventeenth batch accepted")
        t.blc_seen, t.blc_limit = 4, 4  # remaining resume budget after 12 start batches
        try:
            probe.budget_start(t)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Combined start/resume budget exceeded")
        result["training_budget_guards"] = "PASSED"
        class Loader:
            dataset = [0, 1, 2]
            def __iter__(self):
                return iter(self.dataset)
        loader = probe.OneBatchLoader(Loader())
        require(list(loader) == [0] and loader.batches == 1, "Validation not bounded")
        result["real_loader_wrapper_one_batch"] = True

        # Exercise the outer capacity cleanup and phase reporting on interruption.
        with temporary_probe(folder, "outer-fixture", result) as temporary:
            init = temporary / "controlled.pt"; init.write_bytes(b"preserve initial fixture")
            expected = sha256(init)
            def interrupted(variant, owned, report, active):
                (owned / "weights").mkdir()
                (owned / "weights/last.pt").write_bytes(b"probe only")
                report["capacity"]["status"] = "RUNNING"
                raise KeyboardInterrupt("injected bounded training interruption")
            report = {}
            with patch.object(probe, "paths", return_value={"init": init}), patch.object(probe, "_capacity", interrupted):
                try:
                    probe.capacity("cbr_lif_blc_v1", temporary, report)
                except KeyboardInterrupt:
                    pass
                else:
                    raise AssertionError("Interruption swallowed")
            require(report["capacity"]["status"] == "FAILED", "Interrupted phase misreported")
            require(all(report[k]["status"] == "NOT_RUN" for k in ("lifecycle", "native_half_ema_epoch_val", "native_resume")), "Unrun stages lost")
            require(report["temporary_artifacts"][0]["cleaned"] and sha256(init) == expected, "Outer cleanup touched init or leaked checkpoint")
            result["outer_capacity_failure_cleanup"] = report
        import ultralytics.utils as utils
        for fail in (False, True):
            with temporary_probe(folder, "resource-fixture", result) as temporary:
                main, work = temporary / "main", temporary / "work"
                (main / "weights").mkdir(parents=True)
                (main / "weights/yolo26n.pt").write_bytes(b"existing source resource")
                (main / "bus.jpg").write_bytes(b"existing image resource")
                work.mkdir()
                def capacity_fixture(variant, target, report):
                    if fail:
                        raise RuntimeError("injected capacity error")
                    for key in ("capacity", "lifecycle", "native_half_ema_epoch_val", "native_resume"):
                        report[key] = dict(status="PASSED")
                with patch.object(server, "ROOT", work), patch.object(server, "MAIN", main), \
                     patch.object(utils, "ASSETS", work / "assets"), \
                     patch.object(server, "paths", return_value={"evidence": work / "evidence"}), \
                     patch.object(server, "environment"), patch.object(server, "require_evidence"), \
                     patch.object(server, "evidence_context", return_value={"fixture": True}), \
                     patch.object(probe, "capacity", capacity_fixture):
                    try:
                        server.preflight("cbr_lif_blc_v1")
                    except RuntimeError as error:
                        require(fail and str(error) == "injected capacity error", "Unexpected resource fixture failure")
                saved = json.loads(next((work / "evidence").glob("preflight-*/report.json")).read_text())
                require(saved["status"] == ("FAILED" if fail else "PASSED"), "Resource exit status lost")
                require(all(row["cleaned"] for row in saved["native_amp_resources"]), "Borrowed resource leaked")
                require((main / "weights/yolo26n.pt").read_bytes() == b"existing source resource", "Source resource changed")
        result["borrowed_resources_cleanup_success_failure"] = True
        result["status"] = "PASSED"
    except BaseException as error:
        result.update(status="FAILED", error=repr(error))
        raise
    finally:
        write_json(folder / "report.json", result)
        print(folder / "report.json", result["status"], flush=True)


if __name__ == "__main__":
    check()
