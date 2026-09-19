"""One fixed small nonzero BLC file probe; no training, validation or retry loop."""
import argparse
from copy import deepcopy

import torch
from blc_common import ROOT, VARIANTS, paths, recipe, require, sha256, stamp, write_json, runtime, code_identity
from blc_lifecycle import cold_decoder, file_precision_paths
from blc_probe_io import temporary_probe, retained_size
from ultralytics.utils.patches import torch_load


def check(variant, device):
    folder = ROOT / "outputs/blc_v1" / ("backend-check-" + stamp())
    folder.mkdir(parents=True, exist_ok=False)
    initial = paths(variant)["init"]
    initial_hash = sha256(initial)
    report = dict(status="RUNNING", variant=variant, runtime=runtime(), code_sha256=code_identity()["sha256"],
                  initial_sha256=initial_hash, scope="one fixed 1x3x160x160 input, existing FP32/AMP/half modes; no training",
                  server_preflight="PENDING", paths={})
    try:
        torch.set_num_threads(4)
        with temporary_probe(folder, "nonzero-file-probe", report) as tmp:
            # Explicit nonzero test copy, not a trained result or new initialization.
            model = deepcopy(torch_load(initial, map_location="cpu")["model"]).eval()
            with torch.no_grad():
                model.model[5].blc.Wo.weight.fill_(0.001)
            cold_decoder(model)
            checkpoint = tmp / "native-half-model.pt"
            torch.save(dict(model=None, ema=model.half(), train_args=recipe(variant)[0]), checkpoint)
            del model
            sample = torch.linspace(0, 1, 3 * 160 * 160, device=device).reshape(1, 3, 160, 160)
            file_precision_paths(checkpoint, sample, variant, report["paths"])
            completed = [row for row in report["paths"].values() if row["status"] != "NOT_RUN"]
            report["same_path_status"] = "PASSED" if all(row["autobackend"]["status"] == "PASSED" for row in completed) else "FAILED"
            report["status"] = "PASSED" if all(row["status"] in ("PASSED", "PRECISION_NOTE") for row in completed) else "PENDING"
    except BaseException as error:
        report.update(status="FAILED", error=repr(error))
        raise
    finally:
        report["forward_count"] = sum(sum(row.get(key, {}).get("calls", 0) for key in ("direct_blc", "backend_blc"))
                                      + row.get("fusion", {}).get("forward_count", 0) for row in report["paths"].values())
        report["formal_init_untouched"] = sha256(initial) == initial_hash
        report["retained_bytes_before_report"] = retained_size(folder)
        require(report["formal_init_untouched"], "Probe changed controlled_init")
        write_json(folder / "report.json", report)
        print(folder / "report.json", report["status"], "retained_bytes=", retained_size(folder), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="cbr_lif_blc_v1")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = check(args.variant, args.device)
    if result["status"] != "PASSED":
        raise SystemExit("Probe retained a failure/PENDING result; no automatic retry")
