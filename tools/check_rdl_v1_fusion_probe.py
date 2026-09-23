"""Fault-injection check for diagnostic error preservation using a local fixture."""
from pathlib import Path
import argparse
import json
from unittest.mock import patch

import torch

from init_c19_lif_v1 import require
from ultralytics.nn.tasks import BaseModel
from ultralytics.utils.patches import torch_load
from rdl_v1_fusion import check_ema_fusion, restore_rng


def run(fixture, folder, device):
    saved = torch_load(fixture, map_location="cpu")
    model = saved["model"].to(device).eval()
    batch = dict(img=saved["image"].to(device),
                 bboxes=torch.tensor([[.4,.4,.2,.1],[.6,.6,.3,.2],[.2,.7,.1,.25]], device=device),
                 cls=torch.zeros(3,1,device=device), batch_idx=torch.tensor([0,1,1],device=device))
    restore_rng(saved["rng"])
    native_fuse = BaseModel.fuse
    def corrupted_fuse(self, *args, **kwargs):
        result = native_fuse(self, *args, **kwargs)
        # Only disposable diagnostic models: simulate a real wrong fusion.
        with torch.no_grad():
            self.model[-1].cbr.offset_out.bias.add_(3.)
        return result
    with patch.object(BaseModel, "fuse", corrupted_fuse):
        try:
            check_ema_fusion(model, batch, folder)
        except AssertionError:
            pass
        else:
            raise AssertionError("Diagnostic bypassed original ordered assertion")
    report = json.loads((folder/"fusion.json").read_text())
    require(report["original_status"] == "FAIL" and report["original_assertion"]["allclose_failed_count"] > 0,
            "Corrupted fusion was not rejected")
    require(report["diagnostic_status"] == "COMPLETE", "Failure evidence incomplete")
    require(all((folder/name).is_file() for name in ("fixture.pt", "records.pt")), "Failure fixture lost")
    require(any(v.get("status") for v in report["common_head_outputs"].values()),
            "Common-input replay failed to expose corrupted CBR")
    require(BaseModel.fuse is native_fuse, "Test patch leaked")
    print(json.dumps(dict(status="PASS", original_assertion_preserved=True,
                         corrupted_fusion_max_abs=report["original_assertion"]["max_abs_error"],
                         failure_evidence=str(folder/"fusion.json")), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    require(not args.output.exists(), "Use a new output directory")
    torch.set_num_threads(4)
    run(args.fixture, args.output, args.device)
