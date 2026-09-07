"""Create clean FP32 C20 initialization from the SHA-locked original C2 source. No training."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "ultralytics-main"
MODEL_DIR = ULTRALYTICS_ROOT / "ultralytics/cfg/models/rt-detr"
BASE_CFG = MODEL_DIR / "rtdetr-resnet18-lite.yaml"
C20_CFG = MODEL_DIR / "rtdetr-resnet18-lite-cscef-cbr.yaml"
SERVER_MAIN = Path("/root/autodl-tmp/projects/Crack_RTDETR")
DEFAULT_SOURCE = SERVER_MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
DEFAULT_OUTPUT = ROOT / "weights/rtdetr_r18_lite_cscef_cbr_imagenet_backbone_init.pt"
SOURCE_SHA256 = "fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e"
BASE_COMMIT = "67c3078e54a657fd96d65fee657a75fbb1dae0d6"
PROTECTION = ROOT / "docs/c20_protected.json"
sys.path.insert(0, str(ULTRALYTICS_ROOT))

import ultralytics
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import AIFI, RTDETRDecoderCBR, CSCEFv51
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def write_json(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(value):
        if isinstance(value, Path): return str(value)
        raise TypeError(f"Unsupported report value: {type(value)}")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=convert) + "\n", encoding="utf-8")


def verify_protected():
    manifest = json.loads(PROTECTION.read_text(encoding="utf-8"))
    changed = [n for n, digest in manifest["files"].items()
               if (sha256(ROOT / n) if n.endswith(".jpg") else canonical_hash((ROOT / n).read_bytes())) != digest]
    require(not changed, f"Protected C2 sources changed: {changed}")
    return {"status": "passed", "reference_commit": BASE_COMMIT, "files": manifest["files"]}


def runtime_info():
    imported = Path(ultralytics.__file__).resolve()
    require(imported.is_relative_to(ULTRALYTICS_ROOT.resolve()), "Ultralytics imported outside C20 worktree.")
    files = [p for folder in (ULTRALYTICS_ROOT / "ultralytics", ULTRALYTICS_ROOT / "tests", ROOT / "tools")
             for p in folder.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".sh"}]
    files.append(PROTECTION)
    return {"ultralytics_file": str(imported), "ultralytics_version": ultralytics.__version__,
            "python": sys.version, "torch_version": str(torch.__version__), "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
            "code_sha256": {p.relative_to(ROOT).as_posix(): canonical_hash(p.read_bytes()) for p in sorted(files)}}


def clean_checkpoint(checkpoint):
    return checkpoint.get("epoch") == -1 and all(checkpoint.get(k) is None for k in
        ("best_fitness", "ema", "optimizer", "scaler", "updates", "train_metrics", "train_results"))


def read_source(source):
    require(source.is_file() and sha256(source) == SOURCE_SHA256,
            f"Original C2 initialization missing/SHA mismatch: {source}; trained checkpoints forbidden.")
    checkpoint = torch_load(source, map_location="cpu")
    require(clean_checkpoint(checkpoint), "C2 source contains training state.")
    return checkpoint, deepcopy(checkpoint["model"]).float()


def topology(base=None, target=None):
    """Prove the C2 graph survives insertion; derive every index from the actual YAML."""
    base = deepcopy(base) if base is not None else YAML.load(BASE_CFG)
    target = deepcopy(target) if target is not None else YAML.load(C20_CFG)
    a, b = base.pop("backbone") + base.pop("head"), target.pop("backbone") + target.pop("head")
    require(base == target, "Non-graph YAML settings differ from C2.")
    cs = [i for i, node in enumerate(b) if node[2] == "CSCEFv51"]
    require(len(cs) == 1 and len(b) == len(a) + 1, "Expected exactly one CSCEF insertion.")
    cs = cs[0]
    mapping = dict(zip(range(len(a)), (i for i in range(len(b)) if i != cs)))
    def refs(node, index):
        f = node[0]
        return [index + j if j < 0 else j for j in (f if isinstance(f, list) else [f])]
    fusion = cs + 1
    require(b[cs][1:] == [1, "CSCEFv51", []], "C17 CSCEF arguments changed.")
    require(b[fusion][2] == "Concat" and b[fusion + 1][2] == "RepC3", "CSCEF must precede the original concat/RepC3.")
    for old, new in mapping.items():
        expected = deepcopy(a[old][1:])
        if old == len(a) - 1:
            require(expected[1] == "RTDETRDecoder", "C2 last node is not decoder.")
            expected[1] = "RTDETRDecoderCBR"
        require(b[new][1:] == expected, f"Layer arguments/class changed: C2 {old} -> C20 {new}.")
        incoming = [mapping.get(j, j) for j in refs(a[old], old)]
        if new == fusion:
            require(refs(b[cs], cs) == [incoming[1], incoming[0]], "CSCEF inputs must be [projected S3, upsampled P4].")
            require(b[incoming[0]][2] == "nn.Upsample" and b[incoming[1]][2] == "Conv", "Incorrect CSCEF input producers.")
            incoming[1] = cs  # [S, L] becomes [S, F_out], with identical channel order.
        require(refs(b[new], new) == incoming, f"Connection mismatch: C2 {old} -> C20 {new}.")
    decoder_inputs = refs(b[-1], len(b) - 1)
    require(decoder_inputs[0] == fusion + 1 and all(b[i][2] == "RepC3" for i in decoder_inputs),
            "CBR input slot 0 must be final Neck P3, with P3/P4/P5 in C2 order.")
    return {"layer_mapping": mapping, "cscef": cs, "cscef_inputs": refs(b[cs], cs),
            "concat": fusion, "concat_inputs": refs(b[fusion], fusion), "decoder": len(b) - 1,
            "decoder_inputs": decoder_inputs, "status": "passed"}


def remap_key(key, graph=None):
    parts = key.split(".")
    require(len(parts) >= 3 and parts[0] == "model", f"Unexpected C2 state key: {key}")
    parts[1] = str((graph or topology())["layer_mapping"][int(parts[1])])
    return ".".join(parts)


def tensor_rows(source, target, mapping=None):
    rows = []
    for k, v in source.items():
        dest = mapping[k] if mapping is not None else k
        rows.append({"key": k, "target_key": dest, "shape": list(v.shape), "dtype": str(v.dtype),
                     "equal": dest in target and v.shape == target[dest].shape and v.dtype == target[dest].dtype
                     and torch.equal(v.cpu(), target[dest].cpu())})
    return rows


def common_rows(source, target):
    graph = topology()
    mapping = {k: remap_key(k, graph) for k in source}
    require(len(set(mapping.values())) == len(source), "Common mapping contains duplicate destinations.")
    return tensor_rows(source, target, mapping)


def branch_modules(model):
    return {f"model.{i}": m for i, m in enumerate(model.model) if type(m) is CSCEFv51} | {
        new_prefix(model).rstrip("."): model.model[-1].cbr}


def new_state_keys(model):
    return {prefix + "." + k for prefix, module in branch_modules(model).items() for k in module.state_dict()}


def new_prefix(model):
    return f"model.{len(model.model) - 1}.cbr."


def verify_module(model, require_zero=True):
    graph = topology()
    require(len(model.model) == graph["decoder"] + 1, "Wrong combined model length.")
    for index, field in ((graph["cscef"], "cscef_inputs"), (graph["concat"], "concat_inputs"),
                         (graph["decoder"], "decoder_inputs")):
        require(model.model[index].f == graph[field], f"Parsed model connection differs: {field}")
    cs = model.model[graph["cscef"]]
    require(type(cs) is CSCEFv51 and sum(type(m) is CSCEFv51 for m in model.modules()) == 1, "Expected C17 CSCEF class once.")
    require(sum(p.numel() for p in cs.parameters()) == 26912 and len(cs.state_dict()) == 7,
            "C17 CSCEF parameters/buffers changed.")
    if require_zero:
        require(torch.count_nonzero(cs.output_projection.weight) == 0, "CSCEF output projection is not zero.")
    head = model.model[-1]
    require(type(head) is RTDETRDecoderCBR, "Expected CBR subclass head.")
    require(head.num_decoder_layers == 3 and head.num_queries == 300 and head.decoder.eval_idx == 2,
            "C2 Lite decoder configuration changed.")
    require(sum(type(m) is AIFI for m in model.modules()) == 1, "Original AIFI missing.")
    cbr = head.cbr
    require(cbr.rho == .1 and cbr.normal_fraction == .1, "Fixed C20 geometry changed.")
    require(cbr.p3_proj.out_channels == 64 and cbr.query_proj.out_features == 64, "Fixed projection changed.")
    require(sum(p.numel() for p in cbr.parameters()) <= 100000, "CBR exceeds parameter budget.")
    if require_zero:
        require(torch.count_nonzero(cbr.offset_out.weight) == 0 and torch.count_nonzero(cbr.offset_out.bias) == 0,
                "Expected zero residual initialization.")


def verify_reloads(output, expected):
    checkpoint = torch_load(output, map_location="cpu")
    require(clean_checkpoint(checkpoint), "Initialization contains trained state.")
    results = {}
    for name, model in (("raw", checkpoint["model"]), ("RTDETR(checkpoint)", RTDETR(str(output)).model),
                        ("RTDETR(YAML).load(checkpoint)", RTDETR(str(C20_CFG)).load(str(output)).model)):
        verify_module(model)
        rows = tensor_rows(expected, model.state_dict())
        require(set(expected) == set(model.state_dict()) and all(r["equal"] for r in rows), f"{name}: reload inexact.")
        results[name] = {"status": "passed", "states_exact": len(rows)}
    return results


def initialize(source, output):
    # Preserve the caller's CPU and CUDA RNG states, including all reload constructors.
    # All construction is on CPU; do not call torch.manual_seed (which seeds CUDA too).
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=[]):
        report = _initialize(source, output)
    require(torch.equal(cpu, torch.get_rng_state()), "Initialization changed caller CPU RNG.")
    require(all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all() if cuda else [])),
            "Initialization changed caller CUDA RNG.")
    report.update(caller_cpu_rng_preserved=True, caller_cuda_rng_preserved=True if cuda else "unavailable")
    return report


def _initialize(source, output):
    source, output = source.resolve(), output.resolve()
    require(source != output and not output.exists(), f"Refusing overwrite: {output}")
    verify_protected()
    _, source_model = read_source(source)
    torch.random.default_generator.manual_seed(42)
    baseline = RTDETRDetectionModel(str(BASE_CFG), nc=80, verbose=False)
    rng = torch.get_rng_state().clone()
    torch.random.default_generator.manual_seed(42)
    target = RTDETRDetectionModel(str(C20_CFG), nc=80, verbose=False).eval()
    require(torch.equal(rng, torch.get_rng_state()), "Constructor consumed extra CPU RNG.")
    require(all(r["equal"] for r in common_rows(baseline.state_dict(), target.state_dict())), "Fresh common states differ.")
    common = source_model.state_dict()
    require(set(common) == set(baseline.state_dict()), "Source is not the complete C2 model.")
    require(all(common[k].shape == baseline.state_dict()[k].shape for k in common), "C2 source shapes differ.")
    graph = topology()
    mapped = {remap_key(k, graph): v for k, v in common.items()}
    extra = set(target.state_dict()) - set(mapped)
    require(len(common) == 533 and len(mapped) == 533, "Expected all 533 C2 states exactly once.")
    require(extra == new_state_keys(target) and set(mapped) <= set(target.state_dict()), "Unexpected new/missing states.")
    target.load_state_dict({**target.state_dict(), **mapped}, strict=True)
    rows = common_rows(common, target.state_dict())
    require(all(r["equal"] for r in rows), "C2 source values changed.")
    verify_module(target)
    provenance = {"source": str(source), "source_sha256": SOURCE_SHA256, "c2_commit": BASE_COMMIT,
                  "runtime": runtime_info(), "seed": 42, "storage": "FP32; original C2 half values exactly promoted"}
    target.args = {**DEFAULT_CFG_DICT, "model": str(C20_CFG), "task": "detect"}
    target.task, target.pt_path = "detect", str(output)
    checkpoint = {"epoch": -1, "best_fitness": None, "model": deepcopy(target).float(), "ema": None,
                  "updates": None, "optimizer": None, "scaler": None, "train_args": target.args,
                  "train_metrics": None, "train_results": None, "date": datetime.now(timezone.utc).isoformat(),
                  "version": ultralytics.__version__, "license": "AGPL-3.0", "c20_provenance": provenance}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        torch.save(checkpoint, file)
    return {**provenance, "output": str(output), "output_sha256": sha256(output), "common_tensor_audit": rows,
            "topology": graph, "new_state_keys": sorted(extra),
            "new_parameters": {prefix: sum(p.numel() for p in m.parameters()) for prefix, m in branch_modules(target).items()},
            "constructor_common_states_and_rng_equal": True, "reloads": verify_reloads(output, target.state_dict()),
            "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/c20/initialization.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {"status": "failed"}
    try:
        report = initialize(args.source, args.output)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        write_json(args.report, report)
    print(f"Initialization passed: {args.output}\nSHA256: {report['output_sha256']}")


if __name__ == "__main__":
    main()
