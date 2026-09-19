"""Read-only, fixed-session B7 parent control. Never issues an admission decision.

Only new diagnostic files are written. The supervisor enforces 300 seconds,
including imports, input hashing, reconstruction and all <=16 detector calls.
See docs/dpr/B7_SAVED_STATE_REVIEW.md for the non-state reconstruction proof.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import tarfile
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "f0eace10a882049e7053fbe9f69cad0c38149554"
SESSION = "/root/autodl-tmp/projects/Crack_RTDETR-dpr-v1/outputs/dpr/cbr_lif_dpr_v1/r1_20260919T165019_256581"
ARCHIVE_SHA = "c34d69c0565f0e02930156961053e42c2d5a04b503bf3fab81e17c47a1b9aa9b"
MANIFEST_SHA = "ff76e9d8645864a0fe555af9c3ea441dc414bab26de3ff88283f13f801f5b711"
MODES = ("cuda_fp32", "cuda_native_amp")
ROLES = ("parent", "candidate")
ATOL, RTOL = 2e-5, 2e-4
STAGES = ("dpr_input", "effective_kernel", "conv_output", "norm_output", "target",
          "P3", "P4", "P5", "encoder_features", "candidate_scores", "output")
LIMIT = 300


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    def safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite_float": repr(value)}
        if isinstance(value, dict):
            return {k:safe(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        return value
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(safe(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], timeout=15).decode().strip()


def package_inputs(archive, manifest):
    """Read members in memory; never extract an untrusted archive path."""
    need(sha(manifest) == MANIFEST_SHA, "Not the supplemented manifest")
    index = json.loads(manifest.read_text(encoding="utf-8"))
    need(index["archive"]["sha256"] == ARCHIVE_SHA and sha(archive) == ARCHIVE_SHA
         and archive.stat().st_size == index["archive"]["bytes"], "Archive identity differs")
    with tarfile.open(archive) as packed:
        members = packed.getmembers()
        need(len(members) == len({m.name for m in members}) == 20, "Duplicate/missing package members")
        need(all(m.isfile() for m in members) and {m.name for m in members} == set(index["files"]), "Package inventory differs")
        contents = {}
        for member in members:
            expected = index["files"][member.name]
            # Archive and manifest already have fixed, authenticated identities.
            # The original supplement.json is 421,296,366 bytes uncompressed.
            need(member.size == expected["bytes"], "Member size differs: " + member.name)
            data = packed.extractfile(member).read()
            need(hashlib.sha256(data).hexdigest() == expected["sha256"], "Member digest differs: " + member.name)
            contents[member.name] = data
    report = json.loads(contents["supplement.json"])
    need(report["session"] == SESSION and report["variant"] == "cbr_lif_dpr_v1", "Wrong original session/variant")
    need(report["code_identity"]["head"] == SOURCE_SHA and report["source_stable"]
         and report["code_identity"] == report["code_identity_at_end"], "Original code identity differs")
    need(report["status"] == "BLOCKED" and report["collection_status"] == "COMPLETED"
         and report["contract"] == "dpr_acceptance_v2", "Wrong original R1 record")
    source_files = report["code_identity"]["files"]
    verify_protected_sources(source_files)
    for name, data in contents.items():
        if name.startswith("source/"):
            original = "tools/" + name.split("/", 1)[1]
            need(hashlib.sha256(data).hexdigest() == source_files[original], "Packaged source differs: " + name)
    return report, dict(archive_sha256=ARCHIVE_SHA, manifest_sha256=MANIFEST_SHA, members_verified=20,
                        protected_sources_verified=len(source_files), original_source_sha=SOURCE_SHA)


def verify_protected_sources(source_files):
    for name, expected in source_files.items():
        path = PurePosixPath(name)
        need(not path.is_absolute() and ".." not in path.parts, "Unsafe source identity")
        data = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        need(hashlib.sha256(data).hexdigest() == expected, "Protected source changed: " + name)


def references(report):
    rows = [("batch", report["batch"]["artifact"], SESSION + "/batch.pt")]
    for mode in MODES:
        entry = report["modes"][mode]
        need(entry["batch"] == 2 and entry["imgsz"] == 160, "Input geometry changed")
        for role in ROLES:
            runs = entry["B_evidence"][role]["native_replays"]
            need(len(runs) == 2, "Expected exactly two saved updates")
            for i, run in enumerate(runs):
                name = f"{mode}/{role}/native_replay_{i}.pt"
                rows.append((name, run["artifact"], SESSION + "/" + name))
    for name, record, expected_path in rows:
        need(record["path"] == expected_path and record["bytes"] > 0 and len(record["sha256"]) == 64,
             "Saved artifact binding differs: " + name)
    need(len({r[1]["path"] for r in rows}) == 9, "Saved inputs are not distinct")
    return rows


def verify_file(record):
    path = Path(record["path"])
    need(path.is_file() and not path.is_symlink(), "Missing/nonregular server tensor: " + str(path))
    need(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "Tensor identity differs: " + str(path))


class Budget:
    def __init__(self, deadline, progress=lambda: None):
        self.deadline, self.progress = deadline, progress
        self.counts = {}

    def check(self):
        need(time.monotonic() < self.deadline, "300 second overall budget expired")

    def invoke(self, group, model, image):
        self.check()
        need(group in {m + "/" + r for m in MODES for r in ROLES}, "Unknown forward group")
        need(self.counts.get(group, 0) < 4 and sum(self.counts.values()) < 16, "Forward budget exhausted")
        need(not model.training, "Only eval inference is allowed")
        self.counts[group] = self.counts.get(group, 0) + 1
        self.progress()  # Record started calls, even if a CUDA call is later killed.
        return model(image)


def tensor_identity(value):
    value = value.detach().cpu().contiguous()
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def state_identity(model):
    return {name: tensor_identity(value) for name, value in model.state_dict().items()}


def restore_post(payload, model, inventory):
    """Only actual_post.model supplies weights. No pre/EMA fallback or casting."""
    need(isinstance(payload, dict) and "actual_post" in payload, "Missing actual_post; fallback forbidden")
    actual = payload["actual_post"]
    saved, current = actual["model"], model.state_dict()
    need(set(saved) == set(current), "Parameter/buffer state names differ")
    need(set(dict(model.named_parameters())) == set(inventory["parameters"]), "Parameter inventory differs")
    buffers = dict(model.named_buffers())
    need(set(buffers) == set(inventory["buffers"]) == set(actual["buffers"]), "Buffer inventory differs")
    for name, value in saved.items():
        need(isinstance(value, torch.Tensor) and value.dtype == current[name].dtype
             and value.shape == current[name].shape, "Saved dtype/shape differs: " + name)
        need(not value.is_floating_point() or (value.dtype == torch.float32 and bool(torch.isfinite(value).all())),
             "Nonfinite/non-FP32 saved state: " + name)
    for name, value in actual["buffers"].items():
        need(name in saved and tensor_identity(value) == tensor_identity(saved[name]), "Separate saved buffer differs: " + name)
    model.load_state_dict(saved, strict=True)
    identity = state_identity(model)
    need(identity == {n:tensor_identity(v) for n,v in saved.items()}, "Restored state bytes differ")
    modes, grads = payload["model_modes"], payload["parameter_requires_grad"]
    need(set(modes) == set(dict(model.named_modules())) and all(type(v) is bool and v for v in modes.values()),
         "Original training-mode inventory differs; non-state reconstruction unresolved")
    need(set(grads) == set(dict(model.named_parameters())) and all(type(v) is bool for v in grads.values()),
         "Requires-grad inventory differs")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(grads[name])
    return dict(selected="actual_post.model", state=identity, parameter_names=inventory["parameters"],
                buffer_names=inventory["buffers"], exact_bytes=True, original_modes=modes, requires_grad=grads)


def rebuild(payload, original, mode, role):
    from init_dpr import build
    from ultralytics.nn.modules import LIFDown, RTDETRDecoderCBR
    from ultralytics.utils import IterableSimpleNamespace
    model = build("cbr_lif_dpr_v1", nc=1, baseline=role == "parent")
    model.nc, model.names = 1, {0:"crack"}
    model.args = IterableSimpleNamespace(**original["recipe"])
    # The native loss lazily installed this parameter-free child before B6.
    # Recreate its module inventory; inference with an image never calls it.
    model.criterion = model.init_criterion()
    need(not model.criterion.state_dict(), "Criterion acquired persistent state; reconstruction unresolved")
    evidence = restore_post(payload, model, original["modes"][mode]["B_evidence"][role]["inventory"])
    head = model.model[-1]
    need(type(head) is RTDETRDecoderCBR and head.nc == 1 and head.num_queries == 300
         and head.hidden_dim == 256 and head.f == [19,22,25] and head.decoder.eval_idx == 2
         and len(head.decoder.layers) == 3 and not head.export and not head.dynamic
         and not head.learnt_init_query and head.shapes == [] and head.anchors.numel() == 0
         and head.valid_mask.numel() == 0, "Decoder non-state constructor assumptions differ")
    need(type(model.model[20]) is LIFDown and head.cbr.rho == head.cbr.normal_fraction == .1,
         "CBR/LIF constructor attributes differ")
    target = model.get_submodule("model.5.blocks.1.branch2b")
    need((role == "parent" and type(target).__name__ == "ConvNormLayer") or
         (role == "candidate" and type(target).__name__ == "DPRConvNormLayer" and not target.deployed),
         "Target type/deployment state differs")
    # Fixed f0 path: fresh native constructor -> same 160px warmup under
    # mode autocast -> cached anchors -> CPU/CUDA moves -> B7 .float().eval().
    # No model forward is used to generate these data-independent caches.
    # In AMP, retain original autocast behavior of log; do NOT generate FP32
    # anchors directly or clear shapes and let a cold eval regenerate them.
    shapes = [[20,20], [10,10], [5,5]]
    amp = mode == "cuda_native_amp"
    with torch.cuda.amp.autocast(enabled=amp):
        anchors, mask = head._generate_anchors(shapes, dtype=torch.float16 if amp else torch.float32, device="cuda")
    head.shapes, head.anchors, head.valid_mask = shapes, anchors.cpu(), mask.cpu()
    evidence["cache_reconstruction"] = dict(basis="PINNED_SOURCE_PATH_DERIVATION; cache bytes were not serialized",
        source_sha=SOURCE_SHA, generated_under_native_autocast=amp, shapes=shapes,
        generated_anchors=tensor_identity(head.anchors), generated_valid_mask=tensor_identity(head.valid_mask),
        proof_document="docs/dpr/B7_SAVED_STATE_REVIEW.md", old_cache_bytes_independently_compared=False)
    model.to("cuda").float().eval()
    need(state_identity(model) == evidence["state"], "Device/eval preparation changed actual_post bytes")
    evidence["cache_reconstruction"]["eval_anchors"] = tensor_identity(head.anchors)
    evidence["cache_reconstruction"]["eval_valid_mask"] = tensor_identity(head.valid_mask)
    return model, evidence


def flags():
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic=torch.are_deterministic_algorithms_enabled(), warn_only=torch.is_deterministic_algorithms_warn_only_enabled())


def set_flags(value):
    torch.backends.cuda.matmul.allow_tf32 = value["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = value["cudnn_tf32"]
    torch.backends.cudnn.benchmark = value["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = value["cudnn_deterministic"]
    torch.use_deterministic_algorithms(value["deterministic"], warn_only=value["warn_only"])


@contextmanager
def original_flags(start):
    old = flags()
    try:
        desired = {k:start[k] for k in old}
        desired.update(matmul_tf32=False, cudnn_tf32=False)
        set_flags(desired)
        with torch.cuda.amp.autocast(enabled=False):
            yield
    finally:
        set_flags(old)


def observation_identity(model):
    head = model.model[-1]
    target = model.get_submodule("model.5.blocks.1.branch2b")
    return dict(state=state_identity(model), buffers={n:tensor_identity(v) for n,v in model.named_buffers()},
        modes={n:m.training for n,m in model.named_modules()}, requires_grad={n:p.requires_grad for n,p in model.named_parameters()},
        hooks={n:(dict(m._forward_hooks), dict(m._forward_pre_hooks)) for n,m in model.named_modules()},
        shapes=head.shapes.copy(), anchors=tensor_identity(head.anchors), mask=tensor_identity(head.valid_mask),
        decoder_override=head.__dict__.get("_get_decoder_input"), kernel_override=target.__dict__.get("get_equivalent_kernel"))


def raw_rows(left, right):
    from dpr_diagnostics import metric, selection_difference
    need(set(left) == set(right) == set(STAGES) | {"candidate_indices"}, "Incomplete observation coverage")
    for capture in (left, right):
        need(all(capture[k].dtype == torch.float32 for k in STAGES)
             and list(capture["candidate_scores"].shape) == [2,525,1]
             and list(capture["output"].shape) == [2,300,5], "Eval FP32/top-k/output geometry differs")
    rows = {k:metric(left[k], right[k], ATOL, RTOL) for k in STAGES}
    for value in (left["candidate_indices"], right["candidate_indices"]):
        need(list(value.shape) == [2,300] and value.dtype == torch.int64 and bool(((value >= 0) & (value < 525)).all())
             and all(v.unique().numel() == 300 for v in value), "Actual top-k geometry differs")
    return dict(comparisons=rows, selection=selection_difference(left["candidate_indices"], right["candidate_indices"]),
        first_exceeded=next((k for k in STAGES if not rows[k]["raw_allclose"]), None),
        first_exceeded_meaning="first observed stage; not necessarily the first erroneous operation")


def capture_checked(model, image, budget, group):
    from dpr_diagnostics import capture_trace
    before = observation_identity(model)
    try:
        return capture_trace(model, image, invoke=lambda x:budget.invoke(group, model, x))
    finally:
        need(observation_identity(model) == before, "Observation left state/cache/mode/hook/override changes: " + group)


def compare_pair(first, second, model, image, budget, group):
    from dpr_diagnostics import compare_traces
    result = raw_rows(first, second)
    if not result["comparisons"]["output"]["raw_allclose"]:
        before = observation_identity(model)
        try:
            original = compare_traces(first, second, model, image,
                invoke=lambda x:budget.invoke(group, model, x), atol=ATOL, rtol=RTOL)
        finally:
            need(observation_identity(model) == before, "Fixed replay changed model or left hooks")
        need(original["comparisons"] == result["comparisons"] and original["selection"] == result["selection"],
             "Original observation aggregation differs")
        result.update(original)
        result["fixed_replay_status"] = "MEASURED_ONCE"
    else:
        # compare_traces unconditionally performs a fourth forward. Do not call
        # it in this branch: its raw metric/selection logic above is identical.
        result["fixed_replay_status"] = "NOT_NEEDED_RAW_OUTPUT_PASSED"
    return result


def original_b7_consistency(current, old):
    keys = ("comparisons", "selection", "first_exceeded", "fixed_candidate_replay_output", "actual_selections")
    differences = {k:dict(original=old.get(k), reconstructed=current.get(k)) for k in keys if old.get(k) != current.get(k)}
    return dict(exact_record_match=not differences, differences=differences,
                note="Exact recorded-statistic comparison is a reconstruction diagnostic, not a new acceptance tolerance")


def interpretation(groups):
    if any(not g.get("original_B7_consistency", {}).get("exact_record_match", True) for g in groups.values()):
        return "RECONSTRUCTION_OR_OBSERVATION_MISMATCH_FIRST; keep BLOCKED; do not infer a model bug"
    if len(groups) != 4 or any(g.get("status") != "MEASURED" or "cross_update" not in g for g in groups.values()):
        return "INCOMPLETE; keep BLOCKED"
    failures = {k:bool(v["cross_update"]["first_exceeded"]) for k,v in groups.items()}
    if any(failures[m + "/parent"] for m in MODES):
        return "PARENT_ALSO_CROSSES_OBSERVED_BOUNDARIES; shared background is not DPR admission or causal proof; pause DPR or separately review contract"
    if any(failures.values()):
        return "ONLY_DPR_FAILS_IN_THIS_CONTROL; narrows localization, does not prove a formula error; pause DPR or separately review contract"
    return "NO_OBSERVED_FAILURE_IN_THIS_CONTROL; existing R1 BLOCKED is unchanged; no admission granted"


def parallel_table(groups):
    return {mode:{stage:{role:groups.get(mode+"/"+role, {}).get("cross_update", {}).get("comparisons", {}).get(stage,
                {"measurement":"NOT_COLLECTED"}) for role in ROLES} for stage in STAGES} for mode in MODES}


def worker(args, report):
    original, report["inputs"] = package_inputs(args.archive, args.manifest)
    report["diagnostic_sha"] = git("rev-parse", "HEAD")
    report["original_assessment"] = original["assessment"]
    from dpr_acceptance import validate_b
    report["original_B7_rejections"] = {}
    for mode in MODES:
        try:
            validate_b(original["modes"][mode], mode, files=False)
        except RuntimeError as error:
            need("B7 target" in str(error), "Original offline R1 rejection differs: " + str(error))
            report["original_B7_rejections"][mode] = str(error)
        else:
            raise RuntimeError("Original B7 rejection unexpectedly absent")
    rows = references(original)
    report["input_files"] = {n:dict(record=r, verification="PENDING_SERVER") for n,r,_ in rows}
    report["non_state_basis"] = "Pinned source reconstruction, including warmup autocast caches; not a serialized cache-byte comparison"
    if args.metadata_only:
        report.update(collection_status="LOCAL_METADATA_ONLY", server_forward="NOT_RUN", conclusion="BLOCKED; server tensor measurement pending")
        return
    need(str(args.archive.resolve()) == SESSION + "/r1_light.tar.gz"
         and str(args.manifest.resolve()) == SESSION + "/package_manifest.json", "Server session paths differ")
    need(not git("status", "--porcelain", "--untracked-files=no"), "Tracked checkout changes; stop before loading inputs")
    global torch
    import torch
    sys.path.insert(0, str(ROOT / "ultralytics-main"))
    import ultralytics
    from dpr_r1_evidence import batch_hash
    need(Path(ultralytics.__file__).resolve() == ROOT / "ultralytics-main/ultralytics/__init__.py", "Wrong production import")
    env = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    report["environment"] = env
    report["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    need(report["cublas_workspace_config"] == ":4096:8", "Pinned seed42 cuBLAS workspace setting missing")
    need(all(env[k] == original["environment"][k] for k in env), "Original server runtime differs; no dependency changes allowed")
    torch.set_num_threads(4)
    budget = Budget(args.deadline, lambda:write_json(args.output / "diagnostic.json", report))
    report["forward_counts"] = budget.counts
    for name, record, _ in rows:
        budget.check(); verify_file(record)
        report["input_files"][name]["verification"] = "PATH_SIZE_SHA256_VERIFIED"
    # Hash-verified local engineering payloads contain tensors/dicts only.
    # torch.load deserializes their container, but only actual_post.model is
    # selected for reconstruction; pre/EMA/optimizer are never applied.
    batch = torch.load(rows[0][1]["path"], map_location="cpu", weights_only=False)
    need(batch_hash(batch) == original["batch"]["fingerprint"] and list(batch["img"].shape) == [2,3,160,160]
         and batch["img"].dtype == torch.float32 and len(batch["bboxes"]) == original["batch"]["real_GT"], "Original batch/GT differs")
    image = batch["img"].to("cuda").float()
    report["groups"] = {}
    old_flags = flags()
    try:
        for mode in MODES:
            for role in ROLES:
                budget.check()
                group = mode + "/" + role
                current = report["groups"][group] = dict(status="IN_PROGRESS")
                entry = original["modes"][mode]["B_evidence"][role]
                starts = entry["starts"]
                need(len(starts) == 2 and all({k:s[k] for k in old_flags} == {k:starts[0][k] for k in old_flags} for s in starts),
                     "Original forward runtime flags differ")
                need(all(s["amp"] is (mode == "cuda_native_amp") and s["model_dtypes"] == ["torch.float32"]
                         and s["autocast_dtype"] == ("torch.float16" if mode == "cuda_native_amp" else "torch.float32")
                         and s["batch_sha256"] == original["batch"]["fingerprint"] for s in starts),
                     "Original cache-generation dtype/batch history differs")
                with original_flags(starts[0]):
                    traces, selections = [], []
                    current["restorations"] = []
                    for index, run in enumerate(entry["native_replays"]):
                        budget.check()
                        payload = torch.load(run["artifact"]["path"], map_location="cpu", weights_only=False)
                        model, evidence = rebuild(payload, original, mode, role)
                        del payload; gc.collect()
                        current["restorations"].append(evidence)
                        captured, selection = capture_checked(model, image, budget, group)
                        traces.append(captured); selections.append(selection)
                        if index == 0:
                            repeated, repeated_selection = capture_checked(model, image, budget, group)
                            repeat = current["same_state_repeat"] = raw_rows(captured, repeated)
                            repeat["all_bytes_equal"] = all(torch.equal(captured[k], repeated[k]) for k in captured)
                            repeat["actual_selections"] = [selection, repeated_selection]
                            need(repeat["first_exceeded"] is None and repeat["selection"]["changed_positions"] == 0,
                                 "Same-state inference is unstable; stop without retry: " + group)
                            del repeated, model; gc.collect(); torch.cuda.empty_cache()
                    current["cross_update"] = compare_pair(*traces, model, image, budget, group)
                    current["cross_update"]["actual_selections"] = selections
                    del model, traces; gc.collect(); torch.cuda.empty_cache()
                current["tf32_and_other_flags_restored"] = flags() == old_flags
                need(current["tf32_and_other_flags_restored"], "Precision flags were not restored")
                if role == "candidate":
                    current["original_B7_consistency"] = original_b7_consistency(current["cross_update"],
                        original["modes"][mode]["B_evidence"]["updated_function"])
                    if not current["original_B7_consistency"]["exact_record_match"]:
                        current["status"] = "RECONSTRUCTION_OR_OBSERVATION_MISMATCH"
                        raise RuntimeError("DPR differs from original B7 record; stop localization without retry: " + mode)
                current["status"] = "MEASURED"
                write_json(args.output / "diagnostic.json", report)
    finally:
        set_flags(old_flags)
        report["precision_flags_restored"] = flags() == old_flags
        report["protected_sources_unchanged_at_end"] = False
        verify_protected_sources(original["code_identity"]["files"])
        report["protected_sources_unchanged_at_end"] = True
    report["parallel_table"] = parallel_table(report["groups"])
    report["conclusion"] = interpretation(report["groups"])
    report["collection_status"] = "COMPLETED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(SESSION) / "r1_light.tar.gz")
    parser.add_argument("--manifest", type=Path, default=Path(SESSION) / "package_manifest.json")
    parser.add_argument("--metadata-only", action="store_true", help="Package/source validation only; no tensor load or inference")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        need(args.output is not None and args.deadline is not None, "Worker requires supervisor")
        report = dict(schema="dpr_saved_b7_control_v1", status="BLOCKED", admission_eligible=False,
            formal_training="NOT_STARTED", final_test="NOT_RUN", capacity="NOT_RUN", full_preflight="NOT_RUN",
            atol=ATOL, rtol=RTOL, comparison_direction="replay_0 is left, replay_1 is right; original comparison() statistics",
            limits=dict(seconds=LIMIT, per_group=4, total=16), forward_counts={}, collection_status="IN_PROGRESS")
        started = time.monotonic()
        try:
            worker(args, report)
        except Exception:
            report.update(collection_status="STOPPED", traceback=traceback.format_exc())
            print(report["traceback"], flush=True)
        finally:
            if "groups" in report:
                report["parallel_table"] = parallel_table(report["groups"])
                if report["collection_status"] != "COMPLETED":
                    report["conclusion"] = interpretation(report["groups"])
            report["worker_seconds"] = time.monotonic() - started
            write_json(args.output / "diagnostic.json", report)
        return 0 if args.metadata_only and report["collection_status"] == "LOCAL_METADATA_ONLY" else 3
    need(args.output is None and args.deadline is None, "Output/deadline are supervisor-owned")
    started = time.monotonic()
    output = ROOT / "outputs/dpr/cbr_lif_dpr_v1" / ("b7_saved_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f") + "_" + str(os.getpid()))
    output.mkdir(parents=True, exist_ok=False)
    print("NEW_DIAGNOSTIC=" + str(output), flush=True)
    code = ROOT / "tools/compare_dpr_b7.py"
    own_manifest = {}
    for path in (code, ROOT / "tools/test_dpr_b7_saved.py", ROOT / "tools/run_saved_dpr_b7.sh",
                 ROOT / "docs/dpr/B7_SAVED_STATE_REVIEW.md"):
        data = path.read_bytes(); destination = output / "source" / path.name
        destination.parent.mkdir(exist_ok=True); destination.write_bytes(data)
        own_manifest[str(path.relative_to(ROOT))] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    write_json(output / "source_manifest.json", own_manifest)
    command = [sys.executable, str(code), "--worker", "--output", str(output), "--deadline", str(started + LIMIT),
               "--archive", str(args.archive.resolve()), "--manifest", str(args.manifest.resolve())]
    if args.metadata_only: command.append("--metadata-only")
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        # Pinned original seed42 uses this value. Set it before CUDA initializes
        # in the disposable worker; the caller's environment is unchanged.
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"))
        try:
            rc = process.wait(timeout=max(.01, started + LIMIT - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(); rc = 124
    status = dict(status="BLOCKED", admission_eligible=False, exit_code=rc, elapsed_seconds=time.monotonic()-started,
                  deadline_seconds=LIMIT, timed_out=rc == 124, formal_training="NOT_STARTED", final_test="NOT_RUN")
    if rc == 124:
        status["reason"] = "Overall budget exhausted; worker terminated, no retry; partial measurements are not complete evidence"
    write_json(output / "exit_status.json", status)
    print(json.dumps(status), flush=True)
    print("Report: " + str(output / "diagnostic.json"), flush=True)
    print("Light files: diagnostic.json, exit_status.json, worker.log, source_manifest.json, source/ (no tensors)", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
