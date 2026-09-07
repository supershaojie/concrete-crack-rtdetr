"""Final bounded CBR gradient check: C19/C20, same fixed 16 train images, 4x4, no updates."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import sys
import traceback
from types import SimpleNamespace

from cbr_rescue_common import ROOT, C20_SHA, TASK_FILES, sha256, canonical, require, write_json, git, source_state, package
from diagnose_cbr_rescue import RUNS, SETTINGS, Tee, import_runtime, build_dataset, manifest

STEP1 = "d73391f1b8496b16d9b6ba06429c6393b1a21eda"
STEP1_SUMMARY_SHA = "44f1272ab52408fefb8f4ceb663eaa41fc965654f767939533df1220f1472523"
STEP1_PACKAGE_SHA = "8b90e4e28f42ca44581271b9820c8cb5d5a14f427a5054b02f78f6038be05a9f"
DEFAULT_MAIN = Path("/root/autodl-tmp/projects/Crack_RTDETR")
DEFAULT_SUMMARY = Path("/root/autodl-tmp/projects/Crack_RTDETR_diag_c20_cbr_rescue/outputs/cbr_rescue_step1/summary.json")


def code_state():
    # Reuse step-1 protection and add its seven tools/documents, without repeating model audits.
    state = source_state()
    for name in TASK_FILES:
        require(canonical((ROOT / name).read_bytes()) == canonical(git("show", f"{STEP1}:{name}")),
                f"Existing step-1 file changed: {name}")
    state["gradient_base_commit"] = STEP1
    return state


def load_prior(path):
    require(path.is_file(), f"Completed step-1 summary missing: {path}")
    require(sha256(path) == STEP1_SUMMARY_SHA, "Step-1 summary differs from verified 7,986,300-byte result package.")
    result = json.loads(path.read_text(encoding="utf-8"))
    require(result["status"] == "completed" and result["source_before"]["commit"] == STEP1, "Wrong/incomplete step-1 run.")
    require(result["input_hashes_before"] == result["input_hashes_after"], "Step-1 inputs changed.")
    require(result["inputs"]["C20"]["sha256"] == C20_SHA, "Wrong C20 in step-1 evidence.")
    return result


def choose_images(paths):
    ordered = sorted(str(Path(p).resolve()) for p in paths)
    require(len(ordered) >= 16 and len(set(ordered)) == len(ordered), "Need >=16 unique train paths; no resampling/filtering.")
    selected = random.Random(42).sample(range(len(ordered)), 16)
    return ordered, [{"sample_index": i, "batch": i // 4, "slot": i % 4,
                      "train_index": index, "image": ordered[index]} for i, index in enumerate(selected)]


def fixed_data(data, output, settings):
    from unittest.mock import patch
    from ultralytics.models.rtdetr.val import RTDETRDataset
    with patch("ultralytics.data.base.check_file_speeds", lambda *a, **kw: None):
        paths = RTDETRDataset.get_img_files(SimpleNamespace(prefix="gradient: ", fraction=1.0), data["train"])
    ordered, chosen = choose_images(paths)
    full_list = output / "train_sorted_paths.txt"
    full_list.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    selected_list = output / "selected_train_paths.txt"
    selected_list.write_text("\n".join(row["image"] for row in chosen) + "\n", encoding="utf-8")
    # Build only the 16 selected images with the established native read-only loader.
    dataset = build_dataset({**data, "train": str(selected_list.resolve())}, "train", output, settings)
    rows = {row["image"]: row for row in manifest(dataset)}
    require(set(rows) == {r["image"] for r in chosen}, "Selected image set changed during native dataset loading.")
    chosen = [{**r, **rows[r["image"]]} for r in chosen]
    report = {"selection": "Python random.Random(42).sample(range(N),16) after stable absolute-path sorting; order retained",
              "train_images": len(ordered), "sorted_train_list_sha256": sha256(full_list), "rows": chosen,
              "batch_size": 4, "batches": 4, "seed": 42, "native_dataset_warnings": dataset.diagnostic_warnings}
    write_json(output / "fixed_train_manifest.json", report)
    return dataset, report


def verify_fixed(manifest_report):
    for row in manifest_report["rows"]:
        require(sha256(row["image"]) == row["image_sha256"], f"Selected image changed: {row['image']}")
        p = Path(row["label"])
        current = sha256(p) if p.is_file() else None
        require(current == row["label_sha256"], f"Selected label changed: {p}")


def check_model(name, weights, dataset, fixed, data, settings, output):
    import torch
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.models.rtdetr.val import RTDETRValidator
    from cbr_gradient_probe import mixed_graph_mode, state_record, objects, probe_batch
    folder = output / name
    folder.mkdir()
    device = torch.device("cpu" if settings["device"] == "cpu" else "cuda:0")
    model, checkpoint = load_checkpoint(str(weights), device=device, fuse=False)
    del checkpoint
    model.float().eval()
    require(len(model.names) == 1, "Expected one-class crack checkpoint.")
    require(any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()), "Expected unfused model.")
    head, roles, descriptions = objects(model)
    nc_missing = not hasattr(model, "nc")
    if nc_missing:
        model.nc = head.nc  # metadata normally attached by the native trainer
    require(model.nc == head.nc == len(model.names), "Class metadata mismatch.")
    # Native strip_optimizer() preserves the attribute but sets it to None in best.pt.
    if getattr(model, "criterion", None) is None:
        model.criterion = model.init_criterion()
    require((roles["cscef_all"] is not None) == (name == "C20"), f"Incorrect {name} CSCEF topology.")
    loss_config = {"class": type(model.criterion).__name__, "gain": model.criterion.loss_gain,
                   "matcher_cost": model.criterion.matcher.cost_gain, "aux_loss": model.criterion.aux_loss,
                   "use_uni_match": model.criterion.use_uni_match, "uni_match_ind": model.criterion.uni_match_ind,
                   "vfl": type(model.criterion.vfl).__name__, "fl": type(model.criterion.fl).__name__,
                   "num_denoising": head.num_denoising, "label_noise_ratio": head.label_noise_ratio,
                   "box_noise_scale": head.box_noise_scale, "decoder_layers": head.decoder.num_layers,
                   "rho": head.cbr.rho, "normal_fraction": head.cbr.normal_fraction, "samples_per_query": 36}
    require(loss_config["gain"] == {"class": 1, "bbox": 5, "giou": 2, "no_object": .1, "mask": 1, "dice": 1}, "Native loss gains differ from C19 documentation.")
    require(loss_config["aux_loss"] and model.criterion.vfl is not None and head.num_denoising > 0,
            "Native DN/auxiliary/VFL configuration missing.")
    before = state_record(model)
    original_modes = {n: m.training for n, m in model.named_modules()}
    original_flags = {n: p.requires_grad for n, p in model.named_parameters()}
    write_json(folder / "state_before.json", before)
    validator = RTDETRValidator(save_dir=folder / "preprocess", args=settings)
    validator.device, validator.data, validator.training = device, data, False
    index = {str(Path(p).resolve()): i for i, p in enumerate(dataset.im_files)}
    report = {"name": name, "checkpoint": str(weights), "checkpoint_sha256": sha256(weights), "roles": descriptions,
              "nc_metadata_restored_from_actual_head": nc_missing,
              "loss_configuration": loss_config, "batches": [], "status": "failed"}
    try:
        with mixed_graph_mode(model) as mode:
            report["mixed_mode"] = mode
            noise = None
            for batch_i in range(4):
                selected = fixed["rows"][batch_i * 4:(batch_i + 1) * 4]
                batch = dataset.collate_fn([dataset[index[r["image"]]] for r in selected])
                batch = validator.preprocess(batch)
                require(batch["img"].shape == (4, 3, settings["imgsz"], settings["imgsz"]), "Wrong bounded batch shape.")
                result = probe_batch(model, batch, batch_i, noise)
                if batch_i == 0:
                    noise = {role: value["repeat_A_noise_norm"] for role, value in result["gradient_groups"].items()
                             if value["applicable"]}
                require(state_record(model) == before, "Parameter/buffer/.grad changed during probe.")
                result["parameters_buffers_grad_unchanged"] = True
                report["batches"].append(result)
                write_json(folder / f"batch_{batch_i}.json", result)
                print(f"{name}: gradient batch {batch_i + 1}/4, loss={result['loss_A']['total']:.8g}, DN={result['DN']['queries_per_image']}; A/B exact", flush=True)
                del batch, result
        require(original_modes == {n: m.training for n, m in model.named_modules()}, "Training flags not restored.")
        require(original_flags == {n: p.requires_grad for n, p in model.named_parameters()}, "requires_grad flags not restored.")
        report["status"] = "completed"
        return report
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            after = state_record(model)
            write_json(folder / "state_after.json", after)
            require(before == after, "Final parameter/buffer/.grad mismatch.")
            report["parameters_buffers_grad_unchanged"] = True
        finally:
            write_json(folder / "report.json", report)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


def run(args):
    output = args.output.resolve()
    require(not output.exists(), f"Preserve previous gradient output: {output}")
    output.mkdir(parents=True)
    report = {"status": "failed", "started_utc": datetime.now(timezone.utc).isoformat(),
              "conclusion": {"category": "诊断实现/输入问题", "reason": "Run not completed."}}
    files, before, fixed, handler = {}, None, None, None
    try:
        prior = load_prior(args.step1_summary)
        files[str(args.step1_summary.resolve())] = sha256(args.step1_summary)
        before = code_state()
        require(not before["status"].strip(), "Gradient worktree must be clean and committed.")
        report["code_before"] = before
        write_json(output / "source_before.json", before)
        (output / "code_diff.patch").write_bytes(git("diff", "--binary", STEP1, "HEAD"))
        report["prior_evidence"] = {"summary_sha256": STEP1_SUMMARY_SHA, "package_sha256": STEP1_PACKAGE_SHA,
                                    "commit": STEP1, "inputs": prior["inputs"], "no_val_test_rerun": True}
        weights = {name: (getattr(args, name.lower()) or args.main / "runs/c_series" / RUNS[name] / "weights/best.pt").resolve()
                   for name in ("C19", "C20")}
        data_path = (args.data or args.main / "configs/crack_autodl.yaml").resolve()
        for name, path in {**weights, "data": data_path}.items():
            require(path.is_file(), f"Required {name} input missing: {path}; locate it and pass explicit --{name.lower()}.")
            files[str(path)] = sha256(path)
            require(files[str(path)] == prior["inputs"][name]["sha256"], f"{name} hash differs from completed step-1 checkpoint/data.")
        report["runtime"] = import_runtime(output)
        report["runtime"]["mode"] = "Unfused FP32, autograd enabled; root/head/decoder training routes; all other modules eval; TF32 off"
        from unittest.mock import patch
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.utils import LOGGER
        from cbr_gradient_probe import decide, DECISION_RULE
        from cbr_rescue_analysis import write_csv
        handler = logging.FileHandler(output / "framework.log", encoding="utf-8")
        LOGGER.addHandler(handler)
        settings = {**SETTINGS, "imgsz": 640, "batch": 4, "device": args.device, "data": str(data_path), "split": "train"}
        report["settings"], report["preregistered_decision_rule"] = settings, DECISION_RULE
        with patch("ultralytics.data.utils.check_font", lambda *a, **kw: None):
            data = check_det_dataset(str(data_path), autodownload=False)
        require(data["nc"] == 1, "Expected one-class data.")
        report["resolved_data"] = data
        dataset, fixed = fixed_data(data, output, settings)
        report["fixed_manifest_sha256"] = sha256(output / "fixed_train_manifest.json")
        results = {}
        for name in ("C19", "C20"):
            verify_fixed(fixed)
            results[name] = check_model(name, weights[name], dataset, fixed, data, settings, output)
        # Same batch seed and labels yield the same DN geometry/mask, not the same learned class embeddings.
        require(results["C19"]["loss_configuration"] == results["C20"]["loss_configuration"], "C19/C20 native loss/DN settings differ.")
        for a, b in zip(results["C19"]["batches"], results["C20"]["batches"]):
            for key in ("metadata", "initial_noisy_boxes", "attention_mask", "gt_groups"):
                require(a["DN"][key] == b["DN"][key], f"Cross-model DN geometry differs: {key}")
        report["cross_model_DN_geometry_equal"] = True
        report["models"] = results
        rows = []
        for name, model_result in results.items():
            for batch in model_result["batches"]:
                for role, statistics in batch["gradient_groups"].items():
                    if statistics["applicable"]:
                        rows.append({"model": name, "batch": batch["batch"], "role": role, **statistics})
        write_csv(output / "gradient_groups.csv", rows)
        report["conclusion"] = decide({name: r["batches"] for name, r in results.items()})
        write_json(output / "conclusion.json", report["conclusion"])
        (output / "conclusion.md").write_text(json.dumps(report["conclusion"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["status"] = "completed"
    except BaseException as error:
        report["error"], report["traceback"] = repr(error), traceback.format_exc()
        report["conclusion"] = {"category": "诊断实现/输入问题", "reason": repr(error), "candidate_selected": False}
        raise
    finally:
        try:
            after = {p: sha256(p) if Path(p).is_file() else None for p in files}
            report["file_hashes_before"], report["file_hashes_after"] = files, after
            require(files == after, "Checkpoint/data/step-1 evidence changed during gradient check.")
            if fixed is not None:
                verify_fixed(fixed)
                report["selected_images_labels_unchanged"] = True
            if before is not None:
                report["code_after"] = code_state()
                require(report["code_after"] == before, "Source changed during gradient check.")
        except BaseException as error:
            report["status"], report["integrity_error"] = "failed", repr(error)
            report["conclusion"] = {"category": "诊断实现/输入问题", "reason": repr(error), "candidate_selected": False}
            raise
        finally:
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            report["artifacts"] = sorted({"summary.json", "console.log", *(
                p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
                and "runtime_config" not in p.relative_to(output).parts and p.suffix in {".json", ".csv", ".md", ".txt", ".log", ".patch"})})
            write_json(output / "summary.json", report)
            if handler:
                LOGGER.removeHandler(handler)
                handler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    p.add_argument("--step1-summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--output", type=Path, required=True)
    for name in ("c19", "c20", "data"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--device", choices=("0", "cpu"), default="0")
    p = sub.add_parser("pack")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pack":
        package(args.input, args.output)
        return
    require(not args.output.exists(), f"Preserve prior output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_name(args.output.name + ".console.log")
    with log_path.open("x", encoding="utf-8") as log:
        try:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                run(args)
        except BaseException:
            traceback.print_exc(file=log)
            raise
        finally:
            log.flush()
            if args.output.is_dir():
                (args.output / "console.log").write_bytes(log_path.read_bytes())


if __name__ == "__main__":
    main()
