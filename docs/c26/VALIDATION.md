# C26 local validation

Completed locally on Windows, Python3.9.25 / PyTorch2.7.1+cu118 / RTX2060. These are not AutoDL Python3.10 / PyTorch2.1.2+cu121 / RTX4090 results. No dependency was installed/upgraded.

## Actual checks

- Exact C19 `cbr.py` and C24 `scca_aifi.py` source equality, parsed YAML equality under only the requested substitutions, registration and strict candidate exclusion.
- Actual unfused nc1 parameter counts: C2 20,082,772; C19 20,128,661; C24 20,148,312; C26 20,194,201.
- Unified file SHA verified; all 533 public constructor states and mapped values match. 19 added states retain original initialization and match independent original-module models. All 552 states survive initialization round-trip.
- Native `RTDETRTrainer.get_model` and public `RTDETR.train(trainer=...)` reconstruction: 543 exact states, only nine nc80→nc1 classifier shape skips. The public API probe stops before training setup; it does not run a formal epoch or use the real dataset.
- Zero-residual C2/C26 inference at 640×640 and 160×192 has maximum absolute difference **0.0**. Direct observation verifies Neck P3 shape 80×80 at 640. Production modules have no query hook/cache.
- FP32 CPU and CUDA AMP: three native-loss/DN updates at imgsz160, batches 2,2,1. Loss and all existing gradients finite; added parameters are present once in native AdamW. Both output projections update. Zero first-step upstream gradients are accepted as required by zero initialization. After updates, gradients propagate and both modules contribute to actual output.
- Updated full-object checkpoint round-trip retains both modules/state/output exactly (max absolute difference 0.0). Removing learned SCCA/CBR output projections changes actual output; disabling CBR preserves scores. All debug models/weights and synthetic data are removed.
- Decoder's optional query matches the last emitted training/evaluation layer, including eval_idx 0/1/2. Legacy two-tensor return and existing positional arguments remain intact. Original C19 tests also compare full public gradients with C2 and exercise DN/auxiliary/nonzero refinement.
- Strict 109-field recipe/type check, C26-only dispatch/duplicate reservation, shell exit bookkeeping and no automatic OOM recipe change. Dispatch tests are mocked; they do not start tmux/formal training.
- Same-pass evaluation export exercised on **three synthetic images**, including one background, using the actual validator and FP32 640/batch16 settings (the only batch is partial). Each fake split exports all 900 predictions and two GT, plots and a 1×10 AP array. Repeated output and changed-checkpoint test attempts are rejected. Fixture metrics are plumbing evidence only; they are not crack-dataset results.
- Complete packaging fixture exceeds 20 MiB, includes weights and prediction/train images, verifies every manifest member and total SHA, protects old archives, and rejects failed training. It never calls evaluate. This is synthetic packaging evidence, not a completed C26 result package.
- Bash syntax validation for both entrypoint/synchronization scripts. The AutoDL filesystem, tmux worker dispatch and fixed-SHA server synchronization are not executed locally.

Focused suites: 8 C26 tests (3 integration-contract, 4 lifecycle/package, 1 synthetic export), plus 19 unmodified C19/SCCA/related ACR decoder regression tests. Supporting original C19 test source is copied from its fixed commit; existing C24/C25 tests/assertions are unchanged.

## Evidence and reproduction

`evidence/local_validation.json` contains actual native updates and loading records; `mapping.json` lists every public key and added key; `export_fixture.json` explicitly labels its synthetic scope. Logs, the 109-field recipe and source hashes accompany them. Evidence was generated on the working tree before the final commit, so its runtime parent commit is C24; `source_trace.json` records the delivered source hashes. No historical or server check is represented as passing based on local evidence.

```bash
export PYTHONPATH="$PWD/ultralytics-main:$PWD/tools"
export YOLO_AUTOINSTALL=false
python -m unittest discover -s ultralytics-main/tests -p 'test_c26*.py' -v
python -m unittest discover -s ultralytics-main/tests -p test_cbr.py -v
python -m unittest discover -s ultralytics-main/tests -p 'test_scca*.py' -v
python -m unittest discover -s ultralytics-main/tests -p test_acr.py -v
python tools/check_c26.py --source /PATH/TO/rtdetr_r18_lite_imagenet_backbone_init.pt --output outputs/c26_optional_local_check
```

The last command is optional and never a start-direct requirement. Use a new output directory. On machines without CUDA it records AMP as NOT_RUN. CUDA grid_sample backward is checked for finite behavior, not bitwise determinism.

## Not run

`full_server_preflight=NOT_RUN`; AutoDL FP32/AMP model comparison=NOT_RUN; real batch16 server smoke=NOT_RUN; optional CBR before/after localization diagnostics=NOT_RUN; formal C26 training=NOT_RUN; complete real-data val/test=NOT_RUN; formal result package=NOT_RUN; historical C2/C19/C24 reevaluations=NOT_RUN.
