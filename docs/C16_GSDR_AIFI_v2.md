# C16 / GSDR-AIFI-v2

C16 is an implementation revision of C14: rename the registered position MLP to restore ordinary optimizer grouping, isolate only the new sparse branch's construction RNG, and disable CUDA autocast locally around sparse QK/softmax/AV. One C16 run cannot separately quantify these three changes. No accuracy gain is claimed before training.

The branch `exp-rtdetr-r18-lite-gsdr-aifi-v2` starts at C14 `d8954374f7564dc6180424e9058b44403968308e`. C2 reference is `67c3078e54a657fd96d65fee657a75fbb1dae0d6`. Local worktree: `D:/MyProjects/Crack_RTDETR/outputs/worktrees/gsdr-aifi-v2`. C14, the main local checkout and the separate C15 worktree retain their commits and every tracked file.

## Model and initialization

- Original AIFI 256/1024/8, sparse width 128, 4 heads, 4 offset groups, stride 2, offset range 2.0, kernel 3, sampling/padding/align_corners conventions, fusion position and normalization order are retained.
- Added parameters: **117,644**. Unfused whole model: **20,200,416** at nc=1; **20,301,852** at nc=80.
- 564 total states: 533 shared C2 states and 31 sparse states. The initializer checks every shared tensor, including 12 original AIFI states and 69 backbone parameter keys.
- Source checkpoint SHA256 is locked to `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`. C14 best/last and other trained checkpoints are rejected.
- Saved parameters are FP32. Original C2 values are exactly promoted; new states are never half-quantized. Epoch is -1, and EMA/optimizer/scaler/updates/training results are empty. Raw/API/YAML reloads must match every tensor exactly.
- Old GSDR source, YAML, initializer, audit and tests remain unchanged. Only the two public registration files receive V2 additions. `gsdr_aifi_v2_protected.json` records canonical-LF hashes of all 874 other files inherited from C14, including the public trainer/data/validator/default configuration.

## Tools

| File | Behavior |
| --- | --- |
| `tools/init_rtdetr_r18_lite_gsdr_aifi_v2_controlled.py` | `--source`, `--output`, `--report`; creates a new clean FP32 checkpoint and refuses overwrites. |
| `tools/audit_rtdetr_r18_lite_gsdr_aifi_v2.py` | `--source`, `--initialized`, `--report`; complete synthetic CPU/CUDA audit, original 17 tests, supplied 5 tests and 8 launcher tests. No dataset evaluation. |
| `tools/train_rtdetr_r18_lite_gsdr_aifi_v2.py` | Default: prepare only. `--execute`: foreground formal training. `--tmux`: immediately dispatch formal training in an independent detached session. |
| `tools/gsdr_aifi_v2_tmux_worker.py` | Internal standard-library process supervisor; captures bootstrap failures and actual exit code. |

The launcher reads the **original server C2 args.yaml**, preserves all 109 fields, and checks every field. With the default project, only `model`, `name` and `save_dir` change. `--project` is an optional output-only relocation. Sanity checks never supply missing fields. No fallback to the checked-in test fixture or library defaults is permitted.

Preparation verifies checkpoint/audit/source/data-config hashes and the local import path. The audit must pass CPU and CUDA on the same Python/PyTorch/CUDA/GPU/worktree environment. Formal execution requires committed source and rechecks the plan, protected files and hashes. Callbacks compare all actual trainer arguments before data scanning and again before the first batch; all 564 actual nc=1 states must match the audited hashes. The callback does not reseed, reconstruct, copy or randomize any head. Native AMP must remain enabled.

An atomic claim beside the experiment directory prevents duplicate runs across different report directories. Existing experiment directories, console logs and completed/failed launches are never reused. Keep the claim and logs after failure for inspection; do not automatically delete them or resume from last/best. A retry requires resolving the failure and explicitly choosing fresh output identifiers/report paths.

## Actual local verification and limits

Verified using Python 3.9.25, PyTorch 2.7.1+cu118 and RTX 2060:

- Real `RTDETRTrainer.get_model()` at seed42, nc=1 and nc=80: all 533 shared states, all 9 classification states (5 weights), CPU RNG and every tensor in the complete 640x640 CPU FP32 output tree are exactly equal to C2.
- Real AdamW grouping: ordinary weight132 / norm82 / bias143. All 8 position MLP weights (384 elements) use ordinary weight decay 0.0001 and warmup start 0. True biases retain bias warmup 0.1. C2 groups are unchanged.
- First backward reaches the zero output projection. After four actual synthetic AdamW steps, Q/K/V, offset-convolution/normalization/output and every position MLP upstream weight have finite nonzero gradients; no manually randomized boundaries are used for that check.
- CPU FP32 and CUDA FP32/AMP FP16/explicit half module forward/backward pass. Actual sparse QK and AV matmul outputs are FP32. Complete CUDA half forward at 640 passes for nc=1 and nc=80.
- Original GSDR 17 tests + supplied V2 5 tests + launcher 8 tests pass (30 total, no CUDA skips). Launcher tests cover all-field preservation/rejection, actual-args drift, duplicate claims, bootstrap exit logging, actual-model mutation rejection and the real `RTDETR.train -> get_model` path, stopped in a dataset-free fixture before any training.
- The local original C2 args copy has SHA256 `ab0594ac3758b53421dc0a5adbea693505fc9e2591d8e668a25ba950d70234fd`; its 109-field comparison changes only the three identifiers. This offline comparison does not validate the server's data paths or replace the original server file.

Local artifacts under `outputs/gsdr_aifi_v2/`: `initialization.json`, `audit.json`, per-suite logs, `c2_field_comparison.json`, `locked_train_args_reference.yaml`, `external_protection.json`. These generated reports/weights are not Git source files. Development attempts are retained separately in `development/`.

**Not executed locally:** formal training, dataset val/test, Linux tmux dispatch, or PyTorch 2.1.2+cu121 / RTX 4090 checks. Run the server audit below; a local CUDA pass does not certify the server version. CUDA grid_sample backward emits PyTorch's nondeterminism warning under the unchanged `deterministic=True` warn-only policy; no reproducibility setting was weakened.

The new tracked `ultralytics/assets/bus.jpg` is copied from the existing trusted local Ultralytics asset, SHA256 `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`. It supports the native training AMP self-check without changing any C2 arguments. YOLO26n is only that native check's resource, never the C16 initialization.

## Server: first deployment, audit and preparation

Run in Bash. These commands create a new worktree without switching the MAIN/C14/C15 checkouts. An existing V2 directory or local branch causes creation to fail safely; inspect it instead of deleting/resetting it.

```bash
set -euo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
V2_WT=/root/autodl-tmp/projects/Crack_RTDETR_gsdr_aifi_v2
BRANCH=exp-rtdetr-r18-lite-gsdr-aifi-v2
git -C "$MAIN" fetch origin "$BRANCH:refs/remotes/origin/$BRANCH"
git -C "$MAIN" worktree add -b "$BRANCH" "$V2_WT" "origin/$BRANCH"
cd "$V2_WT"
test "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$BRANCH")"
git merge-base --is-ancestor d8954374f7564dc6180424e9058b44403968308e HEAD
git log -1 --format='%H %s'
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rtdetr
export PYTHONPATH="$V2_WT/ultralytics-main"
python -c 'import sys, torch, ultralytics; print(sys.executable); print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name()); print(ultralytics.__file__)'

SOURCE="$MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
C2_ARGS="$MAIN/runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml"
INIT="$V2_WT/weights/rtdetr_r18_lite_gsdr_aifi_v2_imagenet_backbone_init.pt"
AUDIT="$V2_WT/outputs/gsdr_aifi_v2/audit.json"
REPORT="$V2_WT/outputs/gsdr_aifi_v2/launch_c16"
test -f "$C2_ARGS"
test -f "$MAIN/configs/crack_autodl.yaml"
command -v tmux

# Reuse a native AMP-check cache if available; otherwise the unchanged trainer can fetch its official resource.
if [ -f "$MAIN/yolo26n.pt" ] && [ ! -e "$V2_WT/yolo26n.pt" ]; then
  cp -n "$MAIN/yolo26n.pt" "$V2_WT/yolo26n.pt"
fi
python tools/init_rtdetr_r18_lite_gsdr_aifi_v2_controlled.py \
  --source "$SOURCE" --output "$INIT" \
  --report "$V2_WT/outputs/gsdr_aifi_v2/initialization.json"
python tools/audit_rtdetr_r18_lite_gsdr_aifi_v2.py \
  --source "$SOURCE" --initialized "$INIT" --report "$AUDIT"
python tools/train_rtdetr_r18_lite_gsdr_aifi_v2.py \
  --c2-args "$C2_ARGS" --initialized "$INIT" --audit-report "$AUDIT" --report-dir "$REPORT"
python -m json.tool "$REPORT/launch_plan.json"
cat "$REPORT/train_args.yaml"
```

Up to this point, no formal training or dataset val/test has run. Preparation must report 109 fields and only `model`, `name`, `save_dir` changed. Any failure stops execution; resolve its cause without modifying the C2 recipe or using another checkpoint.

## Server: start C16 and inspect logs

Use the same Bash session and variables after the server audit/preparation pass. C15 may continue in its existing session; the commands below never stop it. Check available GPU memory before starting another 200-epoch job on the same device; do not change batch/device/AMP to work around contention.

```bash
nvidia-smi
python tools/train_rtdetr_r18_lite_gsdr_aifi_v2.py \
  --c2-args "$C2_ARGS" --initialized "$INIT" --audit-report "$AUDIT" \
  --report-dir "$REPORT" --tmux
SESSION=gsdr_aifi_v2_c16_rtdetr_r18_lite_gsdr_aifi_v2_e200_b16_onlineaug
tmux list-sessions
tail -F "$REPORT/console.log" "$REPORT/bootstrap.log"
```

`tail -F` follows files and may have no green tmux status bar. Ctrl-C exits tail only. Training output is deliberately redirected to logs, so attaching to the tmux session may show a quiet pane. From another shell, inspect:

```bash
V2_WT=/root/autodl-tmp/projects/Crack_RTDETR_gsdr_aifi_v2
REPORT="$V2_WT/outputs/gsdr_aifi_v2/launch_c16"
cat "$REPORT/tmux.json"
cat "$REPORT/preflight.json"          # appears after actual model/args checks, before the first batch
cat "$REPORT/actual_train_args.yaml"
cat "$REPORT/exit_code.json"          # appears on success/failure; zero means successful exit
cat "$REPORT/process_exit_code.json"  # supervisor's actual process result, including bootstrap failures
```

Training outputs are under `$MAIN/runs/c_series/c16_rtdetr_r18_lite_gsdr_aifi_v2_e200_b16_onlineaug`. Use `--execute` instead of `--tmux` for foreground execution, never both and never in addition to an already dispatched run.

## File changes

New: V2 module/YAML; initializer/audit/launcher/tmux supervisor; two V2 test files; authentic C2 args test fixture; this document; protection manifest; native AMP bus asset. Modified: only `ultralytics/nn/modules/__init__.py` and `ultralytics/nn/tasks.py` for V2 registration. No baseline merge, CSCEF combination model, parameter search, extra seeds, holdout, epoch extension or threshold sweep is included.
