# BFR-P4 v1 fixed-commit server commands

Execution commit: `__BFR_COMMIT__`. Branch: `exp-rtdetr-r18-lite-bfr-p4-v1`.
Run each complete block independently. The first two blocks do not start formal training.
Do not run the ablation concurrently. Existing experiments and processes remain protected.

## 1. Synchronize and activate the existing environment

```bash
bash <<'BASH'
set -Eeuo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR-bfr-p4-v1
SHA=__BFR_COMMIT__
BRANCH=exp-rtdetr-r18-lite-bfr-p4-v1
cd "$MAIN"
case "$(git remote get-url origin)" in
  https://github.com/supershaojie/concrete-crack-rtdetr|https://github.com/supershaojie/concrete-crack-rtdetr.git) ;;
  *) echo 'Unexpected repository remote; stop.' >&2; exit 1 ;;
esac
if ! git cat-file -e "${SHA}^{commit}" 2>/dev/null; then
  for attempt in 1 2 3 4 5; do
    if timeout 120s env GIT_TERMINAL_PROMPT=0 git -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 fetch --progress --no-tags origin "$BRANCH"; then
      break
    fi
    printf 'Fetch attempt %s failed.\n' "$attempt" >&2
  done
fi
git cat-file -e "${SHA}^{commit}" || { echo 'Object missing: use the verified delivery bundle block below.' >&2; exit 1; }
git merge-base --is-ancestor a0459d6a652cb702699087c88fa39a3e4c4087ec "$SHA"
if [ -e "$WT" ]; then
  test -d "$WT"
  test "$(git -C "$WT" rev-parse --show-toplevel)" = "$WT"
  test "$(git -C "$WT" rev-parse HEAD)" = "$SHA"
else
  git worktree add --detach "$WT" "$SHA"
fi
test "$(git -C "$WT" rev-parse HEAD)" = "$SHA"
test -z "$(git -C "$WT" status --porcelain --untracked-files=no)"
mkdir -p "$WT/outputs/bfr_p4"
cat > "$WT/outputs/bfr_p4/environment.sh" <<'ENV'
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
export BFR_P4_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
export BFR_P4_WORKTREE=/root/autodl-tmp/projects/Crack_RTDETR-bfr-p4-v1
export BFR_P4_SHA=__BFR_COMMIT__
export PYTHONPATH="$BFR_P4_WORKTREE/ultralytics-main"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BFR_P4_WORKTREE"
test "$(git rev-parse HEAD)" = "$BFR_P4_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
ENV
source "$WT/outputs/bfr_p4/environment.sh"
python -c 'import pathlib,torch,ultralytics,os; assert pathlib.Path(ultralytics.__file__).resolve().is_relative_to(pathlib.Path(os.environ["BFR_P4_WORKTREE"])); print(torch.__version__,torch.version.cuda,ultralytics.__file__)'
nvidia-smi
BASH
```

If all five fetches fail, upload `bfr_p4_v1.bundle` from the delivery folder to
`/root/autodl-tmp/bfr_p4_v1.bundle`, run the following, then rerun block 1.
The bundle contains only commits after the fixed parent and requires that parent.

```bash
bash <<'BASH'
set -Eeuo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR
git cat-file -e a0459d6a652cb702699087c88fa39a3e4c4087ec^{commit}
git bundle verify /root/autodl-tmp/bfr_p4_v1.bundle
git fetch /root/autodl-tmp/bfr_p4_v1.bundle refs/heads/exp-rtdetr-r18-lite-bfr-p4-v1
test "$(git rev-parse FETCH_HEAD)" = __BFR_COMMIT__
BASH
```

## 2. Controlled initialization and finite preflight

Check the displayed GPU processes first. Run this only when B16/640 fits without
interfering with existing jobs. Failure or missing resources cannot open the start gate.
The preflight uses a fresh directory; preserve a failed report and choose a new
directory for a corrected rerun, then pass that same path to start/resume.

```bash
bash <<'BASH'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-bfr-p4-v1/outputs/bfr_p4/environment.sh
nvidia-smi
SOURCE="$BFR_P4_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
DATA="$BFR_P4_MAIN/configs/crack_autodl.yaml"
for VARIANT in cbr_lif_bfr_p4_v1 bfr_p4_v1; do
  INIT="weights/${VARIANT}_controlled_init.pt"
  if [ -f "$INIT" ]; then
    python -c 'import sys; sys.path.insert(0,"tools"); from init_bfr_p4 import audit_checkpoint; print(audit_checkpoint(sys.argv[1],sys.argv[2],require_untrained=True)["status"])' "$INIT" "$VARIANT"
  else
    python -u tools/init_bfr_p4.py "$VARIANT" --source "$SOURCE" --output "$INIT" --report "outputs/bfr_p4/$VARIANT/initialization.json"
  fi
done
python -u tools/check_bfr_p4_math.py --device cpu --output outputs/bfr_p4/math_cpu_server.json
python -u tools/check_bfr_p4_math.py --device cuda:0 --output outputs/bfr_p4/math_cuda_server.json
python -u tools/preflight_bfr_p4.py --variant cbr_lif_bfr_p4_v1 --source "$SOURCE" --initialized weights/cbr_lif_bfr_p4_v1_controlled_init.pt --data "$DATA" --output outputs/bfr_p4/cbr_lif_bfr_p4_v1/server_preflight --max-batches 16
python -u tools/train_bfr_p4.py plan --variant cbr_lif_bfr_p4_v1
BASH
```

The ablation configuration and initialization are ready. Its formal run requires
its own separate preflight after the main run; the default commands dispatch only the main combination.

## 3. Explicit formal start (only after PASSED server preflight)

This is the first block that starts the 200-epoch experiment. Keep it separate
from initialization. Use an existing dedicated tmux terminal if desired.

```bash
bash <<'BASH'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-bfr-p4-v1/outputs/bfr_p4/environment.sh
nvidia-smi
python -u tools/train_bfr_p4.py start --variant cbr_lif_bfr_p4_v1 2>&1 | tee "outputs/bfr_p4/cbr_lif_bfr_p4_v1/start_$(date -u +%Y%m%dT%H%M%SZ).txt"
BASH
```

## 4. Explicit resume of this unfinished run

Do not resume a completed/early-stopped experiment or after changing code or recipe.

```bash
bash <<'BASH'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-bfr-p4-v1/outputs/bfr_p4/environment.sh
nvidia-smi
python -u tools/train_bfr_p4.py resume --variant cbr_lif_bfr_p4_v1 --checkpoint "$BFR_P4_MAIN/runs/c_series/cbr_lif_bfr_p4_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/last.pt" 2>&1 | tee "outputs/bfr_p4/cbr_lif_bfr_p4_v1/resume_$(date -u +%Y%m%dT%H%M%SZ).txt"
BASH
```

## 5. Independent val, then optional final test of the identical best.pt

These blocks are for after training. They were **not executed** in this delivery.
The first records the val-selected weight SHA256; the second requires the same weight.

```bash
bash <<'BASH'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-bfr-p4-v1/outputs/bfr_p4/environment.sh
python -u tools/eval_bfr_p4.py val --variant cbr_lif_bfr_p4_v1 --checkpoint "$BFR_P4_MAIN/runs/c_series/cbr_lif_bfr_p4_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt" --data "$BFR_P4_MAIN/configs/crack_autodl.yaml" --output outputs/bfr_p4/cbr_lif_bfr_p4_v1/independent_val
BASH
```

```bash
bash <<'BASH'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-bfr-p4-v1/outputs/bfr_p4/environment.sh
python -u tools/eval_bfr_p4.py test --variant cbr_lif_bfr_p4_v1 --checkpoint "$BFR_P4_MAIN/runs/c_series/cbr_lif_bfr_p4_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt" --data "$BFR_P4_MAIN/configs/crack_autodl.yaml" --val-report outputs/bfr_p4/cbr_lif_bfr_p4_v1/independent_val/metrics.json --output outputs/bfr_p4/cbr_lif_bfr_p4_v1/final_test
BASH
```

## 6. Lightweight package and status (no training/evaluation triggered)

```bash
bash <<'BASH'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-bfr-p4-v1/outputs/bfr_p4/environment.sh
python tools/train_bfr_p4.py status --variant cbr_lif_bfr_p4_v1
python tools/pack_bfr_p4_light.py --variant cbr_lif_bfr_p4_v1 --run "$BFR_P4_MAIN/runs/c_series/cbr_lif_bfr_p4_v1_rtdetr_r18_lite_e200_b16_onlineaug" --output "outputs/bfr_p4/bfr_p4_LIGHT_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
BASH
```
