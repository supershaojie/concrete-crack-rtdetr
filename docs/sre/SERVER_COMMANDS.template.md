# SRE fixed-SHA server handoff

Code SHA: `@SRE_SHA@` (full 40 digits). Branch `exp-rtdetr-r18-lite-sre-v1`, successful base `a0459d6a652cb702699087c88fa39a3e4c4087ec`.

This file is generated after the code commit; do not commit the rendered copy merely to update its SHA. It does not report any server execution. Formal training NOT_STARTED; final test NOT_RUN. Every block starts a self-contained Bash and propagates failures. Only execute the first two blocks initially. Do not run GPU preflight concurrently with another experiment; inspect nvidia-smi first. Insufficient B16/640/AMP capacity is a resource issue, not permission to reduce the formal conditions.

## 1. Synchronize and set environment

```bash
bash <<'SRE_SYNC'
set -Eeuo pipefail
SRE_SHA=@SRE_SHA@
SRE_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
SRE_WORKTREE=/root/autodl-tmp/projects/Crack_RTDETR-sre-v1
SRE_BRANCH=exp-rtdetr-r18-lite-sre-v1
case "$(git -C "$SRE_MAIN" remote get-url origin)" in
 https://github.com/supershaojie/concrete-crack-rtdetr|https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr.git|ssh://git@github.com/supershaojie/concrete-crack-rtdetr.git) ;;
 *) echo 'Unexpected repository'; exit 1 ;;
esac
if ! git -C "$SRE_MAIN" cat-file -e "$SRE_SHA^{commit}" 2>/dev/null; then
 for SRE_ATTEMPT in 1 2 3 4 5; do
  if GIT_TERMINAL_PROMPT=0 timeout 120s git -C "$SRE_MAIN" -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 fetch --progress --no-tags --no-write-fetch-head origin "refs/heads/$SRE_BRANCH:refs/remotes/origin/$SRE_BRANCH"; then
   git -C "$SRE_MAIN" cat-file -e "$SRE_SHA^{commit}" 2>/dev/null && break
  fi
 done
fi
if ! git -C "$SRE_MAIN" cat-file -e "$SRE_SHA^{commit}" 2>/dev/null; then
 : "${SRE_BUNDLE:?Bounded fetch failed. Upload the verified incremental bundle and export SRE_BUNDLE=/absolute/path/to/bundle before retrying.}"
 git -C "$SRE_MAIN" bundle verify "$SRE_BUNDLE"
 git -C "$SRE_MAIN" fetch --no-tags --no-write-fetch-head "$SRE_BUNDLE" "refs/heads/$SRE_BRANCH:refs/remotes/sre-bundle/$SRE_BRANCH"
fi
git -C "$SRE_MAIN" cat-file -e "$SRE_SHA^{commit}"
SRE_SCRIPT="$(mktemp /tmp/sync_sre.XXXXXX.sh)"
trap 'rm -f -- "$SRE_SCRIPT"' EXIT
git -C "$SRE_MAIN" show "$SRE_SHA:tools/sync_sre.sh" > "$SRE_SCRIPT"
bash "$SRE_SCRIPT" "$SRE_SHA" "$SRE_MAIN" "$SRE_WORKTREE"
source "$SRE_WORKTREE/outputs/sre/environment-$SRE_SHA.sh"
python -c 'import torch,ultralytics; print(torch.__version__, torch.version.cuda, ultralytics.__file__)'
nvidia-smi
SRE_SYNC
```

## 2. Initialize, plan, and preflight the main combination only

```bash
bash <<'SRE_PREFLIGHT'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-sre-v1/outputs/sre/environment-@SRE_SHA@.sh
nvidia-smi
python tools/init_sre.py --variant cbr_lif_sre_v1 --source "$SRE_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
python tools/train_sre.py plan --variant cbr_lif_sre_v1 --data "$SRE_MAIN/configs/crack_autodl.yaml"
timeout --signal=INT --kill-after=30s 1800s python tools/preflight_sre.py --variant cbr_lif_sre_v1 --data "$SRE_MAIN/configs/crack_autodl.yaml" --device 0 --max-batches 24 --max-seconds 600
SRE_PREFLIGHT
```

The declared budget is at most24 real batches,600 seconds for native setup/capacity, and1800 seconds total including core checks. Timeout/worker startup problems are not a pass; inspect the preserved report and actual exit code. Existing initialization is protected. If initialization already succeeded, rerun only the plan/preflight lines after inspection (preflight archives its old report by content hash); never delete old results to force a pass. The ablation uses variant `sre_v1` in a separate invocation only after explicitly scheduling its resources; the initial dispatch above does not train either model.

## 3. Explicit formal start (only after PASSED server preflight)

Run in a durable terminal or your own tmux session. This block starts only the main combination, with original 200-epoch recipe. Keep actual exit status even if collecting logs with tee.

```bash
bash <<'SRE_START'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-sre-v1/outputs/sre/environment-@SRE_SHA@.sh
nvidia-smi
python -u tools/train_sre.py start --variant cbr_lif_sre_v1 --data "$SRE_MAIN/configs/crack_autodl.yaml"
SRE_START
```

## 4. Resume an actual unfinished run

Resume must retain optimizer/scaler/epoch and learned SRE states. A completed run or stripped best.pt is not a resume source. Completion at 200 epochs and final_eval status are recorded separately; a final validation failure must not restart the 200 epochs.

```bash
bash <<'SRE_RESUME'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-sre-v1/outputs/sre/environment-@SRE_SHA@.sh
python -u tools/train_sre.py resume --variant cbr_lif_sre_v1 --data "$SRE_MAIN/configs/crack_autodl.yaml" --checkpoint "$SRE_MAIN/runs/c_series/cbr_lif_sre_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/last.pt"
SRE_RESUME
```

## 5. Independent val and recall diagnostic, then separately authorized test

Only after training, use best.pt selected by the original fitness rule. The tool records SHA256. Parent validation must load the actual parent architecture; its historical P/R is not R_at_P. Val diagnosis uses fixed TP labels, pooled equal-score groups, IoU matching 0.50 and precision_target 0.8656. Formal CLI IoU remains 0.7. Never optimize a threshold on test.

```bash
bash <<'SRE_VAL'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-sre-v1/outputs/sre/environment-@SRE_SHA@.sh
SRE_BEST="$SRE_MAIN/runs/c_series/cbr_lif_sre_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt"
SRE_PARENT="$SRE_MAIN/runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt"
printf '%s  %s\n' 24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa "$SRE_PARENT" | sha256sum --check -
python tools/eval_sre.py val --weights "$SRE_BEST" --data "$SRE_MAIN/configs/crack_autodl.yaml" --output outputs/sre/cbr_lif_sre_v1/val --architecture cbr_lif_sre_v1 --device 0
python tools/eval_sre.py val --weights "$SRE_PARENT" --data "$SRE_MAIN/configs/crack_autodl.yaml" --output outputs/sre/parent/val --architecture parent_cbr_lif --device 0
python tools/eval_sre.py recall --report outputs/sre/cbr_lif_sre_v1/val/metrics.json --stream outputs/sre/cbr_lif_sre_v1/val/predictions_gt.jsonl.gz --weights "$SRE_BEST" --role candidate --output outputs/sre/cbr_lif_sre_v1/recall_at_precision.json
python tools/eval_sre.py recall --report outputs/sre/parent/val/metrics.json --stream outputs/sre/parent/val/predictions_gt.jsonl.gz --weights "$SRE_PARENT" --role parent --output outputs/sre/parent/recall_at_precision.json
python tools/eval_sre.py compare --parent outputs/sre/parent/recall_at_precision.json --candidate outputs/sre/cbr_lif_sre_v1/recall_at_precision.json --output outputs/sre/cbr_lif_sre_v1/recall_comparison.json
SRE_VAL
```

The parent and candidate have separate thresholds. Archived parent evidence may be diagnosed with `recall --inventory docs/sre/parent_dataset_inventory.json` when the historical report lacks embedded data identity; this reconstructs fixed labels once using the original matcher and audits the historical metrics, without performing new model inference.

Final test is **NOT_RUN** in this delivery. A future separate invocation is provided for after val and frozen best SHA:

```bash
bash <<'SRE_TEST'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-sre-v1/outputs/sre/environment-@SRE_SHA@.sh
SRE_BEST="$SRE_MAIN/runs/c_series/cbr_lif_sre_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt"
python tools/eval_sre.py test --weights "$SRE_BEST" --data "$SRE_MAIN/configs/crack_autodl.yaml" --output outputs/sre/cbr_lif_sre_v1/test --architecture cbr_lif_sre_v1 --device 0 --val-report outputs/sre/cbr_lif_sre_v1/val/metrics.json
SRE_TEST
```

## 6. Lightweight package

This command does not train, evaluate, or download anything. Weights, datasets, module archives and large raw predictions are excluded; hashes record their provenance.

```bash
bash <<'SRE_PACK'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-sre-v1/outputs/sre/environment-@SRE_SHA@.sh
python tools/pack_sre_light.py --metadata outputs/sre/cbr_lif_sre_v1 --run "$SRE_MAIN/runs/c_series/cbr_lif_sre_v1_rtdetr_r18_lite_e200_b16_onlineaug" --output "outputs/sre/SRE_LIGHT_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
SRE_PACK
```
