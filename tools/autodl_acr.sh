#!/usr/bin/env bash
# Usage: bash tools/autodl_acr.sh prepare|start|status|val|test|pack [c22|c23]
# prepare exits after checks and isolated smoke. Only explicit start trains.
set -Eeuo pipefail
MODE="${1:-prepare}"
export ACR_VARIANT="${2:-c22}"
case "$MODE" in prepare|start|status|val|test|pack) ;; *) echo 'Unknown command'; exit 2 ;; esac
case "$ACR_VARIANT" in c22|c23) ;; *) echo 'Variant must be c22 or c23'; exit 2 ;; esac
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
NAME=c22_rtdetr_r18_lite_acr_e200_b16_onlineaug
if [[ "$ACR_VARIANT" == c23 ]]; then NAME=c23_rtdetr_r18_lite_cscef_v51_acr_e200_b16_onlineaug; fi
REPORT="$ROOT/outputs/$ACR_VARIANT/launch"
INIT="$ROOT/weights/${ACR_VARIANT}_acr_controlled_init.pt"
SOURCE="$MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
C2_ARGS="$MAIN/runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml"
DATA="$MAIN/configs/crack_autodl.yaml"
RUN="$MAIN/runs/c_series/$NAME"
STAMP="$(date +%Y%m%d_%H%M%S_%N)"
mkdir -p "$ROOT/outputs/$ACR_VARIANT"
exec > >(tee -a "$ROOT/outputs/$ACR_VARIANT/${MODE}_${STAMP}.bootstrap.log") 2>&1
trap 'rc=$?; if [[ "$MODE" == prepare && -d "$REPORT" ]]; then printf "failed\n" > "$REPORT/prepare.state"; fi; printf "ACR %s failed: exit=%s line=%s command=%s\n" "$MODE" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    CONDA_BASE=/root/miniconda3
else
    echo 'Activate the existing rtdetr conda environment; conda not found.' >&2; exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate rtdetr
cd "$ROOT"
export PYTHONPATH="$ROOT/ultralytics-main"
export YOLO_AUTOINSTALL=false
if [[ "$MODE" != status && "$MODE" != pack ]]; then
    python - <<'PY'
import pathlib, sys, torch, ultralytics
print('Python:',sys.executable,'Torch:',torch.__version__,'Ultralytics:',ultralytics.__file__,flush=True)
assert pathlib.Path(ultralytics.__file__).resolve().is_relative_to(pathlib.Path.cwd()/'ultralytics-main')
assert torch.__version__.split('+')[0]=='2.1.2','Expected existing AutoDL PyTorch 2.1.2; no automatic environment changes'
assert torch.cuda.is_available(),'CUDA required on AutoDL'
print('GPU:',torch.cuda.get_device_name(),flush=True)
PY
fi
case "$MODE" in
    prepare)
        [[ -z "$(git status --porcelain)" ]] || { echo 'Commit all source changes before prepare'; exit 1; }
        if [[ -e "$RUN" || -e "$RUN.acr.launch.lock" || -e "$REPORT/tmux.json" || -e "$REPORT/exit_code.json" ]]; then
            echo 'Existing launch or results; preserve them and use status.' >&2; exit 1
        fi
        mkdir -p "$REPORT"
        # A per-variant lock prevents concurrent prepare processes from changing one plan.
        exec 9>"$ROOT/outputs/$ACR_VARIANT/prepare.lock"
        flock -n 9 || { echo 'Another prepare is running'; exit 1; }
        printf 'running\n' > "$REPORT/prepare.state"
        for path in "$C2_ARGS" "$DATA" "$SOURCE"; do [[ -f "$path" ]] || { echo "Missing $path"; exit 1; }; done
        printf '%s  %s\n' fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e "$SOURCE" | sha256sum -c -
        python tools/acr_resources.py --main "$MAIN" --report "$REPORT/resources.json"
        if [[ ! -f "$INIT" ]]; then
            python tools/init_acr.py --source "$SOURCE" --output "$INIT" --report "$REPORT/initialization.json" \
                2>&1 | tee "$REPORT/initialization.console.log"
        fi
        [[ -s "$REPORT/initialization.json" ]] || { echo 'Initialization report missing; preserve existing weights and investigate'; exit 1; }
        AUDIT="$REPORT/audit_$STAMP.json"
        # Server smoke exercises the requested batch16. Local audits can explicitly use batch2 on small GPUs.
        export ACR_SMOKE_BATCH=16
        python tools/audit_acr.py --source "$SOURCE" --initialized "$INIT" --require-torch 2.1.2 --require-cuda \
            --c2-args "$C2_ARGS" --smoke-data "$DATA" --smoke-dir "$ROOT/outputs/$ACR_VARIANT/smoke_$STAMP" \
            --report "$AUDIT" 2>&1 | tee "$REPORT/audit_$STAMP.console.log"
        cp "$AUDIT" "$REPORT/audit.json"
        python tools/train_acr.py --c2-args "$C2_ARGS" --initialized "$INIT" --audit-report "$REPORT/audit.json" \
            --name "$NAME" --report-dir "$REPORT"
        printf 'passed\n' > "$REPORT/prepare.state"
        echo "Prepared $ACR_VARIANT. Formal training has NOT started. Plan: $REPORT/launch_plan.json"
        ;;
    start)
        command -v tmux >/dev/null
        python tools/train_acr.py --c2-args "$C2_ARGS" --initialized "$INIT" --audit-report "$REPORT/audit.json" \
            --name "$NAME" --report-dir "$REPORT" --tmux
        ;;
    status)
        python tools/train_acr.py --name "$NAME" --report-dir "$REPORT" --status
        ;;
    val|test)
        python - "$REPORT" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])
for name in ('exit_code.json','process_exit_code.json'):
    result=json.loads((p/name).read_text());assert result['exit_code']==0,f'Incomplete/failed training: {result}'
PY
        EXTRA=()
        if [[ "$MODE" == test ]]; then EXTRA=(--val-report "$ROOT/outputs/$ACR_VARIANT/evaluation_val/metrics_summary.json"); fi
        python tools/acr_results.py evaluate --weights "$RUN/weights/best.pt" --data "$DATA" --split "$MODE" \
            --device 0 --batch 16 --policy corrected --label "$ACR_VARIANT" \
            --output "$ROOT/outputs/$ACR_VARIANT/evaluation_$MODE" "${EXTRA[@]}"
        ;;
    pack)
        python tools/acr_results.py pack --run "$RUN" --launch "$REPORT" \
            --val "$ROOT/outputs/$ACR_VARIANT/evaluation_val" --test "$ROOT/outputs/$ACR_VARIANT/evaluation_test" \
            --output "$MAIN/downloads/acr/${ACR_VARIANT}_${STAMP}.tar.gz"
        ;;
esac
