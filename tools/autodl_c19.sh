#!/usr/bin/env bash
# C19 server workflow. Only the explicit "start" mode launches formal training.
set -euo pipefail
MODE="${1:-prepare}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
NAME=c19_rtdetr_r18_lite_cbr_e200_b16_onlineaug
REPORT="$ROOT/outputs/cbr/launch_c19"
INIT="$ROOT/weights/rtdetr_r18_lite_cbr_imagenet_backbone_init.pt"
C2ARGS="$MAIN/runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml"
DATA="$MAIN/configs/crack_autodl.yaml"
SAMPLES="$ROOT/outputs/cbr/fixed_val_samples.json"
RUN="$MAIN/runs/c_series/$NAME"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    CONDA_BASE=/root/miniconda3
else
    echo 'Conda not found. Activate the existing rtdetr environment and put conda on PATH.' >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate rtdetr
cd "$ROOT"
export PYTHONPATH="$ROOT/ultralytics-main"
python -c 'import pathlib,torch,ultralytics; p=pathlib.Path(ultralytics.__file__).resolve(); print("Python/package:",p,"Torch:",torch.__version__); assert p.is_relative_to(pathlib.Path.cwd()/"ultralytics-main"); assert torch.__version__.split("+")[0]=="2.1.2"; assert torch.cuda.is_available()'

case "$MODE" in
    prepare)
        test -z "$(git status --porcelain)"
        test -f "$C2ARGS" && test -f "$DATA"
        mkdir -p "$REPORT"
        # Reuse the existing native AMP self-check weight when available; no dependency installation.
        if [[ ! -e yolo26n.pt && -f "$MAIN/yolo26n.pt" ]]; then ln -s "$MAIN/yolo26n.pt" yolo26n.pt; fi
        if [[ ! -f "$INIT" ]]; then
            python tools/init_rtdetr_r18_lite_cbr_controlled.py \
                --source "$MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" \
                --output "$INIT" --report "$REPORT/initialization.json" \
                2>&1 | tee "$REPORT/initialization.console.log"
        fi
        test -s "$REPORT/initialization.json"
        if [[ ! -f "$SAMPLES" ]]; then
            python tools/cbr_results.py freeze --data "$DATA" --output "$SAMPLES" --limit 16
        fi
        STAMP="$(date +%Y%m%d_%H%M%S_%N)"
        # Always execute a fresh audit. An old audit.json is never treated as sufficient.
        python tools/audit_rtdetr_r18_lite_cbr.py \
            --source "$MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" \
            --initialized "$INIT" --report "$REPORT/audit.json" \
            --require-torch 2.1.2 --require-cuda --c2-args "$C2ARGS" \
            --smoke-data "$DATA" --smoke-dir "$ROOT/outputs/cbr/smoke_$STAMP" \
            2>&1 | tee "$REPORT/audit_$STAMP.console.log"
        python tools/train_rtdetr_r18_lite_cbr.py --c2-args "$C2ARGS" --initialized "$INIT" \
            --audit-report "$REPORT/audit.json" --name "$NAME" --report-dir "$REPORT"
        echo "Preparation finished. No formal training started. Review $REPORT/launch_plan.json"
        ;;
    start)
        command -v tmux >/dev/null
        python tools/train_rtdetr_r18_lite_cbr.py --c2-args "$C2ARGS" --initialized "$INIT" \
            --audit-report "$REPORT/audit.json" --name "$NAME" --report-dir "$REPORT" --tmux
        ;;
    status)
        python tools/train_rtdetr_r18_lite_cbr.py --name "$NAME" --report-dir "$REPORT" --status
        ;;
    val)
        python tools/cbr_results.py evaluate --c2 "$MAIN/runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt" \
            --c19 "$RUN/weights/best.pt" --data "$DATA" --samples "$SAMPLES" \
            --split val --device 0 --batch 16 --output "$ROOT/outputs/cbr/evaluation_val"
        ;;
    pack)
        python tools/cbr_results.py pack --run "$RUN" --launch "$REPORT" \
            --evaluation "$ROOT/outputs/cbr/evaluation_val" \
            --output "$ROOT/outputs/cbr/c19_val_diagnostic.tar.gz"
        ;;
    *) echo 'Usage: bash tools/autodl_c19.sh prepare|start|status|val|pack' >&2; exit 2 ;;
esac
