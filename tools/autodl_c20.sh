#!/usr/bin/env bash
# Only explicit start launches formal training. No package/environment installation.
set -Eeuo pipefail
MODE="${1:-prepare}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
NAME=c20_rtdetr_r18_lite_cscef_cbr_e200_b16_onlineaug
REPORT="$ROOT/outputs/c20/launch_c20"
INIT="$ROOT/weights/rtdetr_r18_lite_cscef_cbr_imagenet_backbone_init.pt"
SOURCE="$MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
C2_ARGS="$MAIN/runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml"
DATA="$MAIN/configs/crack_autodl.yaml"
RUN="$MAIN/runs/c_series/$NAME"
STAMP="$(date +%Y%m%d_%H%M%S_%N)"
mkdir -p "$ROOT/outputs/c20"
exec > >(tee -a "$ROOT/outputs/c20/${MODE}_${STAMP}.bootstrap.log") 2>&1
trap 'rc=$?; printf "C20 %s FAILED: exit=%s line=%s command=%s\n" "$MODE" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR
if [[ "$MODE" == prepare ]]; then printf 'running\n' > "$ROOT/outputs/c20/prepare.state"; fi

case "$MODE" in prepare|start|status|val|test|pack) ;; *) echo 'Usage: bash tools/autodl_c20.sh prepare|start|status|val|test|pack'; exit 2 ;; esac
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    CONDA_BASE=/root/miniconda3
else
    echo 'Conda not found: activate the existing rtdetr environment and put conda on PATH.' >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate rtdetr
cd "$ROOT"
export PYTHONPATH="$ROOT/ultralytics-main"
if [[ "$MODE" != status && "$MODE" != pack ]]; then
    python - <<'PY'
import pathlib, sys, torch, ultralytics
p = pathlib.Path(ultralytics.__file__).resolve()
print('Python:', sys.executable, 'Package:', p, 'Torch:', torch.__version__, 'CUDA:', torch.version.cuda, flush=True)
assert p.is_relative_to(pathlib.Path.cwd() / 'ultralytics-main'), 'Wrong Ultralytics import: expected C20 worktree'
assert torch.__version__.split('+')[0] == '2.1.2', 'Expected existing server PyTorch 2.1.2; do not change environment'
assert torch.cuda.is_available(), 'CUDA unavailable'
print('GPU:', torch.cuda.get_device_name(), flush=True)
PY
fi

case "$MODE" in
    prepare)
        if [[ -n "$(git status --porcelain)" ]]; then echo 'C20 worktree must be clean and committed.' >&2; exit 1; fi
        if [[ -e "$RUN" || -e "$RUN.c20.launch.lock" || -e "$REPORT/tmux.json" || -e "$REPORT/exit_code.json" ]]; then
            echo 'Prior C20 launch/results exist. Preserve them; use status. prepare cannot overwrite launch evidence.' >&2; exit 1
        fi
        for path in "$C2_ARGS" "$DATA" "$SOURCE"; do
            if [[ ! -f "$path" ]]; then echo "Required file missing: $path" >&2; exit 1; fi
        done
        printf '%s  %s\n' fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e "$SOURCE" | sha256sum -c -
        mkdir -p "$REPORT"
        if [[ ! -e yolo26n.pt && -f "$MAIN/yolo26n.pt" ]]; then ln -s "$MAIN/yolo26n.pt" yolo26n.pt; fi
        if [[ ! -f "$INIT" ]]; then
            python tools/init_rtdetr_r18_lite_c20.py --source "$SOURCE" --output "$INIT" \
                --report "$REPORT/initialization.json" 2>&1 | tee "$REPORT/initialization.console.log"
        fi
        if [[ ! -s "$REPORT/initialization.json" ]]; then echo 'Missing initialization.json for existing initialization; investigate without overwriting it.' >&2; exit 1; fi
        AUDIT="$REPORT/audit_$STAMP.json"
        python tools/audit_rtdetr_r18_lite_c20.py --source "$SOURCE" --initialized "$INIT" \
            --report "$AUDIT" --require-torch 2.1.2 --require-cuda --c2-args "$C2_ARGS" \
            --smoke-data "$DATA" --smoke-dir "$ROOT/outputs/c20/smoke_$STAMP" \
            2>&1 | tee "$REPORT/audit_$STAMP.console.log"
        cp "$AUDIT" "$REPORT/audit.json"
        python tools/train_rtdetr_r18_lite_c20.py --c2-args "$C2_ARGS" --initialized "$INIT" \
            --audit-report "$REPORT/audit.json" --name "$NAME" --report-dir "$REPORT"
        printf 'passed\n' > "$ROOT/outputs/c20/prepare.state"
        echo "Preparation passed. Formal training has NOT started. Plan: $REPORT/launch_plan.json"
        ;;
    start)
        command -v tmux >/dev/null
        python tools/train_rtdetr_r18_lite_c20.py --c2-args "$C2_ARGS" --initialized "$INIT" \
            --audit-report "$REPORT/audit.json" --name "$NAME" --report-dir "$REPORT" --tmux
        ;;
    status)
        python tools/train_rtdetr_r18_lite_c20.py --name "$NAME" --report-dir "$REPORT" --status
        ;;
    val|test)
        # Evaluation requires a successfully completed training process, never a live best.pt.
        python - "$REPORT" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
for name in ('exit_code.json', 'process_exit_code.json'):
    result = json.loads((p / name).read_text())
    assert result['exit_code'] == 0, f'Training did not finish successfully: {name}: {result}'
PY
        EXTRA=()
        if [[ "$MODE" == test ]]; then EXTRA=(--val-report "$ROOT/outputs/c20/evaluation_val/metrics_summary.json"); fi
        python tools/c20_results.py evaluate --weights "$RUN/weights/best.pt" --data "$DATA" \
            --split "$MODE" --device 0 --batch 16 --output "$ROOT/outputs/c20/evaluation_$MODE" "${EXTRA[@]}"
        ;;
    pack)
        python tools/c20_results.py pack --run "$RUN" --launch "$REPORT" \
            --val "$ROOT/outputs/c20/evaluation_val" --test "$ROOT/outputs/c20/evaluation_test" \
            --output "$ROOT/outputs/c20/c20_cscef_cbr_small.tar.gz"
        ;;
esac
