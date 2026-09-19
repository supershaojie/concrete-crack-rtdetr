#!/usr/bin/env bash
# Explicit dispatch only; init-preflight never falls through to start.
set -Eeuo pipefail
mode="${1:---help}"
if [[ "$mode" == --help ]]; then
  printf '%s\n' 'Usage: PBI_VARIANT=cbr_lif_pbi_v1|pbi_v1 bash tools/pbi_server.sh {environment|init-preflight|plan|start|resume|val|test|pack|status}'
  exit 0
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PBI_VARIANT="${PBI_VARIANT:-cbr_lif_pbi_v1}"
case "$PBI_VARIANT" in cbr_lif_pbi_v1|pbi_v1) ;; *) printf "Unknown variant\n" >&2; exit 2 ;; esac
PBI_META="$ROOT/outputs/pbi/$PBI_VARIANT"
mkdir -p "$PBI_META/command_logs"
stamp="$(date -u +%Y%m%dT%H%M%SZ)_$$"
log="$PBI_META/command_logs/${mode}_${stamp}.log"
# tee is a process substitution: it cannot hide the command's exit status.
exec > >(tee -a "$log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "${log}.exit_code"; exit "$rc"' EXIT
contract_version=pbi_acceptance_v1
if [[ "$mode" == init-preflight && -e "$PBI_META/passed_gates.sh" ]]; then
  # Revoke an older permit before any new check can fail. Keep its contents as
  # historical evidence; never let a failed rerun look like current admission.
  superseded="$PBI_META/passed_gates_${stamp}.superseded.sh"
  [[ ! -e "$superseded" ]] || { printf 'Permit history path already exists.\n' >&2; exit 2; }
  mv -- "$PBI_META/passed_gates.sh" "$superseded"
  printf 'Previous permit preserved and revoked: %s\n' "$superseded"
fi
source "$ROOT/docs/pbi/environment.sh"
python - <<'PY'
from pathlib import Path
import os, ultralytics, torch
expected=Path(os.environ['PBI_WORKTREE'])/'ultralytics-main'
assert Path(ultralytics.__file__).resolve().is_relative_to(expected.resolve())
print('Import:',ultralytics.__file__,'PyTorch:',torch.__version__,'CUDA:',torch.version.cuda)
PY
train=(python -u tools/train_pbi.py)
common=(--variant "$PBI_VARIANT" --main "$PBI_MAIN" --data "$PBI_DATA" --init "$PBI_INIT")
case "$mode" in
  environment)
    python - <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, 'tools')
from train_pbi import git_head, server_environment
from pbi_common import runtime
observed=runtime()
print(json.dumps(dict(git_head=git_head(), variant=os.environ['PBI_VARIANT'],
                     worktree=os.environ['PBI_WORKTREE'], source_import=observed['ultralytics'],
                     environment=server_environment(), formal_training='NOT_STARTED', final_test='NOT_RUN'), indent=2))
PY
    ;;
  init-preflight)
    # Prepare both fresh initializations, with one audited common PBI state.
    # Only the explicitly selected variant receives capacity checks.
    for init_variant in cbr_lif_pbi_v1 pbi_v1; do
      init_path="$PBI_WORKTREE/weights/${init_variant}_controlled_init.pt"
      init_meta="$PBI_WORKTREE/outputs/pbi/$init_variant"
      mkdir -p "$init_meta"
      existing=()
      [[ ! -e "$init_path" ]] || existing=(--verify-existing)
      python -u tools/init_pbi.py "$init_variant" --source "$PBI_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" \
        --output "$init_path" --report "$init_meta/initialization_${stamp}.json" "${existing[@]}"
    done
    initialization="$PBI_META/initialization_${stamp}.json"
    math="$PBI_META/module_${stamp}.json"
    python -u tools/check_pbi_math.py --output "$math"
    checks="$PBI_META/checks_${stamp}"
    python -u tools/check_pbi.py --variant "$PBI_VARIANT" \
      --source "$PBI_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" --initialized "$PBI_INIT" \
      --real-dataset "$PBI_MAIN/datasets/crack_det" --output "$checks" \
      --initialization-report "$initialization" --math-report "$math"
    capacity="$PBI_META/capacity_${stamp}"
    python -u tools/preflight_pbi.py "${common[@]}" --output "$capacity" --max-batches 16 --target-updates 2
    gate_report="$PBI_META/gate_${stamp}.json"
    python - "$checks/checks.json" "$capacity/preflight.json" "$gate_report" <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ['PBI_WORKTREE'])/'tools'))
from train_pbi import strict_gate
from pbi_common import write_json
admission = strict_gate(Path(sys.argv[2]), Path(sys.argv[1]), os.environ['PBI_VARIANT'],
                        Path(os.environ['PBI_INIT']), Path(os.environ['PBI_DATA']))
assert admission['contract_version'] == 'pbi_acceptance_v1', 'Unexpected acceptance contract'
write_json(Path(sys.argv[3]), admission)
PY
    # Atomically publish only after init, math, full checks and capacity have all
    # passed the current strict gate. This is never a formal training dispatch.
    permit_tmp="$(mktemp "$PBI_META/.passed_gates_${stamp}.XXXXXX")"
    printf 'export PBI_CONTRACT_VERSION=%q\nexport PBI_INITIALIZATION_REPORT=%q\nexport PBI_MATH_REPORT=%q\nexport PBI_CHECKS=%q\nexport PBI_PREFLIGHT=%q\nexport PBI_GATE_REPORT=%q\n' \
      "$contract_version" "$initialization" "$math" "$checks/checks.json" "$capacity/preflight.json" "$gate_report" > "$permit_tmp"
    mv -- "$permit_tmp" "$PBI_META/passed_gates.sh"
    printf '%s\n' 'Preflight complete. Formal training NOT_STARTED; final test NOT_RUN.'
    ;;
  plan|status)
    "${train[@]}" "$mode" "${common[@]}"
    ;;
  start|resume)
    # Ambient variables cannot make an old/incomplete permit appear current.
    unset PBI_CONTRACT_VERSION PBI_INITIALIZATION_REPORT PBI_MATH_REPORT PBI_CHECKS PBI_PREFLIGHT PBI_GATE_REPORT
    source "$PBI_META/passed_gates.sh"
    [[ "${PBI_CONTRACT_VERSION:-}" == "$contract_version" ]] || {
      printf 'Full current-contract preflight permit is required.\n' >&2; exit 2;
    }
    extra=()
    if [[ "$mode" == resume ]]; then
      extra=(--checkpoint "$PBI_MAIN/runs/c_series/${PBI_VARIANT}_rtdetr_r18_lite_e200_b16_onlineaug/weights/last.pt")
    fi
    "${train[@]}" "$mode" "${common[@]}" --checks "$PBI_CHECKS" --preflight "$PBI_PREFLIGHT" "${extra[@]}"
    ;;
  val|test)
    evaluation_extra=()
    if [[ "$mode" == test ]]; then
      evaluation_extra=(--val-report "$PBI_META/evaluation_val_latest.json")
    fi
    python -u tools/eval_pbi.py "$mode" --weights "$PBI_MAIN/runs/c_series/${PBI_VARIANT}_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt" \
      --variant "$PBI_VARIANT" --data "$PBI_DATA" --output "$PBI_META/evaluation_$mode" "${evaluation_extra[@]}"
    ;;
  pack)
    python -u tools/pack_pbi_light.py --variant "$PBI_VARIANT" --input "$PBI_META" \
      --input "$PBI_MAIN/runs/c_series/${PBI_VARIANT}_rtdetr_r18_lite_e200_b16_onlineaug" \
      --output "$PBI_WORKTREE/outputs/pbi/${PBI_VARIANT}_${stamp}_LIGHT.tar.gz"
    ;;
  *) printf 'Unknown command: %s\n' "$mode" >&2; exit 2 ;;
esac
