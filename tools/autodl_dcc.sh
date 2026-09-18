#!/usr/bin/env bash
# Explicit dispatch only; init-preflight never falls through to start.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/docs/dcc/environment.sh"
mode="${1:---help}"
if [[ "$mode" == --help ]]; then
  printf '%s\n' 'Usage: bash tools/autodl_dcc.sh {init-preflight|resume-verify|plan|start|resume|val|test|pack|status}'
  exit 0
fi
mkdir -p "$DCC_META/command_logs"
stamp="$(date -u +%Y%m%dT%H%M%SZ)_$$"
log="$DCC_META/command_logs/${mode}_${stamp}.log"
# tee is a process substitution: it cannot hide the command's exit status.
exec > >(tee -a "$log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "${log}.exit_code"; exit "$rc"' EXIT
contract_version=dcc_acceptance_v2
if [[ "$mode" == init-preflight && -e "$DCC_META/passed_gates.sh" ]]; then
  # Revoke an older permit before any new check can fail. Keep its contents as
  # historical evidence; never let a failed rerun look like current admission.
  superseded="$DCC_META/passed_gates_${stamp}.superseded.sh"
  [[ ! -e "$superseded" ]] || { printf 'Permit history path already exists.\n' >&2; exit 2; }
  mv -- "$DCC_META/passed_gates.sh" "$superseded"
  printf 'Previous permit preserved and revoked: %s\n' "$superseded"
fi
python - <<'PY'
from pathlib import Path
import os, ultralytics, torch
expected=Path(os.environ['DCC_WORKTREE'])/'ultralytics-main'
assert Path(ultralytics.__file__).resolve().is_relative_to(expected.resolve())
print('Import:',ultralytics.__file__,'PyTorch:',torch.__version__,'CUDA:',torch.version.cuda)
PY
train=(python -u tools/train_dcc.py)
common=(--variant "$DCC_VARIANT" --main "$DCC_MAIN" --data "$DCC_DATA" --init "$DCC_INIT")
case "$mode" in
  resume-verify)
    # Affected engineering checks only. Never promotes old reports or writes
    # passed_gates.sh, and never falls through to a formal start/resume.
    python -u tools/check_dcc_checkpoint.py --output "$DCC_META/checkpoint_policy_${stamp}.json"
    checks="$DCC_META/resume_checks_${stamp}"
    python -u tools/check_dcc_resume.py --variant "$DCC_VARIANT" \
      --source "$DCC_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" --initialized "$DCC_INIT" \
      --real-dataset "$DCC_MAIN/datasets/crack_det" --output "$checks"
    capacity="$DCC_META/capacity_resume_fix_${stamp}"
    python -u tools/preflight_dcc.py "${common[@]}" --output "$capacity" --max-batches 16 --target-updates 2
    python - "$checks/resume_checks.json" "$capacity/preflight.json" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'tools')
from dcc_common import sha256, write_json
from train_dcc import code_identity
checks_path, capacity_path = map(Path, sys.argv[1:])
checks, capacity = [json.loads(p.read_text()) for p in (checks_path, capacity_path)]
assert checks.get('contract_version') == capacity.get('contract_version') == 'dcc_acceptance_v2', 'Old acceptance contract is not reusable'
assert checks.get('report_kind') == 'partial_resume_diagnostic', 'Expected newly generated partial diagnostic'
assert capacity.get('report_kind') == 'native_capacity', 'Expected newly generated native capacity report'
assert checks['code_identity'] == capacity['code_identity'] == code_identity(), 'Validation source changed'
complete = checks['status'] in ('PASSED', 'PRECISION_NOTE') and capacity['status'] == 'PASSED'
summary = dict(status=checks['status'] if complete else 'FAILED',
               contract_version='dcc_acceptance_v2', report_kind='partial_verification_summary',
               checks=str(checks_path), checks_sha256=sha256(checks_path),
               capacity=str(capacity_path), capacity_sha256=sha256(capacity_path),
               saving_and_restoring_state={k: v.get('dcc', {}).get('checkpoint_resume', {}).get('checkpoint_correctness', {}).get('status', 'PENDING')
                                          for k, v in checks['devices'].items()},
               trajectory_repeatability={k: v.get('dcc', {}).get('checkpoint_resume', {}).get('trajectory_repeatability', {}).get('status', 'PENDING')
                                          for k, v in checks['devices'].items()},
               raw_next_update_allclose={k: v.get('dcc', {}).get('checkpoint_resume', {}).get('raw_next_update_allclose')
                                         for k, v in checks['devices'].items()},
               formal_start='BLOCKED: targeted diagnostics do not replace the complete strict gate',
               formal_training='NOT_STARTED', final_test='NOT_RUN')
write_json(checks_path.parent / 'verification_summary.json', summary)
print(json.dumps(summary, indent=2))
sys.exit(0 if complete else 3)
PY
    ;;
  init-preflight)
    # Existing controlled weights are verified, never silently overwritten.
    existing=()
    [[ ! -e "$DCC_INIT" ]] || existing=(--verify-existing)
    initialization="$DCC_META/initialization_${stamp}.json"
    math="$DCC_META/module_${stamp}.json"
    python -u tools/init_dcc.py "$DCC_VARIANT" --source "$DCC_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" \
      --output "$DCC_INIT" --report "$initialization" "${existing[@]}"
    python -u tools/check_dcc_math.py --output "$math"
    checks="$DCC_META/checks_${stamp}"
    python -u tools/check_dcc.py --variant "$DCC_VARIANT" \
      --source "$DCC_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" --initialized "$DCC_INIT" \
      --real-dataset "$DCC_MAIN/datasets/crack_det" --output "$checks" \
      --initialization-report "$initialization" --math-report "$math"
    capacity="$DCC_META/capacity_${stamp}"
    python -u tools/preflight_dcc.py "${common[@]}" --output "$capacity" --max-batches 16 --target-updates 2
    gate_report="$DCC_META/gate_${stamp}.json"
    python - "$checks/checks.json" "$capacity/preflight.json" "$gate_report" <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ['DCC_WORKTREE'])/'tools'))
from train_dcc import strict_gate
from dcc_common import write_json
admission = strict_gate(Path(sys.argv[2]), Path(sys.argv[1]), os.environ['DCC_VARIANT'],
                        Path(os.environ['DCC_INIT']), Path(os.environ['DCC_DATA']))
assert admission['contract_version'] == 'dcc_acceptance_v2', 'Unexpected acceptance contract'
write_json(Path(sys.argv[3]), admission)
PY
    # Atomically publish only after init, math, full checks and capacity have all
    # passed the current strict gate. This is never a formal training dispatch.
    permit_tmp="$(mktemp "$DCC_META/.passed_gates_${stamp}.XXXXXX")"
    printf 'export DCC_CONTRACT_VERSION=%q\nexport DCC_INITIALIZATION_REPORT=%q\nexport DCC_MATH_REPORT=%q\nexport DCC_CHECKS=%q\nexport DCC_PREFLIGHT=%q\nexport DCC_GATE_REPORT=%q\n' \
      "$contract_version" "$initialization" "$math" "$checks/checks.json" "$capacity/preflight.json" "$gate_report" > "$permit_tmp"
    mv -- "$permit_tmp" "$DCC_META/passed_gates.sh"
    printf '%s\n' 'Preflight complete. Formal training NOT_STARTED; final test NOT_RUN.'
    ;;
  plan|status)
    "${train[@]}" "$mode" "${common[@]}"
    ;;
  start|resume)
    # Ambient variables cannot make an old/incomplete permit appear current.
    unset DCC_CONTRACT_VERSION DCC_INITIALIZATION_REPORT DCC_MATH_REPORT DCC_CHECKS DCC_PREFLIGHT DCC_GATE_REPORT
    source "$DCC_META/passed_gates.sh"
    [[ "${DCC_CONTRACT_VERSION:-}" == "$contract_version" ]] || {
      printf 'Full current-contract preflight permit is required.\n' >&2; exit 2;
    }
    extra=()
    if [[ "$mode" == resume ]]; then
      extra=(--checkpoint "$DCC_MAIN/runs/c_series/${DCC_VARIANT}_rtdetr_r18_lite_e200_b16_onlineaug/weights/last.pt")
    fi
    "${train[@]}" "$mode" "${common[@]}" --checks "$DCC_CHECKS" --preflight "$DCC_PREFLIGHT" "${extra[@]}"
    ;;
  val|test)
    evaluation_extra=()
    if [[ "$mode" == test ]]; then
      evaluation_extra=(--val-report "$DCC_META/evaluation_val/metrics.json")
    fi
    python -u tools/eval_dcc.py "$mode" --weights "$DCC_MAIN/runs/c_series/${DCC_VARIANT}_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt" \
      --variant "$DCC_VARIANT" --data "$DCC_DATA" --output "$DCC_META/evaluation_$mode" "${evaluation_extra[@]}"
    ;;
  pack)
    python -u tools/pack_dcc_light.py --variant "$DCC_VARIANT" --input "$DCC_META" \
      --input "$DCC_MAIN/runs/c_series/${DCC_VARIANT}_rtdetr_r18_lite_e200_b16_onlineaug" \
      --output "$DCC_WORKTREE/outputs/dcc/${DCC_VARIANT}_${stamp}_LIGHT.tar.gz"
    ;;
  *) printf 'Unknown command: %s\n' "$mode" >&2; exit 2 ;;
esac
