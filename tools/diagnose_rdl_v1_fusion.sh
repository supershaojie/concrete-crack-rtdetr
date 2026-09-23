#!/usr/bin/env bash
# One original lifecycle fixture + <=10 traced eval forwards; never train/start.
set -euo pipefail
rdl_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
test -f /root/miniconda3/etc/profile.d/conda.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
test "$(command -v python)" = /root/miniconda3/envs/rtdetr/bin/python
cd -- "$rdl_root"
export PYTHONPATH="$rdl_root/ultralytics-main"
export PYTHONUNBUFFERED=1
python -c 'import platform,torch; assert platform.python_version()=="3.10.13" and torch.__version__=="2.1.2+cu121" and torch.cuda.is_available(), "Use the original server environment; no automatic upgrade"'
rdl_dir="outputs/rdl_v1/fusion_diag_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p outputs/rdl_v1
set +e
timeout --signal=TERM --kill-after=30s 600s python tools/check_rdl_v1.py \
  --source /root/autodl-tmp/projects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt \
  --device cuda:0 --fusion-diagnostic "$rdl_dir" 2>&1 | tee "${rdl_dir}.log"
rdl_codes=("${PIPESTATUS[@]}")
set -e
python - "$rdl_dir" <<'PY'
import json, pathlib, sys
folder=pathlib.Path(sys.argv[1]); path=folder/'fusion.json'
print('EVIDENCE',path,'LOG',str(folder)+'.log','RESULT',str(folder)+'_result.json')
if not path.is_file():
    print('INCOMPLETE: inspect log/result; failure or timeout before fusion snapshot')
    sys.exit(0)
r=json.loads(path.read_text())
keys=('runtime','controlled_source_sha256','original_status','original_assertion','finding','diagnostic_status',
      'diagnostic_error','validation_invariants','unfused_snapshot_state_exact','fusion_invariants','reproduce_original','inside_score_head')
summary={k:r.get(k) for k in keys}
summary['precision']={k:r[k]['precision'] for k in ('before_predict','before_fuse','after_fuse','after_predict')}
summary['selection_kind']=r.get('selection',{}).get('kind')
summary['selection_images']=[{k:i[k] for k in ('changed_positions','a_only','b_only','score_disturbance_e','cutoff_a','cutoff_b')}
    for i in r.get('selection',{}).get('images',[])]
summary['continuous_failures']={k:[v for v in r.get(k,{}).values() if v.get('status')]
    for k in ('pre_selection','candidate_id_alignment','fixed_ids_A','fixed_ids_B','common_head_inputs','common_candidate_inputs','common_head_outputs')}
summary['mother_exact']=bool(r.get('mother_vs_rdl')) and all(v.get('allclose_failed_count')==0 and not v.get('status')
    for g in r.get('mother_vs_rdl',{}).values() for v in g.values())
print(json.dumps(summary,indent=2))
PY
if (( rdl_codes[0] != 0 )); then exit "${rdl_codes[0]}"; fi
exit "${rdl_codes[1]}"
