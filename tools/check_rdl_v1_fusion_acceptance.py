"""Targeted acceptance tests from the actual server evidence; no lifecycle run."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import torch

from init_c19_lif_v1 import require, sha256
from c19_lif_v1_cutoff import selection_from_scores
from c19_lif_v1_diagnostic import atomic_json, align_to_ids, compare_records
from rdl_v1_fusion_acceptance import review_fusion, POST
import rdl_v1_fusion as fusion


def run(path, output, records_path=None):
    raw = path.read_bytes()
    base = json.loads(raw)
    rows = []
    def test(name, mutate, accepted=False):
        report = deepcopy(base)
        mutate(report)
        actual = review_fusion(report)
        require(actual["accepted"] is accepted, name + ": " + str(actual))
        rows.append(dict(name=name, result=actual))
    test("actual_server_permutation", lambda r: None, True)
    test("finding_string_is_not_authority", lambda r: r.update(finding="UNTRUSTED_LABEL"), True)
    def set_drift(r):
        s=r["selection"]
        row=s["images"][1]
        other=next(i for i in range(len(s["scores_b"][1])) if i not in row["natural_b"])
        s["scores_b"][1][other]=s["scores_b"][1][row["natural_b"][-1]]
        row["natural_b"][-1]=other
        r["selection"]=selection_from_scores([x["natural_a"] for x in s["images"]], [x["natural_b"] for x in s["images"]],
            s["scores_a"],s["scores_b"],s["shapes"],s["score_dtype_a"],s["score_dtype_b"])
        require(r["selection"]["kind"] == "SET_DRIFT", "Set-drift test did not change sets")
    test("candidate_set_drift", set_drift)
    test("duplicate_ids", lambda r: r["selection"]["images"][1]["natural_b"].__setitem__(0,r["selection"]["images"][1]["natural_b"][1]))
    test("wrong_query_count", lambda r: r["selection"]["images"][0]["natural_a"].pop())
    test("gather_not_verified", lambda r: r["traces"]["natural_b"].update(actual_gather_verified=False))
    test("wrong_replay_ids", lambda r: r["traces"]["fixed_A_b"]["candidate_indices"][1].reverse())
    test("aligned_true_error", lambda r: r["candidate_id_alignment"]["boxes"].update(
        allclose_failed_count=1,status="FAILED_REAL_NUMERICAL_MISMATCH",max_abs_error=.1,max_rel_error=1.))
    for value, name in ((float("nan"),"nan"),(float("inf"),"inf")):
        test(name+"_aligned_output", lambda r,v=value: r["candidate_id_alignment"]["boxes"].update(max_abs_error=v))
        test(name+"_logits", lambda r,v=value: r["selection"]["scores_b"][0].__setitem__(0,v))
    test("nonfinite_tensor_mask", lambda r: r["candidate_id_alignment"]["boxes"].update(masked_nonfinite_count=1))
    for name in ("pre_selection","candidate_id_alignment","fixed_ids_A","fixed_ids_B","common_head_inputs","common_head_outputs","common_candidate_inputs","reproduce_original"):
        test("empty_"+name, lambda r,k=name: r.update({k:{}}))
        test("missing_tensor_"+name, lambda r,k=name: r[k].pop(next(iter(r[k]))))
    for name in ("mother_vs_rdl","mother_source","precision_scope","validation_invariants","fusion_invariants","traces"):
        test("missing_"+name, lambda r,k=name: r.pop(k))
    test("missing_tensor_finite_proof", lambda r: r["fixed_ids_A"]["boxes"].pop("masked_nonfinite_count"))
    test("missing_error_statistic", lambda r: r["common_head_outputs"]["scores"].pop("max_rel_error"))
    test("replay_not_exact", lambda r: r["reproduce_original"]["before"].update(max_abs_error=1e-8))
    test("mother_not_exact", lambda r: r["mother_vs_rdl"]["fused"]["boxes"].update(max_abs_error=1e-8))
    test("lif_changed", lambda r: r["fusion_invariants"].update(lif_state_exact=False))
    test("input_changed", lambda r: r.update(input_unchanged_after_fuse=False))
    test("restore_failed", lambda r: r["precision_scope"].update(restored=False))
    test("restore_values_differ", lambda r: r["precision_scope"]["after"].update(cudnn_tf32=False))
    test("widened_tolerance", lambda r: r["candidate_id_alignment"]["boxes"].update(atol=.1))
    # Integration: the original exception is retained, but no longer wins over a
    # verified permutation after scope restoration. No model forward is mocked as evidence.
    before = fusion.precision_settings()
    def finish(model, batch, report, folder, source_sha, reference):
        scope=report["precision_scope"]
        report.update(deepcopy(base)); report["precision_scope"]=scope
        report.pop("evidence_directory",None)
        return report, AssertionError("original ordered failure")
    with patch.object(fusion,"_check_ema_fusion",side_effect=finish):
        actual=fusion.check_ema_fusion(torch.nn.Linear(1,1),{})
        require(actual["original_status"]=="FAIL" and actual["fusion_acceptance"]["accepted"],"Caller discarded permutation acceptance")
    require(fusion.precision_settings()==before,"Acceptance leaked precision settings")
    tensor_tests = "NOT_RUN (records not supplied)"
    if records_path:
        from ultralytics.utils.patches import torch_load
        records=torch_load(records_path,map_location="cpu")
        a=records["natural_a"]
        # Synthesize a known bijection of actual native tensors, never coordinates.
        ids=a["candidate_indices"].clone(); ids[:,[115,116]]=ids[:,[116,115]]
        b=align_to_ids(dict(a,candidate_indices=ids),a)
        aligned=align_to_ids(a,b)
        compare_records(a,aligned,atol=3e-5,rtol=3e-5,keys=POST)
        for name,value in (("true_error",.1),("nonfinite",float("nan"))):
            bad=deepcopy(b); bad["boxes"][0,0,0,0]+=value
            try: compare_records(a,align_to_ids(a,bad),atol=3e-5,rtol=3e-5,keys=POST)
            except RuntimeError: pass
            else: raise AssertionError("Actual aligned tensor corruption accepted: "+name)
        tensor_tests="PASS: actual tensor ID permutation; real error and NaN rejected"
    require(path.read_bytes()==raw,"Modified original server evidence")
    result=dict(status="PASS",server_report_sha256=sha256(path),cases=len(rows),checks=rows,
                caller_original_FAIL_final_PASS=True,tensor_tests=tensor_tests,original_evidence_unchanged=True)
    atomic_json(output,result)
    print(f"acceptance PASS: {len(rows)} evidence cases; {tensor_tests}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--records",type=Path)
    args=parser.parse_args();require(not args.output.exists(),"Use a new output file")
    run(args.report,args.output,args.records)
