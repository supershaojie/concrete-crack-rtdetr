"""Finite operational fixtures: no tmux launch, server access or formal training."""
from __future__ import annotations

import argparse
import ast
import tempfile
from unittest.mock import patch

from tcr_v1_core import *
import tcr_v1_ops as lifecycle
from tcr_v1_eval import all_queries, mother_postprocess, geometry_init, geometry_update


def rejected(fn):
    try: fn()
    except (RuntimeError,FileExistsError): return True
    raise AssertionError("Unsafe action was accepted")


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); folder=args.output.parent.resolve()
    folder.mkdir(parents=True,exist_ok=True)
    report=dict(status="FAILED",scope="isolated synthetic fixtures")
    for p in (ROOT/"tools").glob("*tcr*.py"): ast.parse(p.read_text(encoding="utf-8"),filename=str(p))
    cfg,rows=recipe(); require(len(rows)==109 and sum(r["changed"] for r in rows)==3,"Recipe drift")
    report["recipe"]=dict(fields=109,changed=[r["field"] for r in rows if r["changed"]])
    require(lifecycle.status_from(None,None,[])["phase"]=="NOT_STARTED","No-start fixture")
    state=dict(dispatch="new",phase="RUNNING",epoch=7)
    stale=dict(dispatch="old",python_exit_code=1,tee_exit_code=0)
    good=dict(dispatch="new",python_exit_code=0,tee_exit_code=0)
    bad=dict(dispatch="new",python_exit_code=1,tee_exit_code=0)
    require(lifecycle.status_from(state,stale,[dict(pid=100)])["phase"]=="RUNNING","Stale exit changed active dispatch")
    require(lifecycle.status_from(dict(state,phase="COMPLETED"),good,[])["phase"]=="COMPLETED","Completed fixture")
    require(lifecycle.status_from(state,bad,[])["phase"]=="FAILED","Failure fixture")
    require(lifecycle.status_from(state,None,[])["phase"]=="FAILED","Lost-process fixture")
    require(lifecycle.status_from(dict(state,phase="DISPATCHED"),None,[],True)["phase"]=="DISPATCHED","Pending dispatch fixture")
    processes=[dict(pid=1,ppid=0,argv=["python",str(ROOT/"tools/tcr_v1.py"),"_worker","--dispatch","new"]),
               dict(pid=2,ppid=1,argv=["worker"]),dict(pid=3,ppid=0,argv=["python","/other/tools/tcr_v1.py","_worker"])]
    require({p["pid"] for p in lifecycle.owned_processes(processes)}=={1,2},"Other experiment mistaken for this experiment")
    report["status_fixtures"]=["NOT_STARTED","DISPATCHED","RUNNING","COMPLETED","FAILED","stale_exit","owned_descendants"]
    with tempfile.TemporaryDirectory(dir=folder,prefix="ops_") as temporary:
        temp=Path(temporary); out=temp/"outputs"; run=temp/"run"
        with patch.object(lifecycle,"OUT",out),patch.object(lifecycle,"RUN",run),patch.object(lifecycle,"process_rows",lambda:[]),patch.object(lifecycle.shutil,"which",lambda unused:None):
            write_json(out/"state.json",state)
            lifecycle.update_state("new",phase="SETTING_UP")
            require(read_json(out/"state.json")["dispatch"]=="new","Duplicate dispatch update failed")
            rejected(lambda:lifecycle.update_state("old",phase="FAILED"))
            lifecycle.record_exit("old",1,0)
            require(read_json(out/"state.json")["phase"]=="SETTING_UP","Old exit replaced new phase")
            run.mkdir(); (run/"args.yaml").write_text("batch: 16\n")
            (run/"weights").mkdir(); (run/"weights/last.pt").write_bytes(b"fixture")
            rejected(lifecycle.archive_failed_start)
            (run/"weights/last.pt").unlink()  # only this newly created fixture file
            (run/"results.csv").write_text("epoch,loss\n1,1.0\n")
            rejected(lifecycle.archive_failed_start)
            (run/"results.csv").write_text("epoch,loss\n")
            result=lifecycle.archive_failed_start()
            require(not run.exists() and Path(result["archived"]).is_dir(),"Failed setup not renamed safely")
        evidence=temp/"mechanism.jsonl"; evidence.write_text('{"samples":2}\n')
        archive=temp/"LIGHT.tar.gz"
        lifecycle.write_archive(archive,{"mechanism.jsonl":evidence},{"README.txt":b"fixture only"})
        require(read_json(str(archive)+".verification.json")["status"]=="PASS","Archive integrity")
        report["archive"]=dict(jsonl_retained=True,manifest_verified=True,safe_failed_start=True)
    raw=torch.tensor([[[.1,.2,.3,.4,.1],[.5,.6,.2,.2,.9],[.8,.8,.1,.1,.2]]])
    selected,affected=mother_postprocess(raw,640,.5)
    all_rows=all_queries(raw,640)
    require(affected==1 and all_rows[0]["query_id"].tolist()==[1,2,0],"Query sorting fixture")
    require(torch.equal(selected[0]["bboxes"],all_rows[0]["bboxes"][:1]),"Sorted mask detached box/score identity")
    geo=geometry_init()
    gt=dict(bboxes=torch.tensor([[20.,20.,40.,40.]]),imgsz=(640,640))
    geometry_update(geo,gt,dict(bboxes=gt["bboxes"],conf=torch.tensor([.9])))
    require(sum(v["matched_iou75"] for v in geo["buckets"]["short_side"])==1,"Geometry matching fixture")
    report["evaluation"]=dict(corrected_sorted_mask=True,query_ids=True,geometry=True)
    write_json(folder/"nonfinite_fixture.json",dict(loss=float("nan"),finite=False,overflow=True))
    require(read_json(folder/"nonfinite_fixture.json")==dict(loss=None,finite=False,overflow=True),"Nonfinite value hidden")
    report.update(status="PASS",nonfinite_null_with_flag=True,source=source_identity())
    write_json(args.output,report); print(json.dumps(dict(status="PASS",report=str(args.output))))


if __name__=="__main__": main()
