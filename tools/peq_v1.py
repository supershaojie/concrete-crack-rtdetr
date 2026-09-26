#!/usr/bin/env python
"""PEQ v1 real command entry. Long training/test run only through explicit actions."""
from __future__ import annotations
import argparse
import traceback
from peq_v1_common import *


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("record-delivery","prepare","preflight","preflight-worker","probe","start","worker",
                                         "status","resume","finish","val","test","pack","archive-failed","checks"))
    parser.add_argument("--sha")
    parser.add_argument("--dispatch")
    parser.add_argument("--report",type=Path)
    parser.add_argument("--resume-worker",action="store_true")
    parser.add_argument("--python-exit",type=int)
    parser.add_argument("--tee-exit",type=int)
    parser.add_argument("--weights",type=Path,default=MAIN/"runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt")
    parser.add_argument("--data",type=Path,default=MAIN/"configs/crack_autodl.yaml")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--batch",type=int,default=2,help="Probe batch only; formal settings stay B16")
    parser.add_argument("--device",default="0")
    parser.add_argument("--retry",action="store_true")
    parser.add_argument("--include-predictions",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    from peq_v1_runtime import (record_delivery,prepare,dispatch,worker,status,finish,archive_failed)
    if args.action=="record-delivery":
        result=record_delivery(args.sha)
    elif args.action=="prepare":
        result=prepare()
    elif args.action=="preflight":
        from peq_v1_preflight import preflight
        result=preflight(args.retry)
    elif args.action=="preflight-worker":
        from peq_v1_preflight import preflight_worker
        require(args.report is not None,"--report is required")
        preflight_worker(args.report)
        result=read_json(args.report)
    elif args.action=="probe":
        from peq_v1_eval import probe
        result=probe(args.weights,args.data,args.device,args.batch,args.limit,args.retry)
    elif args.action in ("start","resume"):
        result=dispatch(args.action=="resume")
    elif args.action=="worker":
        require(bool(args.dispatch),"Worker needs current dispatch")
        worker(args.dispatch,args.resume_worker)
        result=status()
    elif args.action=="status":
        result=status()
    elif args.action=="finish":
        require(args.dispatch and args.python_exit is not None and args.tee_exit is not None,"Exit identity/codes are required")
        result=finish(args.dispatch,args.python_exit,args.tee_exit)
    elif args.action in ("val","test"):
        from peq_v1_eval import evaluate
        result=evaluate(args.action,args.data,args.retry)
    elif args.action=="pack":
        from peq_v1_pack import pack
        result=pack(args.include_predictions)
    elif args.action=="archive-failed":
        result=archive_failed()
    else:
        from check_peq_v1 import run_checks
        result=run_checks("cpu" if args.device=="cpu" else "cuda:"+args.device,True,args.report)
    if args.action=="status":
        print(json_bytes(result).decode(),flush=True)
    else:
        print(json_bytes({k:v for k,v in result.items() if k in
                         ("status","phase","reason","dispatch","commit","images","candidate_coverage","inversion_rate","opportunity",
                          "light","error_analysis","path","timeout","effective_updates")}).decode(),flush=True)
        print("Evidence directory:",OUT,flush=True)
    return 2 if result.get("status") in ("PENDING","FAILED") else 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as error:
        append_json(OUT/"command_errors.jsonl",dict(at=utc(),command=sys.argv[1:],error=repr(error),traceback=traceback.format_exc()))
        raise
