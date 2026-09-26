"""TCR v1 server entry. No action implicitly starts training, probe, val or test."""
from __future__ import annotations

import argparse
from tcr_v1_core import *


def main():
    os.chdir(ROOT)  # Absolute entry works independently of the caller's directory.
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","preflight","probe","start","status","resume","val","test","pack","archive-failed-start","commands","_worker","_exit","_preflight"))
    parser.add_argument("--weights",type=Path)
    parser.add_argument("--dataset",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--device",default="0")
    parser.add_argument("--limit",type=int,default=64)
    parser.add_argument("--include-predictions",action="store_true")
    parser.add_argument("--dispatch")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--python-code",type=int)
    parser.add_argument("--tee-code",type=int)
    parser.add_argument("--folder",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    from tcr_v1_ops import prepare,preflight,launch,status,archive_failed_start,package,record_exit
    if args.action=="prepare": result=prepare()
    elif args.action=="preflight": result=preflight()
    elif args.action=="start": result=launch()
    elif args.action=="resume": result=launch(resume=True)
    elif args.action=="status": result=status()
    elif args.action=="archive-failed-start": result=archive_failed_start()
    elif args.action=="pack": result=package(args.include_predictions)
    elif args.action in {"val","test"}:
        from tcr_v1_eval import evaluate
        result=evaluate(args.action)
    elif args.action=="probe":
        from tcr_v1_probe import probe
        weights=args.weights or MAIN/"runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt"
        result=probe(weights,args.dataset or MAIN/"datasets/crack_det",args.output or OUT/"probe",args.limit,args.device)
    elif args.action=="_worker":
        require(args.dispatch,"Missing dispatch")
        from tcr_v1_train import worker
        result=worker(args.dispatch,args.resume)
    elif args.action=="_exit":
        require(args.dispatch and args.python_code is not None and args.tee_code is not None,"Missing exit identity")
        result=record_exit(args.dispatch,args.python_code,args.tee_code)
    elif args.action=="_preflight":
        require(args.folder,"Missing isolated preflight folder")
        from tcr_v1_preflight import run
        result=run(args.folder)
    elif args.action=="commands":
        result=commands()
    if result is not None:
        print(json.dumps(clean_json(result),ensure_ascii=False,indent=2,allow_nan=False))


def commands():
    sha=git("rev-parse","HEAD")
    server="/root/autodl-tmp/projects/Crack_RTDETR-tcr_v1"
    python="/root/miniconda3/envs/rtdetr/bin/python"
    prefix=f"env PYTHONPATH={server}/ultralytics-main PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false {python} -u {server}/tools/tcr_v1.py"
    text=f"# TCR v1 server commands\n\nPinned full SHA: `{sha}`. Each block is independent.\n\n"
    text+=f"## Sync\n\n```bash\ncurl -fL --retry 3 https://raw.githubusercontent.com/supershaojie/concrete-crack-rtdetr/{sha}/tools/sync_tcr_v1.sh -o /tmp/sync_tcr_v1_{sha}.sh && bash /tmp/sync_tcr_v1_{sha}.sh {sha}\n```\n\n"
    for action in ("prepare","preflight","probe","start","status","resume","val","test","pack"):
        text+=f"## {action}\n\n```bash\n{prefix} {action}\n```\n\n"
    text+="## tmux\n\n```bash\ntmux attach -t tcr-v1-training\n```\n\nDetach: Ctrl+B, then D.\n\n"
    text+=f"## Existing train/val prediction archive\n\n```bash\n{prefix} pack --include-predictions\n```\n\n"
    text+=f"## Preserve a failed setup before explicitly starting again\n\n```bash\n{prefix} archive-failed-start\n```\n"
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"server_commands.md").write_text(text,encoding="utf-8")
    return dict(commit=sha,commands=str(OUT/"server_commands.md"))


if __name__=="__main__": main()
