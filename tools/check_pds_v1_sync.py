"""Offline exercise of the official sync against disposable shared Git fixtures."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from pds_v1_common import *


def command(args,cwd=None):
    result=subprocess.run(args,cwd=cwd,capture_output=True,text=True)
    require(result.returncode==0,f"Fixture command failed {args}: {result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def run(output):
    output=output.resolve()
    require(output.is_relative_to(ROOT/"outputs"),"Sync fixture must remain inside this worktree outputs")
    output.mkdir(parents=True,exist_ok=False)
    bash=shutil.which("bash") or ("C:/Program Files/Git/bin/bash.exe" if os.name=="nt" else None)
    require(bash,"bash unavailable")
    sha=git("rev-parse","HEAD")
    origin,main,target=output/"origin.git",output/"main",output/"main-pds_v1"
    command(["git","clone","--shared","--bare",str(ROOT),str(origin)])
    command(["git","clone","--shared",str(origin),str(main)])
    command(["git","checkout","-b","fixture-main",BASE],main)
    expected="https://github.com/supershaojie/concrete-crack-rtdetr.git"
    command(["git","remote","set-url","origin",expected],main)
    # Only the fetch transport below is redirected; get-url still checks the real identity string.
    shim=output/"python_fixture.sh"
    shim_text="#!/usr/bin/env bash\nset -e\n"
    if os.name=="nt":
        shim_text+='export PDS_MAIN="$(cygpath -m "$PDS_MAIN")"\n'
    import shlex
    shim_text+="exec "+shlex.quote(Path(sys.executable).as_posix())+' "$@"\n'
    with shim.open("w",encoding="utf-8",newline="\n") as f:
        f.write(shim_text)
    if os.name!="nt":
        shim.chmod(0o755)
    original=(ROOT/"tools/sync_pds_v1.sh").read_text(encoding="utf-8")
    script=output/"sync_fixture.sh"
    # Interpreter and offline fetch transport are substituted; sync/delivery logic is unchanged.
    with script.open("w",encoding="utf-8",newline="\n") as f:
        f.write(original.replace("PYTHON=/root/miniconda3/envs/rtdetr/bin/python","PYTHON="+shlex.quote(shim.as_posix())))
    env=os.environ.copy()
    # Bash canonical paths are obtained from bash, not guessed on Windows.
    def shell_path(path):
        if os.name=="nt":
            return command([bash,"-c",'cygpath -u "$1"',"path",str(path)])
        return str(path)
    env.update(PDS_MAIN=shell_path(main),PDS_WORKTREE=shell_path(target))
    env.pop("PDS_FETCHED_SHA",None)
    def sync(value=sha,extra_prefix=""):
        transport=('git() { if [[ "$1" == -C && "$3" == fetch && "$4" == origin ]]; then '
                   'command git -C "$2" fetch '+shlex.quote(origin.as_posix())+' "$5:refs/remotes/origin/$5"; '
                   'else command git "$@"; fi; }; export -f git; ')
        line=transport+extra_prefix+"bash "+shlex.quote(shell_path(script))+" "+shlex.quote(value)
        return subprocess.run([bash,"-c",line],env=env,capture_output=True,text=True)
    first=sync()
    require(first.returncode==0,first.stdout+"\n"+first.stderr)
    record=read_json(target/"outputs/pds_v1/delivery.json")
    require(record["sha"]==sha,"Official delivery identity missing")
    require(git("rev-parse","HEAD",cwd=main)==BASE,"Sync moved fixture main HEAD")
    # LBC-named session alone must never block PDS sync.
    lbc='tmux() { [[ "$*" == *lbc-v1-training* ]]; }; export -f tmux; '
    again=sync(extra_prefix=lbc)
    require(again.returncode==0,again.stdout+"\n"+again.stderr)
    require(list((target/"outputs/pds_v1").glob("delivery.json.*.archive")),"Previous metadata not archived")
    path=target/"README.md"
    old=path.read_bytes()
    path.write_bytes(old+b"\nfixture tracked edit\n")
    dirty=sync()
    require(dirty.returncode!=0 and path.read_bytes()==old+b"\nfixture tracked edit\n","Dirty target overwritten")
    path.write_bytes(old)
    active=sync(extra_prefix='tmux() { return 0; }; export -f tmux; ')
    require(active.returncode!=0,"Active PDS session accepted")
    malformed=sync("bad-sha")
    require(malformed.returncode!=0,"Incomplete SHA accepted")
    require(git("rev-parse","HEAD",cwd=main)==BASE and git("rev-parse","HEAD",cwd=target)==sha,"Fixture heads changed unexpectedly")
    result=dict(status="PASS",scope="offline shared Git fixture, interpreter and fetch transport bindings substituted",
                tested_sha=sha,official_delivery_record=True,clean_resync=True,main_head_preserved=True,
                tracked_edits_preserved=True,active_PDS_rejected=True,LBC_session_ignored=True,
                complete_SHA_required=True,old_metadata_archived=True,no_real_remote_access=True)
    write_json(output/"sync.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.output),indent=2))
