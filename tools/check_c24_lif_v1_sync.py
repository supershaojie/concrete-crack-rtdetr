"""Exercise delivered sync against isolated local Git remotes, never user worktrees."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import shlex
from c24_lif_v1_common import *

def checks(bash):
    sha=git('rev-parse','HEAD');folder=Path(tempfile.mkdtemp(prefix='sync_fixture_',dir=ROOT/'outputs')).resolve()
    remote=folder/'remote.git';main=folder/'main';target=folder/'worktree with spaces'
    def run(args,cwd=None,good=True,env=None):
        result=subprocess.run(list(map(str,args)),cwd=cwd,capture_output=True,text=True,env=env)
        if good:require(result.returncode==0,result.stdout+result.stderr)
        return result
    run(['git','clone','--bare','--shared',ROOT,remote]);run(['git','clone','--shared',remote,main])
    url='https://github.com/supershaojie/concrete-crack-rtdetr.git'
    run(['git','remote','set-url','origin',url],main)
    for k,v in [('user.name','Local sync fixture'),('user.email','sync-fixture@example.invalid')]:run(['git','config',k,v],main)
    main_head=run(['git','rev-parse','HEAD'],main).stdout
    (main/'downloads').mkdir(exist_ok=True);marker=main/'downloads/user_keep.txt';marker.write_text('preserve user result')
    routing=folder/'local_fetch.sh'
    routing.write_bytes(('git() {\n if [[ "${1:-}" == -C && "${3:-}" == fetch && "${4:-}" == origin ]]; then\n'
        ' command git "$1" "$2" fetch '+shlex.quote(remote.as_posix())+' "${@:5}"\n'
        ' else command git "$@"; fi\n}\n').encode())
    env={**os.environ,'C24_LIF_V1_PYTHON':sys.executable.replace('\\','/'),'BASH_ENV':routing.as_posix()}
    args=[bash,ROOT/'tools/sync_c24_lif_v1.sh',sha,main,target]
    first=run(args,env=env);again=run(args,env=env)
    require(marker.read_text()=='preserve user result' and run(['git','rev-parse','HEAD'],main).stdout==main_head,'Main modified')
    file=target/'tools/train_c24_lif_v1.py';original=file.read_bytes();file.write_bytes(original+b'\n# fixture dirty\n')
    dirty=run(args,good=False,env=env);require(dirty.returncode!=0 and file.read_bytes()!=original,'Dirty target overwritten');file.write_bytes(original)
    run(['git','commit','--allow-empty','-m','test fixture newer branch'],main)
    new=run(['git','rev-parse','HEAD'],main).stdout.strip()
    run(['git','push',remote,'HEAD:refs/heads/'+BRANCH],main)
    mismatch=run([bash,ROOT/'tools/sync_c24_lif_v1.sh',new,main,target],good=False,env=env)
    require(mismatch.returncode!=0 and run(['git','rev-parse','HEAD'],target).stdout.strip()==sha,'Different SHA target modified')
    unrelated=folder/'unrelated';unrelated.mkdir();(unrelated/'keep').write_text('untouched')
    other=run([bash,ROOT/'tools/sync_c24_lif_v1.sh',sha,main,unrelated],good=False,env=env)
    require(other.returncode!=0 and (unrelated/'keep').read_text()=='untouched','Unrelated directory modified')
    typo=run([bash,ROOT/'tools/autodl_c24_lif_v1.sh','typo'],good=False)
    require(typo.returncode==2,'Unknown action must exit 2 before environment/training')
    for name in ['sync_c24_lif_v1.sh','autodl_c24_lif_v1.sh']:run([bash,'-n',ROOT/'tools'/name])
    report=dict(status='PASSED',tested_commit=sha,first_sync=True,idempotent_same_sha=True,dirty_refused=True,different_sha_preserved=True,
        unrelated_preserved=True,main_head_and_downloads_preserved=True,unknown_action_exit=typo.returncode,bash_syntax='PASSED',
        fixture=str(folder),network='fixture BASH_ENV routes fetch to local bare repository; no external fetch/push',fixture_preserved=True)
    write_json(ROOT/'docs/c24_lif_v1/sync_checks.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--bash',required=True);a=p.parse_args();checks(a.bash)
