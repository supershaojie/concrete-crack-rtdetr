"""Safe detached server worktree creation / fast-forward update, invoked from fetched code."""
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

MAIN=Path('/root/autodl-tmp/projects/Crack_RTDETR')
DEST=MAIN.with_name('Crack_RTDETR-lcd_v1')
ORIGIN='https://github.com/supershaojie/concrete-crack-rtdetr.git'
BRANCH='exp-rtdetr-r18-lite-lcd-v1'


def git(where,*args):
    return subprocess.check_output(['git','-C',str(where),*args],text=True).strip()


def sync(sha):
    assert re.fullmatch('[0-9a-f]{40}',sha), 'Full SHA required'
    assert git(MAIN,'remote','get-url','origin')==ORIGIN
    assert git(MAIN,'rev-parse','FETCH_HEAD')==sha,'Fetch the fixed branch first'
    before=git(MAIN,'rev-parse','HEAD')
    registrations=git(MAIN,'worktree','list','--porcelain')
    other={}
    for block in registrations.split('\n\n'):
        row=dict(line.split(' ',1) for line in block.splitlines() if ' ' in line)
        if row.get('worktree') and Path(row['worktree']).resolve()!=DEST.resolve():
            other[row['worktree']]=row['HEAD']
    assert not subprocess.run(['tmux','has-session','-t','=lcd-v1-training'],capture_output=True).returncode==0,'LCD session active'
    for cmdline in Path('/proc').glob('[0-9]*/cmdline'):
        try: argv=cmdline.read_bytes().split(b'\0')
        except OSError:continue
        assert not (str(DEST/'tools/lcd_v1.py').encode() in argv and b'worker' in argv),'LCD worker active'
    if DEST.exists():
        assert (DEST/'.git').is_file(),'Existing destination is not a linked worktree'
        assert Path(git(DEST,'rev-parse','--path-format=absolute','--git-common-dir'))==Path(git(MAIN,'rev-parse','--path-format=absolute','--git-common-dir'))
        assert not git(DEST,'status','--porcelain','--untracked-files=no'),'Tracked edits protected'
        old=git(DEST,'rev-parse','HEAD')
        subprocess.run(['git','-C',str(DEST),'merge-base','--is-ancestor',old,sha],check=True)
        tracked=set(git(DEST,'ls-files','-z').split('\0'))
        targets=git(MAIN,'ls-tree','-r','--name-only','-z',sha).split('\0')
        for name in filter(None,targets):
            path=DEST/name
            assert name in tracked or not path.exists(),'Untracked/ignored target collision: '+name
            for ancestor in path.parents:
                if ancestor==DEST:break
                assert not ancestor.is_file() or ancestor.relative_to(DEST).as_posix() in tracked,'Untracked ancestor collision'
        subprocess.run(['git','-C',str(DEST),'checkout','--detach','--no-overwrite-ignore',sha],check=True)
    else:
        subprocess.run(['git','-C',str(MAIN),'worktree','add','--detach',str(DEST),sha],check=True)
    assert git(DEST,'rev-parse','HEAD')==sha and git(MAIN,'rev-parse','HEAD')==before
    assert all(git(Path(p),'rev-parse','HEAD')==head for p,head in other.items()),'Other HEAD changed'
    out=DEST/'outputs/lcd_v1';out.mkdir(parents=True,exist_ok=True)
    report=out/'delivery.json'
    if report.exists():
        backup=out/'history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup.mkdir(parents=True);report.rename(backup/report.name)
    report.write_text(json.dumps(dict(commit=sha,branch=BRANCH,origin=ORIGIN,worktree=str(DEST),main=str(MAIN),
                                     other_heads_preserved=other),indent=2)+'\n',encoding='utf-8')
    print('Synced LCD:',sha,'at',DEST)


if __name__=='__main__':sync(sys.argv[1])
