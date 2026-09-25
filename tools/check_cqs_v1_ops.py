"""Bounded operational guard fixtures; no tmux dispatch or formal training."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import cqs_v1 as cli
from cqs_v1_common import OUT, ROOT, RUN_NAME, read_json, require, utc, write_json


def run():
    folder=OUT/('ops_fixture_'+utc());folder.mkdir(parents=True)
    parsed=[]
    for name in ('prepare','preflight','probe','start','status','resume','val','test','pack','archive-failed-run'):
        cli.parser().parse_args([name]);parsed.append(name)
    bash=shutil.which('bash') or 'C:/Program Files/Git/bin/bash.exe'
    for script in ('cqs_v1.sh','sync_cqs_v1.sh'):
        subprocess.run([bash,'-n',str(ROOT/'tools'/script)],check=True)
    invalid=subprocess.run([bash,str(ROOT/'tools/sync_cqs_v1.sh'),'short'],capture_output=True)
    require(invalid.returncode==2,'Sync did not reject non-full SHA before touching repository')
    import torch
    from cqs_v1_results import postprocess, POLICY
    values=torch.tensor([[[.5,.5,.2,.2,.0001],[.5,.5,.2,.2,.9],[.5,.5,.2,.2,.3]]])
    selected,_=postprocess(values,640,.001)
    require(torch.equal(selected[0]['conf'],torch.tensor([.9,.3])),'Corrected sorted confidence mask changed')
    nonfinite=folder/'nonfinite.json'
    write_json(nonfinite,dict(loss=float('nan'),overflow=float('inf')))
    require(read_json(nonfinite)['loss']==dict(value=None,nonfinite=True,original='nan'),'Nonfinite JSON hid original value')
    fake=folder/'fake';(fake/'tools').mkdir(parents=True)
    output=fake/'outputs';output.mkdir()
    py=Path(cli.sys.executable).as_posix()
    pipeline=[]
    for code,bad_log in ((7,False),(0,False),(0,True)):
        (fake/'tools/cqs_v1.py').write_text(f'import sys\nprint("worker-output",flush=True)\nsys.exit({code})\n',encoding='utf-8')
        ident=f'exit_{code}_{bad_log}'
        log=output/('absent/log.txt' if bad_log else ident+'.log')
        with patch.object(cli,'ROOT',fake),patch.object(cli,'OUT',output),patch.object(cli,'sys',SimpleNamespace(executable=py)),\
             patch.dict(cli.os.environ,{'PATH':'/usr/bin:/bin'}):
            script=cli.worker_script(dict(id=ident,log=log.as_posix()))
        p=output/(ident+'.sh');p.write_text(script,encoding='utf-8')
        result=subprocess.run([bash,str(p)],capture_output=True,text=True)
        record=read_json(output/(ident+'.shell-exit.json'))
        require(record['python_exit']==code and (record['tee_exit']!=0)==bad_log,'PIPESTATUS attribution is wrong')
        require(result.returncode==(code if code else 1 if bad_log else 0),'Worker shell hid failure')
        require('worker-output' in result.stdout,'tmux-facing output was hidden')
        pipeline.append(dict(**record,shell_exit=result.returncode,visible_output=True))
    main=fake/'main';run_path=main/'runs/c_series'/RUN_NAME;run_path.mkdir(parents=True)
    (run_path/'args.yaml').write_text('fixture: true\n')
    with patch.object(cli,'MAIN',main),patch.object(cli,'RUN',run_path),patch.object(cli,'OUT',output),\
         patch.object(cli,'active_processes',return_value=[]),patch.object(cli,'session_active',return_value=False):
        checkpoint=run_path/'last.pt';checkpoint.write_bytes(b'fixture')
        try:
            cli.archive_failed_run()
        except RuntimeError:
            protected=checkpoint.exists()
        else:
            raise AssertionError('A run with checkpoint was archived as an initialization failure')
        checkpoint.rename(run_path.parent/'preserved_fixture_checkpoint.pt')
        archived=cli.archive_failed_run()
        require(Path(archived['archived_run']).joinpath('args.yaml').is_file(),'Failed-run evidence was not preserved')
        require(not run_path.exists(),'Archive did not free exact run path')
    # Gate cache is keyed by meaningful identity, never report timestamp.
    write_json(output/'prepare.json',dict(status='PASS',local_only=False,fingerprint=dict(sha256='fixture')))
    write_json(output/'preflight.json',dict(status='PASS',fingerprint='fixture',created='old timestamp'))
    with patch.object(cli,'OUT',output),patch.object(cli,'verify_delivery'),patch.object(cli,'active_processes',return_value=[]),\
         patch.object(cli,'session_active',return_value=False),patch.object(cli,'fingerprint',return_value=dict(sha256='fixture')):
        cached=cli.preflight()
        require(cached['created']=='old timestamp','Timestamp caused needless re-preflight')
    report=dict(status='PASS',scope='operational fixtures only; Linux tmux dispatch remains server PENDING',
                parsers=parsed,bash_syntax=True,full_sha_guard=True,pipeline=pipeline,
                checkpoint_protected=protected,failed_run_archive=archived,preflight_cache_reused=True)
    report.update(eval_policy=POLICY,corrected_sorted_conf_mask=True,strict_nonfinite_json=True)
    write_json(OUT/'ops_validation.json',report)
    return report


if __name__=='__main__':
    print(run())
