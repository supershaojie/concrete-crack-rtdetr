"""Small lifecycle, hard-size-limit and policy regressions; never formal model training."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import base64
import json
import os
import subprocess
import sys
from c24_lif_v1_common import *
from c24_lif_v1_pack import pack_light,verify_archive,LIMIT
import train_c24_lif_v1 as lifecycle

def checks():
    report={}
    with TemporaryDirectory(dir=ROOT/'outputs',prefix='pack_stress_') as temp:
        folder=Path(temp);launch=folder/'launch';launch.mkdir()
        # >100 MB input; incompressible log tails exceed the upload budget before reduction.
        random=base64.b64encode(os.urandom(800000))
        for i in range(90):(launch/f'worker_{i}.log').write_bytes(random)
        massive=dict(status='BLOCKED',first_error=dict(stage='fusion_cuda_half',key='candidate_ids',mode='half',shape=[1,300],
            atol=.008,rtol=.04,max_abs=.05,candidate_ids_a=[list(range(300))],candidate_ids_b=[list(range(1,301))]),
            activations=[i/7000 for i in range(1500000)],large_text=base64.b64encode(os.urandom(1800000)).decode())
        write_json(launch/'large.json',massive)
        write_json(launch/'first_failure.json',dict(status='BLOCKED',first_error=massive['first_error'],replay={'A':{'status':'PASSED'},'B':{'status':'BLOCKED'}}))
        (launch/'best.pt').write_bytes(b'DO NOT LOAD OR PACKAGE')
        dest=pack_light(launch,folder/'light.tar.gz');manifest=verify_archive(dest)
        require(dest.stat().st_size<=LIMIT and all(not k.endswith('.pt') for k in manifest),'Light size/weights policy')
        import tarfile
        with tarfile.open(dest) as t:
            first=json.load(t.extractfile('metadata/first_failure.json'));large=json.load(t.extractfile('metadata/large.json'))
            package=json.load(t.extractfile('PACKAGE.json'))
        require(first['first_error']==large['first_error']==massive['first_error'],'Essential failure evidence lost')
        require(package['reductions'],'Stress did not exercise reductions')
        report['pack_stress']=dict(status='PASSED',source_bytes=sum(f.stat().st_size for f in launch.iterdir()),
            archive_bytes=dest.stat().st_size,limit=LIMIT,sha256=sha256(dest),reductions=len(package['reductions']),first_failure_preserved=True,
            checkpoint_loaded=False,archive_members=len(manifest),all_json_parseable=True)
    with TemporaryDirectory(dir=ROOT/'outputs',prefix='locks_') as temp:
        folder=Path(temp);fake=dict(run=folder/NAME,launch=folder/'launch')
        with patch.object(lifecycle,'paths',lambda:fake),patch.object(lifecycle,'tmux_active',lambda:False),patch.object(lifecycle,'token',lambda pid:'start_123' if pid==os.getpid() else None):
            info=lifecycle.acquire('preflight')
            try:lifecycle.acquire('preflight');raise AssertionError('Duplicate lock accepted')
            except RuntimeError:pass
            lifecycle.release('preflight',dict(info,token='wrong'));require(lifecycle.lock_path('preflight').exists(),'Wrong token released lock')
            lifecycle.release('preflight',info);require(not lifecycle.lock_path('preflight').exists(),'Owned lock not released')
            stale=lifecycle.lock_path('worker');stale.mkdir();write_json(stale/'owner.json',dict(info,pid=99999999,start_time='old'))
            new=lifecycle.acquire('worker');require(list(folder.glob('*.stale.*')),'Stale evidence not archived');lifecycle.release('worker',new)
            ambiguous=lifecycle.lock_path('preflight');ambiguous.mkdir();write_json(ambiguous/'owner.json',dict(info,worktree='/unrelated'))
            try:lifecycle.acquire('preflight');raise AssertionError('Ambiguous owner accepted')
            except RuntimeError:pass
            require(ambiguous.exists(),'Ambiguous lock destroyed')
            report['locks']=dict(status='PASSED',duplicate_refused=True,pid_start_time_checked=True,wrong_token_preserved=True,
                dead_owner_archived=True,unrelated_owner_preserved=True)
    # Parsing and sorted-mask behavior use deliberately reordered scores.
    from c24_lif_v1_results import postprocess
    import torch
    raw=torch.tensor([[[.5,.5,.2,.2,.05],[.5,.5,.2,.2,.9],[.5,.5,.2,.2,.2]]])
    output,affected=postprocess(raw,640,.1)
    require(torch.equal(output[0]['conf'],torch.tensor([.9,.2])) and affected==1,'Sorted mask regression')
    report['evaluation_mask']=dict(status='PASSED',policy='corrected_sorted_conf_mask_v1',inference='NOT_RUN')
    write_json(ROOT/'docs/c24_lif_v1/ops_checks.json',report);return report

if __name__=='__main__':print(json.dumps(checks(),indent=2))
