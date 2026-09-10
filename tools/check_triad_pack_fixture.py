"""Exercise complete packaging with explicitly synthetic train-lifecycle evidence and real tiny eval outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import os
from unittest.mock import patch
from triad_compat import *
import triad_compat_results as results
from train_triad_compat import recipe


def run(pipeline,checkpoint,output):
    pipeline=Path(pipeline);checkpoint=Path(checkpoint);output=Path(output)
    require(not output.exists(),'Preserve previous fixture');output.mkdir(parents=True)
    launch=output/'launch';training=output/'run';launch.mkdir();training.mkdir()
    p=dict(launch=launch,run=training,session='fixture')
    token='synthetic-fixture-only';info=runtime()
    for name in ('best.pt','last.pt'):
        (training/'weights').mkdir(exist_ok=True);os.link(checkpoint,training/'weights'/name)
    for name in ('args.yaml','results.csv','results.png','train_batch0.jpg','labels.jpg'):
        (training/name).write_text('SYNTHETIC PACKAGING FIXTURE; NOT TRAINING',encoding='utf-8')
    # Actual pipeline plots are used, while train artifact completeness is simulated explicitly.
    for f in (pipeline/'val/plots').iterdir():
        if f.is_file():shutil.copyfile(f,training/f.name)
    for split in ('val','test'):shutil.copytree(pipeline/split,launch/f'evaluation_{split}')
    c2=ROOT/'docs/triad_compat/c2_args.yaml';shutil.copyfile(c2,launch/'authoritative_c2_args.yaml')
    shutil.copyfile(pipeline/'subset/data.yaml',launch/'data_config.yaml')
    args,diff=recipe(c2,'triad_v1',checkpoint)
    for name in ('train_args.yaml','actual_train_args.yaml'):YAML.save(launch/name,args)
    YAML.save(training/'args.yaml',args)
    plan=dict(runtime=info,preflight='passed',args=args,c2_args_sha256=sha256(c2),init_sha256=sha256(checkpoint),variant='triad_v1',token=token,fixture=True)
    write_json(launch/'plan.json',plan)
    (launch/'source_snapshot.tar.gz').write_bytes(b'SYNTHETIC SNAPSHOT FIXTURE')
    write_json(launch/'source_record.json',dict(snapshot_sha256=sha256(launch/'source_snapshot.tar.gz'),
        c2_args_sha256=sha256(c2),data_sha256=sha256(launch/'data_config.yaml'),fixture=True))
    write_json(launch/'initialization.json',dict(status='passed',source_sha256=SOURCE_SHA256,output_sha256=sha256(checkpoint),fixture=True))
    for name in ('nc1_loading.json','training_setup.json','parameter_diff.json','amp_resources.json','preflight.json'):
        write_json(launch/name,dict(status='passed',fixture=True))
    for name in ('source_from_base.patch','pip_freeze.txt','console.log','worker.sh','model.yaml'):
        (launch/name).write_text('PACKAGING FIXTURE ONLY')
    write_json(launch/'process.json',dict(pid=0,token=token,fixture=True))
    write_json(launch/'launch_state.json',dict(token=token,status='DISPATCHED',fixture=True))
    write_json(launch/'exit_code.json',dict(exit_code=0,token=token,fixture=True))
    (launch/'process_exit_code.txt').write_text('0')
    write_json(launch/'worker_complete.json',dict(token=token,artifacts_complete=True,fixture=True))
    with patch.object(results,'paths',return_value=p),patch.object(results,'evaluate',side_effect=AssertionError('Pack must never run eval')):
        archive=results.package('triad_v1',output/'fixture_complete.tar.gz')
        inventory=results.verify_archive(archive)
        try:results.package('triad_v1',archive)
        except RuntimeError:pass
        else:raise AssertionError('Overwrote fixture archive')
        (launch/'evaluation_test/metrics.json').write_text('{}')
        try:results.package('triad_v1',output/'invalid.tar.gz')
        except RuntimeError:pass
        else:raise AssertionError('Accepted broken test evidence')
    write_json(output/'pack_validation.json',dict(status='passed',scope='explicit synthetic train-lifecycle fixture + real two-image val/test outputs; NOT formal training',
        archive_sha256=sha256(archive),archive_bytes=archive.stat().st_size,members=len(inventory),
        no_implicit_evaluation=True,overwrite_rejected=True,broken_test_rejected=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pipeline',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.pipeline,a.checkpoint,a.output)
