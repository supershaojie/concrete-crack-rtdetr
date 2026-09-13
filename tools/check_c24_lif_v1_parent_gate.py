"""Finite regression of measured original-C24 certificates and deliberately corrupted copies."""
from copy import deepcopy
from pathlib import Path
import json
import torch
from c24_lif_v1_common import ROOT, read_json, write_json, require
from c24_lif_v1_acceptance import numerical_status, fusion_status, stage_status, aggregate, ACCEPTED
from c24_lif_v1_parent_p3 import WARNING, certificate_status, boundary_facts, outside_facts, tensor_record


def package_check(preflight, output):
    import shutil
    import uuid
    from c24_lif_v1_pack import pack_light
    from c24_lif_v1_light import LightReader
    fixture=(ROOT/'outputs'/('p3pkg_'+uuid.uuid4().hex[:10])).resolve()
    require(fixture.is_relative_to((ROOT/'outputs').resolve()),'Fixture must stay in this worktree')
    launch=fixture/'launch';launch.mkdir(parents=True)
    for path in Path(preflight).rglob('*.json'):
        target=launch/'preflight'/path.relative_to(preflight)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    # Real server diagnostics contain duplicate stage/preflight/certificate JSON;
    # round-trip every reference and every score, not just the top-level manifest.
    destination=pack_light(launch,fixture/'c24_lif_v1_LIGHT.tar.gz')
    reader=LightReader(destination)
    for name in reader.manifest:
        if name.endswith('.json'):reader.read(name)
    restored=reader.read('metadata/preflight/preflight.json')
    original=read_json(Path(preflight)/'preflight.json')
    for mode in ('amp','half'):
        get=lambda r:next(s['result'] for s in r['stages'] if s['name']=='fusion_cuda_'+mode)
        a,b=get(original),get(restored)
        require(a==b and fusion_status(b)==WARNING,'LIGHT lost raw candidate/shared/outside/replay evidence')
    require(max(v['bytes'] for n,v in reader.manifest.items() if n.endswith('.json'))<=64000,'LIGHT JSON shard exceeds 64 KB')
    return dict(status='PASSED',bytes=destination.stat().st_size,members=len(reader.manifest),
        references=len(reader.references),manifest_and_all_references='VERIFIED',
        full_fusion_roundtrip='EXACT',path=str(destination),model_loading=False,preflight_rerun=False)


def checks(preflight, output):
    report=read_json(Path(preflight)/'preflight.json')
    rows={r['name']:r['result'] for r in report['stages']}
    results={}
    for mode in ('amp','half'):
        fusion=rows['fusion_cuda_'+mode];case=fusion['cases'][0]
        require(case['natural_relation']=='SET_DRIFT' and numerical_status(case)==WARNING and
                fusion_status(fusion)==stage_status('fusion_cuda_'+mode,fusion)==WARNING, 'Measured certificate not accepted: '+mode)
        require(aggregate(['PASSED',WARNING])==WARNING,'Aggregate lost parent warning')
        results[mode+'_measured_certificate']=dict(status=WARNING,
            outside=case['parent_p3_certificate']['outside'],input=case['input'])
        tests={}
        def run(name,mutate,expected=None):
            changed=deepcopy(case);mutate(changed)
            status=numerical_status(changed)
            reason=certificate_status(changed,changed.get('parent_control'))
            require(status not in ACCEPTED and (expected is None or status==expected), 'Accepted corruption '+name+': '+str(reason))
            tests[name]=dict(status=status,diagnostic=reason)
        cert=lambda c:c['parent_p3_certificate']
        ex=lambda c,n:cert(c)['executions'][n]
        run('only_comparability_label',lambda c:(c.pop('parent_p3_certificate'),c['parent_control'].update(comparability='VERIFIED')),'REQUIRES_REVIEW')
        run('missing_parent',lambda c:c.pop('parent_control'),'BLOCKED')
        run('missing_certificate',lambda c:c.pop('parent_p3_certificate'),'REQUIRES_REVIEW')
        for side in ('A','B'):
            for who in ('combo','parent'):
                run('missing_'+who+'_replay_'+side,lambda c,s=side,w=who:(c if w=='combo' else c['parent_control'])['replay'].pop(s),'BLOCKED')
        run('missing_shared_field',lambda c:ex(c,'parent_A')['tensors'].pop('p3'),'BLOCKED')
        run('shared_input_changed',lambda c:ex(c,'parent_A')['environment']['input'].update(sha256='0'*64),'REQUIRES_REVIEW')
        run('shared_weight_changed',lambda c:ex(c,'parent_A')['contract']['shared_prefix'].update(sha256='0'*64),'REQUIRES_REVIEW')
        run('shared_torch_equal_failed',lambda c:cert(c)['shared_state_comparisons']['A']['shared_prefix'].update(exact=False,first_difference='model.0.conv1.weight'),'REQUIRES_REVIEW')
        run('token_mixing_path',lambda c:ex(c,'parent_A')['contract'].update(linear_norm_score_tokenwise=False),'REQUIRES_REVIEW')
        run('wrong_shape',lambda c:ex(c,'parent_A')['tensors']['p3']['shape'].__setitem__(2,1),'BLOCKED')
        run('numerical_wrong_shape',lambda c:c['continuous'][0]['shape_b'].__setitem__(2,1),'BLOCKED')
        run('nonfinite_score',lambda c:ex(c,'parent_A')['scores'].__setitem__(0,float('nan')),'BLOCKED')
        run('duplicate_id',lambda c:ex(c,'parent_A')['ids'].__setitem__(0,ex(c,'parent_A')['ids'][1]),'BLOCKED')
        run('wrong_mask',lambda c:ex(c,'parent_A')['invalid_ids'].append(1),'BLOCKED')
        run('changed_tolerance',lambda c:c['replay']['B']['continuous'][0].update(atol=.1),'BLOCKED')
        run('missing_comparison_metric',lambda c:cert(c)['shared_comparisons']['A'][0].pop('max_abs'),'REQUIRES_REVIEW')
        run('unexplained_swap_margin',lambda c:cert(c)['boundary']['combo']['swaps'][0].update(margin_A=10),'BLOCKED')
        def hash_scores(e):
            dtype={'torch.float16':torch.float16,'torch.float32':torch.float32}[e['full_logits']['dtype']]
            tensor=torch.tensor(e['scores'],dtype=dtype).reshape(1,-1,1)
            e['full_logits']=tensor_record(tensor)
            e['tensors']['encoder_logits']=tensor_record(tensor[:,:e['region']['stop']])
        def refresh(c,kind):
            a,b=ex(c,kind+'_A'),ex(c,kind+'_B')
            cert(c)['boundary'][kind]=boundary_facts(a,b)
            cert(c)['outside'][kind]=outside_facts(a,b)
        def one_logit(c):
            e=ex(c,'parent_A');used=set(e['ids'])|set(e['invalid_ids'])
            index=next(i for i in range(e['region']['stop']) if i not in used)
            before=e['scores'][index]
            e['scores'][index]=float(torch.nextafter(torch.tensor(before,dtype=torch.float16),torch.tensor(float('-inf'),dtype=torch.float16)))
            hash_scores(e);refresh(c,'parent')
            # The full 300 selected IDs and all selected scores are unchanged.
            require(e['ids']==case['candidate_ids_a'][0], 'Fixture changed selected IDs')
        run('one_P3_logit_same_topk',one_logit,'REQUIRES_REVIEW')
        def outside_cutoff(c):
            for side in ('A','B'):
                e=ex(c,'combo_'+side);index=next(i for i in range(e['region']['stop'],e['region']['total']) if i not in e['invalid_ids'])
                e['scores'][index]=min(e['scores'][i] for i in e['ids']);hash_scores(e)
            refresh(c,'combo')
        run('outside_competitor_at_cutoff',outside_cutoff,'REQUIRES_REVIEW')
        from c24_lif_v1_acceptance import blocking_summary
        tampered=deepcopy(fusion);tampered['cases'][0]['parent_p3_certificate']['executions']['parent_A']['scores'][0]=float('nan')
        summary=blocking_summary(dict(scope='local',stages=[dict(name='fusion_cuda_'+mode,status='PASSED',result=tampered)]))
        require(summary['unresolved_stages'][0]['name']=='fusion_cuda_'+mode and summary['unresolved_stages'][0]['status']=='BLOCKED',
                'Summary trusted status string over certificate facts')
        tests['summary_recomputes_facts']=dict(status='BLOCKED')
        def different_ids(c):
            p=c['parent_control'];p['candidate_ids_a'][0][0]=next(i for i in range(ex(c,'parent_A')['region']['stop']) if i not in p['candidate_ids_a'][0])
        run('same_change_count_different_IDs',different_ids,'BLOCKED')
        for key in ('lif_bn_state_exact','physical_negatives'):
            broken=deepcopy(fusion);broken[key]=False
            require(fusion_status(broken)=='BLOCKED','Physical failure accepted')
            tests[key]=dict(status='BLOCKED')
        historical=read_json(ROOT/('outputs/c24_lif_v1_parent_gate_fix/audit/historical_'+mode+'.json'))
        require(certificate_status(historical['combo'],historical['parent'])['reason']=='MISSING_SHARED_P3_CERTIFICATE', 'Historical missing evidence promoted')
        tests['historical_LIGHT_stays_missing']=dict(status='REQUIRES_REVIEW')
        results[mode+'_corrupted_evidence']=dict(scope='COUNTERFACTUAL_COPIES_OF_MEASURED_CERTIFICATE_NOT_MODEL_RUNS',tests=tests)
    # Permutation control uses actual small-shape evidence and identity alignment.
    for mode in ('amp','half'):
        case=rows['fusion_cuda_'+mode]['cases'][1]
        require(case['natural_relation'] in ('IDENTICAL','PERMUTATION') and numerical_status(case)=='PASSED','ID alignment regression')
    from check_c24_lif_v1_preflight_fix import server_gate_contracts
    results['server_reader_with_real_warning']=server_gate_contracts(Path(output)/'measured_warning_server_reader',
        {mode:rows['fusion_cuda_'+mode] for mode in ('amp','half')})
    results['light_roundtrip']=package_check(preflight,output)
    results['lock_identity']=lock_checks(output)
    write_json(Path(output)/'parent_gate_contracts.json',dict(status='PASSED',results=results))
    return dict(status='PASSED',negative_cases=sum(len(v.get('tests',{})) for v in results.values()),
                warning_modes=['amp','half'],formal_training='NOT_RUN',full_val_test='NOT_RUN')


def lock_checks(output):
    from unittest.mock import patch
    from types import SimpleNamespace
    import uuid
    import train_c24_lif_v1 as launcher
    folder=(ROOT/'outputs'/('p3lock_'+uuid.uuid4().hex[:10])).resolve()
    require(folder.is_relative_to((ROOT/'outputs').resolve()),'Lock fixture must stay in this worktree')
    result={}
    for kind in ('preflight','worker'):
        path=folder/kind;path.mkdir(parents=True)
        owner=dict(worktree=str(ROOT.resolve()),session=launcher.SESSION,pid=123,token='fixture')
        write_json(path/'owner.json',owner);before=(path/'owner.json').read_bytes()
        with patch.object(launcher,'lock_path',lambda k:path),patch.object(launcher,'tmux_active',lambda:False):
            try:
                launcher.acquire(kind)
                raise AssertionError('Unknown PID start time accepted')
            except RuntimeError as error:
                require('Ambiguous' in str(error),'Wrong lock rejection')
        require((path/'owner.json').read_bytes()==before and not list(folder.glob(kind+'.stale.*')),'Unknown lock modified')
        result[kind]='AMBIGUOUS_OWNER_PRESERVED'
    return dict(status='PASSED',cases=result,scope='Isolated lock fixture; no real lock or process touched')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(checks(a.preflight,a.output),indent=2))
