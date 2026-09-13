"""Original-C24 shared-P3 certificate. Diagnostic evidence, never a forward change.

All scalar scores below come from four natural executions. Replays execute each
model again with its own features. No P3 activations/weights are transplanted.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import torch
from c24_lif_v1_common import ROOT, SOURCE_SHA256, require, sha256
from c24_lif_v1_topology import locate, refs

KIND = 'original_c24_shared_p3_v1'
WARNING = 'ACCEPTED_WITH_PARENT_P3_CUTOFF_WARNING'
PARENT_COMMIT = 'f6e9dfda765046ae7691302cf5ec89d3f76cec5d'
HEAD_HASH = 'c092316743055f02adc84589d843588631ffe705fb0777178e0362d81471f4bc'
SCCA_HASH = '67b0d347d5305c16d48bb031cd10fd1d23b7ea447b58b2423166eb68b81dedca'
LIF_HASH = '26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7'
SHARED_KEYS = ('p3','encoder_input','encoder_features','encoder_logits','valid_mask','anchors_valid')
EXECUTIONS = ('combo_A','combo_B','parent_A','parent_B')
SLACK_ULPS = 8  # FP64 roundoff only; not an FP16 ULP or an adjustable tolerance.


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def tensor_record(value):
    raw=value.detach().cpu().contiguous()
    return dict(shape=list(raw.shape),dtype=str(raw.dtype),sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
                finite=bool(torch.isfinite(raw).all()))


def state_record(state):
    rows={n:tensor_record(v) for n,v in sorted(state.items())}
    return dict(sha256=digest(rows),tensors=len(rows),names=list(rows),finite=all(v['finite'] for v in rows.values()),
                dtype_counts={t:sum(v['dtype']==t for v in rows.values()) for t in sorted({v['dtype'] for v in rows.values()})})


def environment(x,mode):
    return dict(input=tensor_record(x),device=str(x.device),torch=str(torch.__version__),mode=mode,
        autocast=torch.is_autocast_enabled(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic=torch.are_deterministic_algorithms_enabled(),warn_only=torch.is_deterministic_algorithms_warn_only_enabled())


def path_contract(model,kind):
    from init_c24_lif_v1 import CONFIGS,MODEL_DIR,YAML
    from init_c24_lif_v1 import RTDETRDecoder
    layers=model.yaml['backbone']+model.yaml['head'];top=locate(model.yaml)
    expected=YAML.load(MODEL_DIR/CONFIGS[kind]);expected_layers=expected['backbone']+expected['head']
    p3=top['decoder_inputs'][0];head=model.model[top['decoder']]
    ancestors=set()
    def visit(index):
        if index<0 or index in ancestors:return
        require(index<p3+1,'P3 has a future/downsample dependency')
        ancestors.add(index)
        for source in refs(layers[index],index):visit(source)
    visit(p3)
    head_path=ROOT/'ultralytics-main/ultralytics/nn/modules/head.py'
    head_source=hashlib.sha256(head_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    # Pin reviewed source plus live module/method geometry. Future cross-token
    # selection changes cannot silently inherit this certificate contract.
    methods=all(getattr(head,n).__func__ is getattr(RTDETRDecoder,n) for n in ('_get_encoder_input','_get_decoder_input'))
    projections=[]
    for module in head.input_proj:
        good=(type(module) is torch.nn.Sequential and len(module)==2 and type(module[0]) is torch.nn.Conv2d and
            type(module[1]) is torch.nn.BatchNorm2d)
        if good:
            conv,bn=module
            good=conv.kernel_size==conv.stride==conv.dilation==(1,1) and conv.padding==(0,0) and conv.groups==1 and not bn.training
        projections.append(bool(good))
    linear_norm=(type(head.enc_output) is torch.nn.Sequential and len(head.enc_output)==2 and
        type(head.enc_output[0]) is torch.nn.Linear and type(head.enc_output[1]) is torch.nn.LayerNorm and
        head.enc_output[1].normalized_shape==(head.enc_output[0].out_features,) and type(head.enc_score_head) is torch.nn.Linear)
    prefix,selection,scca=path_states(model,ancestors)
    return dict(kind=kind,architecture_sha256=digest(layers),expected_architecture_sha256=digest(expected_layers),
        original_architecture=layers==expected_layers,topology=top,p3=p3,ancestors=sorted(ancestors),
        lif_outside_shared_path=top['p3_to_p4']['downsample'] not in ancestors,
        head_source_sha256=head_source,head_source_raw_sha256=sha256(head_path),native_head=type(head) is RTDETRDecoder,native_methods=methods,
        projections_tokenwise=projections,linear_norm_score_tokenwise=bool(linear_norm),
        nc=head.nc,num_queries=head.num_queries,hidden_dim=head.enc_score_head.in_features,training=model.training,
        shared_prefix=state_record(prefix),selection_state=state_record(selection),
        nonzero_scca=state_record(scca),
        scca_output_nonzero=bool(torch.count_nonzero(model.model[top['aifi']].scca_o.weight)))



def path_states(model,ancestors):
    head=model.model[locate(model.yaml)['decoder']]
    prefix={n:v for n,v in model.state_dict().items() if n.startswith('model.') and int(n.split('.')[1]) in ancestors}
    selection={n:v for n,v in head.state_dict().items() if n.startswith(('input_proj.0.','enc_output.','enc_score_head.'))}
    return prefix,selection,{n:v for n,v in prefix.items() if '.scca_' in n}


def state_comparison(a,b):
    names=sorted(set(a)|set(b));first=None
    for name in names:
        if name not in a or name not in b or a[name].dtype!=b[name].dtype or not torch.equal(a[name],b[name]):
            first=name;break
    return dict(exact=first is None,names_equal=set(a)==set(b),tensors=len(names),first_difference=first)

def shared_tensors(records,stop):
    mask=records['valid_mask'][:,:stop]
    anchors=records['_trace']['anchors'][:,:stop]
    return dict(p3=records['p3'],encoder_input=records['encoder_input'][:,:stop],
        encoder_features=records['encoder_features'][:,:stop],encoder_logits=records['encoder_logits'][:,:stop],
        valid_mask=mask,anchors_valid=anchors[mask.expand_as(anchors)])


def execution(model,x,mode,records,contract):
    from init_c24_lif_v1 import RTDETRDecoder
    shapes=records['_trace']['shapes'];stop=math.prod(shapes[0]);total=sum(math.prod(s) for s in shapes)
    require(records['encoder_logits'].shape==(1,total,1),'P3 certificate requires native nc=1 B1 logits')
    mask=records['valid_mask'];anchors=records['_trace']['anchors']
    expected_a,expected_m=RTDETRDecoder._generate_anchors(shapes,dtype=records['encoder_input'].dtype,device=anchors.device)
    shared=shared_tensors(records,stop)
    logits=records['encoder_logits'];scores=logits.detach().double().max(-1).values[0]
    invalid=(~mask[0,:,0]).nonzero().flatten().tolist()
    return dict(environment=environment(x,mode),contract=contract,shapes=shapes,region=dict(start=0,stop=stop,total=total),
        tensors={k:tensor_record(v) for k,v in shared.items()},scores=scores.cpu().tolist(),
        full_logits=tensor_record(logits),full_encoder_input=tensor_record(records['encoder_input']),
        valid_mask=tensor_record(mask),invalid_ids=invalid,expected_mask=tensor_record(expected_m),
        anchor_generation_dtype=str(records['encoder_input'].dtype),
        anchors_valid=tensor_record(anchors[mask.expand_as(anchors)]),
        expected_anchors_valid=tensor_record(expected_a[expected_m.expand_as(expected_a)]),
        invalid_anchor_count=len(invalid),invalid_anchors_positive_inf=bool(torch.isposinf(anchors[~mask.expand_as(anchors)]).all()),
        ids=records['candidate_ids'][0].tolist())


def slack(*values):
    return SLACK_ULPS*math.ulp(max(1.,*(abs(float(v)) for v in values)))


def boundary_facts(a,b):
    sa,sb=a['scores'],b['scores'];ia,ib=a['ids'],b['ids'];s=a['region']['stop']
    aa,bb=set(ia),set(ib);ta=min(sa[i] for i in ia);tb=min(sb[i] for i in ib)
    e_s=max(abs(x-y) for x,y in zip(sa[:s],sb[:s]));eps=slack(ta,tb,e_s)
    changed=sorted(aa^bb)
    margins=[dict(id=i,score_A=sa[i],score_B=sb[i],distance_A=abs(sa[i]-ta),distance_B=abs(sb[i]-tb)) for i in changed]
    swaps=[]
    for i in sorted(aa-bb):
        for j in sorted(bb-aa):
            ma=sa[i]-sa[j];mb=sb[j]-sb[i];e=abs(sa[i]-sb[i])+abs(sa[j]-sb[j]);e64=slack(sa[i],sa[j],sb[i],sb[j])
            swaps.append(dict(a_only=i,b_only=j,margin_A=ma,margin_B=mb,local_perturbation=e,slack_FP64=e64,
                              explained=ma>=0 and mb>=0 and ma+mb<=e+e64))
    ordered_a=sorted(sa,reverse=True);ordered_b=sorted(sb,reverse=True);k=len(ia)
    return dict(cutoff_A=ta,cutoff_B=tb,next_A=ordered_a[k],next_B=ordered_b[k],
        cutoff_tie_ids_A=[i for i,v in enumerate(sa) if v==ta],cutoff_tie_ids_B=[i for i,v in enumerate(sb) if v==tb],
        cutoff_tie_count_A=sum(v==ta for v in sa),cutoff_tie_count_B=sum(v==tb for v in sb),
        cutoff_band_ids_A=[i for i,v in enumerate(sa[:s]) if abs(v-ta)<=e_s+eps],
        cutoff_band_ids_B=[i for i,v in enumerate(sb[:s]) if abs(v-tb)<=e_s+eps],
        only_A=sorted(aa-bb),only_B=sorted(bb-aa),shared_score_perturbation=e_s,slack_FP64=eps,
        changed_scores=margins,swaps=swaps,
        within_band=all(v['distance_A']<=e_s+eps and v['distance_B']<=e_s+eps for v in margins),
        all_swaps_explained=all(v['explained'] for v in swaps))


def outside_facts(a,b):
    sa,sb=a['scores'],b['scores'];s=a['region']['stop'];total=len(sa)
    invalid_a,invalid_b=set(a['invalid_ids']),set(b['invalid_ids'])
    require(invalid_a==invalid_b,'Fusion changed valid candidate mask')
    ids=[i for i in range(s,total) if i not in invalid_a]
    require(ids,'Outside region has no legal candidate evidence')
    qa=max(sa[i] for i in ids);qb=max(sb[i] for i in ids)
    ta=min(sa[i] for i in a['ids']);tb=min(sb[i] for i in b['ids'])
    e=max(abs(sa[i]-sb[i]) for i in ids);dt=abs(ta-tb);eps=slack(ta,tb,qa,qb,e,dt)
    ga,gb=ta-qa,tb-qb
    return dict(legal_count=len(ids),outside_invalid_ids=sorted(set(range(s,total))&invalid_a),
        max_score_A=qa,max_score_B=qb,max_ids_A=[i for i in ids if sa[i]==qa],max_ids_B=[i for i in ids if sb[i]==qb],
        cutoff_A=ta,cutoff_B=tb,gap_A=ga,gap_B=gb,e_O=e,delta_t=dt,slack_FP64=eps,
        required_strict_lower_bound=e+dt+eps,strict_gap=min(ga,gb)>e+dt+eps)


def build_certificate(models,records,x,mode,contracts):
    from c24_lif_v1_numerics import compare
    executions={name:execution(m,x,mode,r,c) for name,m,r,c in zip(EXECUTIONS,models,records,contracts)}
    stop=executions['combo_A']['region']['stop']
    comparisons={};state_comparisons={}
    for side,ci,pi in [('A',0,2),('B',1,3)]:
        combo,parent=shared_tensors(records[ci],stop),shared_tensors(records[pi],stop)
        comparisons[side]=[compare(combo[k],parent[k],k,0.,0.) for k in SHARED_KEYS]
        cs=path_states(models[ci],contracts[ci]['ancestors']);ps=path_states(models[pi],contracts[pi]['ancestors'])
        state_comparisons[side]={k:state_comparison(a,b) for k,a,b in zip(('shared_prefix','selection_state','nonzero_scca'),cs,ps)}
    return dict(kind=KIND,mode=mode,original_parent_commit=PARENT_COMMIT,public_source_sha256=SOURCE_SHA256,
        module_sha256=dict(scca=sha256(ROOT/'ultralytics-main/ultralytics/nn/modules/scca_aifi.py'),
                           lif=sha256(ROOT/'ultralytics-main/ultralytics/nn/modules/lif_down.py')),
        executions=executions,shared_comparisons=comparisons,shared_state_comparisons=state_comparisons,
        boundary={kind:boundary_facts(executions[kind+'_A'],executions[kind+'_B']) for kind in ('combo','parent')},
        outside={kind:outside_facts(executions[kind+'_A'],executions[kind+'_B']) for kind in ('combo','parent')},
        scope='Four independent natural forwards; same-side shared region only. Natural outputs may differ across fusion.')


def certificate_status(case,parent):
    """Recompute facts; status/comparability labels never supply proof."""
    from c24_lif_v1_acceptance import numerical_evidence_status
    cert=case.get('parent_p3_certificate')
    if not cert:return dict(status='REQUIRES_REVIEW',reason='MISSING_SHARED_P3_CERTIFICATE')
    if not parent:return dict(status='BLOCKED',reason='MISSING_PARENT_NUMERICS')
    try:
        require(numerical_evidence_status(case)=='PASSED' and numerical_evidence_status(parent)=='PASSED','Invalid combo/parent operator or replay evidence')
        require(cert['kind']==KIND and case['mode']==parent['mode']==cert['mode'] in ('amp','half'),'Certificate mode/kind mismatch')
        require(cert['original_parent_commit']==PARENT_COMMIT and cert['public_source_sha256']==SOURCE_SHA256 and
                cert['module_sha256']==dict(scca=SCCA_HASH,lif=LIF_HASH),'Original parent/module/source identity mismatch')
        ex=cert['executions'];require(set(ex)==set(EXECUTIONS),'Four natural executions required')
        env=ex['combo_A']['environment'];region=ex['combo_A']['region'];shapes=ex['combo_A']['shapes']
        require(len(shapes)==3 and all(len(s)==2 and all(type(v) is int and v>0 for v in s) for s in shapes),'Invalid feature shapes')
        require(region==dict(start=0,stop=math.prod(shapes[0]),total=sum(math.prod(s) for s in shapes)),'Region not derived from feature shapes')
        require(env['mode']==case['mode'] and env['input']['shape'][0]==1 and env['input']['finite'] is True,'Input identity missing')
        require(env['autocast'] is (case['mode']=='amp'),'Autocast identity mismatch')
        require(not env['matmul_tf32'] and not env['cudnn_tf32'] and not env['cudnn_benchmark'] and env['cudnn_deterministic'] and env['deterministic'],
                'Uncontrolled numerical backend')
        for name,e in ex.items():
            contract=e['contract'];kind='combo' if name.startswith('combo') else 'c24'
            require(e['shapes']==shapes and e['region']==region,'Four execution geometry mismatch')
            if e['environment']!=env:return dict(status='REQUIRES_REVIEW',reason='PARENT_INPUT_OR_ENVIRONMENT_DIFFERS',node=name)
            supported=(contract['kind']==kind and contract['architecture_sha256']==contract['expected_architecture_sha256'] and
                contract['original_architecture'] and contract['lif_outside_shared_path'] and contract['head_source_sha256']==HEAD_HASH and
                contract['native_head'] and contract['native_methods'] and contract['projections_tokenwise']==[True]*3 and
                contract['linear_norm_score_tokenwise'] and not contract['training'])
            if not supported:return dict(status='REQUIRES_REVIEW',reason='UNPROVEN_OR_CHANGED_TOKENWISE_PATH',node=name)
            require(contract['p3']==contract['topology']['decoder_inputs'][0] and contract['p3'] in contract['ancestors'] and
                    contract['topology']['p3_to_p4']['downsample'] not in contract['ancestors'] and contract['nc']==1 and
                    contract['num_queries']==len(e['ids'])==300,'Shared graph/query geometry changed')
            scores=e['scores'];total=region['total'];stop=region['stop']
            require(len(scores)==total and all(type(v) in (int,float) and math.isfinite(v) for v in scores),'Score shape/nonfinite')
            dtype={'torch.float16':torch.float16,'torch.float32':torch.float32}[e['full_logits']['dtype']]
            score_tensor=torch.tensor(scores,dtype=dtype).reshape(1,total,1)
            require(tensor_record(score_tensor)==e['full_logits'] and tensor_record(score_tensor[:,:stop])==e['tensors']['encoder_logits'],
                    'P3/full score hash or dtype mismatch')
            invalid=e['invalid_ids'];require(invalid==sorted(set(invalid)) and all(type(i) is int and 0<=i<total for i in invalid),'Invalid mask IDs')
            mask=torch.ones((1,total,1),dtype=torch.bool);mask[:,invalid]=False
            from init_c24_lif_v1 import RTDETRDecoder
            generation_dtype={'torch.float16':torch.float16,'torch.float32':torch.float32}[e['anchor_generation_dtype']]
            require(e['anchor_generation_dtype']==e['full_encoder_input']['dtype'],'Anchor construction dtype differs from natural features')
            _,geometry_mask=RTDETRDecoder._generate_anchors(shapes,dtype=generation_dtype,device='cpu')
            require(torch.equal(mask,geometry_mask),'Mask does not match original anchor geometry')
            dim=contract['hidden_dim'];shared_valid=int(mask[:,:stop].sum())
            expected_shapes=dict(p3=[1,dim,*shapes[0]],encoder_input=[1,stop,dim],encoder_features=[1,stop,dim],
                encoder_logits=[1,stop,1],valid_mask=[1,stop,1],anchors_valid=[shared_valid*4])
            require(dim==256 and all(e['tensors'][k]['shape']==v for k,v in expected_shapes.items()),'Shared feature geometry mismatch')
            require(e['anchors_valid']['shape']==[int(mask.sum())*4] and e['expected_anchors_valid']['shape']==e['anchors_valid']['shape'],
                    'Full anchor geometry mismatch')
            require(tensor_record(mask)==e['valid_mask']==e['expected_mask'] and tensor_record(mask[:,:stop])==e['tensors']['valid_mask'] and
                    e['invalid_anchor_count']==len(invalid) and e['invalid_anchors_positive_inf'] and
                    e['anchors_valid']==e['expected_anchors_valid'] and e['anchors_valid']['finite'],'Mask/anchor identity mismatch')
            ids=e['ids'];require(len(ids)==len(set(ids))==300 and all(type(i) is int and 0<=i<stop and i not in invalid for i in ids),'Selected ID outside valid shared region')
            threshold=min(scores[i] for i in ids)
            require(threshold>=max(scores[i] for i in range(total) if i not in set(ids)),'Recorded IDs not a natural top-k')
            target=case if kind=='combo' else parent;side='a' if name.endswith('_A') else 'b'
            require(target['candidate_ids_'+side]==[ids],'Natural gather IDs not bound to certificate')
            f=target['comparison_fingerprints']
            require(f['input_sha256']==env['input']['sha256'] and f['input_dtype']==env['input']['dtype'] and
                    f['device']==env['device'] and f['torch']==env['torch'] and f['encoder_input_'+side]==e['full_encoder_input']['sha256'],
                    'Natural execution fingerprints not bound to numerical report')
        for side in ('A','B'):
            c,p=ex['combo_'+side],ex['parent_'+side]
            require(case['candidate_ids_'+side.lower()]==parent['candidate_ids_'+side.lower()],'Full parent A/B candidate lists differ')
            for key in ('shared_prefix','selection_state','nonzero_scca'):
                a,b=c['contract'][key],p['contract'][key]
                require(a['finite'] and b['finite'] and a['tensors']==len(a['names'])>0 and b['tensors']==len(b['names'])>0,'Missing/invalid shared state')
                measured=cert['shared_state_comparisons'][side][key]
                if a!=b or measured!=dict(exact=True,names_equal=True,tensors=a['tensors'],first_difference=None):
                    return dict(status='REQUIRES_REVIEW',reason='SHARED_PARAMETER_OR_BN_DIFFERS',node=side+'/'+key,first_difference=measured)
            require(c['contract']['scca_output_nonzero'] and p['contract']['scca_output_nonzero'],'Nonzero SCCA pressure missing')
            comparisons=cert['shared_comparisons'][side]
            require([r['key'] for r in comparisons]==list(SHARED_KEYS),'Missing shared-region comparisons')
            for row in comparisons:
                k=row['key'];a,b=c['tensors'][k],p['tensors'][k]
                require(a['finite'] and b['finite'],'Shared tensor nonfinite')
                require(row['shape_a']==a['shape'] and row['shape_b']==b['shape'] and row['dtype_a']==a['dtype'] and row['dtype_b']==b['dtype'],
                        'Shared tensor shape/dtype mismatch')
                require(row.get('atol')==row.get('rtol')==0.,'Shared exactness tolerance changed')
                if a!=b or row['status']!='PASSED' or (row.get('exact') is not True and
                    (row.get('finite') is not True or row.get('over_tolerance')!=0 or row.get('max_abs')!=0)):
                    return dict(status='REQUIRES_REVIEW',reason='SHARED_P3_TENSOR_DIFFERS',node=side+'/'+k,first_difference=row)
        for kind in ('combo','parent'):
            a,b=ex[kind+'_A'],ex[kind+'_B']
            boundary=boundary_facts(a,b);outside=outside_facts(a,b)
            require(boundary==cert['boundary'][kind] and outside==cert['outside'][kind],'Recorded margin/gap facts do not recompute')
            if not boundary['within_band'] or not boundary['all_swaps_explained']:
                return dict(status='REQUIRES_REVIEW',reason='UNEXPLAINED_CUTOFF_EXCHANGE',node=kind)
            if not outside['strict_gap']:return dict(status='REQUIRES_REVIEW',reason='OUTSIDE_COMPETITOR_NOT_SEPARATED',node=kind,outside=outside)
        # Scores for ALL shared candidates are exact, hence exchange records must
        # match too. Whole-encoder perturbations remain intentionally different.
        require(cert['boundary']['combo']==cert['boundary']['parent'],'Parent/shared cutoff facts differ')
        return dict(status=WARNING,reason='ORIGINAL_C24_SHARED_P3_AND_OUTSIDE_SEPARATION_VERIFIED',
                    scope='Finite fusion acceptance with natural-output warning; no performance equivalence claim')
    except (RuntimeError,KeyError,TypeError,ValueError,IndexError,OverflowError) as error:
        return dict(status='BLOCKED',reason=str(error))


def certified_pair(combo_a,combo_b,parent_a,parent_b,x,mode):
    from c24_lif_v1_numerics import capture,compare_pair
    from c24_lif_v1_acceptance import numerical_status
    models=(combo_a,combo_b,parent_a,parent_b)
    contracts=[path_contract(m,k) for m,k in zip(models,('combo','combo','c24','c24'))]
    records=[capture(m,x,trace=True) for m in models]
    combo=compare_pair(combo_a,combo_b,x,mode,records[:2])
    parent=compare_pair(parent_a,parent_b,x,mode,records[2:])
    parent['scope']='Original C24 independently constructed from controlled public/seed-42 mapping'
    combo['parent_control']=parent
    if combo['operator_status']!='PASSED' or parent['operator_status']!='PASSED':
        combo.update(status='BLOCKED',parent_p3_verification=dict(status='BLOCKED',reason='NATURAL_OPERATOR_FAILURE'))
        return combo
    combo['parent_p3_certificate']=build_certificate(models,records,x,mode,contracts)
    combo['parent_p3_verification']=certificate_status(combo,parent)
    combo['status']=numerical_status(combo)
    return combo
