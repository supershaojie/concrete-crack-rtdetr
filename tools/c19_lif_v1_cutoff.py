"""Evidence predicate shared by the finite checker and the real launch gate.

Score arithmetic is FP64 on already-produced logits. No model or selection calls.
"""
import math
import numpy as np

VERSION='c19_lif_fusion_v2'
WARNING='PASS_WITH_BASELINE_CUTOFF_TIE'


def need(condition,message):
    if not condition:raise ValueError(message)


def selection_from_scores(ia,ib,sa,sb,shapes,dtype_a,dtype_b):
    need(dtype_a==dtype_b and dtype_a in ('torch.float16','torch.float32','torch.float64'),'Score dtype mismatch')
    need(len(ia)==len(ib)==len(sa)==len(sb)>0,'Missing candidate batches')
    need(all(len(s)==2 and all(type(v) is int and v>0 for v in s) for s in shapes),'Invalid shapes')
    n=sum(h*w for h,w in shapes);rows=[];same=True;ordered=True
    scalar={'torch.float16':np.float16,'torch.float32':np.float32,'torch.float64':np.float64}[dtype_a]
    def ulp(value):
        v=scalar(value)
        with np.errstate(over='ignore',invalid='ignore'):
            neighbours=[float(np.nextafter(v,scalar(np.inf))),float(np.nextafter(v,scalar(-np.inf)))]
        return max(abs(x-value) for x in neighbours if math.isfinite(x))
    for image,(a,b,x,y) in enumerate(zip(ia,ib,sa,sb)):
        k=len(a)
        need(0<k<n and len(b)==k and len(x)==len(y)==n,'Candidate/score shape mismatch')
        need(all(type(v) is int and 0<=v<n for v in a+b),'Illegal candidate ID')
        need(len(set(a))==len(set(b))==k,'Duplicate candidate ID')
        need(all(type(v) in (int,float) and math.isfinite(v) for v in x+y),'Nonfinite logits')
        # Values must be exactly representable in the declared production dtype.
        need(all(float(scalar(v))==v for v in x+y),'Logits do not match declared production dtype')
        x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
        slack=8*np.finfo(np.float64).eps*max(float(np.abs(x).max()),float(np.abs(y).max()),np.finfo(np.float64).tiny)
        def cutoff(scores,ids):
            mask=np.ones(n,dtype=bool);mask[ids]=False
            kth=float(scores[ids].min());next_score=float(scores[mask].max());u=ulp(kth)
            return dict(k=k,kth=kth,k_plus_1=next_score,gap=kth-next_score,ulp=u,
                        tied_at_k=int((scores==kth).sum()),selected_tied_at_k=int((scores[ids]==kth).sum()),
                        within_one_ulp=int((np.abs(scores-kth)<=u).sum()),
                        native_order_valid=bool(np.all(scores[ids][:-1]>=scores[ids][1:]) and kth>=next_score))
        ca,cb=cutoff(x,a),cutoff(y,b)
        left=sorted(set(a)-set(b));right=sorted(set(b)-set(a));same &= not left and not right;ordered &= a==b
        exchanges=[]
        for i in left:
            for j in right:
                ma=float(x[i]-x[j]);mb=float(y[j]-y[i]);local=float(abs(x[i]-y[i])+abs(x[j]-y[j]))
                unit=max(ulp(float(v)) for v in (x[i],x[j],y[i],y[j]))
                boundary=max(abs(float(x[i])-ca['kth']),abs(float(x[j])-ca['kth']),
                             abs(float(y[i])-cb['kth']),abs(float(y[j])-cb['kth']))
                explained=ma>=-slack and mb>=-slack and ma<=local+slack and mb<=local+slack
                # Deliberately narrow: changed members lie within one original
                # FP16 ULP of each cutoff, not merely inside a broad rtol band.
                near=boundary<=unit+slack and ca['gap']<=ca['ulp']+slack and cb['gap']<=cb['ulp']+slack
                exchanges.append(dict(a_only=i,b_only=j,sA_i=float(x[i]),sA_j=float(x[j]),sB_i=float(y[i]),sB_j=float(y[j]),
                    margin_a=ma,margin_b=mb,e_local=local,slack=slack,local_ulp=unit,boundary_distance=boundary,
                    margin_explained=bool(explained),cutoff_band=bool(near)))
        rows.append(dict(image=image,natural_a=a,natural_b=b,changed_positions=sum(i!=j for i,j in zip(a,b)),
            a_only=left,b_only=right,common_count=len(set(a)&set(b)),score_disturbance_e=float(np.abs(x-y).max()),
            floating_slack=slack,exchanges=exchanges,margin_explained=all(e['margin_explained'] for e in exchanges),
            cutoff_a=ca,cutoff_b=cb,cutoff_gap_a=ca['gap'],cutoff_gap_b=cb['gap'],
            cutoff_eligible=bool(ca['native_order_valid'] and cb['native_order_valid'] and exchanges and
                                 all(e['margin_explained'] and e['cutoff_band'] for e in exchanges))))
    def coords(ids):
        result=[]
        for row in ids:
            mapped=[]
            for value in row:
                offset=value
                for level,(h,w) in enumerate(shapes):
                    if offset<h*w:mapped.append([value,level,offset//w,offset%w]);break
                    offset-=h*w
            result.append(mapped)
        return result
    return dict(kind='IDENTICAL' if ordered else ('PERMUTATION' if same else 'SET_DRIFT'),images=rows,shapes=shapes,
                scores_a=sa,scores_b=sb,score_dtype_a=dtype_a,score_dtype_b=dtype_b,statistics_dtype='float64',
                slack_formula='8 * eps(float64) * max(abs(produced logits), tiny(float64))',
                coordinates_a=coords(ia),coordinates_b=coords(ib))


def _selection(report):
    rows=report['images']
    actual=selection_from_scores([r['natural_a'] for r in rows],[r['natural_b'] for r in rows],
        report['scores_a'],report['scores_b'],report['shapes'],report['score_dtype_a'],report['score_dtype_b'])
    need(report==actual,'Selection statistics/IDs differ from saved logits')
    need(all(r['cutoff_a']['native_order_valid'] and r['cutoff_b']['native_order_valid'] for r in rows),'IDs violate natural top-k order')
    return actual


def _group(group,keys,atol,rtol,allow_row_mismatch=False,precision='fp32'):
    need(isinstance(group,dict) and set(group)==set(keys),'Required comparison tensors missing/extra')
    for key in keys:
        r=group[key]
        need(r.get('key')==key and r.get('atol')==atol and r.get('rtol')==rtol,'Comparison identity/tolerances changed: '+key)
        need(isinstance(r.get('shape'),list) and r['shape'] and all(type(v) is int and v>0 for v in r['shape']),'Invalid tensor shape: '+key)
        need(r.get('candidate_ids_available') is True and isinstance(r.get('dtype_a'),str) and isinstance(r.get('dtype_b'),str),'Missing trace dtype/IDs: '+key)
        expected='torch.bool' if key=='valid_mask' else ('torch.int64' if key=='candidate_indices' else None)
        if expected:need(r['dtype_a']==r['dtype_b']==expected,'Wrong semantic dtype: '+key)
        else:
            need(r['dtype_a'] in ('torch.float16','torch.float32') and r['dtype_b'] in ('torch.float16','torch.float32') and
                 (r['dtype_a']==r['dtype_b'] or precision=='amp'),'Wrong floating dtype: '+key)
            need(type(r.get('masked_nonfinite_count')) is int and r['masked_nonfinite_count']>=0 and
                 (key in ('anchors','selected_anchors') or r['masked_nonfinite_count']==0),'Invalid finite/mask evidence: '+key)
        for field in ('max_abs_error','allclose_failed_count'):
            v=r.get(field);need(type(v) in (int,float) and math.isfinite(v) and v>=0,'Invalid comparison metric: '+key+'.'+field)
        if not allow_row_mismatch:need(r['allclose_failed_count']==0 and not r.get('status'),'Continuous comparison failed: '+key)
        else:need(r.get('status') in (None,'FAILED_REAL_NUMERICAL_MISMATCH'),'Incomplete natural trace: '+key)


def _trace(trace,ids,native=False):
    need(trace.get('actual_gather_verified') is True and trace.get('topk_calls')==1,'Actual gather proof missing')
    need(trace.get('candidate_indices')==ids,'Replay consumed different candidate IDs')
    if native:need(trace.get('native_candidate_indices')==ids,'Natural selection used a forced candidate list')


def review_evidence(r):
    """Recompute acceptance; never accept a caller-supplied status string alone."""
    from c19_lif_v1_diagnostic import schema,PRE_KEYS,LIF_KEYS,MODES
    try:
        need(r.get('schema_version')==VERSION,'New dynamic diagnostic evidence required')
        need(r.get('device') in ('cpu','cuda') and r.get('precision') in ('fp32','half','amp'),'Unknown precision/device')
        atol,rtol=(2e-5,2e-4) if r['precision']=='fp32' else (3e-3,3e-2)
        need(r['tolerances']==dict(atol=atol,rtol=rtol),'Changed tolerance policy')
        expected_dtype='torch.float16' if r['precision']=='half' else 'torch.float32'
        def group(*args,**kwargs):return _group(*args,precision=r['precision'],**kwargs)
        need(r['precision_path']==dict(model_a_dtype=expected_dtype,model_b_dtype=expected_dtype,input_dtype=expected_dtype,
                                      autocast=r['precision']=='amp'),'Actual model/input precision path differs')
        meta=r['record_schema'];need(meta['mode']=='pair' and meta['layers']==3,'Unexpected combination/decoder schema')
        keys=schema(meta);pre=LIF_KEYS+PRE_KEYS;post=[k for k in keys if k not in pre+['candidate_indices']]
        invariants=r['fusion_invariants']
        need(invariants['lif_bn_retained'] is True and invariants['lif_state_exact'] is True and invariants['other_bn_fused'] is True,'LIF/fusion invariant failed')
        s=_selection(r['selection']);ia=[v['natural_a'] for v in s['images']];ib=[v['natural_b'] for v in s['images']]
        need(all(len(v)==300 for v in ia+ib),'Original query count changed')
        _trace(r['traces']['natural_a'],ia,True);_trace(r['traces']['natural_b'],ib,True)
        group(r['pre_selection'],pre,atol,rtol);group(r['natural_outputs'],post,atol,rtol,True)
        group(r['fixed_query_replay'],keys,atol,rtol)
        _trace(r['traces']['replay_A_a'],ia);_trace(r['traces']['replay_A_b'],ia)
        if s['kind']!='SET_DRIFT':
            group(r['common_id_alignment'],post,atol,rtol)
            if s['kind']=='IDENTICAL':group(r['natural_outputs'],post,atol,rtol)
            return dict(accepted=True,status='PASSED' if s['kind']=='IDENTICAL' else 'PASS_WITH_CANDIDATE_PERMUTATION',acceptance='PASSED')
        need(r['device']=='cuda' and r['precision']=='half','Cutoff exception is only for CUDA true-half')
        need(s['score_dtype_a']=='torch.float16','True-half cutoff requires actual half logits')
        need(all(v['cutoff_eligible'] for v in s['images'] if v['a_only']),'Changed member outside measured cutoff/ULP band')
        group(r['fixed_query_replay_B'],keys,atol,rtol)
        _trace(r['traces']['replay_B_a'],ib);_trace(r['traces']['replay_B_b'],ib)
        p=r['parent_selection'];mapping=p['mapping']
        need(mapping['status']=='PASSED' and mapping['common_states']==533 and mapping['all_shapes_exact'] is True and
             mapping['all_values_exact'] is True and mapping['source_dtype']=='torch.float32' and
             mapping['pair_cast_exact'] is True and mapping['same_input_rng_precision'] is True and
             mapping['fuse_order']=='float32_fuse_then_half' and len(mapping['source_state_sha256'])==64,'C2 common-state/source proof missing')
        ps=_selection(p['selection'])
        need(ps['kind']=='SET_DRIFT' and len(ps['images'])==len(s['images']),'C2 selection relation differs')
        for a,b in zip(s['images'],ps['images']):
            need(a['natural_a']==b['natural_a'] and a['natural_b']==b['natural_b'],'C2 complete natural ID sequences differ')
            need(not b['a_only'] or b['cutoff_eligible'],'C2 cutoff evidence failed')
            # Same discrete boundary condition, not a different parent drift.
            for side in ('cutoff_a','cutoff_b'):
                need(a[side]['tied_at_k']==b[side]['tied_at_k'] and a[side]['gap']==b[side]['gap'],'C2 cutoff tie/gap differs')
        parent_keys=schema(dict(mode='C2',layers=3));parent_post=[k for k in parent_keys if k not in PRE_KEYS+['candidate_indices']]
        group(p['pre_selection'],PRE_KEYS,atol,rtol);group(p['natural_outputs'],parent_post,atol,rtol,True)
        for side,ids in [('A',ia),('B',ib)]:
            group(p['replay_'+side],parent_keys,atol,rtol)
            _trace(p['traces']['replay_'+side+'_a'],ids);_trace(p['traces']['replay_'+side+'_b'],ids)
        _trace(p['traces']['natural_a'],ia,True);_trace(p['traces']['natural_b'],ib,True)
        need(r.get('natural_status')=='NATURAL_SELECTION_DRIFT','Natural drift disclosure missing')
        return dict(accepted=True,status=WARNING,acceptance='ACCEPTED_WITH_WARNING')
    except (KeyError,ValueError,TypeError,IndexError,OverflowError) as error:
        return dict(accepted=False,status='REQUIRES_REVIEW',acceptance='BLOCKED',reason=str(error))


def fusion_accepted(report,device=None,precision=None):
    result=review_evidence(report)
    return (result['accepted'] and report.get('status')==result['status'] and report.get('acceptance')==result['acceptance'] and
            report.get('operator_status')=='OPERATOR_CHECK_PASSED' and not report.get('failure') and
            (device is None or report.get('device')==device) and (precision is None or report.get('precision')==precision))
