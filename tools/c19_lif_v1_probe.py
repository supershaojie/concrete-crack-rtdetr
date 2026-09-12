"""Pure tensor probe usable in fresh processes with historical ultralytics imports."""
from __future__ import annotations

import torch


def _capture(model, image, batch, fixed_ids, records, handles, patches):
    """Trace the native gather, without a second topk or a production code change.

    Fixed IDs affect only the one validated encoder selection call. Both features
    and anchors still come from this model. All instrumentation is scoped here.
    """
    from types import SimpleNamespace
    head=model.model[-1]
    active=False; calls=0; score_tensor=None; feature_tensor=None; consumed=None; bbox_delta=None
    records['mode']='pair' if hasattr(head,'cbr') else 'C2'
    records['layers']=len(head.decoder.layers)
    def save(name, value): records[name]=value.detach().cpu().clone()
    def replace(obj,name,value):
        patches.append((obj,name,name in obj.__dict__,obj.__dict__.get(name)))
        setattr(obj,name,value)
    def output(name):
        def hook(module, args, value): save(name,value)
        return hook
    def inputs(module,args):
        for i,v in enumerate(args[0]): records[f'scale_{i}']=v.detach().cpu().clone()
        records['p3_pointer']=args[0][0].data_ptr()
    def decoder(module,args,value):
        records['raw_boxes']=value[0].detach().cpu().clone()
        records['raw_scores']=value[1].detach().cpu().clone()
        if len(value)==3:
            records['returned_query']=value[2].detach().cpu().clone()
    def cbr_inputs(module,args):
        records['cbr_p3_pointer']=args[0].data_ptr()
        save('cbr_p3',args[0])
        records['cbr_query']=args[1].detach().cpu().clone()
        records['cbr_before']=args[2].detach().cpu().clone()
    def cbr_output(module,args,value):
        save('cbr_after',value[0] if isinstance(value,tuple) else value)
    def candidates(module,args,value):
        nonlocal score_tensor
        score_tensor=value
        save('candidate_scores',value)
    def features(module,args,value):
        nonlocal feature_tensor
        feature_tensor=value
        save('encoder_features',value)
    def selected(module,args):
        if consumed is None or feature_tensor is None:
            raise RuntimeError('Incomplete trace: enc_bbox_head ran before actual encoder topk')
        expected=feature_tensor.gather(1,consumed.unsqueeze(-1).expand(-1,-1,feature_tensor.shape[-1]))
        if not torch.equal(args[0],expected):
            raise RuntimeError('Incomplete trace: native gather did not consume captured IDs')
        save('selected_features',args[0])
        records['actual_gather_verified']=True
    def selected_delta(module,args,value):
        nonlocal bbox_delta
        bbox_delta=value
        save('selected_bbox_delta',value)
    original_topk=torch.topk
    def actual_topk(input,k,dim=None,*args,**kwargs):
        nonlocal calls,consumed
        if not active:return original_topk(input,k,dim,*args,**kwargs)
        calls+=1
        if calls!=1 or score_tensor is None or dim!=1 or k!=head.num_queries or not torch.equal(input,score_tensor.max(-1).values):
            raise RuntimeError('Incomplete trace: unexpected topk scope/count/input/shape')
        result=original_topk(input,k,dim,*args,**kwargs)
        save('native_candidate_indices',result.indices)
        if fixed_ids is not None:
            ids=fixed_ids.to(input.device)
            if ids.dtype!=torch.int64 or ids.shape!=result.indices.shape or (ids<0).any() or (ids>=input.shape[1]).any():
                raise RuntimeError('Invalid replay candidate IDs')
            if any(len(row.unique())!=k for row in ids):raise RuntimeError('Duplicate replay candidate IDs')
            result=SimpleNamespace(values=input.gather(1,ids),indices=ids)
        consumed=result.indices.detach().clone()
        save('candidate_indices',consumed)
        return result
    original_encoder=head._get_encoder_input
    def encoder_input(x):
        value=original_encoder(x)
        save('flattened_features',value[0]);records['shapes']=[list(s) for s in value[1]]
        return value
    original_decoder_input=head._get_decoder_input
    def decoder_input(*args,**kwargs):
        nonlocal active
        active=True
        try:value=original_decoder_input(*args,**kwargs)
        finally:active=False
        if calls!=1 or not records.get('actual_gather_verified'):
            raise RuntimeError('Incomplete trace: missing native selection/gather')
        save('anchors',head.anchors);save('valid_mask',head.valid_mask)
        anchors=head.anchors.expand(consumed.shape[0],-1,-1).gather(1,consumed.unsqueeze(-1).expand(-1,-1,4))
        save('selected_anchors',anchors)
        save('reference_boxes',value[1]);save('decoder_embeddings',value[0])
        if bbox_delta is None or not torch.equal(value[1][:,-head.num_queries:],bbox_delta+anchors):
            raise RuntimeError('Incomplete trace: actual reference boxes/anchor gather disagree')
        scores=score_tensor.gather(1,consumed.unsqueeze(-1).expand(-1,-1,score_tensor.shape[-1]))
        if not torch.equal(scores,value[3]):raise RuntimeError('Incomplete trace: selected scores/IDs disagree')
        return value
    replace(torch,'topk',actual_topk)
    replace(head,'_get_encoder_input',encoder_input)
    replace(head,'_get_decoder_input',decoder_input)
    handles += [head.register_forward_pre_hook(inputs),head.decoder.register_forward_hook(decoder)]
    handles += [head.enc_score_head.register_forward_hook(candidates)]
    handles += [head.enc_output.register_forward_hook(features),head.enc_bbox_head.register_forward_pre_hook(selected),
                head.enc_bbox_head.register_forward_hook(selected_delta)]
    handles += [m.register_forward_hook(output(f'projection_{i}')) for i,m in enumerate(head.input_proj)]
    handles += [layer.register_forward_hook(output(f'query_layer_{i}')) for i,layer in enumerate(head.decoder.layers)]
    if hasattr(head,'cbr'):
        handles += [head.cbr.register_forward_pre_hook(cbr_inputs),head.cbr.register_forward_hook(cbr_output),
                    head.cbr.offset_out.register_forward_hook(output('cbr_offsets'))]
    # Derive downsample from the finest decoder input rather than hard-coding 20.
    fine=head.f[0]
    for m in model.model:
        if hasattr(m,'conv') and getattr(m.conv,'stride',None)==(2,2) and m.i>fine:
            handles += [m.register_forward_hook(output('downsample'))]
            if hasattr(m,'residual'):
                records['mode']='pair' if hasattr(head,'cbr') else 'LIF'
                residual=m.residual
                def traced_residual(x):
                    value=residual(x);save('lif_residual',value);return value
                replace(m,'residual',traced_residual)
                handles += [m.register_forward_pre_hook(lambda mod,a:save('lif_input',a[0])),
                            m.conv.register_forward_hook(output('lif_conv')),
                            m.bn.register_forward_pre_hook(lambda mod,a:save('lif_pre_bn',a[0])),
                            m.bn.register_forward_hook(output('lif_post_bn'))]
            elif hasattr(head,'cbr'):records['mode']='C19'
            break
    value=model.predict(image,batch=batch)
    raw=value if model.training else value[1]
    records['boxes']=raw[0].detach().cpu().clone()
    records['scores']=raw[1].detach().cpu().clone()
    records['enc_boxes']=raw[2].detach().cpu().clone()
    records['enc_scores']=raw[3].detach().cpu().clone()
    records['dn_split']=raw[4]['dn_num_split'] if raw[4] is not None else None
    if hasattr(head,'cbr'):
        records['cbr_residual']=records['cbr_after']-records['cbr_before']
        assert records['p3_pointer']==records['cbr_p3_pointer'], 'CBR did not receive slot0'
        assert torch.equal(records['cbr_query'],records['returned_query'])
        assert torch.equal(records['returned_query'],records[f'query_layer_{len(head.decoder.layers)-1}'])
        assert torch.equal(records['cbr_before'],records['raw_boxes'][-1])
        assert torch.equal(records['boxes'][:-1],records['raw_boxes'][:-1])
        assert torch.equal(records['scores'],records['raw_scores'])
    for k in ('p3_pointer','cbr_p3_pointer'):records.pop(k,None)
    records['topk_calls']=calls
    return value,records


def capture(model, image, batch=None, fixed_ids=None):
    records={};handles=[];patches=[]
    try:
        return _capture(model,image,batch,fixed_ids,records,handles,patches)
    except BaseException as error:
        error.trace_records=records
        raise
    finally:
        # Also covers exceptions during hook installation or trace validation.
        for handle in handles:handle.remove()
        for obj,name,existed,previous in reversed(patches):
            if existed:setattr(obj,name,previous)
            elif name in obj.__dict__:delattr(obj,name)


def targets(batch):
    return dict(cls=batch['cls'].long().flatten(),bboxes=batch['bboxes'],batch_idx=batch['batch_idx'].long(),
                gt_groups=[int((batch['batch_idx']==i).sum()) for i in range(len(batch['img']))])


if __name__=='__main__':
    import argparse,sys,hashlib
    from pathlib import Path
    parser=argparse.ArgumentParser()
    for key in ('root','payload','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(Path(args.root)/'ultralytics-main'))
    from ultralytics.nn.tasks import RTDETRDetectionModel
    from ultralytics.utils.patches import torch_load
    import ultralytics
    assert Path(ultralytics.__file__).resolve().is_relative_to(Path(args.root).resolve())
    torch.set_num_threads(4);torch.manual_seed(42)
    payload=torch_load(args.payload,map_location='cpu')
    model=RTDETRDetectionModel(payload['yaml'],nc=80,verbose=False)
    fresh={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
    model=RTDETRDetectionModel(payload['yaml'],nc=1,verbose=False)
    model.load_state_dict(payload['state'],strict=True); model.eval()
    with torch.no_grad(): _,evaluation=capture(model,payload['image'])
    model.nc=1;model.train();torch.manual_seed(123)
    prediction,training=capture(model,payload['batch']['img'],targets(payload['batch']))
    loss=model.loss(payload['batch'],preds=prediction)[0];loss.backward()
    gradients={k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None}
    torch.save(dict(fresh=fresh,eval=evaluation,train=training,loss=float(loss),gradients=gradients,
                    source=ultralytics.__file__),args.output)
