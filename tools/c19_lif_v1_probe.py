"""Pure tensor probe usable in fresh processes with historical ultralytics imports."""
from __future__ import annotations

import torch


def capture(model, image, batch=None):
    head=model.model[-1]; records={}; handles=[]
    def output(name):
        def hook(module, args, value): records[name]=value.detach().cpu().clone()
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
        records['cbr_query']=args[1].detach().cpu().clone()
        records['cbr_before']=args[2].detach().cpu().clone()
    def candidates(module,args,value):
        records['candidate_scores']=value.detach().cpu().clone()
        records['candidate_indices']=torch.topk(value.max(-1).values,head.num_queries,dim=1).indices.detach().cpu().clone()
    handles += [head.register_forward_pre_hook(inputs),head.decoder.register_forward_hook(decoder)]
    handles += [head.enc_score_head.register_forward_hook(candidates)]
    handles += [layer.register_forward_hook(output(f'query_layer_{i}')) for i,layer in enumerate(head.decoder.layers)]
    if hasattr(head,'cbr'): handles += [head.cbr.register_forward_pre_hook(cbr_inputs)]
    # Derive downsample from the finest decoder input rather than hard-coding 20.
    fine=head.f[0]
    for m in model.model:
        if hasattr(m,'conv') and getattr(m.conv,'stride',None)==(2,2) and m.i>fine:
            handles += [m.register_forward_hook(output('downsample'))];break
    try:
        value=model.predict(image,batch=batch)
        raw=value if model.training else value[1]
        records['boxes']=raw[0].detach().cpu().clone()
        records['scores']=raw[1].detach().cpu().clone()
        records['enc_boxes']=raw[2].detach().cpu().clone()
        records['enc_scores']=raw[3].detach().cpu().clone()
        records['dn_split']=raw[4]['dn_num_split'] if raw[4] is not None else None
        if hasattr(head,'cbr'):
            assert records['p3_pointer']==records['cbr_p3_pointer'], 'CBR did not receive slot0'
            assert torch.equal(records['cbr_query'],records['returned_query'])
            assert torch.equal(records['returned_query'],records[f'query_layer_{len(head.decoder.layers)-1}'])
            assert torch.equal(records['cbr_before'],records['raw_boxes'][-1])
            assert torch.equal(records['boxes'][:-1],records['raw_boxes'][:-1])
            assert torch.equal(records['scores'],records['raw_scores'])
        for k in ('p3_pointer','cbr_p3_pointer'):records.pop(k,None)
        return value,records
    finally:
        for handle in handles:handle.remove()


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
