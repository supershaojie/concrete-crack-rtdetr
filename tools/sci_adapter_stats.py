"""Read-only SCI diagnostics: one training batch per epoch, no residual cap."""
from sci_adapter import SCIAdapter, torch, require


@torch.no_grad()
def statistics(module,source,output):
    x,y=source.detach().float(),output.detach().float();delta=y-x
    rms=lambda t:float(t.square().mean().sqrt())
    moments=lambda p:dict(mean=float(p.float().mean()),std=float(p.float().std(unbiased=False)))
    axes=(0,2,3)
    mean_shift=y.mean(axes)-x.mean(axes)
    std_shift=y.std(axes,unbiased=False)-x.std(axes,unbiased=False)
    report=dict(gn_scale=moments(module.norm.weight),gn_bias=moments(module.norm.bias),
        reduce_weight_RMS=rms(module.reduce.weight),restore_weight_RMS=rms(module.restore.weight),
        residual_RMS=rms(delta),Y4_RMS=rms(x),residual_to_Y4_RMS=rms(delta)/(rms(x)+1e-6),
        residual_mean_abs=float(delta.abs().mean()),channel_mean_shift=mean_shift.tolist(),
        channel_std_shift=std_shift.tolist(),channel_mean_shift_mean_abs=float(mean_shift.abs().mean()),
        channel_std_shift_mean_abs=float(std_shift.abs().mean()),shape=list(x.shape),dtype=str(source.dtype))
    require(torch.isfinite(delta).all(),'Nonfinite diagnostic residual')
    return report


def install(model,folder):
    """Attach to trainer-created model; reset observation at each training epoch."""
    import json
    state=dict(armed=False,epoch=None,record=None)
    def begin(trainer):state.update(armed=True,epoch=trainer.epoch,record=None)
    def setup(trainer):
        modules=[m for m in trainer.model.modules() if isinstance(m,SCIAdapter)]
        require(len(modules)==1,'Expected one SCI observer')
        def observe(m,args,out):
            if state['armed'] and m.training:
                state['record']=dict(epoch=state['epoch'],sampling='first training batch each epoch',
                                     **statistics(m,args[0],out))
                state['armed']=False
        modules[0].register_forward_hook(observe)
    def end(trainer):
        require(state['record'] is not None,'SCI training diagnostic missing')
        with (folder/'sci_adapter_stats.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(state['record'],allow_nan=False)+'\n')
    model.add_callback('on_train_start',setup)
    model.add_callback('on_train_epoch_start',begin)
    model.add_callback('on_train_epoch_end',end)
