"""Measured single-class resources; counted operator scope is always explicit."""
from __future__ import annotations
from copy import deepcopy
import statistics
import subprocess
import time
from support import configure, read_json, sha256, validate_checkpoint, write_json


def measure_model(args):
    source=configure(args.output.parent/'measure_runtime',args.source)
    import torch
    from torch.profiler import profile, ProfilerActivity
    from run import load_initial_model
    from adapters import model_identity
    torch.set_num_threads(2)
    model,initialization=load_initial_model(args.source,args.weights)
    model=model.float().cpu().eval()
    values={}
    for name,net in [('full_training_structure_unfused',model),
                     ('official_deployment_fused',deepcopy(model).fuse(verbose=False))]:
        x=torch.zeros(1,3,640,640)
        with torch.no_grad():
            net(x)  # establish dynamic anchor cache outside measurement
            with profile(activities=[ProfilerActivity.CPU],record_shapes=True,with_flops=True) as prof:
                prediction=net(x)
        measured={e.key:{'FLOPs':e.flops,'calls':e.count} for e in prof.key_averages() if e.flops}
        uncounted={e.key:e.count for e in prof.key_averages() if not e.flops and e.key.startswith('aten::')}
        total=sum(e['FLOPs'] for e in measured.values())
        values[name]={'parameters':sum(p.numel() for p in net.parameters()),
            'trainable_parameters_at_measurement':sum(p.numel() for p in net.parameters() if p.requires_grad),
            'GFLOPs_counted_ops':total/1e9,'MACs_equivalent_FLOPs_div_2':total/2,
            'counted_operators':measured,'operators_without_FLOP_formula':uncounted,
            'complete_total_GFLOPs':None,
            'scope':'forward including both heads when unfused; no backward, optimizer or loss; profiling covers conv/matmul/bmm and supported pointwise add/mul only',
            'uncounted_arithmetic':'BN, activation, softmax, division, reductions, decoding and top-k/ranking have no complete profiler FLOP formula; not silently counted as zero',
            'output_shape':list(prediction[0].shape),'candidates_P3_P4_P5':[6400,1600,400],
            'native_one_to_one_topk_limit':300,'identity':model_identity(net,1,deployed=name.endswith('_fused'))}
    result={'source':source,'nc':1,'input':[1,3,640,640],'precision':'FP32','device':'CPU',
        'FLOPs_convention':'conv/matmul multiply-add=2; pointwise add/mul use profiler formula; MACs equivalent=FLOPs/2',
        'method':'torch.profiler with_flops with every unsupported operator reported; counted FLOPs are a partial lower bound',
        'initialization':initialization,'measurements':values,
        'speed':'not measured here; exclusive-hardware speed remains NOT_RUN'}
    write_json(args.output,result)
    return result


