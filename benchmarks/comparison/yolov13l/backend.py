"""Observe the original attention operators; never substitute attention arithmetic."""
from contextlib import contextmanager
from collections import Counter
import math
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class ArithmeticAudit(TorchDispatchMode):
    def __init__(self, require_fp32=False):
        super().__init__()
        self.require_fp32 = require_fp32
        self.operations = Counter()
        self.dtypes = Counter()
        self.macs = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func)
        self.operations[name] += 1
        # Covers attention q/k products, value mixing, HyperACE bmm, and softmax/exp.
        observed = any(x in name for x in ('aten.bmm.', 'aten.mm.', 'aten.addmm.',
                                           'aten.exp.', 'softmax', 'aten.convolution.'))
        if observed:
            tensors = [v for v in args if isinstance(v, torch.Tensor)]
            tensors += [out] if isinstance(out, torch.Tensor) else []
            dtypes = {str(t.dtype) for t in tensors if t.is_floating_point()}
            self.dtypes[name+'|'+','.join(sorted(dtypes))] += 1
            if self.require_fp32 and dtypes != {'torch.float32'}:
                raise RuntimeError('Non-FP32 internal arithmetic: '+name+' '+str(dtypes))
        if name == 'aten.bmm.default':
            a, b = args[:2]
            self.macs['bmm_attention_and_hypergraph'] += a.shape[0]*a.shape[1]*a.shape[2]*b.shape[2]
        elif name in ('aten.mm.default', 'aten.addmm.default'):
            a,b = args[:2] if name == 'aten.mm.default' else args[1:3]
            self.macs['mm_or_addmm'] += a.shape[0]*a.shape[1]*b.shape[1]
        elif name == 'aten.convolution.default':
            weight = args[1]
            if args[6]:  # transposed convolutions are not part of this model
                raise RuntimeError('FLOP counter needs an explicit transposed-convolution rule')
            self.macs['convolution'] += out.numel()*weight.shape[1]*math.prod(weight.shape[2:])
        return out

    def report(self):
        if not any('aten.bmm.' in k for k in self.dtypes):
            raise RuntimeError('No attention matrix arithmetic was observed')
        return {'backend':'official_native_matmul_stable_softmax', 'required_fp32':self.require_fp32,
                'observed_dtypes':dict(self.dtypes), 'MACs_by_operator':dict(self.macs),
                'covered_GFLOPs_2x_MACs':2*sum(self.macs.values())/1e9,
                'observed_operator_calls':dict(self.operations),
                'flops_scope':'Conv/Linear and all executed mm/bmm products, including AAttn and HyperACE',
                'flops_exclusions':'Bias adds, BN, activation, exp/softmax, reductions, pooling, residual/gate/elementwise arithmetic, interpolation, decode and NMS; not a claim of complete FLOPs'}


@contextmanager
def native_fp32():
    from ultralytics.nn.modules import block
    if block.USE_FLASH_ATTN:
        raise RuntimeError('This frozen experiment requires the original native branch')
    with torch.autocast('cuda', enabled=False), torch.autocast('cpu', enabled=False):
        yield

def optional_flash_parity():
    """Compare both original branches on tiny modules only when Flash is actually importable."""
    from copy import deepcopy
    from ultralytics.nn.modules import block
    if not torch.cuda.is_available() or not hasattr(block,'flash_attn_func'):
        return {'status':'NOT_VERIFIED','reason':'Official Flash import/device path unavailable; native remains frozen'}
    rows=[]
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        for area in (1,4):
            module=block.AAttn(64,2,area=area).float().cuda().eval()
            x=torch.randn(1,64,8,8,device='cuda',requires_grad=True)
            block.USE_FLASH_ATTN=False
            ref=module(x);ref_grads=torch.autograd.grad(ref.square().mean(),[x,*module.parameters()])
            other=deepcopy(module);z=x.detach().clone().requires_grad_(True)
            try:
                block.USE_FLASH_ATTN=True
                output=other(z);grads=torch.autograd.grad(output.square().mean(),[z,*other.parameters()])
                output_ok=torch.allclose(ref,output,rtol=.03,atol=.003)
                grads_ok=all(torch.allclose(a,b,rtol=.05,atol=.003) for a,b in zip(ref_grads,grads))
                rows.append({'area':area,'output_allclose':output_ok,'gradients_allclose':grads_ok,
                             'output_max_abs_error':float((ref-output).abs().max()),
                             'gradient_max_abs_error':max(float((a-b).abs().max()) for a,b in zip(ref_grads,grads))})
            except Exception as exc:
                rows.append({'area':area,'status':'NOT_VERIFIED','error':str(exc)})
            finally:
                block.USE_FLASH_ATTN=False
    return {'status':'MEASURED','output_tolerance':{'rtol':.03,'atol':.003},
            'gradient_tolerance':{'rtol':.05,'atol':.003},'rows':rows,
            'scope':'tiny AAttn output/input+parameter gradients; not proof of full-network equivalence',
            'frozen_training_backend':'native'}
