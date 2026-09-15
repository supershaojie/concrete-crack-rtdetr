"""Focused algebra, real-kernel block, nonzero calibration and serialization checks."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from init_sfrd_v1 import require, SITES
from sfrd_v1_svd import factorize, kernel, error
from ultralytics.nn.modules.sfr_d import SpatialFactorConv
from ultralytics.utils.torch_utils import ModelEMA


def close(a, b, rtol=1e-5, atol=1e-6):
    require(torch.isfinite(a).all() and torch.isfinite(b).all(), 'Nonfinite comparison')
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    return dict(max_abs_error=float((a-b).abs().max()), rtol=rtol, atol=atol)


def activate_gate(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, SpatialFactorConv) and hasattr(m, 'gate'):
                for name, p in m.gate.named_parameters():
                    p.copy_(torch.linspace(-.015, .02, p.numel(), device=p.device, dtype=p.dtype).reshape_as(p))


def module_checks():
    torch.manual_seed(317)
    w = torch.randn(3, 3, 3, 3, dtype=torch.float64)
    results = {}
    for rank in (4, 9):
        a, b, _ = factorize(w, rank)
        reconstructed = kernel(a, b)
        if rank == 9: results['full_rank_kernel'] = close(w, reconstructed, 1e-11, 1e-12)
        for h, width in ((5, 7), (1, 1), (1, 6), (8, 1)):
            x = torch.randn(2, 3, h, width, dtype=torch.float64)
            y = F.conv2d(F.conv2d(x, a, padding=(0,1)), b, padding=(1,0))
            results[f'r{rank}_{h}x{width}'] = close(y, F.conv2d(x,reconstructed,padding=1), 1e-11, 1e-12)
    for nonzero in (False, True):
        m = SpatialFactorConv(8, 5, nn.BatchNorm2d(8), True).double().eval()
        if nonzero: activate_gate(m)
        x = torch.randn(2, 8, 7, 9, dtype=torch.float64, requires_grad=True)
        before = x.detach().clone()
        z = m.A_1x3(x); z.retain_grad(); saved_z = z.detach().clone()
        g = 1+.5*torch.tanh(m.gate.dw_h(z)+m.gate.dw_v(z))
        explicit = m.norm(m.B_3x1(z*g))
        actual = m(x)
        row = dict(formula=close(actual,explicit,1e-12,1e-12), gain_shape=list(g.shape),
                   gain_min=float(g.min()), gain_max=float(g.max()))
        require(g.shape == z.shape and (g >= .5).all() and (g <= 1.5).all(), 'Gain shape/range')
        if nonzero:
            require(g.var(dim=(-2,-1)).min() > 0 and g.var(dim=1).mean() > 0, 'Nonzero gate is not spatial/channel dependent')
        else:
            require(torch.equal(g,torch.ones_like(g)), 'Zero gate is not one')
            row['pure_sfr'] = close(actual,m.norm(m.B_3x1(z)),1e-12,1e-12)
        (explicit * torch.randn_like(explicit)).mean().backward()
        names = ['A_1x3.weight','B_3x1.weight','gate.dw_h.weight','gate.dw_v.weight','gate.dw_v.bias']
        row['gradient_norms'] = {k:float(dict(m.named_parameters())[k].grad.norm()) for k in names}
        require(all(v > 0 for v in row['gradient_norms'].values()) and all(torch.isfinite(p.grad).all() for p in m.parameters()), 'Missing/nonfinite genuine path gradient')
        require(torch.equal(x.detach(),before) and torch.equal(z.detach(),saved_z), 'Input/intermediate mutated')
        row['fuse'] = close(m(x.detach()),deepcopy(m).fuse()(x.detach()),1e-11,1e-12)
        results['nonzero' if nonzero else 'zero'] = row
    # Cache mismatch/zero-kernel behavior is validated without writing fake real-source artifacts.
    a,b,s = factorize(torch.zeros(2,2,3,3),2)
    require(torch.count_nonzero(a)+torch.count_nonzero(b)+torch.count_nonzero(s) == 0,'Zero source mishandled')
    results['zero_kernel_synthetic'] = 'PASSED; no real-source factor file created'
    return results


def real_blocks(pair, models):
    rows = {}
    for i, (c, r) in SITES.items():
        original = deepcopy(pair.model[i].blocks[1]).eval()
        block = deepcopy(models['cbr_lif_sfrd_v1'].model[i].blocks[1]).eval()
        control = deepcopy(models['cbr_lif_sfr_control_v1'].model[i].blocks[1]).eval()
        x = torch.randn(1,c,40 if i==6 else 20,40 if i==6 else 20)
        with torch.no_grad():
            baseline, y = original(x), block(x)
            latent = block.branch2b.A_1x3(block.branch2a(x))
            pure = control(x)
            row = dict(input=list(x.shape), latent=list(latent.shape), output=list(y.shape),
                zero_vs_control=close(y,pure), original_output_error=error(baseline.double(),y.double()),
                initial_fusion=block_fusion(block,x), interpretation='Controlled Gaussian input, not detection accuracy')
            a=block.branch2b.A_1x3.weight.double();b=block.branch2b.B_3x1.weight.double()
            h1=block.branch2a(x).double()
            row['real_factor_padding_fp64']=close(F.conv2d(F.conv2d(h1,a,padding=(0,1)),b,padding=(1,0)),
                F.conv2d(h1,kernel(a,b),padding=1),1e-11,1e-12)
            activate_gate(block)
            row['nonzero_fusion'] = block_fusion(block,x)
        rows[str(i)] = row
    return rows


def block_fusion(block,x):
    """Prove the fold in FP64 and quantify FP32 cancellation with an arithmetic bound."""
    fused=deepcopy(block);fused.branch2b.fuse()
    double=deepcopy(block).double();double_fused=deepcopy(double);double_fused.branch2b.fuse()
    a,b=block(x),fused(x)
    proof=close(double(x.double()),double_fused(x.double()),1e-11,1e-12)
    path=block.branch2b;z=path.A_1x3(block.branch2a(x))
    if hasattr(path,'gate'):z=path.gate(z)
    # Dot-product summation bound gamma_n = n*u/(1-n*u), n=3*r.
    # Include coefficient-rounding error from the actual stored fused FP32 weights/bias.
    norm=path.norm;scale=norm.weight.double()/(norm.running_var.double()+norm.eps).sqrt()
    exact_w=path.B_3x1.weight.double()*scale[:,None,None,None]
    exact_bias=norm.bias.double()-norm.running_mean.double()*scale
    folded=fused.branch2b.B_3x1
    magnitude=F.conv2d(z.double().abs(),exact_w.abs(),padding=(1,0))
    coefficient_error=F.conv2d(z.double().abs(),(folded.weight.double()-exact_w).abs(),padding=(1,0))
    coefficient_error+=(folded.bias.double()-exact_bias).abs()[None,:,None,None]
    u=torch.finfo(torch.float32).eps/2;n=3*z.shape[1];gamma=n*u/(1-n*u)
    bound=2*gamma*magnitude+coefficient_error+16*u*(magnitude+exact_bias.abs()[None,:,None,None]+x.double().abs())
    delta=(a.double()-b.double()).abs()
    require((delta<=bound).all(),'FP32 fusion error exceeds dot-product/BN roundoff bound')
    return dict(status='PASSED',fp64_fold=proof,fp32_max_abs=float(delta.max()),
        preliminary_rtol=1e-5,preliminary_atol=1e-6,preliminary_allclose=bool(torch.allclose(a,b,rtol=1e-5,atol=1e-6)),
        arithmetic_bound='2*gamma_(3r)*conv(abs(Z),abs(B_scaled)) + stored_coefficient_error + 16*u*(magnitude+abs(bias)+abs(X)); u=eps32/2',
        max_error_to_bound=float((delta/bound.clamp_min(torch.finfo(torch.float64).tiny)).max()),
        interpretation='FP32 BN folding changes accumulation/cancellation; FP64 identity and arithmetic bound both pass. Whole-network tolerances remain unchanged.')


def ema_check(model):
    source = deepcopy(model); activate_gate(source)
    ema = ModelEMA(source)
    require(set(ema.ema.state_dict()) == set(source.state_dict()), 'EMA structure changed')
    require(all(torch.equal(v,ema.ema.state_dict()[k]) for k,v in source.state_dict().items()), 'EMA initial state differs')
    with torch.no_grad():
        for name,p in source.named_parameters():
            if '.gate.' in name: p.add_(.001)
    ema.update(source)
    require(all(torch.isfinite(v).all() for v in ema.ema.state_dict().values()), 'Nonfinite EMA')
    require(all(torch.count_nonzero(v)>0 for k,v in ema.ema.state_dict().items() if '.gate.' in k), 'EMA gate reset')
    return dict(status='PASSED', states=len(ema.ema.state_dict()), updates=ema.updates, learned_gate_preserved=True)
