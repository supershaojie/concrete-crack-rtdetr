"""CPU float64 spatial SVD, canonical signs, immutable reusable FP32 factors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from init_c19_lif_v1 import SOURCE_SHA256, require, sha256, write_json
from ultralytics.utils.patches import torch_load

ALGORITHM = 'spatial_oh_iw_cpu_f64_canonical_u_pivot_v1'
SITES = {6: (256, 256), 7: (512, 384)}


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def factorize(weight, rank):
    c = weight.shape[0]
    require(tuple(weight.shape) == (c, c, 3, 3) and 0 < rank <= 3*c, 'Invalid W/rank')
    w = weight.detach().to(device='cpu', dtype=torch.float64)
    require(torch.isfinite(w).all(), 'Nonfinite source W')
    matrix = w.permute(0, 2, 1, 3).contiguous().reshape(3*c, 3*c)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    require(all(torch.isfinite(t).all() for t in (u, s, vh)) and (s >= 0).all(), 'Invalid SVD')
    pivot = u[u.abs().argmax(dim=0), torch.arange(u.shape[1])]
    sign = torch.where(pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot))
    before = (u * s.unsqueeze(0)) @ vh
    u, vh = u * sign.unsqueeze(0), vh * sign.unsqueeze(1)
    after = (u * s.unsqueeze(0)) @ vh
    require(torch.equal(before, after), 'Canonical sign altered the decomposition')
    torch.testing.assert_close(after, matrix, rtol=1e-11, atol=1e-12)
    sqrt_s = s[:rank].sqrt()
    a = (sqrt_s.unsqueeze(1) * vh[:rank]).reshape(rank, c, 1, 3).contiguous()
    b = (u[:, :rank] * sqrt_s.unsqueeze(0)).reshape(c, 3, rank).permute(0, 2, 1).unsqueeze(-1).contiguous()
    return a, b, s


def kernel(a, b):
    return torch.einsum('ork,riq->oikq', b[..., 0], a[:, :, 0, :])


def error(reference, approximation):
    absolute = float((reference - approximation).norm())
    denominator = float(reference.norm())
    return dict(absolute_error=absolute, reference_norm=denominator,
                relative_error=absolute / denominator if denominator else None,
                status='PASSED' if denominator else 'ZERO_REFERENCE_NORM')


def factors(reference, cache):
    """Use one hash-checked file for all variants; never overwrite a prior cache."""
    cache = Path(cache)
    manifest = cache.with_suffix('.json')
    contract = dict(algorithm=ALGORITHM, source_sha256=SOURCE_SHA256,
                    ranks={str(k): r for k, (_, r) in SITES.items()}, dtype='torch.float32')
    if cache.exists() or manifest.exists():
        require(cache.is_file() and manifest.is_file(), 'Incomplete factor cache; use a new path')
        report = json.loads(manifest.read_text(encoding='utf-8'))
        require(report['contract'] == contract, 'Cache source/rank/algorithm/dtype mismatch')
        require(sha256(cache) == report['file_sha256'], 'Factor file hash mismatch')
        stored = torch_load(cache, map_location='cpu')
        require(stored['contract'] == contract, 'Factor contract mismatch')
        expected = {f'model.{i}.blocks.1.branch2b.{n}.weight' for i in SITES for n in ('A_1x3', 'B_3x1')}
        require(set(stored['state']) == expected, 'Factor key inventory mismatch')
        for i, (c, r) in SITES.items():
            prefix = f'model.{i}.blocks.1.branch2b.'
            require(tensor_hash(reference.state_dict()[prefix+'conv.weight']) == report['sites'][str(i)]['source_tensor_sha256'], 'Cached W2 differs')
            for name, shape in [('A_1x3', (r, c, 1, 3)), ('B_3x1', (c, r, 3, 1))]:
                k = prefix+name+'.weight'; v = stored['state'][k]
                require(tuple(v.shape) == shape and v.dtype == torch.float32 and torch.isfinite(v).all(), 'Invalid cached factor')
                require(tensor_hash(v) == report['tensor_sha256'][k], 'Cached tensor hash mismatch')
        return stored['state'], {**report, 'cache_reused': True, 'path': str(cache.resolve())}
    state, sites = {}, {}
    for i, (c, r) in SITES.items():
        prefix = f'model.{i}.blocks.1.branch2b.'
        layer = reference.model[i].blocks[1].branch2b
        w = layer.conv.weight.detach().cpu().double()
        print(f'SVD site {i}, M={3*c}x{3*c}, rank={r}', flush=True)
        a, b, s = factorize(w, r)
        a32, b32 = a.float(), b.float()
        require(torch.isfinite(a32).all() and torch.isfinite(b32).all(), 'Nonfinite FP32 factors')
        eps = torch.finfo(torch.float64).eps
        threshold = float(3*c*s[0]*eps)
        energy = float(s.square().sum())
        bn = layer.norm
        scale = bn.weight.detach().cpu().double() / (bn.running_var.detach().cpu().double()+bn.eps).sqrt()
        weighted = scale[:, None, None, None]
        fp64, fp32 = kernel(a, b), kernel(a32, b32).double()
        # Also separate factor quantization from FP32 reconstruction accumulation.
        quantized64 = kernel(a32.double(), b32.double())
        sites[str(i)] = dict(source_key=prefix+'conv.weight', source_tensor_sha256=tensor_hash(layer.conv.weight),
            W_shape=list(w.shape), M_shape=[3*c, 3*c], A_shape=list(a.shape), B_shape=list(b.shape), rank=r,
            source_dtype=str(layer.conv.weight.dtype), svd_dtype='torch.float64', target_dtype='torch.float32',
            singular_values=s.tolist(), numerical_rank=int((s > threshold).sum()), rank_threshold=threshold,
            rank_rule='max(M.shape) * S[0] * torch.finfo(float64).eps',
            energy_retained=float(s[:r].square().sum())/energy if energy else None,
            status='PASSED' if energy else 'ZERO_SOURCE_KERNEL',
            kernel_fp64=error(w, fp64), kernel_fp32=error(w, fp32), factor_quantized_fp64=error(w, quantized64),
            bn2_weighted_fp64=error(weighted*w, weighted*fp64), bn2_weighted_fp32=error(weighted*w, weighted*fp32),
            bn2_eps=bn.eps, bn2_gamma_folded_into_svd=False, sign_reconstruction_exact=True)
        state[prefix+'A_1x3.weight'], state[prefix+'B_3x1.weight'] = a32, b32
    report = dict(contract=contract, path=str(cache.resolve()), sites=sites,
                  torch=str(torch.__version__), numerical_libraries=torch.__config__.show(),
                  tensor_sha256={k: tensor_hash(v) for k, v in state.items()},
                  total_elements=sum(v.numel() for v in state.values()), cache_reused=False,
                  interpretation='Kernel Frobenius approximation only; no image/semantic/detection optimality claim')
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open('xb') as f:
        torch.save(dict(contract=contract, state=state), f)
    report['file_sha256'] = sha256(cache)
    write_json(manifest, report)
    return state, report
