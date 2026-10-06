"""Observe the original attention operators; never substitute attention arithmetic."""
from contextlib import contextmanager
from collections import Counter
from copy import deepcopy
import importlib.metadata as metadata
import inspect
from pathlib import Path
import threading
import math
import torch
from torch.utils._python_dispatch import TorchDispatchMode

_scope_lock = threading.RLock()
_phase = 'native_setup'
_flash_counts = Counter()
_flash_rows = Counter()


def flash_identity():
    """Import the pinned extension, including its ABI, rather than guessing availability."""
    import flash_attn
    import flash_attn_2_cuda
    from flash_attn.flash_attn_interface import flash_attn_func
    from support import sha256
    version = metadata.version('flash-attn')
    if version.split('+')[0] != '2.7.3':
        raise RuntimeError('FlashAttention must be pinned at 2.7.3, got '+version)
    if 'deterministic' not in inspect.signature(flash_attn_func).parameters:
        raise RuntimeError('FlashAttention API lacks deterministic backward')
    return {'version':version, 'import_path':str(Path(flash_attn.__file__).resolve()),
            'interface_path':str(Path(inspect.getfile(flash_attn_func)).resolve()),
            'interface_sha256':sha256(inspect.getfile(flash_attn_func)),
            'extension_path':str(Path(flash_attn_2_cuda.__file__).resolve()),
            'extension_sha256':sha256(flash_attn_2_cuda.__file__),
            'torch_cxx11_abi':bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            'deterministic_backward_supported':True}


def install_flash_observer():
    """A scalar-only observer around the real author call; retains no tensor references."""
    from ultralytics.nn.modules import block
    original = getattr(block, 'flash_attn_func', None)
    if original is None or getattr(original, '_comparison_observer', False):
        return
    def observed(q, k, v, *args, **kwargs):
        if not block.USE_FLASH_ATTN:
            raise RuntimeError('Flash called inside a native scope')
        if kwargs.get('deterministic') != block.FLASH_ATTN_DETERMINISTIC:
            raise RuntimeError('Flash deterministic argument differs from scoped policy')
        if kwargs.get('dropout_p') != 0.0 or kwargs.get('causal') is not False:
            raise RuntimeError('Author attention dropout/causal semantics changed')
        output = original(q, k, v, *args, **kwargs)
        _flash_counts[_phase+'/forward'] += 1
        key = (_phase, str(q.dtype), str(k.dtype), str(v.dtype), str(output.dtype),
               tuple(q.shape), kwargs['deterministic'], kwargs.get('softmax_scale'))
        _flash_rows[key] += 1
        if output.requires_grad:
            phase = _phase
            def gradient_seen(gradient):
                _flash_counts[phase+'/output_gradient'] += 1
                return gradient
            output.register_hook(gradient_seen)
        return output
    observed._comparison_observer = True
    block.flash_attn_func = observed


def flash_evidence():
    return {'counts':dict(_flash_counts), 'calls':[
        {'phase':key[0], 'q_dtype':key[1], 'k_dtype':key[2], 'v_dtype':key[3],
         'output_dtype':key[4], 'q_shape':list(key[5]), 'deterministic':key[6],
         'softmax_scale':key[7], 'calls':count} for key,count in _flash_rows.items()],
        'backward_observation':'output_gradient counts gradient flow; successful operator backward also requires finite Q/K/V gradients',
        'tensor_references_retained':False}


def flash_forward_count():
    return sum(v for k,v in _flash_counts.items() if k.endswith('/forward'))


@contextmanager
def attention_scope(backend, phase='unspecified', deterministic=True):
    """Only select the author's branch; restore it even if forward/validation fails."""
    global _phase
    from ultralytics.nn.modules import block
    if backend not in ('native', 'flash'):
        raise ValueError('A runtime scope requires a resolved native/flash backend')
    with _scope_lock:
        previous = (block.USE_FLASH_ATTN, block.FLASH_ATTN_DETERMINISTIC, _phase)
        if backend == 'flash':
            if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8:
                raise RuntimeError('Strict Flash requires CUDA capability >= 8.0; no native fallback')
            flash_identity()
            install_flash_observer()
        block.USE_FLASH_ATTN = backend == 'flash'
        block.FLASH_ATTN_DETERMINISTIC = bool(deterministic)
        _phase = phase
        try:
            yield
        finally:
            block.USE_FLASH_ATTN, block.FLASH_ATTN_DETERMINISTIC, _phase = previous


def flash_operator_probe():
    """Execute the pinned CUDA kernel and deterministic backward, preserving all RNGs."""
    from flash_attn.flash_attn_interface import flash_attn_func
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        tensors = [torch.randn(1,16,2,32,device='cuda:0',dtype=torch.float16,requires_grad=True) for _ in range(3)]
        output = flash_attn_func(*tensors, dropout_p=0.0, softmax_scale=32**-.5,
                                 causal=False, deterministic=True)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        if not torch.isfinite(output).all() or any(t.grad is None or not torch.isfinite(t.grad).all() for t in tensors):
            raise RuntimeError('Real Flash CUDA forward/backward produced missing/nonfinite gradients')
        return {'status':'PASSED_REAL_CUDA_OPERATOR', 'function':'flash_attn.flash_attn_interface.flash_attn_func',
                'input_dtype':str(tensors[0].dtype), 'output_dtype':str(output.dtype),
                'shape':list(tensors[0].shape), 'deterministic_backward':True,
                'QKV_gradient_max_abs':[float(t.grad.abs().max()) for t in tensors]}


def resolve_train_backend(requested, amp=True, deterministic=True):
    if requested not in ('native','flash','auto'):
        raise ValueError('train_attention_backend must be native, flash or auto')
    if requested == 'native':
        return {'requested':'native','resolved':'native','reason':'explicit native request',
                'deterministic':bool(deterministic),'flash_parity':'NOT_VERIFIED'}
    try:
        if not amp or not deterministic:
            raise RuntimeError('Flash training requires amp=true and deterministic=true')
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8:
            raise RuntimeError('Flash requires CUDA and Ampere or newer (capability >= 8.0)')
        identity = flash_identity()
        probe = flash_operator_probe()
    except Exception as exc:
        if requested == 'flash':
            raise RuntimeError('Strict Flash request failed; no fallback: '+str(exc)) from exc
        return {'requested':'auto','resolved':'native','reason':type(exc).__name__+': '+str(exc),
                'deterministic':bool(deterministic),'flash_parity':'NOT_VERIFIED'}
    return {'requested':requested,'resolved':'flash','reason':'pinned import/ABI and actual deterministic CUDA forward/backward passed',
            'deterministic':True,'flash':identity,'operator_probe':probe,'flash_parity':'NOT_VERIFIED'}


class ArithmeticAudit(TorchDispatchMode):
    def __init__(self, require_fp32=False):
        super().__init__()
        self.require_fp32 = require_fp32
        self.operations = Counter()
        self.dtypes = Counter()
        self.macs = Counter()
        self.flash_calls_before = flash_forward_count()
        self.half_casts = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func)
        if '_to_copy' in name and isinstance(out, torch.Tensor) and out.dtype in (torch.float16,torch.bfloat16):
            self.half_casts += 1
            if self.require_fp32:
                raise RuntimeError('Internal half/bfloat16 cast in native FP32 scope: '+name)
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
        calls = flash_forward_count() - self.flash_calls_before
        if self.require_fp32 and calls:
            raise RuntimeError('Flash called during independent native FP32 arithmetic audit')
        return {'backend':'official_flash_attention' if calls else 'official_native_matmul_stable_softmax',
                'flash_forward_calls':calls, 'required_fp32':self.require_fp32,
                'internal_half_cast_calls':self.half_casts,
                'observed_dtypes':dict(self.dtypes), 'MACs_by_operator':dict(self.macs),
                'covered_GFLOPs_2x_MACs':2*sum(self.macs.values())/1e9,
                'observed_operator_calls':dict(self.operations),
                'flops_scope':'Conv/Linear and all executed mm/bmm products, including AAttn and HyperACE',
                'flops_exclusions':'Opaque Flash kernel MACs (if used), bias adds, BN, activation, exp/softmax, reductions, pooling, residual/gate/elementwise arithmetic, interpolation, decode and NMS; not complete FLOPs'}


@contextmanager
def native_fp32():
    before = flash_forward_count()
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        with attention_scope('native', 'native_fp32'), torch.autocast('cuda', enabled=False), torch.autocast('cpu', enabled=False):
            yield
            if flash_forward_count() != before:
                raise RuntimeError('Flash call in native FP32 scope')
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old

def optional_flash_parity():
    """FP16 quantization bounds, fixed before testing; never relax on failure."""
    from ultralytics.nn.modules import block
    resolution = resolve_train_backend('auto')
    if resolution['resolved'] != 'flash':
        return {'status':'NOT_VERIFIED','reason':resolution['reason']}
    rows=[]
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        for area in (1,4):
            module=block.AAttn(64,2,area=area).float().cuda().eval()
            x=torch.randn(1,64,8,8,device='cuda',requires_grad=True)
            with native_fp32():
                ref=module(x);ref_grads=torch.autograd.grad(ref.square().mean(),[x,*module.parameters()])
            other=deepcopy(module);z=x.detach().clone().requires_grad_(True)
            try:
                with attention_scope('flash','AAttn_parity',deterministic=True):
                    output=other(z);grads=torch.autograd.grad(output.square().mean(),[z,*other.parameters()])
                output_ok=torch.allclose(ref,output,rtol=.03,atol=.003)
                grads_ok=all(torch.allclose(a,b,rtol=.05,atol=.003) for a,b in zip(ref_grads,grads))
                rows.append({'area':area,'output_allclose':output_ok,'gradients_allclose':grads_ok,
                             'output_max_abs_error':float((ref-output).abs().max()),
                             'input_gradient_max_abs_error':float((ref_grads[0]-grads[0]).abs().max()),
                             'parameter_gradient_max_abs_error':max(float((a-b).abs().max()) for a,b in zip(ref_grads[1:],grads[1:]))})
            except Exception as exc:
                rows.append({'area':area,'status':'NOT_VERIFIED','error':str(exc)})
    passed = all(r.get('output_allclose') and r.get('gradients_allclose') for r in rows)
    return {'status':'PASSED' if passed else 'FAILED','output_tolerance':{'rtol':.03,'atol':.003},
            'gradient_tolerance':{'rtol':.05,'atol':.003},'rows':rows,
            'scope':'tiny AAttn output/input+parameter gradients; not proof of full-network equivalence',
            'tolerance_basis':'FP16 Q/K/V quantization and reduction order vs native FP32; fixed 3% output/5% gradients plus 0.003 absolute floor for near-zero entries; no bitwise claim',
            'flash_evidence':flash_evidence()}
