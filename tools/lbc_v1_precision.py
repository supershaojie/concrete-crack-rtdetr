"""Strict isolated FP32 fusion check; original ordered evidence is never discarded."""
from contextlib import contextmanager
from copy import deepcopy
import os
import torch
from init_c19_lif_v1 import require
from lbc_v1_reporting import write_json, finalize_json


def precision_settings():
    return dict(cudnn_tf32=torch.backends.cudnn.allow_tf32,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cuda_autocast=torch.is_autocast_enabled(), cpu_autocast=torch.is_autocast_cpu_enabled(),
                cuda_autocast_dtype=str(torch.get_autocast_gpu_dtype()),
                cpu_autocast_dtype=str(torch.get_autocast_cpu_dtype()),
                autocast_cache_enabled=torch.is_autocast_cache_enabled(),
                nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'))


@contextmanager
def strict_fusion_precision(evidence):
    # Selective reuse: only the precision scope from RDL 91abf119...;
    # no RDL loss, checkpoint, monkey patch or experiment dependency is imported.
    before = evidence['before'] = precision_settings()
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast('cuda', enabled=False), torch.autocast('cpu', enabled=False):
            evidence['inside'] = precision_settings()
            yield
    finally:
        torch.backends.cudnn.allow_tf32 = before['cudnn_tf32']
        torch.backends.cuda.matmul.allow_tf32 = before['matmul_tf32']
        torch.set_float32_matmul_precision(before['float32_matmul_precision'])
        evidence['after'] = precision_settings()
        evidence['restored'] = evidence['after'] == before
        require(evidence['restored'], 'Fusion precision scope failed to restore settings')


def check_fusion(model, image, folder):
    from pathlib import Path
    from c19_lif_v1_diagnostic import tensor_compare, DiagnosticError, fusion_protocol
    from c19_lif_v1_cutoff import fusion_accepted
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    report=dict(status='FAILED',precision_scope={},tolerance=dict(atol=2e-5,rtol=2e-4))
    try:
        with strict_fusion_precision(report['precision_scope']):
            a=deepcopy(model).float().eval()
            b=deepcopy(a).fuse(verbose=False).eval()
            require(hasattr(b.model[20],'bn'),'LIF BN protection missing')
            with torch.no_grad():
                x=a(image.float())[0];y=b(image.float())[0]
            try:
                ordered=tensor_compare(x.cpu(),y.cpu(),'ordered_output','fusion',2e-5,2e-4)
                report['ordered']=ordered
                ok=ordered['allclose_failed_count']==0
            except DiagnosticError as error:
                report['ordered']=error.detail;ok=False
            write_json(folder/'fusion.json',report)
            if not ok:
                # Reuse the mother's candidate-ID recorder/acceptance, without widening tolerances.
                detail=fusion_protocol(a,b,image.float(),folder/'candidate_evidence',image.device.type,'fp32')
                report['candidate_evidence']=str(folder/'candidate_evidence/fuse_diagnostic.json')
                require(fusion_accepted(detail,image.device.type,'fp32'), 'Fusion candidate/numeric mismatch')
                require(detail['selection']['kind']=='PERMUTATION','Only verified same-set permutation may pass')
                report['status']='PASS_CANDIDATE_PERMUTATION'
            else: report['status']='PASS_ORDERED'
            report['unfused_parameters']=sum(p.numel() for p in a.parameters())
            report['fused_parameters']=sum(p.numel() for p in b.parameters())
        return report
    finally:
        finalize_json(folder/'fusion.json',report)
