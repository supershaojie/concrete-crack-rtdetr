"""Frozen Grad-CAM++ gradient-power formulation, CH4 implementation 1.0.

Equation 19 weighting, as used by pytorch-grad-cam GradCAMPlusPlus; this is
the conventional first-gradient squared/cubed formulation, not explicit Hessians.
Reference: https://arxiv.org/abs/1710.11063
https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/grad_cam_plusplus.py
"""
from __future__ import annotations
import numpy as np

METHOD_VERSION = "ch4-gradcampp-gradient-powers-1.0"


def gradcampp(activation, gradient, epsilon=1e-6):
    a=np.asarray(activation,dtype=np.float64)
    if a.ndim!=4 or a.shape[0]!=1 or not np.isfinite(a).all():
        raise ValueError("Grad-CAM++ requires finite batch=1 BCHW activations")
    if gradient is None:
        return np.zeros(a.shape[2:],dtype=np.float32), dict(gradient_exists=False,status="unused_scale")
    g=np.asarray(gradient,dtype=np.float64)
    if g.shape!=a.shape or not np.isfinite(g).all():
        raise ValueError("Missing shape agreement/finite CAM gradients")
    stats=dict(gradient_exists=True,gradient_min=float(g.min()),gradient_max=float(g.max()),
               gradient_max_abs=float(np.abs(g).max()),gradient_l1=float(np.abs(g).sum()),
               activation_shape=list(a.shape),activation_min=float(a.min()),activation_max=float(a.max()))
    if np.abs(g).max()<=1e-12:
        return np.zeros(a.shape[2:],dtype=np.float32), {**stats,"status":"zero_gradient_scale"}
    g2,g3=g*g,g*g*g
    alpha=g2/(2*g2+a.sum(axis=(2,3),keepdims=True)*g3+epsilon)
    alpha=np.where(g!=0,alpha,0)
    weights=(np.maximum(g,0)*alpha).sum(axis=(2,3),keepdims=True)
    raw=np.maximum((weights*a).sum(axis=1)[0],0)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite Grad-CAM++ response")
    stats.update(status="passed" if raw.max()>1e-12 and np.ptp(raw)>1e-12 else "zero_response_scale",
                 cam_min=float(raw.min()),cam_max=float(raw.max()),native_resolution=list(raw.shape),
                 weighting_epsilon=epsilon,method_version=METHOD_VERSION)
    return raw.astype(np.float32),stats
