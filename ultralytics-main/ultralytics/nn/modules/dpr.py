# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""DPR-1: zero-initialized channel-diagonal difference-kernel parameterization.

The original dense kernel, BatchNorm and activation keep their names and values.
This changes optimization parameterization, not the deployed convolution family.
CD/HD/VD/AD have redundant directions (including CD/AD center nullspaces).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import BasicBlock, Blocks, ConvNormLayer


class DPRConvNormLayer(nn.Module):
    """One original 128-channel ConvNormLayer plus exactly 3,072 zero parameters."""

    parameter_names = ("dpr_cd", "dpr_hd", "dpr_vd", "dpr_ad")

    def __init__(self, original: ConvNormLayer):
        super().__init__()
        conv = original.conv
        if not (
            conv.in_channels == conv.out_channels == 128
            and conv.kernel_size == (3, 3)
            and conv.stride == conv.padding == conv.dilation == (1, 1)
            and conv.groups == 1 and conv.bias is None
            and isinstance(original.norm, nn.BatchNorm2d)
            and isinstance(original.act, nn.Identity)
        ):
            raise ValueError("DPR-1 requires the original P3 second BasicBlock branch2b contract")
        self.conv, self.norm, self.act = original.conv, original.norm, original.act
        # zeros_like/new_zeros do not consume RNG or change any original module.
        for name, width in zip(self.parameter_names, (9, 3, 3, 9)):
            self.register_parameter(name, nn.Parameter(conv.weight.new_zeros((128, width))))
        self.deployed = False

    def difference_kernels(self):
        """Return CD/HD/VD/AD in FP32 (FP64 is retained for mathematical checks)."""
        if self.deployed:
            raise RuntimeError("Deploy-only DPR has no difference parameters")
        dtype = torch.float64 if self.conv.weight.dtype == torch.float64 else torch.float32
        t, h, v, a = (getattr(self, name).to(dtype=dtype) for name in self.parameter_names)
        zero = h[:, 0] * 0
        # Scalar slices/stack avoid creating or transferring a CPU permutation tensor.
        center = torch.stack([zero, zero, zero, zero, t.sum(1), zero, zero, zero, zero], 1)
        cd = (t - center).reshape(-1, 3, 3)
        hd = torch.stack((h, h * 0, -h), 2)
        vd = torch.stack((v, v * 0, -v), 1)
        permutation = (3, 0, 1, 6, 4, 2, 7, 8, 5)
        ad = (a - torch.stack([a[:, j] for j in permutation], 1)).reshape(-1, 3, 3)
        return cd, hd, vd, ad

    def get_equivalent_kernel(self):
        """Construct the effective dense kernel once without detaching autograd."""
        if self.deployed:
            return self.conv.weight
        cd, hd, vd, ad = self.difference_kernels()
        delta = cd + hd + vd + ad
        diagonal = torch.diag_embed(delta.permute(1, 2, 0)).permute(2, 3, 0, 1)
        return self.conv.weight.to(dtype=delta.dtype) + diagonal

    def forward(self, x):
        if self.deployed:
            y = self.conv(x)
        else:
            kernel = self.get_equivalent_kernel().to(dtype=self.conv.weight.dtype)
            y = F.conv2d(x, kernel, self.conv.bias, self.conv.stride, self.conv.padding,
                         self.conv.dilation, self.conv.groups)
        return self.act(self.norm(y))

    @torch.no_grad()
    def switch_to_deploy(self):
        """Fold DPR once, keep original norm, and remove training-only parameters.

        Call on an independent eval copy; deploy-only objects cannot resume training.
        The persistent Python flag and missing parameters survive full-model pickle.
        """
        if self.deployed:
            return self
        if self.training:
            raise RuntimeError("DPR deployment requires an independent eval model copy")
        kernel = self.get_equivalent_kernel().to(dtype=self.conv.weight.dtype)
        self.conv.weight.copy_(kernel)
        for name in self.parameter_names:
            delattr(self, name)
        self.deployed = True
        return self


class BlocksDPR(Blocks):
    """Original Blocks, replacing only P3 blocks[1].branch2b without extra nesting."""

    def __init__(self, ch_in, ch_out, block, count, stage_num, act="relu", variant="d"):
        if (ch_in, ch_out, block, count, stage_num) != (64, 128, BasicBlock, 2, 3):
            raise ValueError("BlocksDPR is restricted to the original R18-Lite P3 stage")
        super().__init__(ch_in, ch_out, block, count, stage_num, act, variant)
        self.blocks[1].branch2b = DPRConvNormLayer(self.blocks[1].branch2b)


def count_dpr_conv_norm(module, inputs, output):
    """THOP hook: count the functional dense convolution plus original norm/act.

    THOP otherwise misses F.conv2d and reports an incorrect reduction in GFLOPs.
    Kernel-construction arithmetic/allocation is reported separately by check_dpr.
    """
    conv = module.conv
    macs = output.numel() * (conv.in_channels // conv.groups) * conv.kernel_size[0] * conv.kernel_size[1]
    if conv.bias is not None:
        macs += output.numel()
    norm_ops = getattr(module.norm, "total_ops", 0)
    act_ops = getattr(module.act, "total_ops", 0)
    module.total_ops += macs + norm_ops + act_ops
