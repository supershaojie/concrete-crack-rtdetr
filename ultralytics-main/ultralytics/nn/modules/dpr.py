# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""DPR-1: zero-initialized channel-diagonal difference-kernel parameterization.

The original dense kernel, BatchNorm and activation keep their names and values.
This changes optimization parameterization, not the deployed convolution family.
CD/HD/VD/AD have redundant directions (including CD/AD center nullspaces).
"""

from functools import lru_cache

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


@lru_cache(maxsize=None)
def _probe_thop_semantics(profile):
    """Measure custom-parent aggregation, without version guesses or model/RNG state."""
    class Leaf(nn.Module):
        def forward(self, x):
            return x

    class Parent(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Leaf()

        def forward(self, x):
            return self.child(x)

    def parent_hook(module, inputs, output):
        module.total_ops += 7

    def child_hook(module, inputs, output):
        module.total_ops += 11

    # The custom module must be nested: legacy THOP treats a root differently.
    operations = float(profile(nn.Sequential(Parent()), inputs=(torch.zeros(1),),
                               custom_ops={Parent: parent_hook, Leaf: child_hook}, verbose=False)[0])
    if operations not in (7.0, 18.0):
        raise RuntimeError(f"Unsupported THOP custom-parent aggregation: probe={operations}, expected 7 or 18")
    return operations


def thop_dpr_semantics():
    """Return the installed profiler's measured nested-module counting convention."""
    import thop

    operations = _probe_thop_semantics(thop.profile)
    return {"version": getattr(thop, "__version__", "unknown"),
            "mode": "accumulate_children" if operations == 18 else "replace_subtree",
            "nested_probe_ops": operations, "parent_probe_ops": 7, "child_probe_ops": 11}


def count_dpr_conv_norm(module, inputs, output):
    """Count only missing operations under the installed THOP aggregation rules.

    New THOP accumulates executed children: only an unfolded functional conv is
    missing. Legacy THOP replaces the entire custom parent's subtree, so its
    hook must also include the executed conv (when deployed), norm and act.
    Kernel-construction arithmetic/allocation remains separately reported.
    """
    accumulates_children = thop_dpr_semantics()["mode"] == "accumulate_children"
    if module.deployed:
        if accumulates_children:
            return  # conv/norm/act children already account for everything.
        macs = getattr(module.conv, "total_ops", 0)
    else:
        conv = module.conv
        macs = output.numel() * (conv.in_channels // conv.groups) * conv.kernel_size[0] * conv.kernel_size[1]
        if conv.bias is not None:
            macs += output.numel()
    if not accumulates_children:
        macs = macs + getattr(module.norm, "total_ops", 0) + getattr(module.act, "total_ops", 0)
    module.total_ops += macs


def profile_dpr(model, inputs, custom_ops=None, **kwargs):
    """Profile an isolated model copy, restoring THOP counters/hooks even on failure.

    Callers own the copy. Legacy THOP leaves counters on unsupported containers
    and restores only a single global training flag. Preserve pre-existing
    counters/hooks and every module's mode instead of leaking profiling state.
    """
    import thop

    # Legacy THOP always descends into its root even when that root has a
    # custom hook. Match the detector's nested DPR layout for standalone probes.
    if isinstance(model, DPRConvNormLayer) and thop_dpr_semantics()["mode"] == "replace_subtree":
        model = nn.Sequential(model)
    hook_attributes = ("_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called")
    snapshots = []
    for module in model.modules():
        counters = {key: module._buffers[key] for key in ("total_ops", "total_params") if key in module._buffers}
        hooks = {key: set(getattr(module, key, {})) for key in hook_attributes}
        snapshots.append((module, module.training, counters, set(module._non_persistent_buffers_set), hooks))
    try:
        return thop.profile(model, inputs=inputs,
                            custom_ops={DPRConvNormLayer: count_dpr_conv_norm} if custom_ops is None else custom_ops,
                            **kwargs)
    finally:
        for module, training, counters, non_persistent, hooks in snapshots:
            for key in ("total_ops", "total_params"):
                module._buffers.pop(key, None)
            module._buffers.update(counters)
            module._non_persistent_buffers_set.clear()
            module._non_persistent_buffers_set.update(non_persistent)
            for key, original_ids in hooks.items():
                collection = getattr(module, key, {})
                for hook_id in set(collection) - original_ids:
                    collection.pop(hook_id)
            module.training = training
