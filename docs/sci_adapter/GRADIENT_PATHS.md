# Preserved semantic gradient

`proxy = y + A(stopgrad(y))` has exactly `d proxy / d y = I`, for both zero and
nonzero restore. This is a local Jacobian statement. Learned proxy values can change
the downstream CSCEF gradient numerically; no claim is made that trained whole-model
gradients remain equal to C17/C25 after the adapter becomes nonzero.

`check_sci_adapter_gradients.py` activates original CSCEF/SCCA first, aligns every
shared state tensor, and isolates a CSCEF-residual objective in the actual model graph.
For control and pair, zero-SCI CSCEF gradients at upsampled Y4 and Y4 are exactly
equal to C17/C25 (max abs/rel=0, L2 ratio=1). Nonzero-SCI gradients at semantic and
proxy are exactly equal. An isolated correction has no input gradient and nonzero
gradients for every SCI parameter. Original SCCA and CSCEF parameters receive finite,
nonzero gradients. See gradient_paths.json for per-parameter values.

The real all-zero initialization has a natural three-update learning sequence:

1. Original CSCEF output projection learns; SCI gradients are connected but zero.
2. SCI restore learns after CSCEF opens; GN/reduce remain behind zero restore.
3. GN/reduce and every new parameter have nonzero gradients.

The FP32 and CUDA AMP native loss/DN smoke explicitly checks this sequence, without
changing initialization or adding any warmup, shortcut or gradient multiplier.
Synthetic fixed affine/channel-mix fitting is separate and proves only engineering
learnability, not detector accuracy.

Training diagnostics sample the first **training** batch of each epoch and append
GN scale/bias mean/std, reduce/restore RMS, residual/Y4 RMS, residual mean absolute,
and all 256 channel mean/std shifts. They detach observations, never store an autograd
graph and never limit the residual. Validation forwards cannot consume the observation.
