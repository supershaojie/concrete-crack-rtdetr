# One semantic gradient edge

`CSCEFv52Compat(CSCEFv51)` delegates the entire successful content/confidence/residual computation to the parent. Its sole numerical intervention is:

```python
detached = semantic.detach()
semantic_proxy = detached + 0.25 * (semantic - detached)
return super().forward((lateral, semantic_proxy))
```

The subtraction is exactly zero for finite identical inputs, so the proxy has the same forward value. Its derivative with respect to semantic is 0.25. The parent's convolution, normalization, structure computation and residual are unchanged. The parameter derivative sees the original forward input and therefore is unchanged; the lateral derivative is also unchanged. No hook, learned gate or persistent activation cache is used.

`semantic_grad_scale` is a Python float fixed to 0.25; non-0.25 constructor values are rejected. It is serialized in the complete model object, excluded from state_dict and optimizer. A strict v5.1→v5.2 state load succeeds without any renamed keys. Production YAMLs use the fixed default. Main upY4 and SCCA x/s are never detached. The detach operations in the algebra implement partial gradient scaling; the semantic edge retains 25% gradient and is not fully cut.

The nonzero module objective is output.square().mean(). FP32 and AMP report exact forward identity, semantic L1/L2 ratios 0.25, lateral ratios 1.0 and all five parameter-gradient max_abs=0. True half module forward also has max_abs/max_rel=0.

Pair probes use the actual old replay and new model graph. The residual-only objective reports a 0.25 ratio at both upY4 and Y4. A separate main-Concat objective freezes only its CSCEF contribution inside the diagnostic, ensuring the objective reaches Y4 solely through Concat→RepC3; that diagnostic detach does not exist in production. Its ratio is 1.0 (CUDA roundoff max_abs ≤3.56e-15). Full pair objectives give finite nonzero SCCA gradients; SCCA's own x/s branch gradients remain nonzero. Only the separately attributed CSCEF side component is expected to scale 0.25, not total SCCA gradient.

Tools: `check_mincompat_v2_gradients.py`, `check_mincompat_v2.py`. Evidence: [gradient_edge.json](gradient_edge.json). These are controlled synthetic checks, not formal detection performance or a trained-gradient conflict study.
