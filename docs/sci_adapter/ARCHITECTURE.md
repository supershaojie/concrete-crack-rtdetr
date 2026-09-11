# SCI semantic interface

```text
Original SCCAAIFI (pair) / original AIFI (control)
  -> original FPN -> Y4 (15) -> original nearest upsample (16)
                       |                     |
+                       |                     +----------------> original Concat (20)
                       |                     +-> SCI (18) -> original CSCEF (19) --+
                       |                                      ^                 |
                       |                           original lateral (17)        |
                       +-> original PAN          Concat -> RepC3/P3 -> PAN P4/P5
```

The adapter receives the **existing upsampled** Y4 that original C17 consumed.
No extra interpolation is introduced. CSCEF remains before the original P3 Concat
and RepC3. Adding a graph row changes later numeric indices only, not their location
in the computation or source operations. Original model/YAML/state keys are unchanged;
new variants have an explicitly audited index mapping.

`proxy = y + restore(SiLU(reduce(GN(y.detach()))))`

- GN: 8 groups, 256 channels, affine=True, eps=1e-5; 512 parameters.
- reduce: 1x1, 256 -> 32, bias=False, normal Kaiming initialization; 8,192 parameters.
- SiLU; restore: 1x1, 32 -> 256, bias=True, weight=bias=0; 8,448 parameters.
- Exactly 17,152 trainable parameters. No alpha, shortcut, gate, attention, spatial
  convolution, residual clamp, new loss, matcher, optimizer, or searched preset.
- Constructor preserves CPU RNG as the original successful modules do. The residual
  follows input dtype under AMP. Addition is out of place.

| Variant ID | YAML | nc=1 unfused parameters |
|---|---|---:|
| cscef_v51_sci_control | rtdetr-resnet18-lite-cscef-v51-sci-control.yaml | 20,126,836 |
| scca_sci_cscef_v51 | rtdetr-resnet18-lite-scca-sci-cscef-v51.yaml | 20,192,376 |
| scca_sci_cscef_v51_cbr | rtdetr-resnet18-lite-scca-sci-cscef-v51-cbr.yaml | 20,238,265 |

Round B retains original C19 query, final P3 source, 36 sample points, rho=0.10,
normal_fraction=0.10 and box residual. It is prepared, not trained. Dual Semantic
Stream remains future work and has no implementation here.

All experiments start from the SHA-locked C2 public initialization, never from a
trained experiment best. C2 layers 0..17 map unchanged; 18..26 map to 20..28. Original
C17/C25 layers 18..27 map to 19..28. Only the allowed class-specific tensors adapt
from nc=80 to nc=1, using the native trainer and seed 42.
