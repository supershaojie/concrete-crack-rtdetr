# Failure history and scope of inference

| Experiment | Test mAP50-95 | AP75 |
|---|---:|---:|
| C2 | 0.469636231908 | 0.458360400552 |
| Original C17 / CSCEF-v5.1 | 0.512045085044 | 0.532829775858 |
| Original C24 / SCCA | 0.499872058481 | 0.506461002298 |
| Original C19 / CBR | 0.503892419455 | 0.519995255214 |
| Old C25 direct pair | 0.425179978084 | 0.404977566059 |
| Aggressive CSCEF-v6 | 0.416523743097 | 0.395762350015 |
| Aggressive SCCA-v2 | 0.442671778186 | 0.428146003059 |
| mincompat-v2 standalone | 0.423002181114 | 0.404070857125 |
| mincompat-v2 pair | 0.440251986984 | 0.435543139281 |

Original C17 and C24 each improved over C2. Direct combination failed. The first
compatibility round changed module internals, insertion/propagation and gradients;
its standalone results did not preserve the successful originals. The second round
preserved forward but attenuated the CSCEF semantic gradient to 0.25. This damaged
C17 standalone by about 8.90 pp. Its pair improved over old C25 by
1.507 pp mAP50-95 and
3.057 pp AP75, while remaining below the successful originals.

These experiments support preserving joint optimization and testing interface
calibration. They do not prove a causal gradient conflict or feature mismatch.
This implementation tests the forward-interface hypothesis without changing the
original identity semantic Jacobian. It is not detection-performance evidence.

Evidence: [history_audit.json](history_audit.json), original archived metrics under
[../mincompat_v2/evidence](../mincompat_v2/evidence), and the supplied v2 archives
under evidence/. Both archive SHA256 sidecars, source commit, original attenuation
code, initialization SHA, and all 109 typed C2 recipe fields were checked. Historical
evaluation provenance is not silently harmonized: old C25 recorded seed 0; SCI uses 42.
The full user specification is preserved verbatim as IMPLEMENTATION_SPEC.md.
