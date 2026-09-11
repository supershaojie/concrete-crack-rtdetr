# Failure analysis and v2 decision

Engineering correctness ≠ detection effectiveness. v1 results falsified the assumption that aggressive architectural/gradient decoupling would preserve standalone capability; therefore v2 preserves successful forward paths and changes only one cross-module backward edge.

The verified [failure evidence](failure_evidence.md) shows CSCEF-v6 mAP50-95=0.4165237430966872 versus original C17=0.5120450850441516, and GI-SCCA-v2=0.4426717781864705 versus original C24=0.4998720584810606. Recipe drift or public-weight mapping errors were not found in the archived evidence. One run per variant cannot establish a unique causal explanation.

DR-CSCEF-v6 changed the semantic input from upsampled Y4 to projected backbone P4, moved the residual after PAN, used final P3 as the residual base, removed its propagation through PAN P4/P5, and detached side references. These simultaneous changes can remove useful content adaptation and multiscale propagation. Their separate contributions are unmeasured. v2 restores original CSCEF content, confidence, location, inputs and propagation.

GI-SCCA-v2 detached x and spatial-MHA s inside its channel branch, preserving instantaneous forward values while changing representation learning. The observed standalone loss is consistent with that extra gradient being useful, but is not a multiseed causal proof. v2 uses the original SCCAAIFI file byte for byte (canonical LF), including original pre/post norm insertion, width 64, heads 4, dimension 16, non-affine LayerNorm, centered normalized Q/K, bounded temperature, FP32 attention core and zero output initialization.

SR-CBR-v2 changed final P3 to a stable reference, detached query and displacement geometry, and reduced rho from 0.10 to 0.075. No independent SR-CBR-v2 failure result was supplied; do not claim its ineffectiveness was measured. Original C19 is proven useful in the supplied run and is retained. Its existing sampling-grid and geometry-projection box detaches remain; its query, P3, original-box addition and width/height displacement scale gradients remain live.

C20/C25/C26 negative interactions are historical observations. C25 is the most severe: 0.4251799780843431. v2 targets its actual extra autograd edge, CSCEF semantic content → upY4 → Y4/SCCA/AIFI, with fixed gain 0.25. Whether that edge has conflicting trained task gradients remains a hypothesis; the synthetic probes establish dependency and attenuation, not opposing gradient directions.

Original C17/C19/C24 classes and YAMLs remain intact. v1 classes, YAMLs, reports and tools remain available for history/regression. The v1 performance variants (`cscef_v6`, `scca_v2`, `cbr_v2`, their pair variants, `triad_v1`) are **deprecated_for_performance_experiments**. New formal launch choices contain only the three v2 variants; C25 replay is local-only.

Round A runs standalone v5.2 and the compatible pair from the common C2 initializer. Round B adds original C19 only if Round A supports the compatibility direction. No test-based gamma scan, residual beta change, v3 module, new loss, or hyperparameter tuning is implemented. Better pair performance would support this intervention in the tested setting. Continued pair failure would motivate investigation of forward/residual interaction; it would not by itself prove that alternative cause.
