# SRE v1 structure and mathematical contract

SRE means Support-based Response Enrichment (支持驱动响应补偿). The formal module name has no scale suffix. This experiment starts at `a0459d6a652cb702699087c88fa39a3e4c4087ec` on `exp-rtdetr-r18-lite-sre-v1`.

Both configurations replace only node 19's RepC3 class with SRERepC3. Its original cv1/cv2/m/cv3 keys and three internal RepConv blocks remain. A single SRE receives the complete original RepC3 output. The 27 nodes and routing remain unchanged: corrected node 19 feeds node 20 and decoder 26 via `[19,22,25]`. No second residual surrounds SRE. The main variant retains the original CBR and LIF-Down; the ablation retains C2's original downsampling and decoder. Node 17, AIFI, backbone, loss, matcher, DN and query selection are unchanged.

For X with 256 channels, all SRE arithmetic runs inside disabled autocast in FP32, using differentiable FP32 views of parameters, even with explicit half models:

```
Z = GN4(W_d X)                       # 256 -> 32, GN eps=1e-5
Q = normalize(W_q Z, dim=1, eps=1e-6) # 32 -> 8
V = softplus(Z, beta=1, threshold=20)
w(p,j) = relu(clamp(dot(Qp,Qj), -1,1))**2
s(p) = sum_valid_neighbors w(p,j)
D(p,c) = sum_j w(p,j)*relu(V(j,c)-V(p,c)) / (1+s(p))
Y = X + (W_o D).to(X.dtype)          # 32 -> 256
```

One simultaneous aggregation uses exactly the 24 offsets of the 5x5 neighborhood excluding its center. Invalid neighbors contribute zero; no wraparound, repeated boundary votes, softmax, direction pairs, iterative propagation, temperature, threshold, or learned offsets are introduced. Positive differences are computed before summation. Q normalization is over descriptor channels. A fixed zero message gives the denominator's constant 1.

In exact arithmetic `0 <= D <= s/(1+s)*max_j relu(Vj-Vp)`. No positive similarity, equal values, or a locally maximal channel gives zero compensation. Single-sided support is permitted. Similarity is symmetric but the positive differences are channel-dependent and directional. V is a nonnegative latent feature, not crack probability. Signed W_o means final features/scores/recall need not increase monotonically. Background texture can also receive support; absent target information is not guaranteed recoverable.

Only W_o is zero-initialized; W_d/W_q use nonzero Xavier uniform, GN affine starts at 1/0. CPU RNG is isolated so later parent construction is unchanged. There are 16,704 new parameters: 8,192 + 64 + 256 + 8,192. At 640 input the projection subtotal is 106,496,000 MAC (0.212992 GFLOPs at 2 FLOPs/MAC). This excludes aggregation, GN, normalization and activation and is not whole-network FLOPs. Per-offset accumulation avoids an explicit full unfolded neighbor tensor; autograd still saves intermediate tensors. B16/640 native AMP capacity must be measured separately.

| nc=1 | Unfused | Fused |
|---|---:|---:|
| Original CBR + LIF | 20,149,765 | 19,944,965 |
| CBR + LIF + SRE | 20,166,469 | 19,961,669 |
| Original C2 | 20,082,772 | 19,877,716 |
| C2 + SRE | 20,099,476 | 19,894,420 |

The unchanged original files are checked using LF-normalized hashes: lif_down.py `26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7`, cbr.py `d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787`.

AutoBackend warmup now uses zeros rather than uninitialized torch.empty. This only makes synthetic warmup input finite; no numerical errors in actual data are hidden. GN is never folded into BN; the original LIF fusion safeguard remains.

## Relationship to prior designs

BSC uses fixed directional pairs and their bilateral mean/difference. SRE uses content similarity over the complete local neighborhood, accepting stronger responses per channel, including single-sided support. Both are local residual enhancement methods; they are related, not equivalent.

[Neighborhood Attention](https://arxiv.org/abs/2204.07143) and [RFAConv](https://arxiv.org/abs/2304.03198) were consulted as technical background. They do not establish this design's novelty or efficacy. No NATTEN, custom CUDA, or reference package installation is required.

## Protocol boundaries

The archived `parent_args.yaml` is the actual successful 200-epoch online-augmentation recipe. Only model/name/output identity and verified equivalent data paths may change. No pretrained successful best/last is used for initialization. Both variants derive from the exact nc80 epoch=-1 public source SHA `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`, then audit native Trainer nc1 adaptation.

Formal metrics preserve maximum-F1 P/R and best-checkpoint selection. The separate val diagnostic uses original fixed-candidate one-to-one TP@IoU0.50 labels, raw unique score thresholds with whole tie groups, target P=0.8656, floor=0.001 and max_det=300. It is neither a shared-confidence comparison nor a threshold search on test. Candidate R_at_P, AP75, mAP50–95 and false positives must be assessed together; engineering checks are not accuracy evidence.
