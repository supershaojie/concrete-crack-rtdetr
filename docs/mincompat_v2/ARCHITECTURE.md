# Successful forward paths preserved

Production changes relative to 05e6de8 are one small inherited module, additive export/parser registration, and four YAMLs. No successful module implementation, existing YAML, decoder, PAN, loss, matcher, denoising code or training augmentation is edited.

```text
P5 projection → original AIFI or SCCAAIFI → Y5(10) → original FPN → Y4(15)
                                                                  ↓ Upsample(16)
projected backbone P3(17) ──────────────┐                            ├─ main Concat(19)
                                      ↓                            ↓
                                  CSCEF(18) ← semantic identity, backward ×0.25
                                      ↓ enhanced lateral           │
                                      └────────────────────────────┘
                                              ↓ RepC3(20), P3
                                              ↓ original PAN → P4(23), P5(26)
                                              ↓ Decoder(27): [20,23,26]
```

| Variant | YAML suffix | AIFI | CSCEF at 18 | Decoder | nc1 parameters |
|---|---|---|---|---|---:|
| cscef_v52_compat | cscef-v52-compat | AIFI | CSCEFv52Compat | RTDETRDecoder | 20,109,684 |
| cscef_v52_scca_compat | cscef-v52-scca-compat | SCCAAIFI | CSCEFv52Compat | RTDETRDecoder | 20,175,224 |
| triad_mincompat_v2 | triad-mincompat-v2 | SCCAAIFI | CSCEFv52Compat | RTDETRDecoderCBR | 20,221,113 |
| c25_replay (local only) | c25-replay | SCCAAIFI | CSCEFv51 | RTDETRDecoder | 20,175,224 |

All filenames have prefix `rtdetr-resnet18-lite-` and suffix `.yaml`, in `ultralytics-main/ultralytics/cfg/models/rt-detr/`. Parser uses the original two-input CSCEF channel-injection rule. The original classes are never aliased to compatibility classes.

The CSCEF branch has 26,912 parameters and seven state tensors (five learned weights and two Scharr buffers), identical to v5.1. The configuration float adds zero parameters and zero state keys. The original detached structure confidence is still per-image spatial mean; lateral content gradients are unchanged. Residual gain remains one. Main upY4→Concat and all PAN propagation remain live.

Triad CBR reads decoder input slot 0, final neck P3; there is no fourth reference input. Original rho=0.10 and normal_fraction=0.10, 36 samples/query and original box-size displacement gradients remain. All formal models reject v1 classes through exact class/topology checks.

The public C2 layer mapping is 0..17→0..17 and 18..26→19..27. The enhanced lateral replaces only the corresponding Concat input; CSCEF has the original two inputs [17,16]. All 533 C2 states map exactly. Innovation state counts are 7/12/26, total states 540/545/559; nc80→nc1 adapts only the original nine classification states. See initialization reports in evidence/ for every source/target key, shape and error.

Formal initialization is exclusively `weights/rtdetr_r18_lite_imagenet_backbone_init.pt`, SHA256 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`. Historical best.pt files are not loaded as initializers. The real original C2 archive and all 109 typed recipe fields were checked; only model/name/save_dir differ in v2.
