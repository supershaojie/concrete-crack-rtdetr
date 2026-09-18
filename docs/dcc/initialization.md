# Controlled initialization audit

Both variants were built and checked with the real local public source at
`D:/MyProjects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt`.
Its SHA256 was `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`;
the checkpoint had `nc=80`, `epoch=-1`, and no optimizer/EMA/scaler training state.
The recorded runtime was Python 3.9.25 / PyTorch 2.7.1+cu118. These results are
local evidence; they do not certify the server's PyTorch 2.1.2 runtime or capacity.

| Variant | Public nc=1 state tensors compared exactly | New trainable tensors | Unfused parameters |
|---|---:|---:|---:|
| `cbr_lif_dcc_v1` | 552 | 4 | 20,168,197 |
| `dcc_v1` | 533 | 4 | 20,101,204 |

Every public key, shape, value and tensor SHA256 is recorded in
`initialization_cbr_lif_dcc_v1.json` and `initialization_dcc_v1.json`.
Both report zero missing/unexpected/shape-mismatched keys. Each variant added
exactly 18,432 trainable parameters. Their four new parameter tensors were
compared directly and are identical. `W_d/W_q/W_k` use the PyTorch Conv2d
default initialization; only `W_o` is zero. The identity matrix is a registered
nonpersistent buffer and is recorded separately from checkpoint state tensors.

The main parent uses the unchanged `init_c19_lif_v1.controlled_models` tool,
including its original independent CBR/LIF initialization checks. DCC construction
does not consume the CPU RNG stream used by later public layers. Native
`RTDETRTrainer.get_model` adapts exactly nine classification tensors from 80 to
1 class; those nine tensors are exactly equal between the corresponding parent
and DCC target after synchronized construction. The stored initializations have
`nc=1`; subsequent native training reconstruction preserves all of their values.

Each initialization was saved, reloaded through `RTDETR`, reconstructed by the
real `RTDETRTrainer.setup_model`, and passed through actual `RTDETR.train`
dispatch to native `get_model`. This last check uses a minimal constructor shim
and intentionally raises at `Trainer.train` entry, before dataset/optimizer
setup or any training loop. It proves reconstruction and is **not** a claim of
native training or resume verification; those have separate engineering reports.

The existing checkpoints were then re-audited from the public source with
`init_dcc.py --verify-existing`; checkpoint bytes were not changed.

| Preserved local checkpoint | SHA256 |
|---|---|
| `outputs/dcc/cbr_lif_dcc_v1/controlled_init.pt` | `b93ce912a49ccdcc8965c45cb24371f2ab936b6044abc66750bf4b93b198adc6` |
| `outputs/dcc/dcc_v1/controlled_init.pt` | `3563fcdae038c6e48cb4c490aa6f42327b735570a5b84a4b453093abb56ad3bf` |

These files are deliberately excluded from Git. Fresh server checkpoint file
hashes can differ because checkpoint metadata includes the local paths/date;
the source SHA and per-tensor audits identify the controlled initialization.

`initialization_attempt1_failed.json` preserves the first verifier failure: the
original YAML writes the convolution padding as the string `None`, which the
native parser later interprets as Python `None`. The verifier initially compared
it with Python `None` before parsing. The verifier was corrected to accept the
actual parent representation while preserving all original YAML arguments, and
both complete audits passed. No model or mathematical contract was changed.

The native verbose model summary prints THOP GFLOPs during setup. These numbers
are not accepted as complete DCC operation counts: functional projections and
matrix products need explicit accounting. Parameter counts above are direct
unfused model parameter sums.

Formal 200-epoch training: **NOT_STARTED**. Final test: **NOT_RUN**.
