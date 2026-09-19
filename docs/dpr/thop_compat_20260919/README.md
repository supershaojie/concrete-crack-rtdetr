# DPR THOP aggregation compatibility

Change base: `bf23c6d4ad43aa2182e11e8a0726cef950425728`, branch
`exp-rtdetr-r18-lite-dpr-v1`. This is a counting/diagnostic correction only.
The original DPR classes (forward, four mappings and fold), CBR/LIF, both
configurations, initialization, training/AMP/evaluation protocols and all
precision thresholds are unchanged. Earlier reports are retained in place.

## Fix

The installed profiler is measured with a nested custom parent (7 operations)
and child (11 operations), without consuming model RNG. THOP 2.0.18 returns 7:
the parent rule replaces its subtree. THOP 2.1.6 returns 18: it accumulates
parent and children. The result is cached by the actual profiler function;
unknown aggregation fails rather than guessing from a version string.

For accumulating THOP, the unfolded DPR hook adds only the missing functional
convolution, and the deployed hook adds nothing. Executed conv/BN/activation
children count themselves. For legacy THOP, the hook retains the full subtree
count, using the executed conv child's count after deployment. The functional
conv hook is retained. Kernel construction remains separately listed, outside
deployment convolution GFLOPs.

The profiling wrapper restores THOP counters, hooks and per-module training
flags, including legacy unsupported-container residue and exception paths.
Standalone DPR probes are nested under legacy THOP to account for its special
root traversal. Production callers still profile independent model copies.

All three original FLOPs comparisons and their `1e-9` tolerance remain. Their
diagnostic now prints all six values, three residuals and measured THOP mode;
the same diagnostic is carried by an AssertionError and saved traceback.

## Actual local verification

Windows / Python 3.9.25 / PyTorch 2.7.1+cu118. The existing installed THOP is
2.0.18. Official `ultralytics-thop==2.1.6` was downloaded and extracted into an
ignored output directory, selected only through a process-local PYTHONPATH.
No environment package was installed, upgraded or downgraded. Official wheel
identity and PyPI SHA verification are in `wheel_source.json.gz`.

Both versions ran `tools/check_dpr.py --variant both --imgsz 640 --threads 4`
with new output paths. All three FLOPs assertions passed for all four cases:

| THOP | Variant | Parent = corrected GFLOPs | Raw DPR GFLOPs | Parent fused = deployed GFLOPs |
|---|---|---:|---:|---:|
| 2.0.18 | dpr_v1 | 58.276608 | 56.3891712 | 57.1657728 |
| 2.0.18 | cbr_lif_dpr_v1 | 58.672512 | 56.7850752 | 57.5649536 |
| 2.1.6 | dpr_v1 | 59.42076 | 57.5333232 | 58.3099248 |
| 2.1.6 | cbr_lif_dpr_v1 | 59.816664 | 57.9292272 | 58.7091056 |

The missing functional conv is 1.8874368 GFLOPs in every case. Corrected/parent
and deployed/parent-fused differences are exactly zero; missing-conv residuals
are at most 3.78e-15. These totals are measured, never hardcoded into the hook.

Each of five full-model representations is profiled twice on the same isolated
copy. Both versions and variants passed unchanged-state checks for original
and copy: state_dict tensors/keys, parameter and module identities, all buffers
and their identities/persistence, hooks, mixed training flags and input values.
The tiny fixture additionally passes deliberate hook exceptions, pre-existing
counter buffers and user hooks, cold-probe RNG preservation, and repeated
standalone/nested unfolded/deployed DPR counts.

`diagnostics_probe.json.gz` records three deliberately failing scalar cases run
through the actual diagnostic/print/assert AST: each carries all six values
and three residuals. It also verifies the original three assertion expressions,
both complete DPR class ASTs, and 17 protected source/config/protocol files
against the base commit. Compilation and server handoff Bash syntax passed.
Both `check_dpr.py --help` and the server wrapper's real `--help` were checked.
The latter initially failed in the local Windows Bash process because its
POSIX utility PATH was absent; it passed with process-local `/usr/bin:/bin`.
This is not evidence of server Conda activation, which remains pending.

The complete 2.0.18 module report is PASSED. The 2.1.6 report is
**PRECISION_NOTE**, not an unconditional pass: the existing learned synthetic
native-fusion check changes two candidate indices in each variant. Continuous
features and fixed-candidate replay pass the original thresholds; raw output
allclose is false. The existing classification policy and its evidence are
unchanged. This does not alter or dismiss the previous CUDA AMP/half failures.

Both existing controlled initialization files were re-audited successfully,
including public-source/parent equality, serialized reload and actual native
Trainer reconstruction stopped before training. Their bytes were preserved.
The first dpr_v1 initialization-audit attempt failed at local Git ownership
checking because the process-local safe.directory setting was omitted. Its
failure JSON/log are archived; the corrected invocation used a new report path
and no global Git setting.

## Server status and retained failure evidence

The user's server is PyTorch 2.1.2+cu121 / THOP 2.1.6. Its reported pre-fix
dpr_v1 values are: parent 59.42076, raw 57.5333232, corrected 59.4273136,
deploy 60.2039152, parent-fused 58.3099248, missing conv 1.8874368 GFLOPs.
These are user-provided server observations, not a new server execution.

The THOP 2.1.6 counting behavior was reproduced locally, but the exact server
PyTorch/CUDA environment and full bounded preflight remain **PENDING**. No
server login was performed. Previous AMP independent-resume and half failures
remain recorded; this change does not grant formal-start permission.

After commit, a separate fixed-SHA handoff is generated under `outputs/dpr/`.
It validates and advances only the existing clean DPR worktree from the known
base commit, preserving old reports and refusing untracked/ignored collisions.
It then re-audits initialization and both variant structures, and runs the
main variant's complete bounded preflight in a new timestamped directory.

**Formal training NOT_STARTED; independent final evaluation NOT_RUN;
final test NOT_RUN.**

## Evidence

`manifest.json` identifies each archived file by byte size and SHA256; gzip
members are lossless copies of raw local reports/logs. `probe_thop_versions.py`
is the standalone tiny fixture (output path and repository root arguments).
The `local_2_0_18/` and `local_2_1_6/` directories hold module reports; the
former also holds the successful initialization audits and retained failed
attempt. The vendor wheel and extracted dependencies are not committed.
