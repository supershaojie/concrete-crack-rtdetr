# Admission gate unit fixture

`check_admission_fixture.py` is a mocked gate-logic test. Its runtime, CUDA flags,
capacity metrics and A/B tensor metrics are synthetic fixtures. It does not
perform server preflight, grant a start permit, run training, or run final test.
`admission_fault_checks.json` is exclusively the result of that unit test.

Run from the DPR worktree with the project Python environment:

```bash
python docs/dpr/check_admission_fixture.py
```

The fixture uses the existing, unmodified local reports at
`outputs/dpr/cbr_lif_dpr_v1/init.json` and `outputs/dpr/module_check_03.json`.
The latter must match the current mathematical-source content hashes. If those
local artifacts are absent, produce new initialization/mathematical evidence
with the documented tools and update the fixture's two explicit input paths;
do not invent replacement passed reports. It writes only the small unit-test
report `docs/dpr/admission_fault_checks.json`.

The positive fixture checks schema compatibility. Negative cases remove hard
checks, corrupt raw tensor evidence, substitute CPU for CUDA, and supply
unsupported precision notes. A supported candidate-index precision note is also
accepted without changing the underlying FP32 tolerance. These unit tests never
substitute for the real device, detection-loss, save/resume, or capacity checks.
