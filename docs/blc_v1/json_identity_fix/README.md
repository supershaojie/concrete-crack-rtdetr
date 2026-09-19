# BLC JSON identity repair

Baseline: `010038a1115cc757d2aba6fad480e73e0a36b820`.

`dataset_identity()` previously returned the YAML class map with integer key `0`.
JSON persistence converts that key to the string `"0"`, so the unchanged
`require_evidence()` comparison rejected valid init-preflight evidence as stale data.

The production change only normalizes the returned identity through strict JSON
serialization/deserialization, **after** the original split/class check and exact
full-inventory comparison. The inventory, parsed YAML, resolved root and YAML
SHA256 all remain in the comparison. Input YAML still requires `{0: "crack"}`.
No model, CBR/LIF, recipe, precision threshold, evidence gate or existing report
has been changed. Existing controlled initialization files are retained.

## Local verification

- `identity.json`: PASSED. Executes the historical function extracted from the
  baseline commit and reproduces its failed save/read comparison using the real
  6048/1728/864 dataset inventory. The fixed function differs only in JSON
  representation. Both variants' complete `evidence_context` values round-trip
  without mismatches in any field, including recipe and runtime.
- The unchanged gate accepts the serialized fixed context and rejects image-list,
  label-hash, class-name, train/val/test-path, resolved-root and YAML-hash changes.
  Equivalent-inventory relocation and a comment-only YAML change are also rejected.
- The validations before normalization still reject changed names/splits, a string
  class key in the input YAML, and changes to all four inventory fields in each
  split (image count, box count, image-list SHA256, label-inventory SHA256).
  Fault injection uses isolated fixtures/mocks; the real dataset is never modified.
- `ops-gates.json`: PASSED. Existing failure/PENDING, stale identity/runtime,
  capacity and lifecycle gates remain enforced.
- `update-fixtures.json`: isolated Git fixtures exercise the delivered update
  procedure: known-old update and repeat invocation succeed; dirty tracked files,
  unexpected HEAD, untracked collisions and ignored collisions stop safely.
  Existing weights, reports, user notes and main-worktree state remain intact.
  This is local verification, not server execution.

The reports were generated before this repair commit: `runtime.commit` records
the baseline HEAD, while `code_sha256` identifies the actual repaired working
files. No historical report was rewritten to claim a new code identity.

Local rerun (activate the existing environment and set `BLC_MAIN`, `BLC_DATA`,
and `PYTHONPATH` to the actual local repository, data YAML, vendored ultralytics
and tools):

```sh
python tools/check_blc_identity.py
python tools/check_blc_ops.py
```

`check_blc_identity.py` needs the baseline Git object, real data, source checkpoint
and both existing controlled initialization files. It only reads their identities;
it does not load models, run forward, train or evaluate. Reports use new exclusive
filenames under `outputs/blc_v1`.

## Server handoff

The final fixed-SHA commands are delivered after commit in a separate handoff.
The update verifies repository/worktree identity, accepts only the known old HEAD
or the exact fixed SHA, refuses tracked changes and uses detached checkout with
`--no-overwrite-ignore`. Fetch uses the existing bounded HTTP/1.1 procedure only
if the fixed object is missing. It never resets/cleans or switches the main tree.

The repaired source changes code identity. Preserve the old reports and both
controlled initialization files, and execute these actions on the fixed SHA:

```sh
BLC_VARIANT=cbr_lif_blc_v1 bash tools/blc_server.sh init-preflight --both
BLC_VARIANT=cbr_lif_blc_v1 bash tools/blc_server.sh preflight
```

The second action must only run after the first exits successfully. Do not run
`init` to regenerate the persistent weights. The existing `init-preflight`
implementation creates a temporary reference to re-audit every saved tensor;
the existing controlled initialization file is not overwritten.

Fixed-code server `init-preflight --both`: **PENDING**.
Fixed-code server main-combination `preflight`: **PENDING**.
No server login, formal training or final test was performed for this repair.
The local JSON/gate regressions do not certify server capacity or precision.
