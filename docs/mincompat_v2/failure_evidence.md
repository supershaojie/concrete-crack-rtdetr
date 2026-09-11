# v1 failure evidence

Read before production edits. The two failure packages were identified from their internal `test/metrics.json` `variant` values, not their outer filenames. Both report commit `05e6de8b582f4a5f6c11cd50f8bc0406dacd7d0b`, clean source, 864 test images and the same SHA-locked C2 initialization. Their archived implementation files match the preserved v1 source. Training has 109 typed C2 fields; only model/name/save_dir differ. The original C2 archive's train_run/args.yaml also matches our reference in all 109 fields.

| Model | Precision | Recall | mAP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
|CSCEF-v6|0.753867062836|0.751013057181|0.788689347806|0.395762350015|0.416523743097|
|SCCA-v2|0.803170512796|0.777127420081|0.829686909229|0.428146003059|0.442671778186|
|C17|0.844895323441|0.825003752064|0.879297454830|0.532829775858|0.512045085044|
|C19|0.849033844558|0.837311905432|0.885186356567|0.519995255215|0.503892419455|
|C24|0.855563273975|0.819900945520|0.878243875820|0.506461002298|0.499872058481|
|C20|0.837320605748|0.814948221522|0.866302563893|0.494810709544|0.487043012576|
|C25|0.793837773204|0.777127420081|0.818477876119|0.404977566059|0.425179978084|
|C26|0.842167659938|0.814948221522|0.860372099817|0.496019333872|0.490730335855|
|C2|0.833960007851|0.808044424433|0.858162985651|0.458360400552|0.469636231908|

CSCEF-v6 loses 9.552134 percentage points of mAP50-95 from C17; SCCA-v2 loses 5.720028 points from C24. These observations falsify preservation of standalone capability for these runs. They do not isolate individual causal effects or prove the sign of task-gradient conflicts.

The attached C19 small package contains test evidence; historical C19 source is verified against its full git commit. Original C24 and C2/C20/C25/C26 archives were found alongside the supplied paths and read. Raw selected metrics are in evidence/; exact archive/member paths, SHA256, available sidecar checks, typed recipe differences, checkpoint hashes and failure source records are in [attachment_audit.json](attachment_audit.json).

Historical evaluation provenance remains distinct: C24/C25/C26 use the corrected sorted-confidence-mask policy and record zero affected test images. C20/C25/C26 effective evaluation seed was 0 in historical reports; the v2 entry fixes seed=42. No historical metric was regenerated or silently harmonized.
