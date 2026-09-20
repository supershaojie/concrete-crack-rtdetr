# BLC training admission v2

This authorized rule revision separates permission to begin a native training
experiment from unfused/fused numerical diagnostics. Its source revision is
0d2335122a09203dfa2ebf4b469a68eede27cd61.

training_admission.status=PASSED only means the recorded native execution
evidence permits an experiment. It does not prove fusion equivalence, convergence,
accuracy improvement, or that observed differences are harmless or caused by BLC.
Original FP32/AMP/half fusion statuses and every natural/replay/feature/score,
candidate-position and candidate-set observation are copied unchanged into
fusion_diagnostic.modes. AMP/half PENDING stays PENDING. Differing positions do
not measure the number of newly selected queries or a detection accuracy drop.

The shared evaluator in tools/blc_admission.py is called by new preflight
summaries, read-only reassessment, and the normal start/resume admission gate.
It checks both source/structure/actual Trainer initialization audits, B16/640
native AMP, at least two effective updates and finite nonzero upstream gradients,
exact same-dtype restoration, a nonzero BLC increment, one real native half EMA
validation batch, native optimizer/scaler/EMA/epoch restoration and a resumed
update. All three same-file AutoBackend checks must pass with independent state
and an active nonzero branch. All fusion diagnostics must execute and be finite.
Missing evidence, NOT_RUN, runtime errors, nonfinite values, identity mismatches
and inconsistent update counts still block. The combined 16 training batch and
one validation batch budgets are unchanged. No tolerance or computation changed.

## Reusing the existing server evidence

Use the delivered fixed-SHA update commands first, then:

    export BLC_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
    export BLC_DATA="$BLC_MAIN/configs/crack_autodl.yaml"
    export BLC_VARIANT=cbr_lif_blc_v1
    cd /root/autodl-tmp/projects/Crack_RTDETR-blc-v1
    bash tools/blc_server.sh reassess --from-report \
      /root/autodl-tmp/projects/Crack_RTDETR-blc-v1/outputs/blc_v1/cbr_lif_blc_v1/preflight-20260919T193011968491Z/report.json

Reassessment hashes and reads existing JSON. It never deserializes a checkpoint,
constructs a model, runs a forward/backward pass, trains, validates, or downloads
resources. Existing controlled_init weights, old reports, logs and failures stay
in place. It exclusively creates a new admission-UTC/report.json; errors list
specific missing/conflicting evidence and return nonzero without triggering a
preflight rerun. The wrapper also creates its usual small log and exit-code file.
No init, init-preflight or preflight rerun is required by this migration.

tools/blc_admission_identity.py verifies the original preflight's entire
LF-normalized context.code.files and its aggregate against the fixed source.
Fresh preflights may instead match the current reviewed revision. Each init
audit is separately matched to the actual file manifest of 010038a, 343df78,
0d23351 or the current revision. A recorded runtime HEAD is only an observation;
precommit reports can correctly match a later committed content tree.

The supported older init trees are immutable and explicitly enumerated.
Initialization/reconstruction functions are compared by AST, while all original
initializer dependencies, model sources/configs, recipe and data helper files
must be unchanged. Historical differences are explicitly enumerated reporting
and probe files; they are not blanket tools exclusions. Source and controlled_init
hashes, full data inventory/YAML/root identity, full recipe and the entire runtime
except the observed commit must agree with current context.

Migration is limited to a clean, single commit directly after the source revision.
Only the seven exact admission/CLI/test/document paths enumerated in
ALLOWED_FILES may differ. The server AST must equal the source with the exact
listed admission/CLI edits applied; the shell may change only usage text. All
other runtime files are byte-checked with LF normalization. New admission code
and tests remain in the existing full code identity. Future or unknown changes
need a new reviewed migration, and are not automatically accepted.

The new report binds the full current contexts of both variants, original JSON
SHA256/path bindings, verified source revisions, actual changed-file hashes and
the old/new identity relation. admission, start and resume reread/hash the
bound originals, revalidate migration/current context, and rerun the same
evaluator. Editing a PASSED summary alone cannot open the gate. The latest
admission report is used, including a latest failed reassessment.

After reassessment succeeds, bash tools/blc_server.sh admission checks the
binding without writing a new report. Normal start repeats that gate and uses
the original controlled_init with the unchanged 109-field parent recipe: 200
epochs, B16, 640, seed 42, native AdamW/AMP/half EMA and online augmentation.
An existing formal run is preserved and blocks start; an unfinished run must use
the existing checked native resume entry. No deleted temporary probe checkpoint
is used for formal training. Single-module ablation still requires explicit later
review. Formal val/test and best selection are unchanged.

## Local validation and limits

python tools/check_blc_admission.py runs 10 metadata-only tests with synthetic
FIXTURE evidence clearly separated from real server evidence. They cover finite
fusion differences, failed/missing updates and validation, same-file failures,
NaN/Inf, missing diagnostics, branch/state failures, context changes, original
JSON mutations, unknown revisions, restricted diffs and the complete reassess/
start gate. Existing local initialization JSON is read to check actual schema and
source manifests, without rerunning initialization.

Python syntax, CLI help and Bash syntax are checked. The source review verifies
that BLC/CBR/LIF, both model YAMLs, initialization, dataset/recipe, Trainer math,
loss/optimizer, AMP/half EMA, precision checks/tolerances, best selection and
formal val/test calculation are untouched.

This task adds zero GPU training batches and zero model checkpoints. Committed
code/docs plus small local verification records and an expected server admission
report should remain below 1 MiB, within the 10 MiB target. Each reassessment
retains its new report/log; repeated invocations naturally accumulate records.

The actual server preflight JSON is absent locally. Server reassessment is
**PENDING** until the user runs the commands against that file. No server admission,
formal training, final test, convergence or accuracy result is claimed here.
