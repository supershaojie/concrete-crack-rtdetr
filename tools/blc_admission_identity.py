"""Pinned two-segment admission migration; no model or checkpoint execution."""
import ast
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

from blc_admission import VARIANTS, read_report

SOURCE = "0d2335122a09203dfa2ebf4b469a68eede27cd61"
DEPLOYED_ADMISSION = "3906d10280a2f1236ea54a6cef9fb8853ea1f85c"
INIT_SOURCES = (SOURCE, "343df78240277301a03fa520a532312008bf6850",
                "010038a1115cc757d2aba6fad480e73e0a36b820")
ALLOWED_FILES = {
    "tools/blc_admission.py", "tools/blc_admission_identity.py", "tools/check_blc_admission.py",
    "tools/blc_server.py", "tools/blc_server.sh",
    "docs/blc_v1/training_admission_v2/README.md",
    "docs/blc_v1/training_admission_v2/validation.json",
}
AMP_FIX_FILES = {
    "tools/blc_admission.py", "tools/blc_admission_identity.py", "tools/check_blc_admission.py",
    "docs/blc_v1/admission_amp_skip_fix/README.md",
    "docs/blc_v1/admission_amp_skip_fix/validation.json",
}
# Exact allowed entry-point edits. The remainder of the server AST must match SOURCE.
# This also protects start/resume computation, init, pack, and val/test dispatch.
SERVER_EDITS = (
    ('from blc_probe_io import temporary_probe, retained_size, compact_audit',
     'from blc_probe_io import temporary_probe, retained_size, compact_audit\n'
     'from blc_admission import evaluate as admission_evaluate, read_report, reassess, require_admission'),
    ('def require_evidence(variant, phase):\n    current = evidence_context(variant)',
     'def require_evidence(variant, phase):\n'
     '    if phase == "preflight":\n        return require_admission(variant)\n'
     '    current = evidence_context(variant)'),
    ('    if phase == "preflight":\n'
     '        require(report["capacity"]["status"] == "PASSED" and report["capacity"]["batch"] == 16\n'
     '                and report["capacity"]["imgsz"] == 640 and report["capacity"]["AMP"] is True\n'
     '                and report["capacity"]["effective_updates"] >= 2, "Missing capacity evidence")\n'
     '        require(report["lifecycle"]["status"] == report["native_half_ema_epoch_val"]["status"] == report["native_resume"]["status"] == "PASSED",\n'
     '                "Missing lifecycle/half EMA/native resume evidence")\n', ''),
    ('    for other in VARIANTS:\n        require_evidence(other, "init-preflight")',
     '    init_reports = {other: read_report(require_evidence(other, "init-preflight")) for other in VARIANTS}'),
    ('report = dict(status="FAILED", phase="preflight", variant=variant, stage="native_amp_resources",',
     'report = dict(status="PENDING", phase="preflight", variant=variant, stage="native_amp_resources",'),
    ('        report["status"] = "PASSED" if all(report[key]["status"] == "PASSED" for key in\n'
     '            ("capacity", "lifecycle", "native_half_ema_epoch_val", "native_resume")) else "PENDING"\n', ''),
    ('        report["retention_target_bytes"] = 10 * 1024 * 1024\n'
     '        write_json(folder/"report.json", report)',
     '        report["retention_target_bytes"] = 10 * 1024 * 1024\n'
     '        report.update(admission_evaluate(report, init_reports))\n'
     '        report["status"] = report["training_admission"]["status"]\n'
     '        write_json(folder/"report.json", report)'),
    ('        raise SystemExit("PENDING: capacity finished but lifecycle precision evidence requires review; start remains blocked")',
     '        raise SystemExit("Training admission blocked; inspect report.training_admission.errors")\n'
     '    reassess(variant, folder/"report.json")'),
    ('    require_evidence(variant, "init-preflight")\n'
     '    # The single-module ablation', '    # The single-module ablation'),
    ('"val", "test", "pack"])', '"val", "test", "pack", "reassess", "admission"])'),
    ('    args = parser.parse_args()', '    parser.add_argument("--from-report", type=Path, help="Existing preflight JSON; reassess only")\n'
     '    args = parser.parse_args()\n'
     '    require((args.from_report is not None) == (args.action == "reassess"), "--from-report is required only for reassess")'),
    ('        elif args.action == "plan":',
     '        elif args.action == "reassess":\n            reassess(variant, args.from_report)\n'
     '        elif args.action == "admission":\n            require_admission(variant)\n'
     '        elif args.action == "plan":'),
)
SHELL_OLD = "preflight|plan|start|resume|val|test|pack [--both]"
SHELL_NEW = "preflight|reassess|admission|plan|start|resume|val|test|pack [--both] [--from-report PATH]"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(root, *args, input=None):
    return subprocess.check_output(["git", *args], cwd=root, input=input)


def tracked_code(name):
    p = Path(name)
    return (name.startswith("ultralytics-main/ultralytics/") and p.suffix == ".py"
            or name.startswith("ultralytics-main/ultralytics/cfg/") and p.suffix == ".yaml"
            or p.parent.as_posix() == "tools" and (p.suffix == ".py" or "blc" in p.name and p.suffix == ".sh")
            or name in ("docs/blc_v1/parent_args.yaml", "docs/blc_v1/parent_dataset_inventory.json"))


@lru_cache(maxsize=8)
def tree(root, revision):
    rows = git(root, "ls-tree", "-r", revision).decode().splitlines()
    refs = {line.split("\t", 1)[1]: line.split()[2] for line in rows
            if tracked_code(line.split("\t", 1)[1])}
    payload = git(root, "cat-file", "--batch", input=("\n".join(refs.values()) + "\n").encode())
    offset, blobs = 0, {}
    for name in refs:
        end = payload.index(b"\n", offset)
        header = payload[offset:end].split()
        need(header[1] == b"blob", "Expected Git source blob: " + name)
        size = int(header[2])
        blobs[name] = payload[end + 1:end + 1 + size].replace(b"\r\n", b"\n")
        offset = end + size + 2
    return blobs


def manifest(blobs):
    rows = {name: sha(raw) for name, raw in blobs.items()}
    return dict(sha256=sha(json.dumps(rows, sort_keys=True).encode()), files=rows)


def expected_server(source):
    for old, new in SERVER_EDITS:
        need(source.count(old) == 1, "Entry-point patch no longer uniquely applies: " + old[:80])
        source = source.replace(old, new, 1)
    return source


def syntax(source):
    return ast.dump(ast.parse(source), include_attributes=False)


def scope_check(before, after, changed, allowed=ALLOWED_FILES):
    need(set(changed) <= allowed, "Unknown changed files: " + repr(sorted(set(changed) - allowed)))
    need(syntax(after["tools/blc_server.py"]) == syntax(expected_server(before["tools/blc_server.py"].decode())),
         "Server changes exceed the exact admission/CLI edits; computation protected")
    expected_shell = before["tools/blc_server.sh"].decode().replace(SHELL_OLD, SHELL_NEW)
    need(after["tools/blc_server.sh"].decode() == expected_shell, "Shell execution changed beyond usage text")
    for name in set(before) | set(after):
        if name not in allowed:
            need(before.get(name) == after.get(name), "Protected source changed: " + name)


def amp_fix_scope_check(before, after, changed):
    need(set(changed) <= AMP_FIX_FILES, "Unknown AMP-fix changed files: " + repr(sorted(set(changed) - AMP_FIX_FILES)))
    for name in set(before) | set(after):
        if name not in AMP_FIX_FILES:
            need(before.get(name) == after.get(name), "AMP fix changed protected source: " + name)


def changed_files(root, before_revision, after_revision):
    files = []
    for name in git(root, "diff", "--name-only", before_revision, after_revision).decode().splitlines():
        # Include docs in the scoped commit proof, even though they are not runtime inputs.
        old = git(root, "ls-tree", before_revision, "--", name)
        new = git(root, "ls-tree", after_revision, "--", name)
        files.append(dict(path=name,
            before_lf_sha256=sha(git(root, "show", before_revision + ":" + name).replace(b"\r\n", b"\n")) if old else None,
            after_lf_sha256=sha(git(root, "show", after_revision + ":" + name).replace(b"\r\n", b"\n")) if new else None))
    return files


def current_proof(root, current):
    head = git(root, "rev-parse", "HEAD").decode().strip()
    chain = ((SOURCE, DEPLOYED_ADMISSION), (DEPLOYED_ADMISSION, head))
    need(head not in (SOURCE, DEPLOYED_ADMISSION), "AMP admission fix requires its new committed revision")
    for parent, child in chain:
        parents = git(root, "rev-list", "--parents", "-n", "1", child).decode().split()
        need(parents == [child, parent], "Migration is limited to the exact SOURCE -> 3906d102 -> AMP-fix chain")
    need(not git(root, "status", "--porcelain", "--untracked-files=no").strip(), "Tracked changes: commit/review first")
    before, deployed, after = (tree(str(root), revision) for revision in (SOURCE, DEPLOYED_ADMISSION, head))
    need(current["code"] == manifest(after), "Current LF-normalized code manifest differs from committed source")
    segments = [dict(from_revision=parent, to_revision=child, actual_changed_files=changed_files(root, parent, child))
                for parent, child in chain]
    scope_check(before, deployed, [r["path"] for r in segments[0]["actual_changed_files"]])
    amp_fix_scope_check(deployed, after, [r["path"] for r in segments[1]["actual_changed_files"]])
    files = changed_files(root, SOURCE, head)
    # Retain the original entry-point AST proof as well as the narrower repair diff.
    scope_check(before, after, [r["path"] for r in files], ALLOWED_FILES | AMP_FIX_FILES)
    return dict(source_revision=SOURCE, current_revision=head, actual_changed_files=files,
                migration_chain=segments,
                source_code_sha256=manifest(before)["sha256"], current_code_sha256=current["code"]["sha256"],
                scope="Two explicit parent edges and scoped diffs; original entry AST and all computation inputs protected")


def named_nodes(raw, names):
    nodes = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names}
    need(set(nodes) == set(names), "Initialization proof missing named dependency")
    return nodes


def init_semantics(root, revision):
    before, after = tree(str(root), revision), tree(str(root), SOURCE)
    changed = {p for p in set(before) | set(after) if before.get(p) != after.get(p)}
    # These are reviewed historical reporting/probe changes, not permanent exclusions.
    historical = {"tools/blc_common.py", "tools/blc_server.py", "tools/blc_lifecycle.py",
                  "tools/blc_preflight.py", "tools/blc_probe_io.py",
                  "tools/check_blc_backend.py", "tools/check_blc_identity.py"}
    need(changed <= historical, "Unknown initialization dependency change: " + repr(changed - historical))
    protected = {
        "tools/blc_common.py": ("paths", "build", "verify_model", "audit_states", "rebuild",
                                "initialize", "BLCTrainer", "recipe", "tensor_hash"),
        "tools/blc_preflight.py": ("ProbeComplete", "ReconstructionProbe", "api_reconstruction"),
        "tools/blc_server.py": ("init_one",),
    }
    for name, nodes in protected.items():
        need(named_nodes(before[name], nodes) == named_nodes(after[name], nodes), "Initialization semantics changed: " + name)
    # All vendor sources/configs, data inventory/recipe, original initializer and
    # topology/data helpers are outside historical and therefore byte-verified above.
    # The three fixed historical trees are immutable: common.py's only other
    # changes were JSON report serialization/canonicalization, after strict data checks.
    return dict(verified_revision=revision, to_source_revision=SOURCE,
                historical_changed_files=sorted(changed), initialization_functions_exact=True,
                original_initializers_models_configs_recipe_data_helpers_exact=True,
                historical_reporting_changes="Only the enumerated immutable BLC history is supported")


def context_relation(recorded, current):
    need(isinstance(recorded, dict) and set(recorded) == set(current), "Missing/extra context fields")
    before, after = deepcopy(recorded), deepcopy(current)
    before.pop("code"); after.pop("code")
    need("commit" in before["runtime"] and "commit" in after["runtime"], "Missing runtime commit observation")
    before["runtime"].pop("commit"); after["runtime"].pop("commit")
    mismatches = [key for key in after if before.get(key) != after[key]]
    need(not mismatches, "Stale evidence context: " + ", ".join(mismatches))
    return dict(unchanged_fields=sorted(after), runtime_except_observed_commit_exact=True,
                observed_old_commit=recorded["runtime"]["commit"], current_commit=current["runtime"]["commit"],
                old_code_sha256=recorded["code"]["sha256"], new_code_sha256=current["code"]["sha256"])


def evidence_proof(root, report, current, kind):
    recorded = report.get("context", {})
    candidates = ((SOURCE, DEPLOYED_ADMISSION, current["runtime"]["commit"]) if kind == "preflight"
                  else (*INIT_SOURCES, DEPLOYED_ADMISSION, current["runtime"]["commit"]))
    revision = next((rev for rev in dict.fromkeys(candidates)
                     if recorded.get("code") == manifest(tree(str(root), rev))), None)
    need(revision is not None, kind + ": context.code.files/aggregate does not match any supported source tree")
    proof = dict(verified_source_revision=revision, verified_file_count=len(recorded["code"]["files"]),
                 context_relation=context_relation(recorded, current))
    if kind == "init":
        proof["initialization_migration"] = init_semantics(root, revision if revision in INIT_SOURCES else SOURCE)
        # Embedded source audit identity must agree with the outer audit.
        need(report.get("current_source_audit", {}).get("code") == recorded["code"],
             "Initialization source audit and outer context disagree on code identity")
    return proof


def binding(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha(path.read_bytes()))


def verify_bindings(bindings):
    need(set(bindings) == {"preflight", *VARIANTS}, "Missing raw preflight/initialization bindings")
    for key, saved in bindings.items():
        need(binding(saved["path"]) == saved, "Raw evidence SHA256/path changed: " + key)


def collect(variant, from_report, saved_bindings=None):
    from blc_common import ROOT, SOURCE_SHA256, evidence_context, paths
    reports, bindings, contexts, proof, errors = {}, {}, {}, {}, []
    locations = {"preflight": Path(from_report)}
    for other in VARIANTS:
        if saved_bindings:
            locations[other] = Path(saved_bindings[other]["path"])
        else:
            choices = sorted(paths(other)["evidence"].glob("init-preflight-*/report.json"))
            if choices:
                locations[other] = choices[-1]
            else:
                errors.append("Missing init-preflight report for " + other)
    for key, path in locations.items():
        try:
            bindings[key] = binding(path)
            reports[key] = read_report(path)
            need(isinstance(reports[key], dict), key + ": report must be an object")
        except (OSError, ValueError) as error:
            errors.append(key + ": " + str(error))
            reports[key] = {}
    for other in VARIANTS:
        try:
            contexts[other] = evidence_context(other)  # Full existing dataset + weights + recipe checks.
            need(contexts[other]["source_sha256"] == SOURCE_SHA256, "Specified public source SHA256 changed")
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
            errors.append(other + ": current context unavailable: " + str(error))
    try:
        proof["current_revision_scope"] = current_proof(ROOT, contexts[variant])
        for key, report in reports.items():
            target = variant if key == "preflight" else key
            try:
                proof[key] = evidence_proof(ROOT, report, contexts[target], "preflight" if key == "preflight" else "init")
            except (ValueError, KeyError, subprocess.CalledProcessError) as error:
                errors.append(key + ": identity migration: " + str(error))
    except (ValueError, KeyError, subprocess.CalledProcessError) as error:
        errors.append("Current source migration: " + str(error))
    # Prevent source-report races between hash, parse and identity checks.
    if set(bindings) == {"preflight", *VARIANTS}:
        try:
            verify_bindings(bindings)
        except (ValueError, OSError) as error:
            errors.append(str(error))
    return reports.get("preflight", {}), {v: reports.get(v, {}) for v in VARIANTS}, bindings, contexts, proof, errors
