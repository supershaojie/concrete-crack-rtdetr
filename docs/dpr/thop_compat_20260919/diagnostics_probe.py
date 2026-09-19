"""Bounded AST diagnostics/scope audit; executes no model or training code."""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = "bf23c6d4ad43aa2182e11e8a0726cef950425728"
OUT = Path(__file__).with_suffix(".json")
assert not OUT.exists(), "Refusing to overwrite an earlier audit"
env = dict(os.environ)
env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="safe.directory", GIT_CONFIG_VALUE_0=ROOT.as_posix())


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, env=env)


def source(path, baseline=False):
    data = git("show", f"{BASE}:{path}") if baseline else (ROOT / path).read_bytes()
    return data.replace(b"\r\n", b"\n")


def structure(tree):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "structure")


def flops_assertions(function):
    return [node for node in function.body if isinstance(node, ast.Assert) and
            any(isinstance(child, ast.Name) and child.id == "parent_flops" for child in ast.walk(node.test)) or
            isinstance(node, ast.Assert) and any(isinstance(child, ast.Name) and child.id == "parent_fused_flops"
                                                 for child in ast.walk(node.test))]


path = "tools/check_dpr.py"
current = structure(ast.parse(source(path)))
original = structure(ast.parse(source(path, True)))
current_asserts, original_asserts = flops_assertions(current), flops_assertions(original)
assert len(current_asserts) == len(original_asserts) == 3
for actual, expected in zip(current_asserts, original_asserts):
    assert ast.dump(actual.test) == ast.dump(expected.test), "FLOPs assertion expression changed"
    assert isinstance(actual.test, ast.Compare) and isinstance(actual.test.comparators[0], ast.Constant)
    assert actual.test.comparators[0].value == 1e-9

start = next(i for i, node in enumerate(current.body) if isinstance(node, ast.Assign) and
             any(isinstance(target, ast.Name) and target.id == "flops_diagnostic" for target in node.targets))
end = current.body.index(current_asserts[-1]) + 1
fragment = ast.Module(body=current.body[start:end], type_ignores=[])
compiled = compile(ast.fix_missing_locations(fragment), "actual_structure_diagnostics_ast", "exec")
fields = ("parent_flops", "raw_dpr_flops", "corrected_flops", "deploy_flops", "parent_fused_flops", "missed_conv_flops")
values = dict(zip(fields, (100.0, 90.0, 100.0, 80.0, 80.0, 10.0)))
cases = []
for index, (changed, value) in enumerate((("corrected_flops", 100.1), ("raw_dpr_flops", 89.9), ("deploy_flops", 80.1))):
    namespace = {"json": json, "variant": "controlled_assertion_fixture", "thop_semantics": {"mode": "fixture"}, **values}
    namespace[changed] = value
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(compiled, namespace)
    except AssertionError as error:
        message = str(error)
    else:
        raise AssertionError(f"Expected assertion {index + 1} failure")
    prefix = "DPR FLOPs diagnostics: "
    assert message.startswith(prefix)
    diagnostic = json.loads(message[len(prefix):])
    assert all(diagnostic[key] == namespace[key] for key in fields)
    differences = diagnostic["differences"]
    assert len(differences) == 3 and diagnostic["absolute_tolerance"] == 1e-9
    expected_residuals = (namespace["parent_flops"] - namespace["corrected_flops"],
                          namespace["parent_flops"] - namespace["raw_dpr_flops"] - namespace["missed_conv_flops"],
                          namespace["parent_fused_flops"] - namespace["deploy_flops"])
    assert tuple(differences[key] for key in ("parent_minus_corrected", "parent_minus_raw_minus_missed",
                                             "parent_fused_minus_deploy")) == expected_residuals
    assert all(abs(value) < 1e-9 for value in expected_residuals[:index])
    assert abs(expected_residuals[index]) >= 1e-9
    assert output.getvalue().strip() == message
    cases.append({"assertion_number": index + 1, "status": "PASSED", "diagnostics": diagnostic})

path = "ultralytics-main/ultralytics/nn/modules/dpr.py"
classes = []
for name in ("DPRConvNormLayer", "BlocksDPR"):
    actual = next(node for node in ast.parse(source(path)).body if isinstance(node, ast.ClassDef) and node.name == name)
    expected = next(node for node in ast.parse(source(path, True)).body if isinstance(node, ast.ClassDef) and node.name == name)
    assert ast.dump(actual) == ast.dump(expected), f"Original model class changed: {name}"
    classes.append(name)

sensitive = [
    "tools/init_dpr.py", "tools/train_dpr.py", "tools/preflight_dpr.py", "tools/eval_dpr.py",
    "tools/dpr_checkpoint.py", "tools/dpr_data.py", "tools/dpr_server.sh", "tools/sync_dpr.sh",
    "ultralytics-main/ultralytics/nn/modules/cbr.py", "ultralytics-main/ultralytics/nn/modules/lif_down.py",
    "ultralytics-main/ultralytics/nn/tasks.py", "ultralytics-main/ultralytics/nn/modules/__init__.py",
    "docs/dpr/parent_args.yaml", "docs/dpr/recipe_diff.json", "docs/dpr/parent_dataset_inventory.json",
    "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-dpr-v1.yaml",
    "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-dpr-v1.yaml",
]
hashes = {}
for path in sensitive:
    actual, expected = source(path), source(path, True)
    assert actual == expected, f"Sensitive file changed: {path}"
    hashes[path] = hashlib.sha256(actual).hexdigest()

report = {"status": "PASSED", "base_sha": BASE,
          "method": "AST executes only actual structure diagnostic construction/print/three asserts with controlled scalar fixtures; no model, train or test run",
          "assertion_tests_identical_to_base": True, "absolute_tolerance": 1e-9,
          "failure_cases": cases, "unchanged_complete_model_class_ast": classes,
          "unchanged_sensitive_source_sha256_lf": hashes,
          "git_diff_name_only": git("diff", "--name-only", BASE).decode().splitlines()}
with OUT.open("x", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps({"status": report["status"], "assertion_cases": len(cases),
                  "unchanged_classes": classes, "unchanged_sensitive_files": len(hashes), "output": str(OUT)}, indent=2))
