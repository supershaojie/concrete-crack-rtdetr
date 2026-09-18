"""Inert orchestration fixtures for autodl_dcc.sh; never imports Torch or runs a model.

The real shell wrapper runs in disposable directories containing deliberately
fake Python entry points. These fixtures verify control flow and permit handling;
they do not claim that the mocked mathematical or model checks have run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "dcc_acceptance_v2"

COMMON_STUB = r'''
import hashlib, json, os
from pathlib import Path
def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def event(name):
    with Path(os.environ['DCC_FIXTURE_EVENTS']).open('a',encoding='utf-8') as f:
        f.write(json.dumps(name)+'\n')
'''

TRAIN_STUB = r'''
import json, sys
from pathlib import Path
from dcc_common import sha256, event
CONTRACT='dcc_acceptance_v2'
def code_identity(): return {'fixture_source':'current'}
def strict_gate(preflight_path, checks_path, variant, init, data):
    event('strict_gate')
    checks=json.loads(Path(checks_path).read_text())
    capacity=json.loads(Path(preflight_path).read_text())
    init_record=checks['prerequisites']['initialization']
    math_record=checks['prerequisites']['math']
    initialization=json.loads(Path(init_record['path']).read_text())
    math=json.loads(Path(math_record['path']).read_text())
    reports=[(checks,'full_preflight_engineering'),(capacity,'native_capacity'),
             (initialization,'controlled_initialization_audit'),(math,'dcc_math_audit')]
    for report,kind in reports:
        assert report.get('contract_version')==CONTRACT, 'stale contract'
        assert report.get('report_kind')==kind, 'wrong report kind'
        assert report.get('code_identity')==code_identity(), 'source changed'
        assert report.get('runtime')=={'fixture_runtime':'current'}, 'runtime changed'
        assert report.get('status') in ('PASSED','PRECISION_NOTE'), 'failed required check'
    assert capacity['status']=='PASSED' and initialization['status']=='PASSED' and math['status']=='PASSED'
    assert checks['fusion']=='PASSED', 'fusion failed'
    assert sha256(init_record['path'])==init_record['sha256'], 'initializer report modified'
    assert sha256(math_record['path'])==math_record['sha256'], 'math report modified'
    return dict(contract_version=CONTRACT,initialization_report_sha256=sha256(init_record['path']),
                math_report_sha256=sha256(math_record['path']),checks_report_sha256=sha256(checks_path),
                capacity_report_sha256=sha256(preflight_path))
if __name__=='__main__':
    # Reached only by an explicit start/resume fixture. There is no model code.
    mode=sys.argv[1]; event('explicit_'+mode+'_stub_entry')
    value=lambda flag: sys.argv[sys.argv.index(flag)+1]
    strict_gate(value('--preflight'),value('--checks'),value('--variant'),value('--init'),value('--data'))
    event('formal_stub_admitted')
'''

STAGE_STUB = r'''
import json, os, sys
from pathlib import Path
from dcc_common import write_json, sha256, event
from train_dcc import code_identity
stage=Path(sys.argv[0]).stem
scenario=os.environ['DCC_FIXTURE_SCENARIO']
event(stage)
value=lambda flag: sys.argv[sys.argv.index(flag)+1]
report=dict(status='PASSED',contract_version='dcc_acceptance_v2',code_identity=code_identity(),
            runtime={'fixture_runtime':'current'})
exit_code=0
if stage=='init_dcc':
    report['report_kind']='controlled_initialization_audit'
    dest=Path(value('--report'))
    init=Path(value('--output'));init.parent.mkdir(parents=True,exist_ok=True)
    if not init.exists(): init.write_bytes(b'fixture-only-not-a-checkpoint')
    if scenario=='initializer_failed': report['status']='FAILED'; exit_code=21
elif stage=='check_dcc_math':
    report['report_kind']='dcc_math_audit';dest=Path(value('--output'))
    if scenario=='math_failed': report['status']='FAILED';exit_code=22
elif stage=='check_dcc':
    report['report_kind']='full_preflight_engineering';report['fusion']='PASSED'
    dest=Path(value('--output'))/'checks.json'
    report['prerequisites']={name:dict(path=value(flag),sha256=sha256(value(flag)))
                            for name,flag in [('initialization','--initialization-report'),('math','--math-report')]}
    if scenario=='fusion_failed': report['status']='FAILED';report['fusion']='FAILED'
    if scenario in ('full_precision_note','full_then_changed_math'):report['status']='PRECISION_NOTE'
    if scenario=='full_legacy_rejected':report['contract_version']='dcc_acceptance_v1'
elif stage=='preflight_dcc':
    report['report_kind']='native_capacity';dest=Path(value('--output'))/'preflight.json'
    if scenario=='capacity_failed':report['status']='FAILED';exit_code=24
elif stage=='check_dcc_resume':
    report['report_kind']='partial_resume_diagnostic';dest=Path(value('--output'))/'resume_checks.json'
    report['status']='FAILED' if scenario=='partial_failed' else 'PRECISION_NOTE'
    report['devices']={key:dict(acceptance=dict(A='PASSED',B='PASSED'),dcc=dict(checkpoint_resume=dict(
        restoration=dict(status='PASSED'),checkpoint_correctness=dict(status='FAILED' if scenario=='partial_failed' else 'PASSED'),
        trajectory_repeatability=dict(status='PRECISION_NOTE'),raw_next_update_allclose=False)))
        for key in ('cpu_fp32','cuda_fp32','cuda_native_amp')}
    if scenario=='partial_legacy_rejected':report['contract_version']='dcc_acceptance_v1'
else:
    assert stage=='check_dcc_checkpoint';dest=Path(value('--output'))
write_json(dest,report)
sys.exit(exit_code)
'''


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.encode("utf-8"))


def prepare(folder, scenario, wrapper):
    write(folder / "tools/autodl_dcc.sh", wrapper)
    write(folder / "tools/dcc_common.py", COMMON_STUB)
    write(folder / "tools/train_dcc.py", TRAIN_STUB)
    for name in ("init_dcc", "check_dcc_math", "check_dcc", "preflight_dcc", "check_dcc_resume", "check_dcc_checkpoint"):
        write(folder / "tools" / (name + ".py"), STAGE_STUB)
    write(folder / "ultralytics-main/ultralytics/__init__.py", "# Deliberate inert fixture package.\n")
    write(folder / "ultralytics-main/torch.py", "from types import SimpleNamespace\n__version__='fixture'\nversion=SimpleNamespace(cuda='fixture')\n")
    sep = ";" if os.name == "nt" else ":"
    worktree = folder.resolve().as_posix()
    environment = "\n".join([
        "# INERT FIXTURE ENVIRONMENT; no conda, CUDA, or real project entry points.",
        "export DCC_WORKTREE=" + shlex.quote(worktree),
        "export DCC_MAIN=\"$DCC_WORKTREE/main\"",
        "export DCC_VARIANT=cbr_lif_dcc_v1",
        "export DCC_META=\"$DCC_WORKTREE/outputs/dcc/cbr_lif_dcc_v1\"",
        "export DCC_INIT=\"$DCC_WORKTREE/weights/fixture.pt\"",
        "export DCC_DATA=\"$DCC_WORKTREE/main/configs/fixture.yaml\"",
        "export DCC_FIXTURE_EVENTS=\"$DCC_WORKTREE/events.jsonl\"",
        "export DCC_FIXTURE_SCENARIO=" + shlex.quote(scenario),
        "export PYTHONPATH=" + shlex.quote(worktree + "/ultralytics-main" + sep + worktree + "/tools"),
        "python() { " + shlex.quote(Path(sys.executable).resolve().as_posix()) + " \"$@\"; }",
        "cd \"$DCC_WORKTREE\"", "",
    ])
    write(folder / "docs/dcc/environment.sh", environment)


def execute(bash, folder, mode):
    # Explicit argv forwarding keeps workspace names and quotation marks inert.
    prefix = 'export PATH="/usr/bin:/mingw64/bin:$PATH"; ' if os.name == "nt" else ""
    return subprocess.run([bash, "-lc", prefix + 'bash "$@"', "dcc-shell-fixture",
                           (folder / "tools/autodl_dcc.sh").as_posix(), mode],
                          cwd=folder, capture_output=True, text=True, timeout=60)


def check(output, bash):
    wrapper = (ROOT / "tools/autodl_dcc.sh").read_text(encoding="utf-8")
    subprocess.run([bash, "-n", str(ROOT / "tools/autodl_dcc.sh")], check=True, capture_output=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    cases = [("full_passed", "init-preflight", 0), ("full_precision_note", "init-preflight", 0),
             ("initializer_failed", "init-preflight", 21), ("math_failed", "init-preflight", 22),
             ("fusion_failed", "init-preflight", 1), ("capacity_failed", "init-preflight", 24),
             ("full_legacy_rejected", "init-preflight", 1),
             ("partial_precision_note", "resume-verify", 0), ("partial_failed", "resume-verify", 3),
             ("partial_legacy_rejected", "resume-verify", 1),
             ("full_then_changed_math", "init-preflight", 0)]
    with TemporaryDirectory(prefix="dcc_inert_shell_", dir=output.parent.resolve()) as temporary:
        for scenario, mode, expected in cases:
            folder = Path(temporary) / scenario
            prepare(folder, scenario, wrapper)
            meta = folder / "outputs/dcc/cbr_lif_dcc_v1"
            permit = meta / "passed_gates.sh"
            old = "# Historical fixture permit, never a real training authorization.\nexport DCC_CONTRACT_VERSION=dcc_acceptance_v1\n"
            if mode == "init-preflight":
                write(permit, old)
            run = execute(bash, folder, mode)
            assert run.returncode == expected, (scenario, run.returncode, run.stdout, run.stderr)
            events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
            assert not any(event.startswith("explicit_") or event == "formal_stub_admitted" for event in events), events
            should_permit = mode == "init-preflight" and expected == 0
            assert permit.exists() == should_permit, scenario
            if mode == "init-preflight":
                history = list(meta.glob("passed_gates_*.superseded.sh"))
                assert len(history) == 1 and history[0].read_text() == old
            if should_permit:
                contents = permit.read_text()
                assert CONTRACT in contents and "DCC_INITIALIZATION_REPORT=" in contents and "DCC_MATH_REPORT=" in contents
                assert "DCC_GATE_REPORT=" in contents and list(meta.glob("gate_*.json"))
            summaries = list(meta.glob("resume_checks_*/verification_summary.json"))
            if scenario == "partial_precision_note":
                summary = json.loads(summaries[0].read_text())
                assert summary["status"] == "PRECISION_NOTE" and not any(summary["raw_next_update_allclose"].values())
            if scenario in ("partial_precision_note", "partial_failed"):
                summary = json.loads(summaries[0].read_text())
                expected_a = "FAILED" if scenario == "partial_failed" else "PASSED"
                assert set(summary["saving_and_restoring_state"].values()) == {expected_a}
                assert set(summary["trajectory_repeatability"].values()) == {"PRECISION_NOTE"}
            if scenario == "full_then_changed_math":
                math = next(meta.glob("module_*.json"))
                changed = json.loads(math.read_text());changed["code_identity"] = {"fixture_source": "changed"}
                write(math, json.dumps(changed))
                start = execute(bash, folder, "start")
                assert start.returncode != 0
                after = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
                assert "explicit_start_stub_entry" in after and "formal_stub_admitted" not in after
            results[scenario] = dict(status="PASSED", exit_code=run.returncode, permit_emitted=should_permit,
                                     no_automatic_start=True, events=events, stdout=run.stdout, stderr=run.stderr)
    report = dict(status="PASSED", contract_version=CONTRACT, scope="Inert shell control-flow fixtures; all model/check commands are stubs",
                  wrapper_lf_sha256=hashlib.sha256(wrapper.replace("\r\n", "\n").encode()).hexdigest(),
                  bash_syntax="PASSED", cases=results, real_model_execution=False, formal_training="NOT_STARTED", final_test="NOT_RUN")
    write(output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bash", default=shutil.which("bash") or ("C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else "bash"))
    args = parser.parse_args()
    assert not args.output.exists(), "Existing fixture report is preserved"
    result = check(args.output.resolve(), args.bash)
    print(json.dumps(dict(status=result["status"], cases=len(result["cases"]), output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
