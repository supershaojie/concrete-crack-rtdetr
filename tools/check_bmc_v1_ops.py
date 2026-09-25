"""Small operational fixtures: CLI, shell exit code, sync guards, strict JSON and eval mask."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from bmc_v1_common import ROOT, OUT, require, read, write, runtime


def script(path, text):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def run():
    from bmc_v1 import parser
    from bmc_v1_training import EpochCollector
    import torch
    from c19_lif_v1_results import postprocess
    report = dict(status="FAIL", runtime=runtime(), fixture_scope="local parser/shell fixtures; server tmux/fetch remain user-run")
    try:
        parsed = {}
        for action in ("prepare", "preflight", "probe", "start", "status", "resume", "val", "test", "pack"):
            parsed[action] = parser().parse_args([action]).action
        require(parser().parse_args(["start", "--archive-failed"]).archive_failed, "Missing failed-start recovery")
        report["parser_actions"] = parsed
        bash = shutil.which("bash") or str(Path("C:/Program Files/Git/bin/bash.exe"))
        require(Path(bash).is_file(), "Bash unavailable for script validation")
        for name in ("bmc_v1.sh", "sync_bmc_v1.sh"):
            subprocess.run([bash, "-n", str(ROOT / "tools" / name)], check=True)
        invalid = subprocess.run([bash, str(ROOT / "tools/sync_bmc_v1.sh"), "shortsha"], capture_output=True, text=True)
        require(invalid.returncode == 2, "Sync accepted an abbreviated SHA")
        with tempfile.TemporaryDirectory(prefix="ops_", dir=OUT) as temporary:
            folder = Path(temporary)
            require(folder.resolve().is_relative_to(OUT.resolve()), "Fixture escaped output directory")
            (folder / "tools").mkdir()
            (folder / "outputs/bmc_v1").mkdir(parents=True)
            shell = folder / "tools/bmc_v1.sh"
            script(shell, (ROOT / "tools/bmc_v1.sh").read_text(encoding="utf-8"))
            fake = folder / "python-fixture"
            script(fake, '#!/usr/bin/env bash\nif [[ "$2" == "_worker" ]]; then\n echo fixture-python-failed\n exit 7\nfi\nprintf "%s\\n" "$@" > "$BMC_EXIT_CAPTURE"\n')
            fake.chmod(0o755)
            capture = folder / "captured-exit.txt"
            env = dict(os.environ, BMC_PYTHON=fake.as_posix(), BMC_EXIT_CAPTURE=capture.as_posix())
            result = subprocess.run([bash, shell.as_posix(), "_session", "20260926T000000.000000Z_aaaaaaaa"],
                                    env=env, capture_output=True, text=True)
            require(result.returncode == 7 and "fixture-python-failed" in result.stdout, "tee masked Python exit/output")
            args = capture.read_text().splitlines()
            require(args[args.index("--code")+1] == "7" and args[args.index("--tee-code")+1] == "0", "Wrong PIPESTATUS capture")
            report["shell_python_failure_preserved"] = dict(python_exit=7, tee_exit=0, visible_output=True)
            write(folder / "nonfinite.json", dict(loss=float("nan"), gradient=float("inf"), error="original training error"))
            encoded = read(folder / "nonfinite.json")
            require(encoded["loss"] is None and len(encoded["nonfinite"]) == 2 and encoded["error"] == "original training error", "Nonfinite JSON lost evidence")
            report["strict_nonfinite_json"] = True
            # Read-only synchronization guard fixture; fake git represents a dirty owned linked worktree.
            bin_dir = folder / "bin"
            bin_dir.mkdir()
            fake_git = bin_dir / "git"
            script(fake_git, '''#!/usr/bin/env bash
printf '%s\n' "$*" >> "$BMC_GIT_TRACE"
[[ "$1" == '-C' ]] && shift 2
case "$1" in
remote) echo 'https://github.com/supershaojie/concrete-crack-rtdetr.git';;
fetch|merge-base) exit 0;;
rev-parse) if [[ "$2" == 'FETCH_HEAD' ]]; then printf '%040d\n' 0; else echo /fixture/common; fi;;
status) echo ' M user-file';;
*) echo 'Unexpected operation' >&2; exit 9;;
esac
''')
            fake_git.chmod(0o755)
            target = folder / "dirty-worktree"
            target.mkdir()
            (target / ".git").write_text("fixture")
            (target / "user-file").write_text("user changes preserved")
            trace = folder / "git-trace.txt"
            env.update(PATH=bin_dir.as_posix() + os.pathsep + os.environ["PATH"], BMC_MAIN=folder.as_posix(),
                       BMC_WORKTREE=target.as_posix(), BMC_GIT_TRACE=trace.as_posix())
            env["BMC_FIXTURE_BIN"] = ("/" + bin_dir.drive[0].lower() + bin_dir.as_posix()[2:]) if os.name == "nt" else str(bin_dir)
            result = subprocess.run([bash, "-c", 'export PATH="$BMC_FIXTURE_BIN:$PATH"; exec bash "$1" "$2"',
                                     "fixture", str(ROOT / "tools/sync_bmc_v1.sh"), "0"*40], env=env, capture_output=True, text=True)
            report["sync_fixture"] = dict(code=result.returncode, stdout=result.stdout, stderr=result.stderr,
                                          trace=trace.read_text() if trace.exists() else None)
            require(result.returncode != 0 and "has changes; preserved" in result.stderr, f"Dirty worktree sync was not refused: {result.stderr}")
            require((target / "user-file").read_text() == "user changes preserved", "Sync modified user work")
            require(not any(word in trace.read_text() for word in ("reset", "clean", "checkout")), "Destructive sync operation")
            report["sync_dirty_guard"] = True
        predictions = torch.tensor([[[.5, .5, .1, .1, .0001], [.5, .5, .1, .1, .9], [.5, .5, .1, .1, .2]]])
        corrected, _ = postprocess(predictions, 640, .001)
        require(torch.allclose(corrected[0]["conf"], torch.tensor([.9, .2])), "Corrected sorted mask not reused")
        collector = EpochCollector()
        collector.record(19, dict(state="DISABLED_WARMUP", images=[], matched_gt=3), {"loss_bbox": torch.tensor(2.)})
        summary = collector.summary()
        require(summary["matched"] == 3 and summary["changed"] is None and summary["disagreement_accepted"] is None, "Warmup is mislabeled as zero acceptance")
        report.update(status="PASS", corrected_sorted_mask=True, disabled_warmup_nulls=True)
        return report
    finally:
        write(OUT / "operations.json", report)


if __name__ == "__main__":
    run()
