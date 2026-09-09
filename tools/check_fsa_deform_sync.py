"""Run real sync shell with an explicit Git stub; no network or real worktree mutation."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from init_fsa_deform import ROOT, BASE_COMMIT, require, write_json


def run(bash, output):
    (ROOT / "outputs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fsa_sync_fixture_", dir=ROOT / "outputs") as directory:
        root = Path(directory)
        (root / "bin").mkdir(); (root / "main/.git").mkdir(parents=True)
        stub = r'''#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$FIXTURE_LOG"
repo="$PWD"
if [[ "${1:-}" == -C ]]; then repo="$2"; shift 2; fi
case "$*" in
  'remote get-url origin') echo https://github.com/supershaojie/concrete-crack-rtdetr.git ;;
  'status --short') : ;;
  'status --porcelain') printf '%s' "${FIXTURE_DIRTY:-}" ;;
  'rev-parse --show-toplevel') echo "$repo" ;;
  'rev-parse --git-common-dir') echo "$FIXTURE_COMMON" ;;
  'rev-parse FETCH_HEAD') echo "${FIXTURE_REMOTE:-$FIXTURE_SHA}" ;;
  'rev-parse HEAD') echo "${FIXTURE_HEAD:-$FIXTURE_SHA}" ;;
  'fetch origin codex/fsa-deform'|'cat-file -e '*|'merge-base --is-ancestor '*) : ;;
  'worktree add --detach '*) mkdir -p "$4" ;;
  *) echo "Unexpected Git operation: $*" >&2; exit 99 ;;
esac
'''
        (root / "bin/git").write_bytes(stub.encode())
        (root / "bin/git").chmod(0o755)
        # Export posix paths inside bash, so this test works under Git Bash as well as Linux.
        command = '''set -eu
fixture="$(cd "$1" && pwd)"; script="$2"; sha="$3"
export PATH="$fixture/bin:$PATH" FIXTURE_LOG="$fixture/git.log" FIXTURE_SHA="$sha" FIXTURE_COMMON="$fixture/main/.git"
bash "$script" "$sha" "$fixture/main" "$fixture/worktree"
'''
        def invoke(**extra):
            env = dict(os.environ, **extra)
            return subprocess.run([bash, "-c", command, "fixture", root.as_posix(),
                                   (ROOT / "tools/sync_fsa_deform.sh").as_posix(), BASE_COMMIT],
                                  env=env, capture_output=True, text=True)
        fresh=invoke(); require(fresh.returncode==0, fresh.stderr)
        repeated=invoke(); require(repeated.returncode==0, repeated.stderr)
        # Stub intentionally has no branch-name command: detached reuse must not ask for a branch.
        dirty=invoke(FIXTURE_DIRTY=" M protected.txt"); require(dirty.returncode!=0, "Dirty worktree accepted")
        different=invoke(FIXTURE_HEAD="1"*40); require(different.returncode!=0, "Different existing SHA accepted")
        wrong_remote=invoke(FIXTURE_REMOTE="2"*40); require(wrong_remote.returncode!=0, "Remote mismatch accepted")
        log=(root / "git.log").read_text()
        require("worktree add --detach" in log and "branch --show-current" not in log, "Detached flow wrong")
        require(not any(f" {x} " in log for x in ("reset", "clean", "kill", "stash")), "Destructive operation")
        write_json(output, dict(status="passed", scope="Actual bash sync script with explicit Git stub; no remote access or actual git worktrees",
                   new_detached=True, existing_detached_same_sha=True, dirty_preserved=True,
                   different_sha_preserved=True, remote_mismatch_rejected=True, branch_name_independent=True))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bash",default="bash")
    parser.add_argument("--output",type=Path,default=ROOT / "docs/fsa_deform/sync_validation.json")
    args=parser.parse_args(); run(args.bash,args.output)
