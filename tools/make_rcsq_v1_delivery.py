"""Create a small fixed-SHA server handoff after committing RCS-Q; never sync/train."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-rcsq-v1"


def git(*args):
    return subprocess.check_output(
        ["git", "-c", "safe.directory=" + ROOT.as_posix(), *args], cwd=ROOT, text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--server-delivery-dir", default="/root/autodl-tmp/rcsq-v1-delivery")
    parser.add_argument("--without-bundle", action="store_true")
    args = parser.parse_args()
    if git("status", "--porcelain"):
        raise RuntimeError("Commit/review all task changes before generating fixed-SHA delivery")
    commit = git("rev-parse", "HEAD")
    git("merge-base", "--is-ancestor", BASE, commit)
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("Generate delivery from the dedicated RCS-Q branch")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "DELIVERY_SHA.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "BASE_SHA.txt").write_text(BASE + "\n", encoding="utf-8")
    (output / "sync_rcsq_v1.sh").write_bytes((ROOT / "tools/sync_rcsq_v1.sh").read_bytes().replace(b"\r\n", b"\n"))
    handoff = (ROOT / "docs/rcsq_v1/SERVER_HANDOFF.md").read_text(encoding="utf-8")
    handoff = handoff.replace("@DELIVERY_SHA@", commit).replace("@DELIVERY_DIR@", args.server_delivery_dir)
    (output / "SERVER_HANDOFF_DELIVERED.md").write_bytes(handoff.replace("\r\n", "\n").encode("utf-8"))
    environment = """# Source this file in each new terminal; no training is launched here.
set -Eeuo pipefail
export RCSQ_DELIVERY_SHA={sha}
export RCSQ_DELIVERY_DIR={delivery}
export RCSQ_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
export RCSQ_ROOT=/root/autodl-tmp/projects/Crack_RTDETR-rcsq-v1
export RCSQ_SOURCE="$RCSQ_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
export RCSQ_DATA="$RCSQ_MAIN/configs/crack_autodl.yaml"
export RCSQ_INIT="$RCSQ_ROOT/weights/rcsq_v1"
export RCSQ_PLAN="$RCSQ_ROOT/outputs/rcsq_v1/plan"
export RCSQ_PREFLIGHT="$RCSQ_ROOT/outputs/rcsq_v1/server_preflight"
export PYTHONPATH="$RCSQ_ROOT/ultralytics-main:$RCSQ_ROOT/tools"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
cd "$RCSQ_ROOT"
test "$(git rev-parse HEAD)" = "$RCSQ_DELIVERY_SHA"
""".format(sha=shlex.quote(commit), delivery=shlex.quote(args.server_delivery_dir))
    (output / "task.env").write_bytes(environment.encode("utf-8"))
    if not args.without_bundle:
        git("bundle", "create", str(output / "rcsq_v1.bundle"), "HEAD", "^" + BASE)
        git("bundle", "verify", str(output / "rcsq_v1.bundle"))
    manifest = {
        "commit": commit,
        "base_sha": BASE,
        "branch": BRANCH,
        "formal_training": "NOT_STARTED",
        "final_test": "NOT_RUN",
        "bundle_requires_existing_base": BASE,
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(output.iterdir()) if p.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"delivery": str(output), "commit": commit, "files": list(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
