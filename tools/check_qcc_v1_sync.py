"""Temporary Git/identity fixtures for safe updates; no dataset inventory or GPU work."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import qcc_v1 as cli
import qcc_v1_common as c
from ultralytics.utils import YAML


def load_sync():
    source = (c.ROOT / "tools/sync_qcc_v1.sh").read_text(encoding="utf-8")
    body = source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    namespace = {"__name__": "sync_fixture"}
    exec(compile(body, "sync_qcc_v1.sh:embedded-python", "exec"), namespace)
    return namespace["synchronize"]


class Sync(unittest.TestCase):
    def git(self, *args, cwd=None):
        return subprocess.check_output(["git", *args], cwd=cwd or self.main, text=True,
                                       stderr=subprocess.PIPE, timeout=30).strip()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.main, self.work = self.root / "main", self.root / "work"
        self.main.mkdir()
        self.git("init", "-b", "fixture-main")
        self.git("config", "user.name", "QCC Fixture")
        self.git("config", "user.email", "qcc-fixture@example.invalid")
        self.git("config", "core.autocrlf", "false")
        (self.main / ".gitignore").write_text("outputs/\nignored-collision.txt\n")
        (self.main / "source.txt").write_text("old source\n")
        self.git("add", ".")
        self.git("commit", "-m", "old fixture")
        self.old = self.git("rev-parse", "HEAD")
        (self.main / "source.txt").write_text("new source\n")
        self.git("commit", "-am", "new fixture")
        self.new = self.git("rev-parse", "HEAD")
        self.git("worktree", "add", "-b", c.BRANCH, str(self.work), self.old)
        self.out = self.work / "outputs/qcc_v1"
        self.out.mkdir(parents=True)
        self.previous = dict(worktree=str(self.work), main=str(self.main), commit=self.old, branch=c.BRANCH)
        c.write_json(self.out / "sync.json", self.previous)
        (self.out / "data_identity.json").write_text('{"unchanged": true}\n')
        self.synchronize = load_sync()
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.multiple(cli, OUT=self.out, ROOT=self.work))
        stack.enter_context(patch.object(c, "RUN", self.root / "formal_run"))
        stack.enter_context(patch.object(cli, "active_workers", return_value=[]))
        stack.enter_context(patch.object(cli, "has_tmux", return_value=False))
        stack.enter_context(patch.object(sys, "path", sys.path.copy()))

    # unittest.TestCase.enterContext is not available in the fixed Python 3.9.
    def enterContext(self, context):
        result = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return result

    def sync(self):
        with redirect_stdout(io.StringIO()):
            self.synchronize(self.work, self.main, self.new)

    def assert_unchanged(self, head=None):
        self.assertEqual(self.git("rev-parse", "HEAD", cwd=self.work), head or self.old)
        self.assertEqual(c.read_json(self.out / "sync.json"), self.previous)
        self.assertEqual((self.out / "data_identity.json").read_text(), '{"unchanged": true}\n')

    def test_fast_forward_history_and_idempotent_retry(self):
        self.sync()
        self.assertEqual(self.git("rev-parse", "HEAD", cwd=self.work), self.new)
        self.assertEqual(c.read_json(self.out / "sync.json")["commit"], self.new)
        self.assertEqual(c.read_json(self.out / "history" / ("sync_" + self.old + ".json")), self.previous)
        self.assertEqual((self.out / "data_identity.json").read_text(), '{"unchanged": true}\n')
        # Simulate interruption after Git FF, before atomic sync.json replacement.
        c.write_json(self.out / "sync.json", self.previous)
        self.sync()
        self.sync()
        self.assertEqual(c.read_json(self.out / "sync.json")["commit"], self.new)

    def test_tracked_changes_are_preserved(self):
        (self.work / "source.txt").write_text("user changes\n")
        with self.assertRaisesRegex(RuntimeError, "Worktree changes"):
            self.sync()
        self.assert_unchanged()
        self.assertEqual((self.work / "source.txt").read_text(), "user changes\n")

    def test_untracked_files_are_preserved(self):
        path = self.work / "user-notes.txt"
        path.write_text("user notes\n")
        with self.assertRaisesRegex(RuntimeError, "Worktree changes"):
            self.sync()
        self.assert_unchanged()
        self.assertEqual(path.read_text(), "user notes\n")

    def test_ignored_collision_is_preserved(self):
        (self.main / "ignored-collision.txt").write_text("incoming tracked file\n")
        self.git("add", "-f", "ignored-collision.txt")
        self.git("commit", "-m", "collision fixture")
        self.new = self.git("rev-parse", "HEAD")
        path = self.work / "ignored-collision.txt"
        path.write_text("user ignored file\n")
        with self.assertRaises(subprocess.CalledProcessError):
            self.sync()
        self.assert_unchanged()
        self.assertEqual(path.read_text(), "user ignored file\n")

    def test_divergent_commits_are_preserved(self):
        (self.work / "source.txt").write_text("independent commit\n")
        self.git("commit", "-am", "user fixture commit", cwd=self.work)
        old = self.git("rev-parse", "HEAD", cwd=self.work)
        with self.assertRaises(subprocess.CalledProcessError):
            self.sync()
        self.assert_unchanged(old)

    def test_active_lock_and_training_identity_refuse_update(self):
        lock = self.out / "operation.lock"
        c.write_json(lock, cli.process_identity(__import__("os").getpid()))
        with self.assertRaisesRegex(RuntimeError, "operation is active"):
            self.sync()
        self.assertTrue(lock.exists())
        self.assert_unchanged()
        lock.unlink()  # Only our temporary fixture's own lock.
        c.write_json(self.out / "training_identity.json", {"fixture": True})
        with self.assertRaisesRegex(RuntimeError, "after formal dispatch"):
            self.sync()
        self.assert_unchanged()

    def test_reprepare_updates_code_history_without_inventory(self):
        # Exercise the existing prepare() update path with cached data identity.
        data_root = self.root / "dataset"
        config = dict(path=str(data_root), names={"0": "crack"},
                      train="images/train", val="images/val", test="images/test")
        stamps = {}
        for split in c.COUNTS:
            for kind in ("images", "labels"):
                path = data_root / kind / split
                path.mkdir(parents=True)
                stamps[str(path)] = path.stat().st_mtime_ns
        config_path = self.root / "data.yaml"
        YAML.save(config_path, config)
        data = dict(config=config, config_sha256=c.sha256(config_path), content_sha256="fixed-fixture")
        c.write_json(self.out / "data_identity.json", dict(status="PASS", data=data, directory_stamps=stamps))
        manifest = self.out / "data_manifest.jsonl.gz"
        manifest.write_bytes(b"existing manifest is not read by cached prepare")
        source, init = self.root / "source.pt", self.out / "qcc_v1_init.pt"
        source.write_bytes(b"public fixture")
        init.write_bytes(b"existing init fixture")
        c.write_json(self.out / "initialization.json", dict(output_sha256=c.sha256(init)))
        args = dict(batch=16, imgsz=640, amp=True)
        snapshot = self.out / "source_snapshot.tar.gz"
        self.git("archive", "--format=tar.gz", "--output=" + str(snapshot), "HEAD", cwd=self.work)
        previous = dict(status="PASS", code={"commit": self.old}, args=args, data=data,
                        source_sha256=c.sha256(source), init_sha256=c.sha256(init),
                        source_snapshot_sha256=c.sha256(snapshot))
        c.write_json(self.out / "prepare.json", previous)
        self.sync()
        preserved = {p: p.read_bytes() for p in (self.out / "data_identity.json", manifest, init)}
        with patch.multiple(c, ROOT=self.work, OUT=self.out, SOURCE=source, INIT=init, SOURCE_SHA256=c.sha256(source)), \
                patch.object(c, "code_identity", return_value={"commit": self.new}), \
                patch.object(c, "source_contract"), patch.object(c, "runtime", return_value={"fixture": True}), \
                patch.object(c, "data_config", return_value=config_path), patch.object(c, "recipe", return_value=(args, {})), \
                patch.object(c, "initialize", side_effect=AssertionError("must reuse existing init")), \
                patch.object(c, "inventory", side_effect=AssertionError("raw inventory forbidden")):
            updated = c.prepare()
            self.assertEqual(updated["code"]["commit"], self.new)
            self.assertEqual(updated["data"], data)
            self.assertEqual(c.read_json(self.out / "history" / ("prepare_" + self.old + ".json")), previous)
            self.assertEqual(updated["source_snapshot_sha256"], c.sha256(snapshot))
            self.assertNotEqual(updated["source_snapshot_sha256"], previous["source_snapshot_sha256"])
        for path, original in preserved.items():
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Sync))
    c.write_json(c.OUT / "sync_checks.json", dict(status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun, commit=c.git("rev-parse", "HEAD"), dirty=c.git("status", "--porcelain"),
        scope="Temporary Git fixtures exercise shipped embedded sync and cached prepare; no real server fetch or raw inventory",
        errors=[str(x) for x in result.errors + result.failures]))
    raise SystemExit(0 if result.wasSuccessful() else 1)
