"""ARG identity regression tests with real temporary YAML, image, label and JSON IO.

Source/model/runtime checks use fixtures. No dataset scan, GPU preflight or training.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import arg_v2_common as common


class DataIdentity(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.out = self.root / "outputs"
        self.out.mkdir()
        self.source = self.root / "source.pt"
        self.source.write_bytes(b"public initialization fixture")
        self.init = self.out / "arg_v2_init.pt"
        self.init.write_bytes(b"existing initialized model fixture")
        self.code = dict(commit="identity-test-fixture", dirty="")
        self.args = dict(seed=42, batch=16, imgsz=640, model=str(self.init))
        common.write_json(self.out / "initialization.json", dict(output_sha256=common.sha256(self.init)))
        common.write_json(self.out / "sync.json", dict(commit=self.code["commit"], worktree=str(self.root)))
        self.data_path = self.root / "data.yaml"
        self.config = dict(path=str(self.root / "dataset"), names={0: "crack"})
        self.images, self.labels = {}, {}
        for split in ("train", "val", "test"):
            self.config[split] = "images/" + split
            self.images[split] = self.root / "dataset/images" / split / "one.png"
            self.labels[split] = self.root / "dataset/labels" / split / "one.txt"
            self.images[split].parent.mkdir(parents=True)
            self.labels[split].parent.mkdir(parents=True)
            Image.new("RGB", (8, 6), "white").save(self.images[split])
            self.labels[split].write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        common.YAML.save(self.data_path, self.config)
        self.stack.enter_context(patch.multiple(common, ROOT=self.root, OUT=self.out, SOURCE=self.source,
            INIT=self.init, SOURCE_SHA256=common.sha256(self.source),
            COUNTS={split: (1, 1) for split in self.images}))
        self.stack.enter_context(patch.object(common, "code_identity", return_value=self.code))
        self.stack.enter_context(patch.object(common, "data_config", return_value=self.data_path))
        self.stack.enter_context(patch.object(common, "recipe", return_value=(self.args, {})))
        self.stack.enter_context(patch.object(common, "source_contract", return_value={}))
        self.stack.enter_context(patch.object(common, "runtime", return_value={"scope": "fixture"}))
        self.stack.enter_context(patch.object(common, "initialize", side_effect=AssertionError("Must reuse original init")))
        self.stack.enter_context(patch.object(common.subprocess, "run"))  # git archive only
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def test_integer_ids_prepare_json_roundtrip_and_binding(self):
        yaml_bytes, init_bytes = self.data_path.read_bytes(), self.init.read_bytes()
        prepared = common.prepare()
        saved = common.read_json(self.out / "prepare.json")
        current = common.binding()  # real inventory and complete data equality check
        self.assertEqual(prepared["data"]["config"]["names"], {"0": "crack"})
        self.assertEqual(prepared["data"], saved["data"])
        self.assertEqual(current["data"], saved["data"])
        self.assertEqual(common.prepare(), saved)  # idempotent reuse, no new initialization
        self.assertEqual(common.YAML.load(self.data_path)["names"], {0: "crack"})
        self.assertEqual(self.data_path.read_bytes(), yaml_bytes)
        self.assertEqual(self.init.read_bytes(), init_bytes)
        self.assertEqual(saved["args"], self.args)

    def test_string_ids_and_config_are_preserved(self):
        self.config["names"] = {"0": "crack"}
        before = deepcopy(self.config)
        normalized = common.data_identity_config(self.config)
        self.assertEqual(normalized, before)
        self.assertEqual(self.config, before)
        self.assertIsNot(normalized["names"], self.config["names"])
        common.YAML.save(self.data_path, self.config)
        common.prepare()
        self.assertEqual(common.binding()["data"]["config"], before)

    def test_colliding_ids_rejected_even_with_identical_names(self):
        for name in ("crack", "different class"):
            with self.subTest(second_name=name):
                self.config["names"] = {0: "crack", "0": name}
                common.YAML.save(self.data_path, self.config)
                with self.assertRaisesRegex(RuntimeError, "Class ID collision"):
                    common.prepare()
                self.assertFalse((self.out / "prepare.json").exists())

    def test_changed_class_name_rejected(self):
        common.prepare()
        self.config["names"][0] = "not crack"
        common.YAML.save(self.data_path, self.config)
        with self.assertRaisesRegex(RuntimeError, "Data config changed"):
            common.binding()

    def test_real_image_label_and_yaml_byte_changes_rejected(self):
        common.prepare()
        files = (self.images["train"], self.labels["val"], self.data_path)
        for path in files:
            original = path.read_bytes()
            with self.subTest(file=str(path)):
                try:
                    if path.suffix == ".png":
                        Image.new("RGB", (8, 6), "black").save(path)
                    else:
                        # Counts and parsed labels/config stay equal; byte hash must differ.
                        path.write_bytes(original + (b"\n" if path.suffix == ".txt" else b"\n# byte identity change\n"))
                    with self.assertRaisesRegex(RuntimeError, "Data content identity changed"):
                        common.recheck_data()
                    with self.assertRaisesRegex(RuntimeError, "data snapshot"):
                        common.binding()
                finally:
                    path.write_bytes(original)
                    common.recheck_data()
        self.assertEqual(common.binding()["data"], common.read_json(self.out / "prepare.json")["data"])

    def test_real_split_image_and_box_counts_rejected(self):
        common.prepare()
        label = self.labels["train"]
        original = label.read_bytes()
        label.write_bytes(original + original)
        with self.assertRaisesRegex(RuntimeError, "train counts"):
            common.recheck_data()
        label.write_bytes(original)
        common.recheck_data()
        self.images["train"].with_name("two.png").write_bytes(self.images["train"].read_bytes())
        label.with_name("two.txt").write_bytes(original)
        with self.assertRaisesRegex(RuntimeError, "Data directory changed"):
            common.binding()
        with self.assertRaisesRegex(RuntimeError, "train counts"):
            common.recheck_data()

    def test_binding_rejects_each_identity_field_independently(self):
        prepared = common.prepare()
        # Isolate each comparison so other hashes cannot hide a removed names/count check.
        changes = ((('config', 'names', '0'), "renamed"),
                   (('config', 'path'), "/different/data"),
                   (('config_sha256',), "different config hash"),
                   (('content_sha256',), "different aggregate hash"),
                   (('splits', 'train', 'content_sha256'), "different split hash"),
                   (('splits', 'train', 'images'), 2),
                   (('splits', 'train', 'boxes'), 2))
        for keys, value in changes:
            with self.subTest(field=".".join(keys)):
                changed = deepcopy(prepared["data"])
                target = changed
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with patch.object(common, "cached_data", return_value=changed), \
                     self.assertRaisesRegex(RuntimeError, "Data identity changed since prepare"):
                    common.binding()

    def test_repeated_prepare_and_binding_never_reinventory(self):
        prepared = common.prepare()
        with patch.object(common, "inventory", side_effect=AssertionError("Unexpected full data scan")):
            self.assertEqual(common.prepare()["data"], prepared["data"])
            self.assertEqual(common.binding()["data"], prepared["data"])

    def test_reuse_verified_legacy_manifest_without_raw_file_hashing(self):
        source = self.root / "old_v1"
        source.mkdir()
        data = common.inventory(self.data_path, source / "data_manifest.jsonl.gz")
        common.write_json(source / "prepare.json", dict(status="PASS", created=datetime.now(timezone.utc).isoformat(), data=data))
        with patch.object(common, "inventory", side_effect=AssertionError("Reused snapshot must not rescan raw data")):
            prepared = common.prepare(reuse_from=source / "prepare.json")
        self.assertEqual(prepared["data"], data)
        snapshot = common.read_json(self.out / "data_snapshot.json")
        self.assertEqual(snapshot["reused_from"]["record"], str((source / "prepare.json").resolve()))
        self.assertFalse(snapshot["reused_from"]["raw_files_rehashed"])

    def test_inplace_edit_requires_explicit_recheck_and_invalidates_snapshot(self):
        common.prepare()
        self.labels["train"].write_text("0 .4 .4 .25 .25\n", encoding="utf-8")
        # Directory/config guards cannot honestly claim to detect same-path content edits.
        common.binding()
        with self.assertRaisesRegex(RuntimeError, "Data content identity changed"):
            common.recheck_data()
        self.assertEqual(common.read_json(self.out / "data_snapshot.json")["status"], "INVALID")
        with self.assertRaises(RuntimeError):
            common.binding()


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DataIdentity))
    common.write_json(common.OUT / "local_identity.json", dict(
        status="PASS" if result.wasSuccessful() else "FAIL", tests=result.testsRun,
        scope="temporary YAML/image/label/JSON IO; actual prepare/inventory/binding; fixture source/runtime/recipe and git archive; no server or GPU"))
    raise SystemExit(0 if result.wasSuccessful() else 1)
