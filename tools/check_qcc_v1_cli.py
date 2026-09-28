"""CLI regression checks only; never run training, GPU checks or split evaluation."""
from __future__ import annotations

import ast
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import inspect
import io
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import qcc_v1 as cli
import qcc_v1_preflight as gpu
from qcc_v1_common import OUT, ROOT, git, write_json


HELP_RESULTS = []
COMMAND_CONTRACTS = []


def child_commands(function, **values):
    """Read the actual producer's argv expressions without executing GPU work."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    expressions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_bounded":
            expressions.append(node.args[0])
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "command" for t in node.targets):
            expressions.append(node.value)
    return [eval(compile(ast.Expression(e), inspect.getfile(function), "eval"), function.__globals__, values)
            for e in expressions]


class CLI(unittest.TestCase):
    def test_real_help_entrypoints(self):
        for script in ("qcc_v1.py", "qcc_v1_preflight.py", "check_qcc_v1.py", "check_qcc_v1_lifecycle.py"):
            with self.subTest(script=script):
                command = [sys.executable, str(ROOT / "tools" / script), "--help"]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
                HELP_RESULTS.append(dict(command=command, returncode=result.returncode,
                                         stdout=result.stdout, stderr=result.stderr))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("usage:", result.stdout)
                self.assertNotIn("Traceback", result.stderr)

    def test_parent_and_reload_child_arguments(self):
        folder = Path("fixture folder with spaces")
        checkpoint = folder / "last.pt"
        commands = child_commands(cli.preflight, folder=folder)
        self.assertEqual(len(commands), 2)
        cpu, cuda = commands
        self.assertEqual(cpu[1], str(ROOT / "tools/check_qcc_v1.py"))
        self.assertEqual(cpu[2:], ["--math-only", "--output", str(folder / "cpu")])
        self.assertEqual(Path(cuda[1]), Path(gpu.__file__))
        args = gpu.build_parser().parse_args(cuda[2:])
        self.assertEqual(args.folder, folder)
        self.assertIsNone(args.reload)
        reload_commands = child_commands(gpu.run, folder=folder, checkpoint=checkpoint)
        self.assertEqual(len(reload_commands), 1)
        reload_command = reload_commands[0]
        self.assertEqual(Path(reload_command[1]), Path(gpu.__file__))
        args = gpu.build_parser().parse_args(reload_command[2:])
        self.assertEqual((args.folder, args.reload), (folder, checkpoint))
        COMMAND_CONTRACTS.extend(commands + reload_commands)

    def test_guarded_and_worker_child_arguments(self):
        guarded = child_commands(cli.guarded_preflight, seconds=900, micro_batches=16)
        self.assertEqual(len(guarded), 1)
        self.assertEqual(Path(guarded[0][1]), Path(cli.__file__))
        args = cli.build_parser().parse_args(guarded[0][2:])
        self.assertEqual((args.command, args.seconds, args.micro_batches), ("_preflight", 900, 16))
        worker = child_commands(cli.dispatch, identifier="fixture_dispatch")
        self.assertEqual(len(worker), 1)
        self.assertEqual(worker[0][1], "-u")
        self.assertEqual(Path(worker[0][2]), Path(cli.__file__))
        args = cli.build_parser().parse_args(worker[0][3:])
        self.assertEqual((args.command, args.dispatch), ("_worker", "fixture_dispatch"))
        COMMAND_CONTRACTS.extend(guarded + worker)

    def test_gpu_entry_routes_paths(self):
        folder = Path("fixture folder with spaces")
        checkpoint = folder / "last.pt"
        with patch.object(gpu, "run") as run, patch.object(gpu, "reload_and_val") as reload, \
                patch.object(gpu.torch, "set_num_threads"):
            gpu.main(["--folder", str(folder)])
            run.assert_called_once_with(folder)
            reload.assert_not_called()
            run.reset_mock()
            gpu.main(["--folder", str(folder), "--reload", str(checkpoint)])
            reload.assert_called_once_with(folder, checkpoint)
            run.assert_not_called()

    def test_main_preflight_routes_and_keeps_unqualified_exit(self):
        with patch.object(cli, "operation_lock", side_effect=nullcontext), \
                patch.object(cli, "guarded_preflight", return_value={"start_eligible": False}) as guarded, \
                patch.object(cli, "preflight", return_value={"start_eligible": False}) as inner, \
                redirect_stdout(io.StringIO()):
            for command in ("preflight", "_preflight"):
                with self.subTest(command=command), self.assertRaises(SystemExit) as exit:
                    cli.main([command, "--seconds", "900", "--micro-batches", "16"])
                self.assertEqual(exit.exception.code, 2)
            guarded.assert_called_once_with(900, 16)
            inner.assert_called_once_with(900, 16)

    def test_required_and_unknown_arguments_still_rejected(self):
        for parser, argv in ((gpu.build_parser(), []),
                             (gpu.build_parser(), ["--folder", "fixture", "--unknown"]),
                             (cli.build_parser(), ["_worker"]),
                             (cli.build_parser(), ["_preflight", "--seconds", "invalid"])):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit:
                parser.parse_args(argv)
            self.assertEqual(exit.exception.code, 2)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CLI))
    write_json(OUT / "cli_checks.json", dict(status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun, commit=git("rev-parse", "HEAD"), dirty=git("status", "--porcelain"),
        help=HELP_RESULTS, subprocess_argv=COMMAND_CONTRACTS,
        scope="CLI parsing/help/argv only; no GPU or training qualification; no data inventory",
        errors=[str(x) for x in result.errors + result.failures]))
    raise SystemExit(0 if result.wasSuccessful() else 1)
