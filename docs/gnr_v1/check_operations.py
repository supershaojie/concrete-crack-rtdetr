"""Offline lifecycle regression checks; no model, dataset scan or GPU required."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import gnr_v1 as app
from gnr_v1_common import metadata, sha256, digest, validate_snapshot, write_json


def main():
    assert 'torch' not in sys.modules
    checks = {}
    app.status()
    assert 'torch' not in sys.modules
    checks['status_no_model_import'] = True
    assert app.process_identity(-1) is None
    import psutil
    class UnreadableProcess:
        def cmdline(self):
            raise OSError('process information is temporarily unavailable')
    with patch.object(psutil, 'process_iter', return_value=iter([UnreadableProcess()])):
        assert app.active_workers() == []
    checks['inaccessible_process_and_invalid_pid_do_not_break_status'] = True
    with tempfile.TemporaryDirectory(prefix='gnr-ops-') as temp:
        root = Path(temp)
        with patch.object(app, 'OUT', root / 'out'), patch.object(app, 'RUN', root / 'run'):
            with app.operation_lock():
                code = (
                    "import sys,pathlib;sys.path.insert(0,sys.argv[1]);import gnr_v1 as a;"
                    "a.OUT=pathlib.Path(sys.argv[2]);\n"
                    "try:\n with a.operation_lock(): pass\n"
                    "except RuntimeError: sys.exit(23)\n"
                    "sys.exit(5)"
                )
                child = subprocess.run([sys.executable, '-c', code, str(ROOT / 'tools'), str(app.OUT)], capture_output=True, timeout=20)
                assert child.returncode == 23, child.stderr
            write_json(app.OUT / 'lock_owner.json', {'pid': 999999999, 'created': -1})
            with app.operation_lock():
                pass
            checks['kernel_lock_rejects_concurrent_writer_and_ignores_stale_owner_file'] = True
            package = app.package()
            assert package['status'] == 'INCOMPLETE' and '_INCOMPLETE_' in package['path']
            assert package['archive_readback'] == 'PASS'
            assert 'torch' not in sys.modules
            checks['offline_incomplete_pack_manifest_readback_no_torch'] = True
            fake = {'sha256': 'test', 'identity': {}}
            mapping = {
                'prepare.json': {'development': False, 'recipe_audit': {'actual_status': 'CHECKED'}},
                'preflight.json': {'status': 'TECHNICAL_PASS', 'binding': fake},
                'diagnose.json': {'status': 'ZERO_INTERVENTION', 'binding': fake, 'reason': 'fixture'},
            }
            with patch.object(app, 'binding', return_value=fake), patch.object(app, 'read_json', side_effect=lambda p, *a: mapping.get(Path(p).name)), patch.object(sys, 'platform', 'linux'):
                try:
                    app.ready('start')
                except RuntimeError as error:
                    assert 'ZERO_INTERVENTION' in str(error)
                else:
                    raise AssertionError('Zero activation accepted')
            checks['zero_intervention_refuses_start'] = True

        data = root / 'data'; data.mkdir()
        image, label, yaml = data / 'image.jpg', data / 'label.txt', data / 'data.yaml'
        image.write_bytes(b'image fixture'); label.write_text('0 .5 .5 .2 .2'); yaml.write_text('fixture')
        saved = {'version': 'gnr_content_stat_v1', 'root': str(data), 'data_yaml': str(yaml), 'data_sha256': sha256(yaml),
                 'records': [{'image': image.name, 'label': label.name, 'image_stat': metadata(image), 'label_stat': metadata(label)}],
                 'directories': {}}
        saved['sha256'] = digest(saved)
        validate_snapshot(saved)
        stamp = label.stat().st_mtime_ns
        os.utime(label, ns=(stamp + 10000000, stamp + 10000000))
        try:
            validate_snapshot(saved)
        except RuntimeError as error:
            assert '--refresh-data' in str(error)
        else:
            raise AssertionError('Stale snapshot accepted')
        checks['metadata_change_invalidates_snapshot'] = True

        bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else Path('/bin/bash')
        if bash.is_file():
            subprocess.run([str(bash), '-n'], input=(ROOT / 'tools/sync_gnr_v1.sh').read_bytes(), check=True, timeout=10)
            command = f"set +e; {shlex.quote(Path(sys.executable).as_posix())} -c 'import sys;sys.exit(7)' | tee {shlex.quote((root/'pipe.log').as_posix())} >/dev/null; codes=(\"${{PIPESTATUS[@]}}\"); printf '%s %s' \"${{codes[0]}}\" \"${{codes[1]}}\""
            result = subprocess.run([str(bash), '-c', command], capture_output=True, text=True, timeout=15)
            assert result.stdout == '7 0', result
            checks['bash_syntax_and_python_tee_exit_separation'] = True
    report = {'status': 'PASS', 'checks': checks, 'linux_tmux_end_to_end': 'PENDING: no Linux tmux server in this Windows development environment'}
    write_json(ROOT / 'docs/gnr_v1/operations_validation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
