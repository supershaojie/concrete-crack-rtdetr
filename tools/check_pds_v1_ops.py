"""Bounded safety checks for PDS operations, parser, tee exit and test locking."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest.mock import patch
import pds_v1 as ops
from pds_v1_common import *


def rejects(fn):
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError("Expected rejection")


def run(output):
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ops_",dir=output) as temp:
        root=Path(temp).resolve()
        require(root.is_relative_to(output.resolve()), "Temporary cleanup must remain inside test output")
        run_dir=root/"main/runs/c_series"/RUN_NAME
        out=root/"reports"
        out.mkdir()
        with patch.multiple(ops,OUT=out,MAIN=root/"main",RUN=run_dir):
            assert ops.parser().parse_args(["sync","a"*40]).sha=="a"*40
            for name in ("prepare","preflight","start","status","resume","val","test","pack","archive-failed-start"):
                assert ops.parser().parse_args([name]).action==name
            with patch.object(ops,"no_active"):
                run_dir.mkdir(parents=True)
                (run_dir/"args.yaml").write_text("epochs: 200\n")
                archived=ops.archive_failed_start()
                assert Path(archived["to_path"]).exists() and not run_dir.exists()
                run_dir.mkdir()
                (run_dir/"results.csv").write_text("epoch,loss\n1,2\n")
                rejects(ops.archive_failed_start)
                (run_dir/"results.csv").rename(run_dir/"results.old")
                (run_dir/"weights").mkdir()
                (run_dir/"weights/last.pt").write_bytes(b"preserve")
                rejects(ops.archive_failed_start)
            # Strict JSON keeps nonfinite evidence, including nested paths.
            write_json(out/"nonfinite.json",dict(exception="original",norm=float("inf"),nested=[float("nan")]))
            checked=read_json(out/"nonfinite.json")
            assert checked["norm"] is None and checked["nested"]==[None] and checked["nonfinite"]
            # Same completed test returns identity/results without loading a model or re-running test.
            evaluation=out/"evaluation"
            evaluation.mkdir()
            deploy=evaluation/"deploy.pt"
            deploy.write_bytes(b"locked-deploy")
            best=run_dir/"weights/best.pt"
            best.write_bytes(b"trained-best")
            lock=dict(deploy_sha256=sha256(deploy),source_sha256=sha256(best),identity={"fixed":True})
            write_json(evaluation/"selection.json",lock)
            write_json(evaluation/"test/metrics.json",dict(status="completed",checkpoint_sha256=sha256(deploy)))
            with patch.object(ops,"no_active"),patch.object(ops,"verify_delivery"),patch.object(ops,"current_identity",return_value={"fixed":True}):
                result=ops.evaluate("test")
                assert result["status"]=="EXISTING_RESULT"
                best.write_bytes(b"changed-best")
                rejects(lambda:ops.evaluate("test"))
            # An old exit record must never cover the current dispatch.
            write_json(out/"dispatch.json",dict(dispatch="new",phase="DISPATCHED",log=str(out/"missing.log")))
            write_json(out/"state.json",dict(dispatch="new",phase="SETTING_UP"))
            write_json(out/"exit.json",dict(dispatch="old",phase="COMPLETED"))
            with patch.object(ops,"live_processes",return_value=[]),patch.object(ops,"session_alive",return_value=False):
                assert ops.status()["state"]["phase"]=="FAILED"
            bash=shutil.which("bash") or ("C:/Program Files/Git/bin/bash.exe" if os.name=="nt" else None)
            require(bash,"bash required for syntax/pipeline checks")
            for script in ("pds_v1.sh","sync_pds_v1.sh"):
                subprocess.run([bash,"-n",str(ROOT/"tools"/script)],check=True)
            # Execute the actual generated tee worker on a harmless failing shell command.
            script=root/"worker.sh"
            log=out/"pipeline.log"
            bash_path=subprocess.check_output([bash,"-lc",'printf %s "$PATH"'],text=True)
            with patch.dict(os.environ,{"PATH":bash_path}):
                text=ops.worker_script(["bash","-c","echo PDS_PIPELINE_VISIBLE; exit 7"],log,"probe")
            with script.open("w",encoding="utf-8",newline="\n") as f:
                f.write(text)
            p=subprocess.run([bash,str(script)],capture_output=True,text=True)
            assert p.returncode==7 and "PDS_PIPELINE_VISIBLE" in p.stdout, (p.returncode,p.stdout,p.stderr)
            exit_info=read_json(out/"shell_exit_probe.json")
            assert exit_info["python_exit"]==7 and exit_info["tee_exit"]==0
            assert "PDS_PIPELINE_VISIBLE" in log.read_text()
            with patch.object(ops,"verify_delivery"),patch.object(ops,"status",return_value={"phase":"NOT_STARTED"}):
                packed=ops.pack()
            import tarfile
            with tarfile.open(packed["path"]) as archive:
                names=archive.getnames()
            assert "training/results.csv" not in names  # earlier renamed to results.old
            assert not any(n.endswith(".pt") for n in names)
    result=dict(status="PASS",all_public_commands_parse=True,bash_syntax=True,
                tee_visible_output_and_python_exit_preserved=True,strict_nonfinite_JSON=True,
                setup_only_archive_preserves_old_run=True,checkpoint_and_results_protected=True,
                repeated_completed_test_reuses_identity=True,changed_test_weight_rejected=True,
                stale_exit_not_reused=True,LIGHT_excludes_weights=True,live_server_tmux="PENDING")
    write_json(output/"ops.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=OUT/"ops")
    a=p.parse_args()
    print(json.dumps(run(a.output),indent=2))
