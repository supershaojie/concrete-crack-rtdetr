"""Small real-file checks for evaluation ordering, immutable test cache, commands and archives."""
from __future__ import annotations
import gzip
import re
import shutil
import tempfile
from types import SimpleNamespace
from peq_v1_common import *
from peq_v1_eval import EVAL, evaluation_cache, metric_summary
from peq_v1_delivery import server_commands
from peq_v1_pack import archive_files, verify_archive
from ultralytics.models.rtdetr.peq_train import PEQValidator


def workflow_checks():
    with tempfile.TemporaryDirectory(prefix="peq_workflows_",dir=str(OUT)) as temporary:
        folder=Path(temporary)
        y=torch.zeros(1,300,5)
        y[0,0]=torch.tensor([.5,.5,.2,.2,.49])
        y[0,1]=torch.tensor([.1,.1,.1,.1,.51])
        adjusted=y.clone()
        adjusted[0,:2,4]=torch.tensor([.52,.48])
        batch=dict(img=torch.zeros(1,3,640,640),bboxes=torch.tensor([[.5,.5,.2,.2]]),
                   cls=torch.zeros(1,1),batch_idx=torch.zeros(1,dtype=torch.long),
                   im_file=["synthetic.png"],ori_shape=[(640,640)],ratio_pad=[None])
        metrics={}
        orders={}
        for mode,predictions in (("raw",y),("peq",adjusted)):
            validator=PEQValidator(args=dict(EVAL,plots=False,save_json=False,save_txt=False),save_dir=folder/mode)
            validator.data=dict(val="fixture",names={0:"crack"},nc=1)
            validator.training=False
            validator.device=torch.device("cpu")
            validator.init_metrics(SimpleNamespace(names={0:"crack"},end2end=False))
            selected=validator.postprocess(predictions)
            orders[mode]=selected[0]["query_ids"].tolist()
            validator.update_metrics(selected,batch)
            validator.get_stats()
            validator.finalize_metrics()
            metrics[mode]=metric_summary(validator.metrics)
        require(orders=={"raw":[1,0],"peq":[0,1]},"Each score mode needs its own sorted query mask")
        require(torch.equal(y[...,:4],adjusted[...,:4]),"Same-box fixture mutated")
        require(metrics["peq"]["AP50"]>metrics["raw"]["AP50"],"Real native metrics ignored score ordering")
        cached=folder/"cache";cached.mkdir()
        stream=cached/"predictions_gt.jsonl.gz"
        with gzip.open(stream,"wt",encoding="utf-8") as output:
            output.write(json.dumps(dict(image="synthetic.png",query_ids=list(range(300))))+"\n")
        original=dict(status="COMPLETED",key="fixture",stream=identity(stream))
        write_json(cached/"metrics.json",original)
        require(evaluation_cache(cached,"fixture",["synthetic.png"],"test",True)==original,
                "Completed test --retry must reuse exactly the same report")
        require(evaluation_cache(cached,"fixture",["synthetic.png"],"val",True) is None,"Explicit val retry broken")
        try:
            evaluation_cache(cached,"different",["synthetic.png"],"test",True)
            raise AssertionError("Changed completed test identity accepted")
        except RuntimeError:
            pass
        packed=archive_files(folder/"fixture.tar.gz",{"existing.json":cached/"metrics.json"},
                             {"events.jsonl":b'{"event":"fixture"}\n'})
        require(verify_archive(folder/"fixture.tar.gz")["status"]=="PASS","Archive failed manifest verification")
        commands=server_commands(BASE)
        blocks=re.findall(r"```bash\n(.*?)\n```",commands,re.S)
        require(len(blocks)==13,"Missing independently copyable commands")
        bash=shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
        for index,block in enumerate(blocks):
            path=folder/f"command_{index}.sh"
            path.write_bytes(block.encode())
            subprocess.run([bash,"-n",str(path)],check=True)
        require(BASE in blocks[0] and "sync_peq_v1_" in blocks[0],"Bootstrap is not commit-addressed")
        for block in blocks[1:-1]:
            require("PYTHONPATH=" in block and "PYTHONUNBUFFERED=1" in block and
                    "YOLO_AUTOINSTALL=false" in block and "/root/miniconda3/envs/rtdetr/bin/python" in block,
                    "Action block depends on another block's environment")
        return dict(synthetic_only=True,same_boxes=True,independent_query_orders=orders,
                    metric_fixture=metrics,completed_test_retry_reused=True,changed_test_identity_rejected=True,
                    archive_manifest=packed["verification"],server_command_blocks=len(blocks),bash_syntax="PASS")


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(status="PASS",scope="WORKFLOW_FIXTURES_NOT_FORMAL_EVAL",checks=workflow_checks(),environment=environment())
    write_json(OUT/"checks_workflows.json",result)
    print(json_bytes(result).decode())
