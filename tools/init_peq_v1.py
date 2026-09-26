"""Reuse the pinned mother's controlled source; strict public state and Trainer audits."""
from __future__ import annotations
from copy import deepcopy
import torch
from peq_v1_common import *
from init_c19_lif_v1 import controlled_models, build as build_mother
from ultralytics.models.rtdetr.peq_model import PEQDetectionModel
from ultralytics.models.rtdetr.peq_train import checked_training_model, is_peq
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.utils.patches import torch_load
from ultralytics import RTDETR


def build(nc=80):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return PEQDetectionModel(str(MODEL_YAML), nc=nc, verbose=False)


def initialize(source=SOURCE, output=INIT):
    require(sha256(source)==SOURCE_SHA, "Public source weight SHA256 differs")
    output=Path(output)
    provenance=OUT/"initialization.json"
    if output.exists():
        previous=read_json(provenance)
        require(previous and previous["source"]["sha256"]==SOURCE_SHA and previous["output"]["sha256"]==sha256(output),
                "Existing init has no matching verified provenance; preserved")
        model=torch_load(output,map_location="cpu")["model"].float()
        require(isinstance(model,PEQDetectionModel), "Init type mismatch")
        return previous
    with isolated_rng():
        _, mother, mother_report=controlled_models(source)
        fresh_mother=build_mother()
        target=build()
        before, after=fresh_mother.state_dict(),target.state_dict()
        common_equal={k: torch.equal(v,after[k]) for k,v in before.items()}
        require(len(before)==552 and all(common_equal.values()), "PEQ construction consumed public RNG")
        added=set(after)-set(before)
        require(added and all(is_peq(k) for k in added), "Unknown additional state")
        state={**after,**mother.state_dict()}
        target.load_state_dict(state,strict=True)
        training,trainer_report=checked_training_model(target.yaml,target,dict(nc=1,channels=3))
        require(sum(v.numel() for v in training.parameters())==20167759,"Wrong nc1 parameter count")
        common_rows=[dict(name=k,shape=list(v.shape),equal=torch.equal(v,target.state_dict()[k]),
                          sha256=hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()) for k,v in mother.state_dict().items()]
        require(all(row["equal"] for row in common_rows), "Public initialized values changed")
        target.eval()
        target.args={**DEFAULT_CFG_DICT,"model":str(MODEL_YAML),"task":"detect"}
        target.task,target.pt_path="detect",str(output)
        report=dict(status="PASS",source=identity(source),base=BASE,seed=42,
                    constructor_common_equal=all(common_equal.values()),common=common_rows,
                    added_trainable=sorted(added),added_buffers=[],unknown_differences=[],
                    parameters_nc1=20167759,peq_parameters=17994,
                    mother_controlled_provenance=mother_report,actual_trainer_loading=trainer_report,
                    module_hashes=module_contract(),environment=environment())
        checkpoint=dict(epoch=-1,best_fitness=None,model=deepcopy(target).float(),ema=None,updates=None,
                        optimizer=None,scaler=None,train_args=target.args,train_metrics=None,train_results=None,
                        date=utc(),peq_v1_provenance=report)
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("xb") as stream:
            torch.save(checkpoint,stream)
        restored=RTDETR(str(output)).model
        require(type(restored) is PEQDetectionModel and all(torch.equal(v,restored.state_dict()[k]) for k,v in target.state_dict().items()),
                "Serialized initialization changed")
        report["output"]=identity(output)
        report["reload_exact"]=True
        write_json(provenance,report)
        return report
