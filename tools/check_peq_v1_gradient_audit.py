"""Negative controls for the gradient audit; no formal weights, data or optimizer."""
from peq_v1_gradient_audit import *
from peq_v1_common import *


def check_guards():
    entry=precision_state();entry_rng=fingerprint(rng_state())
    # Exercise restoration even when a diagnostic raises inside active autocast.
    try:
        with isolated_rng(),diagnostic_precision(False),torch.autocast("cpu",enabled=True):
            torch.set_float32_matmul_precision("medium")
            torch.backends.cudnn.allow_tf32=True
            with torch.autocast("cuda",enabled=torch.cuda.is_available()):
                active=precision_state()
                try:
                    with diagnostic_precision(True):
                        torch.rand(3);random.random();np.random.random()
                        require(not precision_state()["cpu_autocast"] and not precision_state()["cuda_autocast"],
                                "Diagnostic autocast was not disabled")
                        raise ValueError("intentional diagnostic failure")
                except ValueError as error:
                    require(str(error)=="intentional diagnostic failure","Unexpected exception")
                require(precision_state()==active,"Active autocast/TF32 did not restore after exception")
    finally:
        require(precision_state()==entry and fingerprint(rng_state())==entry_rng,"Exception fixture leaked state")

    name="model.0.conv.weight"
    raw={name:torch.ones(32,3,3,3)}
    clipped={name:torch.full((32,3,3,3),.1)}
    reference=dict(raw=raw,clipped=clipped)
    repeat=differences(raw,raw,2e-5,3e-4)
    wrong=dict(raw={name:raw[name].clone()},clipped={name:clipped[name].clone()})
    wrong["clipped"][name][0,0,0,0]+=.001
    comparison=compare_snapshots(reference,wrong,repeat,torch.device("cpu"))
    require(comparison["failed_clipped"]==[name] and not comparison["passed"],
            "An injected clipped-gradient error was accepted or lost its full parameter name")

    # A tiny injected gradient can hide below numeric tolerances, but must still fail the detach check.
    report={}
    with isolated_rng(),diagnostic_precision(True):
        torch.manual_seed(42)
        model=build(1).train()
        with torch.no_grad():model.model[-1].peq.output.weight.normal_(std=.01)
        batch=dict(img=torch.rand(2,3,160,160),bboxes=torch.tensor([[.5,.5,.35,.2],[.3,.4,.15,.3]]),
                   cls=torch.zeros(2,1),batch_idx=torch.zeros(2,dtype=torch.long))
        def inject_leak(module,args,output):
            return {**output,"quality_logits":output["quality_logits"]+1e-12*args[0].sum()}
        handle=model.model[-1].peq.register_forward_hook(inject_leak)
        try:
            profile(model,batch,report,lambda:None,torch.device("cpu"))
        except RuntimeError as error:
            require("L_Q crosses detach boundary" in str(error) and "'F'" in str(error),
                    f"Injected leak failed for the wrong reason: {error}")
            leak_error=str(error)
        else:
            raise AssertionError("An actual injected quality-to-F gradient was accepted")
        finally:
            handle.remove()
    require(precision_state()==entry and fingerprint(rng_state())==entry_rng,"Negative control leaked diagnostic state")
    return dict(status="PASS",exception_restores_active_autocast_precision_and_rng=True,
                injected_clipped_error_rejected=comparison["failed_clipped"],injected_quality_leak_rejected=leak_error,
                all_original_numeric_comparisons_passed_despite_tiny_leak=all(r["passed"] for r in report["comparisons"].values()),
                diagnostic_only=True)


if __name__=="__main__":
    torch.set_num_threads(4)
    result=check_guards()
    write_json(OUT/"gradient_guard_checks.json",result)
    print(json_bytes(result).decode())
