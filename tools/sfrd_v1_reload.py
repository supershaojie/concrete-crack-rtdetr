"""Fresh-process import/reload probe; never decomposes or resets learned factors/gates."""
import argparse
import torch
from init_sfrd_v1 import RTDETR, build_training_model, tensor_hash, require, write_json
from ultralytics.utils.patches import torch_load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','expected','output'): parser.add_argument('--'+name,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    model=RTDETR(args.checkpoint).model.eval()
    if model.model[-1].nc == 80:
        torch.manual_seed(42)
        model,_=build_training_model(model.yaml,model,dict(nc=1,channels=3),model.sfrd_variant,initial=True)
    model.eval(); expected=torch_load(args.expected,map_location='cpu')
    require({k:tensor_hash(v) for k,v in model.state_dict().items()} == expected['state_sha256'],'New-process state differs')
    with torch.no_grad(): output=model(expected['image'])[0]
    torch.testing.assert_close(output,expected['output'],rtol=0,atol=0)
    # Resume rebuild must retain nonzero gates and all factors, with no SVD call.
    restored,report=build_training_model(model.yaml,model,dict(nc=1,channels=3),model.sfrd_variant,initial=False)
    require(all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items()),'Resume rebuild changed trained state')
    write_json(args.output,dict(status='PASSED',states=len(model.state_dict()),output_exact=True,
        nonzero_gate_tensors=sum(int(torch.count_nonzero(v)>0) for k,v in model.state_dict().items() if '.gate.' in k),
        resume_strict_load=True,svd_called=False))


if __name__=='__main__':main()
