"""Fresh-process checkpoint/EMA/rebuild regression; only disposable fixtures."""
import argparse
from copy import deepcopy
import torch
from init_nbrg_v1 import RTDETR, verify_model, tensor_hash, require, write_json
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import ModelEMA, initialize_weights
from init_c19_lif_v1 import native_rebuild


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    fixture = torch_load(args.fixture, map_location='cpu')
    model = RTDETR(fixture['checkpoint']).model.eval()
    verify_model(model, fixture['variant'], zero=fixture['zero'])
    require({k: tensor_hash(v) for k, v in model.state_dict().items()} == fixture['state_hashes'], 'Reload changed parameters/buffers')
    with torch.no_grad():
        torch.testing.assert_close(model(fixture['image'])[0], fixture['prediction'], rtol=1e-5, atol=1e-6)
    ema = ModelEMA(model).ema
    require({k: tensor_hash(v) for k, v in ema.state_dict().items()} == fixture['state_hashes'], 'EMA changed gate/state')
    rebuilt = native_rebuild(model.yaml, model, nc=model.model[-1].nc).eval()
    require({k: tensor_hash(v) for k, v in rebuilt.state_dict().items()} == fixture['state_hashes'], 'Native rebuild lost learned gate')
    initialize_weights(rebuilt)
    require({k: tensor_hash(v) for k, v in rebuilt.state_dict().items()} == fixture['state_hashes'], 'Generic initialize_weights changed gate')
    write_json(args.output, dict(status='PASSED', fresh_process=True, zero=fixture['zero'], state_exact=True,
        prediction_close=True, model_ema_exact=True, native_rebuild_exact=True, generic_initialize_preserves_gate=True))
