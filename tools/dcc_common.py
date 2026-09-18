"""Shared DCC identities, structural checks and controlled native reconstruction.

Only public untrained source weights are accepted. Original CBR/LIF initialization
is delegated to the unchanged successful parent's initialization tool.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
import ultralytics
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load

MODEL_DIR = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr"
BASE_COMMIT = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
SOURCE_SHA256 = "fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e"
C2_COMMIT = "67c3078e54a657fd96d65fee657a75fbb1dae0d6"
VARIANTS = {
    "cbr_lif_dcc_v1": ("rtdetr-resnet18-lite-cbr-lif-dcc-v1.yaml",
                       "rtdetr-resnet18-lite-cbr-lif-down.yaml",
                       "cbr_lif_dcc_v1_rtdetr_r18_lite_e200_b16_onlineaug"),
    "dcc_v1": ("rtdetr-resnet18-lite-dcc-v1.yaml", "rtdetr-resnet18-lite.yaml",
               "dcc_v1_rtdetr_r18_lite_e200_b16_onlineaug"),
}
MODULE_HASHES = {
    "lif_down": "26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7",
    "cbr": "d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787",
}
PARAMETERS = {"cbr_lif_dcc_v1": (20168197, 19963397), "dcc_v1": (20101204, 19896148)}
DCC_KEYS = {"model.17.dcc." + name + ".weight" for name in ("W_d", "W_q", "W_k", "W_o")}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_contract():
    result = {}
    for name, expected in MODULE_HASHES.items():
        path = ROOT / "ultralytics-main/ultralytics/nn/modules" / (name + ".py")
        normalized = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        require(normalized == expected, "Original module changed: " + name)
        result[name] = dict(path=str(path), raw_sha256=sha256(path), lf_sha256=normalized)
    return result


def runtime():
    require(Path(ultralytics.__file__).resolve().is_relative_to(ROOT / "ultralytics-main"), "Import is not from this worktree")
    paths = [ROOT / "ultralytics-main/ultralytics/nn/modules/dcc.py",
             ROOT / "ultralytics-main/ultralytics/nn/tasks.py",
             ROOT / "ultralytics-main/ultralytics/nn/modules/__init__.py"]
    paths += sorted(ROOT.glob("tools/*dcc*.py"))
    paths += [MODEL_DIR / item[0] for item in VARIANTS.values()]
    git = ["git", "-c", "safe.directory=" + ROOT.as_posix()]
    return dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__),
                cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                ultralytics=str(Path(ultralytics.__file__).resolve()), worktree=str(ROOT),
                commit=subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                module_hashes=source_contract(),
                files={p.relative_to(ROOT).as_posix(): sha256(p) for p in paths if p.is_file()})


def is_added(key):
    return key.startswith("model.17.dcc.")


def added_buffers(model):
    state = model.state_dict()
    return [dict(name=name, shape=list(value.shape), persistent=name in state, trainable=False)
            for name, value in model.named_buffers() if is_added(name)]


def class_adaptation_keys():
    prefix = "model.26."
    return {prefix + "denoising_class_embed.weight", prefix + "enc_score_head.weight", prefix + "enc_score_head.bias"} | {
        prefix + "dec_score_head.%s.%s" % (i, suffix) for i in range(3) for suffix in ("weight", "bias")}


def build(variant="cbr_lif_dcc_v1", nc=80, baseline=False):
    require(variant in VARIANTS, "Unknown DCC variant")
    with torch.random.fork_rng(devices=[]):
        # CPU modules only. Never seed CUDA in this CPU isolation scope.
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[variant][int(baseline)]), nc=nc, verbose=False)


def verify_model(model, variant="cbr_lif_dcc_v1", zero=False):
    from ultralytics.nn.modules import Conv, DCCConv, DCC, LIFDown, RTDETRDecoder, RTDETRDecoderCBR
    require(variant in VARIANTS, "Unknown DCC variant")
    source_contract()
    expected = YAML.load(MODEL_DIR / VARIANTS[variant][1])
    projection = expected["head"][9]
    require(projection[:3] == [5, 1, "Conv"] and projection[3][:3] == [256, 1, 1]
            and projection[3][3] in (None, "None") and projection[3][4:] == [1, 1, False], "Parent projection contract differs")
    expected["head"][9][2] = "DCCConv"
    require(type(model) is RTDETRDetectionModel, "Native detection model required")
    require(len(model.model) == 27, "DCC must preserve 27 graph nodes")
    require(all(model.yaml[k] == expected[k] for k in ("backbone", "head", "scales")), "Only parent layer 17 class may change")
    node, head = model.model[17], model.model[26]
    require(type(node) is DCCConv and node.f == 5, "DCCConv must replace original P3 projection")
    require(sum(isinstance(m, DCC) for m in model.modules()) == 1, "Exactly one DCC required")
    require(model.model[18].f == [-2, -1] and model.model[20].f == -1 and head.f == [19, 22, 25], "Downstream topology changed")
    require((node.conv.in_channels, node.conv.out_channels, node.conv.kernel_size, node.conv.stride) == (128, 256, (1, 1), (1, 1)), "Projection dimensions changed")
    require(type(model.model[20]) is (LIFDown if variant == "cbr_lif_dcc_v1" else Conv), "Parent downsample changed")
    require(type(head) is (RTDETRDecoderCBR if variant == "cbr_lif_dcc_v1" else RTDETRDecoder), "Parent decoder changed")
    require(head.hidden_dim == 256 and head.num_queries == 300 and len(head.decoder.layers) == 3 and head.decoder.eval_idx == 2, "Decoder contract changed")
    if variant == "cbr_lif_dcc_v1":
        require(head.cbr.rho == head.cbr.normal_fraction == .10, "Original CBR fractions changed")
    require(set(k for k in model.state_dict() if is_added(k)) == DCC_KEYS, "Unexpected DCC state/buffers")
    require(sum(p.numel() for name, p in model.named_parameters() if is_added(name)) == 18432, "DCC parameter count must be 18,432")
    fused = not hasattr(node, "bn")
    if head.nc == 1:
        require(sum(p.numel() for p in model.parameters()) == PARAMETERS[variant][int(fused)], "Model parameter count differs")
    if zero:
        require(torch.count_nonzero(node.dcc.W_o.weight) == 0, "DCC output projection must start zero")
        require(all(torch.count_nonzero(getattr(node.dcc, n).weight) > 0 for n in ("W_d", "W_q", "W_k")), "DCC upstream projections must be nonzero")
    return dict(projection=17, concat=18, p3=19, p4=22, p5=25, decoder=26,
                p3_to_p4=dict(input=19, downsample=20, output=22), nodes=27, new_parameters=18432)


def native_rebuild(cfg, weights, nc=1, channels=3):
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.data = dict(nc=nc, channels=channels)
    return RTDETRTrainer.get_model(trainer, cfg=deepcopy(cfg), weights=weights, verbose=False)


def build_training_model(cfg, weights, data, variant="cbr_lif_dcc_v1"):
    """Native reconstruction with synchronized reference classification-head RNG.

    Trained DCC state is preserved; this function never reapplies zero initialization.
    """
    channels = data.get("channels", 3)
    with torch.random.fork_rng(devices=[]):
        parent = native_rebuild(str(MODEL_DIR / VARIANTS[variant][1]), weights, data["nc"], channels)
    target = native_rebuild(cfg, weights, data["nc"], channels)
    verify_model(target, variant)
    before, after, public = weights.state_dict(), target.state_dict(), parent.state_dict()
    require(set(before) == set(after), "Native Trainer changed state inventory")
    changed = {key for key in before if before[key].shape != after[key].shape}
    expected = class_adaptation_keys() if weights.model[-1].nc != data["nc"] else set()
    require(changed == expected, "Unexpected native class adaptation")
    require(all(torch.equal(value, after[key]) for key, value in before.items() if key not in changed), "Trainer lost initialized or learned state")
    require(all(torch.equal(value, after[key]) for key, value in public.items()), "Parent/target native class adaptation differs")
    report = dict(native_get_model=True, target_nc=data["nc"],
                  COMMON=[dict(name=key, shape=list(value.shape), equal=True, sha256=tensor_sha256(value)) for key, value in public.items()],
                  NEW_TRAINABLE=sorted(DCC_KEYS), NEW_BUFFER=added_buffers(target), STATE_DICT_NEW_BUFFER=[],
                  MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[],
                  ALLOWED_CLASS_ADAPTATION=[dict(name=key, source_shape=list(before[key].shape), target_shape=list(after[key].shape), parent_target_equal=True) for key in sorted(changed)],
                  all_nonadapted_values_exact=True, parent_target_common_exact=True, no_zero_reset=True)
    return target, report


def controlled_models(source, variant="cbr_lif_dcc_v1"):
    source = Path(source)
    require(source.is_file() and sha256(source) == SOURCE_SHA256, "Public source missing or SHA256 mismatch")
    checkpoint = torch_load(source, map_location="cpu")
    require(checkpoint.get("epoch") == -1 and all(checkpoint.get(key) is None for key in
            ("optimizer", "ema", "scaler", "updates", "train_metrics", "train_results", "best_fitness")), "Public source contains trained state")
    original = deepcopy(checkpoint["model"]).float()
    require(original.model[-1].nc == 80, "Public source must have nc=80")
    if variant == "cbr_lif_dcc_v1":
        from init_c19_lif_v1 import controlled_models as original_pair_models
        _, parent80, parent_report = original_pair_models(source)
    else:
        parent80 = build(variant, baseline=True)
        require(original.yaml["backbone"] == parent80.yaml["backbone"] and original.yaml["head"] == parent80.yaml["head"], "Public C2 semantics changed")
        parent80.load_state_dict(original.state_dict(), strict=True)
        parent_report = dict(method="Original C2 model, exact public state", source_sha256=SOURCE_SHA256)
    fresh_parent, target80 = build(variant, baseline=True), build(variant)
    public, target_state = fresh_parent.state_dict(), target80.state_dict()
    require(set(target_state) - set(public) == DCC_KEYS and set(public) <= set(target_state), "Unexpected added/missing state")
    require(all(torch.equal(value, target_state[key]) for key, value in public.items()), "DCC constructor consumed public RNG")
    target80.load_state_dict({**target_state, **parent80.state_dict()}, strict=True)
    verify_model(target80, variant, zero=True)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        target, native_report = build_training_model(target80.yaml, target80, dict(nc=1, channels=3), variant)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        parent = native_rebuild(parent80.yaml, parent80, nc=1)
    require(all(torch.equal(value, target.state_dict()[key]) for key, value in parent.state_dict().items()), "Controlled nc1 parent values differ")
    verify_model(target, variant, zero=True)
    counterpart = "dcc_v1" if variant == "cbr_lif_dcc_v1" else "cbr_lif_dcc_v1"
    other_state = build(counterpart).state_dict()
    require(all(torch.equal(target.state_dict()[key], other_state[key]) for key in DCC_KEYS), "Variant DCC initialization differs")
    report = dict(variant=variant, kind="controlled_initialization", source=str(source.resolve()), source_sha256=SOURCE_SHA256,
                  base_commit=BASE_COMMIT, c2_commit=C2_COMMIT, source_nc=80, target_nc=1, seed=42,
                  COMMON=native_report["COMMON"], NEW_TRAINABLE=sorted(DCC_KEYS), NEW_BUFFER=added_buffers(target), STATE_DICT_NEW_BUFFER=[],
                  MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[],
                  ALLOWED_CLASS_ADAPTATION=native_report["ALLOWED_CLASS_ADAPTATION"], new_parameters=18432,
                  new_initial_values={key: dict(shape=list(target.state_dict()[key].shape), sha256=tensor_sha256(target.state_dict()[key]), nonzero=int(torch.count_nonzero(target.state_dict()[key]))) for key in sorted(DCC_KEYS)},
                  variant_new_state_exact=True, public_constructor_equal=True, parent_target_nc1_exact=True,
                  original_parent_initialization=parent_report, native_trainer=native_report,
                  storage="FP32, source values promoted exactly; nine class tensors rebuilt natively with synchronized CPU RNG")
    return parent, target, report
