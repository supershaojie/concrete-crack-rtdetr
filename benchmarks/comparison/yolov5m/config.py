"""One YAML configuration, resolved once per run. No detector imports or CLI overrides."""
from __future__ import annotations
from copy import deepcopy
import math
from pathlib import Path
import re
import yaml
from support import canonical, digest, read_json, sha256

DEFAULTS = {
    'epochs': 200, 'patience': 50, 'batch': 16, 'imgsz': 640, 'workers': 8,
    'device': '0', 'seed': 42, 'deterministic': True, 'amp': True, 'cache': False,
    'rect': False, 'multi_scale': False, 'nbs': 64, 'optimizer': 'SGD',
    'lr0': .003, 'lrf': .01, 'momentum': .937, 'weight_decay': .0005,
    'warmup_epochs': 3., 'warmup_momentum': .8, 'warmup_bias_lr': .01,
    'cos_lr': True, 'box': .05, 'cls': .3, 'obj': .7, 'cls_pw': 1.,
    'obj_pw': 1., 'iou_t': .2, 'anchor_t': 4., 'fl_gamma': 0.,
    'label_smoothing': 0., 'hsv_h': .015, 'hsv_s': .5, 'hsv_v': .35,
    'degrees': 5., 'translate': .1, 'scale': .4, 'shear': 0., 'perspective': 0.,
    'flipud': 0., 'fliplr': .5, 'mosaic': .5, 'mixup': 0., 'cutmix': 0.,
    'copy_paste': 0., 'close_mosaic': 10, 'plots': True, 'save_period': -1,
}
FIXED = {
    'schema_version': 1, 'experiment': 'yolov5m-coco-native-ft-v1',
    'model': 'original_anchor_based_yolov5m_coco_native_ft_v1',
    'initialization_type': 'coco_detection_pretrained', 'pretraining_source': 'COCO',
    'model_yaml': 'models/yolov5m.yaml', 'scale_name': 'm', 'nc': 1,
    'freeze': [0], 'resume': False, 'save': True, 'val': True, 'exist_ok': False,
    'image_weights': False, 'quad': False, 'albumentations': None,
    'training_preprocess': 'v7.0_native_resize_letterbox_random_perspective_multiplicative_hsv',
    'selection': 'native_training_val_mAP50_95_latest_tie',
    'evaluation': {
        'precision': 'FP32', 'imgsz': 640, 'batch': 16, 'workers': 0,
        'conf': .001, 'nms_iou': .7, 'max_det': 300, 'agnostic': False,
        'multi_label': False, 'max_nms': 30000, 'augment': False, 'rect': False,
        'seed': 42, 'policy': 'corrected_sorted_conf_mask_v1',
        'inverse': 'actual_resize_xy_gain_and_integer_letterbox_padding_no_clipping',
        'nms_timeout': 'disabled_to_prevent_silent_missing_predictions',
    },
}
HYP_KEYS = (
    'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 'warmup_momentum',
    'warmup_bias_lr', 'box', 'cls', 'obj', 'cls_pw', 'obj_pw', 'iou_t', 'anchor_t',
    'fl_gamma', 'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
    'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste',
)
OPTION_KEYS = (
    'epochs', 'patience', 'workers', 'device', 'seed', 'deterministic', 'amp',
    'cos_lr', 'multi_scale', 'rect', 'nbs', 'optimizer', 'close_mosaic',
    'label_smoothing', 'save_period', 'imgsz',
)

class UniqueLoader(yaml.SafeLoader):
    pass

def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError('YAML fields must be strings')
        if key in result:
            raise ValueError('Duplicate YAML field: ' + key)
        result[key] = loader.construct_object(value_node, deep=deep)
    return result

UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)

def read_yaml(path):
    try:
        value = yaml.load(Path(path).read_text(encoding='utf-8-sig'), Loader=UniqueLoader)
    except yaml.YAMLError as e:
        raise ValueError('Invalid config YAML: ' + str(e)) from e
    if not isinstance(value, dict):
        raise ValueError('Config must be a nonempty YAML mapping')
    return value

def run_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', value):
        raise ValueError('run-id must be 1-96 letters/digits/_/- and begin with a letter/digit')
    return value

def resolve(user):
    unknown = set(user) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown or fixed config field(s): ' + ', '.join(sorted(map(str, unknown))))
    result = {**deepcopy(FIXED), **DEFAULTS, **user}
    integer_ranges = {'epochs': (1, 100000), 'patience': (0, 100000), 'batch': (1, 4096),
                      'imgsz': (64, 4096), 'workers': (0, 256), 'seed': (0, 2**32-1),
                      'nbs': (1, 4096), 'close_mosaic': (0, result['epochs']),
                      'save_period': (-1, 100000)}
    for key, (lo, hi) in integer_ranges.items():
        value = result[key]
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f'{key} must be an integer in [{lo}, {hi}] (no autobatch)')
    if result['imgsz'] % 32:
        raise ValueError('imgsz must be a multiple of 32; implicit size rounding is forbidden')
    if result['save_period'] == 0:
        raise ValueError('save_period must be -1 (disabled) or positive')
    for key in ('deterministic', 'amp', 'cache', 'rect', 'multi_scale', 'cos_lr', 'plots'):
        if type(result[key]) is not bool:
            raise ValueError(key + ' must be a YAML boolean')
    bounds = {'lr0': (0, 1), 'lrf': (0, 1), 'momentum': (0, 1), 'weight_decay': (0, 1),
              'warmup_epochs': (0, 100000), 'warmup_momentum': (0, 1),
              'warmup_bias_lr': (0, 1), 'box': (0, 100), 'cls': (0, 100), 'obj': (0, 100),
              'cls_pw': (0, 100), 'obj_pw': (0, 100), 'iou_t': (0, 1), 'anchor_t': (1, 100),
              'fl_gamma': (0, 100), 'label_smoothing': (0, 1), 'degrees': (0, 180),
              'translate': (0, 1), 'scale': (0, 1), 'shear': (0, 89), 'perspective': (0, .001),
              **{k: (0, 1) for k in ('hsv_h', 'hsv_s', 'hsv_v', 'flipud', 'fliplr',
                                    'mosaic', 'mixup', 'cutmix', 'copy_paste')}}
    for key, (lo, hi) in bounds.items():
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{key} must be a finite number in [{lo}, {hi}]')
        result[key] = float(value)
    for key in ('lr0', 'cls_pw', 'obj_pw'):
        if result[key] <= 0:
            raise ValueError(key + ' must be positive')
    if result['anchor_t'] <= 1:
        raise ValueError('anchor_t must be greater than 1')
    if result['momentum'] >= 1 or result['warmup_momentum'] >= 1:
        raise ValueError('momentum and warmup_momentum must be less than 1')
    if result['optimizer'] not in ('SGD', 'Adam', 'AdamW'):
        raise ValueError('optimizer must be SGD, Adam, or AdamW')
    if result['optimizer'] == 'SGD' and result['momentum'] <= 0:
        raise ValueError('Native Nesterov SGD requires momentum > 0')
    device = result['device']
    if type(device) is int and device >= 0:
        device = str(device)
    if not isinstance(device, str) or not re.fullmatch(r'cpu|[0-9]+', device):
        raise ValueError('device must be one GPU index or cpu; multi-GPU/DDP is unsupported')
    result['device'] = device
    if device == 'cpu' and result['amp']:
        raise ValueError('CPU checks require amp: false; AMP is never silently disabled')
    if result['cutmix'] or result['copy_paste']:
        raise ValueError('This native detection pipeline requires cutmix=0 and copy_paste=0')
    if result['rect'] and (result['mosaic'] or result['mixup']):
        raise ValueError('Native rect training disables Mosaic/MixUp; set both probabilities to 0')
    if result['mixup'] and not result['mosaic']:
        raise ValueError('Native v7.0 MixUp runs only in the Mosaic branch; mosaic must be > 0')
    return result

def native_hyp(config):
    return {key: config[key] for key in HYP_KEYS}

def recipe(config):
    return {**deepcopy(config), 'batch_size': config['batch'],
            'native_validation': {'batch': config['batch'], 'imgsz': config['imgsz'], 'rect': True,
                                  'conf': .001, 'nms_iou': .6, 'max_det': 300, 'TTA': False}}

def freeze_config(source, run):
    """Parse exactly the bytes that are archived, avoiding a source-edit race."""
    raw = Path(source).read_bytes()
    target = Path(run) / 'user_config.yaml'
    target.write_bytes(raw)
    config = resolve(read_yaml(target))
    (Path(run) / 'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (Path(run) / 'train_hyp.yaml').write_text(yaml.safe_dump(native_hyp(config), sort_keys=False), encoding='utf-8')
    return config

def snapshot(run):
    run = Path(run)
    frozen = read_json(run / 'frozen.json')
    for name, expected in frozen['config_checksums'].items():
        if sha256(run / name) != expected:
            raise ValueError('Run config snapshot changed: ' + name + '; use a new run-id for new parameters')
    config = read_yaml(run / 'resolved_config.yaml')
    user_values = {key: config[key] for key in DEFAULTS}
    if config != resolve(user_values) or digest(canonical(config)) != frozen['config_sha256']:
        raise ValueError('Resolved run configuration identity differs')
    if read_yaml(run / 'train_hyp.yaml') != native_hyp(config):
        raise ValueError('Frozen raw hyperparameters differ')
    return config
