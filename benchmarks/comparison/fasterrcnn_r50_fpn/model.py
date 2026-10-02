"""Unmodified TorchVision Faster R-CNN architecture, explicit scratch construction."""
from __future__ import annotations

from contextlib import ExitStack
import os
import random
from unittest.mock import patch

import numpy as np
import torch
import torchvision

from support import recipe

LOSS_KEYS = {'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'}


def seed_all(seed=42, deterministic=True):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    # Preserve native CUDA RoIAlign/RPN behavior in 0.16.2, warn for nondeterministic ops.
    # This is recorded and is not a promise of bitwise reproducibility.
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def build_kwargs(cfg):
    if cfg['weights'] is not None or cfg['weights_backbone'] is not None:
        raise ValueError('Scratch construction rejects either pretrained weights parameter')
    if cfg['num_classes'] != 2 or cfg['trainable_backbone_layers'] != 5:
        raise ValueError('Expected background 0, crack 1, and all backbone layers trainable')
    return dict(weights=None, weights_backbone=None, num_classes=2, trainable_backbone_layers=5,
        min_size=cfg['model_transform_min_size'], max_size=cfg['model_transform_max_size'],
        **{k: cfg[k] for k in ('box_score_thresh', 'box_nms_thresh', 'box_detections_per_img',
            'rpn_nms_thresh', 'rpn_pre_nms_top_n_train', 'rpn_post_nms_top_n_train',
            'rpn_pre_nms_top_n_test', 'rpn_post_nms_top_n_test')})


def assert_backbone_size(module, inputs):
    expected = module.comparison_input_size
    if tuple(inputs[0].shape[-2:]) != (expected, expected):
        raise ValueError('Actual backbone input is not the configured square: ' + str(inputs[0].shape))
    module.comparison_observed_size = list(inputs[0].shape[-2:])


def structure(model):
    from torchvision.ops.misc import FrozenBatchNorm2d
    trainable = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.backbone.named_parameters() if not p.requires_grad]
    batch_norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    if frozen or not batch_norms or any(isinstance(m, FrozenBatchNorm2d) for m in model.modules()):
        raise ValueError('Scratch backbone must use trainable native BatchNorm and no frozen parameters')
    if model.roi_heads.box_predictor.cls_score.out_features != 2:
        raise ValueError('Wrong foreground/background head')
    return {'model_class': type(model).__module__ + '.' + type(model).__name__,
        'backbone_class': type(model.backbone).__name__, 'resnet_blocks': [len(model.backbone.body[k]) for k in
            ('layer1', 'layer2', 'layer3', 'layer4')],
        'box_head': type(model.roi_heads.box_head).__name__, 'box_predictor': type(model.roi_heads.box_predictor).__name__,
        'roi_align': str(model.roi_heads.box_roi_pool),
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'trainable_backbone_parameters': trainable, 'frozen_backbone_parameters': frozen,
        'batchnorm_count': len(batch_norms), 'batchnorm_eps': sorted(set(m.eps for m in batch_norms)),
        'batchnorm_momentum': sorted(set(m.momentum for m in batch_norms)), 'frozen_batchnorm_count': 0,
        'anchors': {'sizes': model.rpn.anchor_generator.sizes, 'aspect_ratios': model.rpn.anchor_generator.aspect_ratios},
        'rpn': {'nms_thresh': model.rpn.nms_thresh, 'score_thresh': model.rpn.score_thresh,
            'pre_nms_top_n': model.rpn._pre_nms_top_n, 'post_nms_top_n': model.rpn._post_nms_top_n,
            'positive_fraction': model.rpn.fg_bg_sampler.positive_fraction,
            'batch_size_per_image': model.rpn.fg_bg_sampler.batch_size_per_image,
            'fg_iou': model.rpn.proposal_matcher.high_threshold, 'bg_iou': model.rpn.proposal_matcher.low_threshold},
        'roi': {'score_thresh': model.roi_heads.score_thresh, 'nms_thresh': model.roi_heads.nms_thresh,
            'detections_per_img': model.roi_heads.detections_per_img,
            'positive_fraction': model.roi_heads.fg_bg_sampler.positive_fraction,
            'batch_size_per_image': model.roi_heads.fg_bg_sampler.batch_size_per_image,
            'fg_iou': model.roi_heads.proposal_matcher.high_threshold, 'bg_iou': model.roi_heads.proposal_matcher.low_threshold},
        'transform': {'min_size': model.transform.min_size, 'max_size': model.transform.max_size,
            'size_divisible': model.transform.size_divisible,
            'image_mean': model.transform.image_mean, 'image_std': model.transform.image_std},
        'fusion': 'none; same full architecture for training and FP32 export',
        'native_constants': 'Official BN constants/running buffers and official initialization retained'}


def build_model(cfg=None):
    cfg = cfg or recipe()
    kwargs = build_kwargs(cfg)
    intercepted = []
    def forbidden(*args, **kwargs):
        intercepted.append('external_weight_or_state_load')
        raise RuntimeError('External weights/state/network access forbidden during scratch construction')
    targets = ('torch.load', 'torch.nn.Module.load_state_dict', 'torch.hub.load_state_dict_from_url',
        'torch.hub.download_url_to_file', 'torchvision.models._api.load_state_dict_from_url',
        'torchvision._internally_replaced_utils.load_state_dict_from_url', 'urllib.request.urlopen')
    with ExitStack() as guards:
        for name in targets:
            guards.enter_context(patch(name, side_effect=forbidden))
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(**kwargs)
    if intercepted:
        raise RuntimeError('Initialization attempted an external state load')
    model.backbone.comparison_input_size = cfg['input_height']
    model.backbone.register_forward_pre_hook(assert_backbone_size)
    observed = structure(model)
    if observed['resnet_blocks'] != [3, 4, 6, 3] or observed['box_head'] != 'TwoMLPHead':
        raise ValueError('Expected standard ResNet50-FPN Faster R-CNN, not v2 or another detector')
    return model, {'initialization_type': 'random', 'pretraining_source': None,
        'pretrained_tensors_loaded': 0, 'function': 'torchvision.models.detection.fasterrcnn_resnet50_fpn',
        'constructor_args': kwargs, 'seed': cfg['seed'], 'guarded_calls': list(targets),
        'intercepted_calls': intercepted, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'structure': observed}


def finite_losses(losses):
    if set(losses) != LOSS_KEYS or not all(torch.isfinite(v).all() for v in losses.values()):
        raise FloatingPointError('Expected four finite native losses: ' + str(losses))
    total = sum(losses.values())
    if not torch.isfinite(total).all():
        raise FloatingPointError('Nonfinite total loss')
    return total


def finite_gradients(model):
    count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            count += 1
            if not torch.isfinite(param.grad).all():
                raise FloatingPointError('Nonfinite unscaled gradient: ' + name)
    if count == 0:
        raise FloatingPointError('No gradients from native detection losses')
    return count
