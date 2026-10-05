"""Audit the pinned native optimizer without replacing its grouping or updates."""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from support import digest, read_json, LOCK

LR_PATTERN = re.compile(r'(?=.*23)(?=.*cv3)|proto\.semseg|flow_model')
SUPPORTED = {v.lower(): v for v in ('MuSGD', 'SGD', 'Adam', 'AdamW')}


def normalize_optimizer(name):
    if not isinstance(name, str) or name.lower() not in SUPPORTED:
        raise ValueError('optimizer must be MuSGD/SGD/Adam/AdamW; auto is forbidden')
    return SUPPORTED[name.lower()]


def expected_class(name):
    import torch
    if normalize_optimizer(name) == 'MuSGD':
        from ultralytics.optim.muon import MuSGD
        return MuSGD
    return getattr(torch.optim, normalize_optimizer(name))


def role_map(model):
    import torch
    norm = tuple(v for k, v in torch.nn.__dict__.items() if 'Norm' in k)
    roles = {}
    for module_name, module in model.named_modules():
        for param_name, p in module.named_parameters(recurse=False):
            name = f'{module_name}.{param_name}' if module_name else param_name
            role = ('bias' if 'bias' in name else 'normalization' if isinstance(module, norm)
                    or 'logit_scale' in name else 'weight')
            roles[id(p)] = (name, role)
    return roles


def annotate_groups(optimizer, model, name, lr, decay):
    """Preserve every native group, including empty and 3x groups; add role metadata."""
    name = normalize_optimizer(name)
    if type(optimizer) is not expected_class(name):
        raise ValueError('Actual optimizer class differs from requested configuration')
    if name == 'MuSGD' and (optimizer.muon != .5 or optimizer.sgd != .5):
        raise ValueError('Pinned MuSGD requires muon=sgd=0.5')
    roles = role_map(model)
    seen = []
    for group in optimizer.param_groups:
        entries = [roles[id(p)] for p in group['params']]
        names = [n for n, _ in entries]
        group_roles = {r for _, r in entries}
        if 'bias' in group_roles and len(group_roles) != 1:
            raise ValueError('Native group mixes bias and non-bias parameters')
        group['comparison_bias'] = bool(entries) and group_roles == {'bias'}
        group['comparison_names'] = names
        group['comparison_roles'] = sorted(group_roles)
        seen.extend(id(p) for p in group['params'])
        if entries:
            boosted = [bool(LR_PATTERN.search(n)) for n in names] if name == 'MuSGD' else [False]*len(names)
            if len(set(boosted)) != 1:
                raise ValueError('Native group combines different LR multipliers')
            multiplier = 3 if boosted[0] else 1
            if abs(group['lr'] - lr*multiplier) > 1e-12:
                raise ValueError('Native group LR multiplier differs')
            muon = bool(group.get('use_muon', False))
            if name == 'MuSGD' and any((p.ndim >= 2) != muon for p in group['params']):
                raise ValueError('MuSGD matrix parameter coverage differs from native grouping')
            expected_decay = decay if muon or group_roles == {'weight'} else 0.0
            if group['weight_decay'] != expected_decay:
                raise ValueError('Native group weight decay differs')
    required = {id(p) for p in model.parameters() if p.requires_grad}
    if len(seen) != len(set(seen)) or set(seen) != required:
        raise ValueError('Optimizer has duplicate, missing or frozen parameters')
    if name == 'MuSGD' and not any(g.get('use_muon') and g['params'] for g in optimizer.param_groups):
        raise ValueError('MuSGD requires a nonempty native mixed group')


def optimizer_identity(optimizer, model, config):
    name = normalize_optimizer(config['optimizer'])
    if type(optimizer) is not expected_class(name):
        raise ValueError('Optimizer class identity changed')
    if name == 'MuSGD' and (optimizer.muon != .5 or optimizer.sgd != .5):
        raise ValueError('MuSGD constructor coefficients changed')
    roles = role_map(model)
    groups = []
    for i, g in enumerate(optimizer.param_groups):
        names = [roles[id(p)][0] for p in g['params']]
        if names != g.get('comparison_names'):
            raise ValueError('Optimizer parameter names/order changed')
        groups.append({'index': i, 'names': names, 'shapes': [list(p.shape) for p in g['params']],
            'roles': g['comparison_roles'], 'bias_warmup': g['comparison_bias'],
            'tensors': len(names), 'elements': sum(p.numel() for p in g['params']),
            'empty': not names, 'use_muon': bool(g.get('use_muon', False)),
            'initial_lr': g.get('initial_lr', g['lr']), 'weight_decay': g['weight_decay'],
            'nesterov': g.get('nesterov'), 'betas': list(g['betas']) if 'betas' in g else None,
            'eps': g.get('eps')})
    file = Path(inspect.getfile(type(optimizer)))
    return {'class': type(optimizer).__module__+'.'+type(optimizer).__name__,
        'source_sha256': digest(file.read_bytes().replace(b'\r\n', b'\n')),
        'name': name, 'muon': getattr(optimizer, 'muon', None), 'sgd': getattr(optimizer, 'sgd', None),
        'target_momentum_or_beta1': config['momentum'], 'lr0': config['lr0'],
        'groups': groups, 'lr_multiplier_pattern': LR_PATTERN.pattern if name == 'MuSGD' else None,
        'warmup': {'patch_sha256': read_json(LOCK)['patch_sha256'],
            'before': 'native j==0 identifies bias; mismatches native v8.4.0 group order',
            'after': 'real bias roles start at warmup_bias_lr, others at zero; target each initial_lr*lf(epoch)',
            'warmup_bias_lr': config['warmup_bias_lr'], 'warmup_momentum': config['warmup_momentum'],
            'warmup_epochs': config['warmup_epochs']},
        'precision': 'Muon Newton-Schulz native bfloat16; model AMP does not change optimizer algorithm',
        'decay_semantics': 'mixed group applies decay to SGD component after Muon update; other groups use native optimizer rules'}


def group_values(optimizer):
    return [{k: g.get(k) for k in ('lr', 'initial_lr', 'momentum', 'betas', 'eps',
                                 'weight_decay', 'nesterov', 'use_muon', 'comparison_bias')}
            for g in optimizer.param_groups]


def verify_buffers(optimizer):
    """Every populated mixed state must contain both native momentum buffers."""
    import torch
    mixed = ordinary = 0
    for g in optimizer.param_groups:
        for p in g['params']:
            state = optimizer.state.get(p, {})
            if not state:
                continue
            if g.get('use_muon'):
                if not {'momentum_buffer', 'momentum_buffer_SGD'} <= state.keys():
                    raise ValueError('MuSGD mixed state is missing one of its two buffers')
                mixed += 1
            else:
                ordinary += 1
            for value in state.values():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise ValueError('Nonfinite optimizer state')
    return {'populated_mixed_tensors': mixed, 'populated_other_tensors': ordinary}
