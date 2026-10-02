"""Evidence-based augmentation table; static analysis is distinct from a runtime probe."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from common import digest, load_yaml, write_json

ROOT = Path(__file__).resolve().parents[2]


def build_table(args, project, template, archived_freeze=None):
    table = load_yaml(template)
    table['training_args_evidence'] = 'observed_actual' if args is not None else 'MISSING; values not reconstructed from references'
    for item in table['items']:
        key = item['field']
        item['reference_value'] = item['configured_value']
        item['configured_value'] = args.get(key) if args is not None and key in args else None
        item['configuration_status'] = 'observed_actual' if args is not None and key in args else 'not_an_args_field' if key.startswith('implicit_') else 'UNKNOWN'
        if not key.startswith('implicit_') and (args is None or key not in args or args[key] != item['reference_value']):
            item['effective'] = None
            item['status'] = 'ACTUAL_CONFIG_MISSING_OR_DIFFERS_REVIEW_REQUIRED'
    table['source_identity'] = {}
    for name in table['source_files']:
        path = Path(project) / name
        table['source_identity'][name] = {'path': str(path), 'lf_sha256': digest(path.read_bytes().replace(b'\r\n', b'\n'))} if path.is_file() else {'status': 'MISSING'}
    table['archived_albumentations_evidence'] = {
        'pip_freeze_available': archived_freeze is not None,
        'entries': [s for s in (archived_freeze or '').splitlines() if s.lower().startswith('albumentations')],
        'conclusion': 'not listed in archived pip freeze; historical transform activation not directly probed' if archived_freeze is not None and 'albumentations' not in archived_freeze.lower() else 'inspect archived dependency record',
    }
    table['runtime_probe'] = {'status': 'NOT_RUN'}
    return table


def probe_transforms(project, args):
    """Construct only transform objects. No images, caches, detector, GPU allocation, or trainer."""
    if not args:
        return {'status': 'UNAVAILABLE', 'reason': 'Actual training args missing'}
    sys.path.insert(0, str(Path(project) / 'ultralytics-main'))
    try:
        import ultralytics
        from ultralytics.models.rtdetr.val import RTDETRDataset
        import ultralytics.data.augment as augment
        dataset = object.__new__(RTDETRDataset)
        dataset.augment, dataset.rect, dataset.imgsz = True, False, int(args['imgsz'])
        dataset.use_segments, dataset.use_keypoints = False, False
        dataset.data, dataset.cache = {'names': {0: 'crack'}}, False
        hyp = SimpleNamespace(**args)
        def describe(transform):
            result = {'class': type(transform).__module__ + '.' + type(transform).__name__}
            for key in ('p', 'degrees', 'translate', 'scale', 'shear', 'perspective', 'hgain', 'sgain', 'vgain', 'direction', 'bgr', 'n'):
                value = getattr(transform, key, None)
                if isinstance(value, (int, float, str, bool)):
                    result[key] = value
            if hasattr(transform, 'transforms'):
                result['children'] = [describe(t) for t in transform.transforms]
            if getattr(transform, 'pre_transform', None) is not None:
                result['pre_transform'] = describe(transform.pre_transform)
            if type(transform).__name__ == 'Albumentations':
                result['active_transform'] = getattr(transform, 'transform', None) is not None
            return result
        before = describe(dataset.build_transforms(hyp))
        dataset.close_mosaic(hyp)
        return {'status': 'TRANSFORM_OBJECTS_CONSTRUCTED', 'image_execution': 'NOT_RUN',
                'ultralytics_path': ultralytics.__file__, 'version': ultralytics.__version__, 'augment_module': augment.__file__,
                'albumentations_installed': importlib.util.find_spec('albumentations') is not None,
                'before_close': before, 'after_close': describe(dataset.transforms)}
    except Exception as e:
        return {'status': 'UNAVAILABLE', 'error': repr(e)}
