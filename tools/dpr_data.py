"""Read-only split identity: compare actual configured paths/labels with successful parent."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from init_lif_down import ROOT, require, sha256
from ultralytics.utils import YAML


def dataset_identity(data):
    config = YAML.load(data)
    root = Path(config['path']).resolve(strict=True)
    names = config['names']
    require(names == {0: 'crack'} or names == ['crack'], 'Expected one crack class')
    result = {}
    for split in ('train', 'val', 'test'):
        # This experiment deliberately accepts only the verified directory split.
        directory = (root / config[split]).resolve(strict=True)
        require(directory == root / 'images' / split, 'Configured split differs from verified parent: ' + split)
        files = sorted(p for p in directory.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
        require(files, 'Empty split: ' + split)
        paths, labels, count = [], [], 0
        for image in files:
            label = root / 'labels' / split / image.relative_to(directory).with_suffix('.txt')
            rows = [row.split() for row in label.read_text(encoding='utf-8').splitlines() if row.strip()]
            for row in rows:
                require(len(row) == 5 and row[0] == '0', 'Invalid class/label: ' + str(label))
                values = [float(v) for v in row[1:]]
                require(all(0 <= v <= 1 for v in values) and values[2] > 0 and values[3] > 0,
                        'Invalid normalized box: ' + str(label))
            paths.append(image.relative_to(root).as_posix())
            labels.append(label.relative_to(root).as_posix() + ':' + sha256(label))
            count += len(rows)
        result[split] = dict(images=len(files), boxes=count,
            split_paths_sha256=hashlib.sha256('\n'.join(paths).encode()).hexdigest(),
            label_inventory_sha256=hashlib.sha256('\n'.join(labels).encode()).hexdigest())
    reference = json.loads((ROOT / 'docs/dpr/parent_dataset_inventory.json').read_text(encoding='utf-8'))
    require(result == reference, 'Actual split paths/label hashes differ from successful parent; no automatic remapping')
    return result
