"""Reuse existing native AMP resources; explicit --download permits official asset retrieval."""
import argparse
from pathlib import Path
import shutil
import urllib.request

from init_lif_down import ROOT, require, sha256, write_json
from ultralytics.utils import ASSETS


def prepare(main, report, download=False):
    rows = []
    for name, destination, candidates, url in (
        ('yolo26n.pt', ROOT/'yolo26n.pt', [main/'yolo26n.pt', main/'weights/yolo26n.pt'],
         'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt'),
        ('bus.jpg', ASSETS/'bus.jpg', [main/'ultralytics-main/ultralytics/assets/bus.jpg', main/'bus.jpg'],
         'https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg'),
    ):
        origin = 'existing worktree resource'
        if not destination.is_file():
            source = next((p for p in candidates if p.is_file()), None)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source:
                with destination.open('xb') as target, source.open('rb') as stream:
                    shutil.copyfileobj(stream, target)
                origin = str(source)
            else:
                require(download, f'Missing {name}. Supply a compatible local file at {destination}, '
                                  'or rerun this command with --download for the official resource. No AMP skip.')
                temporary = destination.with_name(destination.name+'.csr-download.partial')
                require(not temporary.exists(), f'Inspect preserved prior partial download: {temporary}')
                with urllib.request.urlopen(url, timeout=60) as response, temporary.open('xb') as target:
                    shutil.copyfileobj(response, target)
                with destination.open('xb') as target, temporary.open('rb') as stream:
                    shutil.copyfileobj(stream, target)
                temporary.unlink()
                origin = url
        require(destination.stat().st_size > 0, 'Empty resource: '+str(destination))
        rows.append(dict(name=name, path=str(destination.resolve()), source=origin, sha256=sha256(destination)))
    require(not Path(report).exists(), 'Preserve existing resource report: '+str(report))
    write_json(report, dict(status='READY_FOR_NATIVE_CHECK', resources=rows,
                           note='Existence/hash only; actual compatibility is verified by native check_amp.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', type=Path, default=Path('/root/autodl-tmp/projects/Crack_RTDETR'))
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    prepare(args.main, args.report, args.download)
