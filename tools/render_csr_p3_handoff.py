"""Render complete server commands with the current full commit and optional verified bundle hash."""
import argparse
from pathlib import Path
import subprocess

from init_lif_down import ROOT, require, sha256


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing handoff')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    template = (ROOT/'docs/csr_p3/SERVER_COMMANDS.template.md').read_text(encoding='utf-8')
    if args.bundle:
        subprocess.run(['git', 'bundle', 'verify', str(args.bundle.resolve())], cwd=ROOT, check=True)
        heads = subprocess.check_output(['git', 'bundle', 'list-heads', str(args.bundle.resolve()),
            'refs/heads/exp-rtdetr-r18-lite-csr-p3-v1'], cwd=ROOT, text=True).split()
        require(heads == [commit, 'refs/heads/exp-rtdetr-r18-lite-csr-p3-v1'], 'Bundle branch SHA differs from current HEAD')
    text = template.replace('@HEAD@', commit).replace('@BUNDLE_SHA@', sha256(args.bundle) if args.bundle else 'BUNDLE_NOT_SUPPLIED')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding='utf-8')
    print(args.output.resolve())
