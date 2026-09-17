"""Generate an untracked, fixed-HEAD server handoff after the final code commit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def generate(output):
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert len(sha) == 40
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip(), 'Commit all source changes first'
    template = (ROOT/'docs/sre/SERVER_COMMANDS.template.md').read_text(encoding='utf-8')
    rendered = template.replace('@SRE_SHA@', sha)
    assert '@SRE_SHA@' not in rendered
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(rendered)
    record = dict(commit=sha, branch='exp-rtdetr-r18-lite-sre-v1',
                  base='a0459d6a652cb702699087c88fa39a3e4c4087ec',
                  commands=str(output.resolve()), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  formal_training='NOT_STARTED', final_test='NOT_RUN')
    with output.with_suffix('.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    generate(parser.parse_args().output)
