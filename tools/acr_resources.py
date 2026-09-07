"""Restore the native AMP image from a trusted copy or hash-verified official asset."""
import argparse
import io
import logging
from pathlib import Path
import shutil
from urllib.request import urlopen

from init_acr import ROOT, SERVER_MAIN, require, sha256, write_json

BUS_SHA256 = "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63"


def ensure_resources(main=SERVER_MAIN):
    target=ROOT / "ultralytics-main/ultralytics/assets/bus.jpg"
    origin="existing worktree asset"
    if not target.is_file():
        source=Path(main)/"ultralytics-main/ultralytics/assets/bus.jpg"
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.is_file():
            require(sha256(source)==BUS_SHA256,"Existing bus.jpg is not the verified official image")
            shutil.copyfile(source,target);origin=str(source)
        else:
            import hashlib
            origin="https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
            with urlopen(origin,timeout=30) as response:payload=response.read(2*1024*1024)
            require(hashlib.sha256(payload).hexdigest()==BUS_SHA256,"Official bus.jpg SHA mismatch")
            with target.open('xb') as file:file.write(payload)
    require(sha256(target)==BUS_SHA256,"Worktree bus.jpg SHA mismatch")
    helper=ROOT/'yolo26n.pt';shared=Path(main)/'yolo26n.pt'
    if not helper.exists() and shared.is_file():shutil.copyfile(shared,helper)
    return {'bus_path':str(target),'bus_sha256':BUS_SHA256,'origin':origin,
            'yolo_helper':str(helper),'helper_present':helper.is_file(),
            'note':'Native check_amp may download its official helper if absent; no AMP bypass.'}


def native_amp_check(model):
    from ultralytics.utils import LOGGER
    from ultralytics.utils.checks import check_amp
    stream=io.StringIO();handler=logging.StreamHandler(stream);LOGGER.addHandler(handler)
    try:
        enabled=check_amp(model)
    finally:LOGGER.removeHandler(handler)
    log=stream.getvalue()
    passed=enabled and 'checks passed' in log and 'checks skipped' not in log
    return {'status':'passed' if passed else 'unverified_or_failed','enabled':enabled,'log':log}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main',type=Path,default=SERVER_MAIN)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();write_json(args.report,ensure_resources(args.main))
