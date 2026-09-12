"""Run all finite server gates in a fresh diagnostic directory; never dispatch.

This creates a separate disposable initialization. A later start-direct still
creates its own fresh formal initialization and repeats all mandatory gates.
"""
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
import uuid

from init_c19_lif_v1 import ROOT,initialize,runtime
from c19_lif_v1_diagnostic import atomic_json
from train_c19_lif_v1 import (verify_delivery,verify_server_environment,paths,MAIN,require,
                             duplicate_processes,verify_data_config,recipe,require_preflight)


def main():
    verify_delivery();verify_server_environment(runtime())
    require(os.environ.get('CONDA_DEFAULT_ENV')=='rtdetr','Activate existing rtdetr environment')
    require(not duplicate_processes(),'Existing experiment worker protected')
    p=paths('c19_lif_v1')
    require(not p['run'].exists(),'Existing formal results protected')
    out=ROOT/'outputs'/('c19_lif_v1_manual_preflight_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='FAILED',formal_training='NOT_RUN',full_val_test='NOT_RUN',runtime=runtime(),directory=str(out))
    print('Finite preflight directory:',out,flush=True)
    try:
        init=out/'controlled_init.pt'
        args,_=recipe(p['c2_args'],'c19_lif_v1',init);verify_data_config(args['data'])
        atomic_json(out/'initialization.json',initialize(p['source'],init))
        with (out/'preflight.log').open('x',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'tools/check_c19_lif_v1.py'),'--source',str(p['source']),
                '--initialized',str(init),'--real-dataset',str(MAIN/'datasets/crack_det'),'--output',str(out/'checks'),
                '--capacity-batch','16'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        check=json.loads((out/'checks/checks.json').read_text(encoding='utf-8'));require_preflight(check)
        report['status']='PASSED'
    except BaseException as error:
        report['error']=repr(error);raise
    finally:
        report['checks']=str(out/'checks/checks.json');report['log']=str(out/'preflight.log')
        atomic_json(out/'preflight_only.json',report)


if __name__=='__main__':main()
