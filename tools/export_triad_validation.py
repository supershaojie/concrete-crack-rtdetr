"""Collect small local evidence reports; never copy checkpoints or result packages into Git."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import shutil
from triad_compat import *
from ultralytics import RTDETR


def export(validation,pipeline,package):
    validation=Path(validation);dest=ROOT/'docs/triad_compat';evidence=dest/'evidence';evidence.mkdir(exist_ok=True)
    raw=json.loads((validation/'validation.json').read_text(encoding='utf-8'))
    summary=dict(runtime=raw['runtime'],scope=raw['scope'],legacy_regression=raw['legacy_regression'],variants={})
    for variant,v in VARIANTS.items():
        record=raw['variants'][variant]
        checkpoint=record.pop('checkpoint')
        loaded=RTDETR(str(validation/variant/'init.pt')).model
        from_yaml=RTDETR(str(MODEL_DIR/v['yaml'])).load(str(validation/variant/'init.pt')).model
        src,dst=loaded.state_dict(),from_yaml.state_dict()
        skip=[k for k in src if src[k].shape!=dst[k].shape]
        require(len(skip)==9 and all(torch.equal(t,dst[k]) for k,t in src.items() if k not in skip),'YAML API load discrepancy')
        checkpoint.update(yaml_load_exact_states=len(src)-len(skip),yaml_load_class_adaptation=skip)
        record['initialization_report']=f'{variant}_initialization.json'
        write_json(evidence/f'{variant}_initialization.json',checkpoint)
        write_json(evidence/f'{variant}_validation.json',record)
        def leaves(value):
            if isinstance(value,dict):
                for k,x in value.items():
                    if k=='max_abs_error':yield x
                    else:yield from leaves(x)
        summary['variants'][variant]=dict(yaml=v['yaml'],parameters=record['parameters'],
            max_zero_init_abs=max(leaves(record['zero_init'])),max_common_abs=max(x['max_abs'] for x in checkpoint['COMMON']),
            common_states=len(checkpoint['COMMON']),new_states=len(checkpoint['NEW']),classification_adaptations=len(checkpoint['CLASS_ADAPTATION']),
            DN_queries=[s['query_count'] for s in record['AMP_loss_DN'].get('steps',[])],
            CPU_FP32_loss_DN=record['FP32_loss_DN']['status'],CUDA_AMP_loss_DN=record['AMP_loss_DN']['status'],
            true_half=record['precision']['status'],save_load_max_abs=max(leaves(record['FP32_loss_DN']['save_load'])),
            optimizer_all_once=record['FP32_loss_DN']['optimizer_coverage']['all_parameters_once'],yaml_load_exact_states=len(src)-len(skip))
    write_json(dest/'validation_summary.json',summary)
    for source,name in [(Path(pipeline)/'pipeline_validation.json','evaluation_pipeline.json'),(Path(package)/'pack_validation.json','pack_validation.json')]:
        shutil.copyfile(source,dest/name)
    # Readable YAML formatting only; assert semantic identity after serialization.
    for variant,v in VARIANTS.items():
        cfg=expected_config(variant)
        text='# Decoupled Reference Triad: original C2 main graph + explicit side-branch routing.\n'
        text+='nc: 80\nscales:\n  l: [1.0, 1.0, 1024]\n'
        index=0
        for group in ('backbone','head'):
            text+='\n'+group+':\n'
            for row in cfg[group]:
                # YAML 1.1 needs a decimal point to recognize exponent notation as a float.
                text+='  - '+json.dumps(row,separators=(', ',': ')).replace('1e-06','1.0e-06')+f' # {index}\n';index+=1
        path=MODEL_DIR/v['yaml'];path.write_text(text,encoding='utf-8')
        parsed=YAML.load(path);require(all(parsed[k]==cfg[k] for k in ('backbone','head','scales')),'YAML formatting changed graph')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validation',required=True);p.add_argument('--pipeline',required=True);p.add_argument('--package',required=True)
    a=p.parse_args();torch.set_num_threads(4);export(a.validation,a.pipeline,a.package)
