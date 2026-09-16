"""Explicit fixed-protocol LSRT val/test and light evidence packaging. No implicit evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from init_lsrt_v1 import ROOT, require, sha256, write_json, runtime, verify_model, fingerprint
from train_lsrt_v1 import read_json, dataset_inventory, require_preflight
from c19_lif_v1_results import EVAL, POLICY, postprocess, verify_archive
import numpy as np
import torch
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import YAML


def require_training(plan_path, launch):
    plan = read_json(plan_path)
    state = read_json(launch/'training_state.json')
    require(state['status'] in {'COMPLETED_200_EPOCHS', 'EARLY_STOPPED_PATIENCE'}, 'Formal training incomplete')
    require(read_json(launch/'exit.json')['exit_code'] == 0, 'Training exit failed')
    provenance = read_json(launch/'launch.json')
    require(provenance['plan_sha256'] == sha256(plan_path), 'Training plan differs')
    require(plan['fingerprints'] == fingerprint(plan['variant'], plan['source'], plan['initialized']), 'Training source/config/init differs')
    require(dataset_inventory(plan['args']['data']) == plan['data_inventory'], 'Frozen data content changed')
    require(not subprocess.check_output(['git', '-c', 'safe.directory='+ROOT.as_posix(), 'status', '--porcelain', '--untracked-files=no'],
                                        cwd=ROOT, text=True).strip(), 'Commit evaluation source before evaluation')
    return plan


def evaluate(options):
    plan = require_training(options.plan, options.launch)
    variant, output, split = plan['variant'], options.output.resolve(), options.mode
    weights = Path(plan['args']['save_dir'])/'weights/best.pt'
    require(weights.is_file(), 'Real training-selected best.pt missing')
    require(not output.exists(), 'Existing evaluation preserved')
    checkpoint_hash = sha256(weights)
    settings = dict(EVAL, data=plan['args']['data'], device=options.device, split=split,
                    project=str(output), name='plots', exist_ok=False, plots=True,
                    save_json=False, save_txt=False)
    if split == 'test':
        require(options.val_report and options.val_report.is_file(), 'Test requires same selected best checkpoint completed val')
        val = read_json(options.val_report)
        require(val.get('status') == 'COMPLETED' and val.get('split') == 'val' and val.get('variant') == variant,
                'Missing matching completed val')
        require(val['checkpoint_sha256'] == checkpoint_hash and val['fingerprints'] == plan['fingerprints']
                and val['data_inventory'] == plan['data_inventory'] and val['policy'] == POLICY, 'Val/test identity changed')
        require(all(type(val['settings'][k]) is type(v) and val['settings'][k] == v for k,v in EVAL.items()), 'Val/test protocol changed')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='FAILED', split=split, variant=variant, checkpoint=str(weights), checkpoint_sha256=checkpoint_hash,
                  fingerprints=plan['fingerprints'], data_inventory=plan['data_inventory'], settings=settings,
                  policy=POLICY, runtime=runtime(), selection='native training best/EMA checkpoint; no test selection',
                  precision_recall_policy='each model own maximum-F1 working point',
                  boxes='original final decoder output; original CBR refined final boxes in main variant',
                  precision='original independent FP32, half=False', per_image_predictions_saved=False)
    try:
        model = RTDETR(str(weights))  # Native checkpoint loader selects EMA when present.
        verify_model(model.model, variant, zero=False)
        require(model.model.model[-1].nc == 1, 'Evaluation requires true nc=1 checkpoint')
        report['parameters_unfused'] = sum(p.numel() for p in model.model.parameters())

        class FixedValidator(RTDETRValidator):
            def init_metrics(self, backend):
                super().init_metrics(backend)
                require(not self.training and not self.args.half, 'Independent original FP32 evaluation required')
                require(all(type(getattr(self.args,k)) is type(v) and getattr(self.args,k) == v for k,v in EVAL.items()),
                        'Effective evaluation settings changed')
                report['actual_settings'] = vars(self.args).copy()
                report['expected_images'] = len(self.dataloader.dataset.im_files)
                self.sorted_mask_affected = 0

            def postprocess(self, predictions):
                selected, affected = postprocess(predictions, self.args.imgsz, self.args.conf)
                self.sorted_mask_affected += affected
                report['historical_sorted_mask_affected_images'] = self.sorted_mask_affected
                return selected

        metrics = model.val(validator=FixedValidator, **settings)
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1,10) and np.isfinite(ap).all(), 'Invalid AP values')
        expected = plan['data_inventory']['splits'][split]
        require(report['expected_images'] == expected['images'], 'Incomplete evaluation split')
        require(sha256(weights) == checkpoint_hash, 'Checkpoint changed during evaluation')
        report.update(status='COMPLETED', precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                      AP50=float(metrics.box.map50), AP75=float(ap[:,5].mean()), AP50_95=float(metrics.box.map),
                      ap_iou_thresholds=[round(.5+i*.05,2) for i in range(10)], ap_by_class=ap.tolist(),
                      speed_ms_per_image=metrics.speed, images=expected['images'], ground_truth=expected['boxes'])
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        write_json(output/'metrics.json', report)
        YAML.save(output/'evaluation_args.yaml', settings)
    return report


def package(options):
    """Explicit allowlist: no checkpoints, datasets, reference ZIPs or per-image prediction streams."""
    destination = options.output.resolve()
    require(not destination.exists() and not Path(str(destination)+'.sha256').exists(), 'Existing package preserved')
    plan = read_json(options.plan)
    require(plan['fingerprints'] == fingerprint(plan['variant'], plan['source'], plan['initialized']),
            'Packaging source/config/source-weight/initialization differs from plan')
    preflight = read_json(options.preflight/'report.json')
    require(preflight.get('variant') == plan['variant'] and preflight.get('fingerprints') == plan['fingerprints'],
            'Preflight package evidence belongs to another model/source')
    initialization = read_json(options.initialization_report)
    require(initialization.get('variant') == plan['variant']
            and initialization.get('source_sha256') == plan['fingerprints']['source_sha256']
            and initialization.get('output_sha256') == plan['fingerprints']['initialization_sha256'],
            'Initialization package evidence differs from plan')
    if options.launch and (options.launch/'launch.json').is_file():
        provenance = read_json(options.launch/'launch.json')
        require(provenance.get('plan_sha256') == sha256(options.plan)
                and provenance.get('fingerprints') == plan['fingerprints']
                and provenance.get('variant') == plan['variant'], 'Launch package evidence differs from plan')
    elif options.launch:
        require(not (options.launch/'training_state.json').exists(), 'Unidentified training state cannot enter this package')
    best = Path(plan['args']['save_dir'])/'weights/best.pt'
    evaluation_reports = {}
    for split in ('val','test'):
        folder = getattr(options,split)
        if folder and (folder/'metrics.json').is_file():
            report = read_json(folder/'metrics.json')
            require(report.get('variant') == plan['variant'] and report.get('split') == split
                    and report.get('fingerprints') == plan['fingerprints']
                    and report.get('data_inventory') == plan['data_inventory']
                    and report.get('policy') == POLICY, 'Evaluation package identity differs: '+split)
            require(best.is_file() and report.get('checkpoint_sha256') == sha256(best),
                    'Evaluation package checkpoint differs: '+split)
            require(all(type(report['settings'].get(k)) is type(v) and report['settings'][k] == v for k,v in EVAL.items()),
                    'Evaluation package protocol differs: '+split)
            evaluation_reports[split] = report
    files = {}
    allowed = {'.json', '.yaml', '.yml', '.csv', '.txt', '.md', '.png', '.jpg', '.py', '.sh', '.patch'}

    def collect(folder, prefix, source=False):
        folder = Path(folder)
        if not folder.is_dir():
            return
        for path in folder.rglob('*'):
            if not path.is_file() or path.is_symlink() or path.suffix.lower() not in allowed:
                continue
            rel = path.relative_to(folder)
            if any(part in {'weights','datasets','__pycache__','.git'} for part in rel.parts):
                continue
            if path.name.startswith(('predictions', 'train_batch', 'val_batch')) or path.stat().st_size > 16*1024*1024:
                continue
            if source and path.suffix.lower() not in {'.py','.yaml','.yml','.sh','.md','.json'}:
                continue
            files[prefix+'/'+rel.as_posix()] = path

    collect(options.plan.parent, 'audit/plan')
    collect(options.preflight, 'audit/preflight')
    require(options.initialization_report.is_file(), 'Initialization audit missing')
    files['audit/initialization.json'] = options.initialization_report
    if options.launch:
        collect(options.launch, 'audit/launch')
    collect(Path(plan['args']['save_dir']), 'train')
    for name in ('val','test'):
        folder = getattr(options, name)
        if folder:
            collect(folder, name)
    collect(ROOT/'docs/lsrt_v1', 'source/docs/lsrt_v1', source=True)
    collect(ROOT/'ultralytics-main/ultralytics', 'source/ultralytics-main/ultralytics', source=True)
    # Include the bounded tools source tree so probe/fusion helpers imported transitively are retained.
    collect(ROOT/'tools', 'source/tools', source=True)
    for name in ('init_lsrt_v1.py','preflight_lsrt_v1.py','check_lsrt_module.py','train_lsrt_v1.py','lsrt_v1_results.py','sync_lsrt_v1.sh',
                 'init_c19_lif_v1.py','init_lif_down.py','train_c19_lif_v1.py','c19_lif_v1_results.py',
                 'c19_lif_v1_data.py','c19_lif_v1_diagnostic.py','c19_lif_v1_cutoff.py','lif_down_topology.py',
                 'check_c19_lif_v1.py','preflight_c19_lif_v1.py','lif_down_fusion.py'):
        path = ROOT/'tools'/name
        if path.is_file():
            files['source/tools/'+name] = path
    for name in ('c2_args.yaml','c2_data.yaml','checks.json'):
        files['source/docs/c19_lif_v1/'+name] = ROOT/'docs/c19_lif_v1'/name
    state = read_json(options.launch/'training_state.json') if options.launch and (options.launch/'training_state.json').is_file() else {'status':'NOT_STARTED'}
    evaluations = {split: evaluation_reports.get(split, {}).get('status', 'NOT_RUN') for split in ('val','test')}
    complete = state['status'] in {'COMPLETED_200_EPOCHS','EARLY_STOPPED_PATIENCE'} and all(v=='COMPLETED' for v in evaluations.values())
    if complete:
        require_training(options.plan, options.launch)
        require_preflight(preflight, plan)
        require(initialization.get('status') == 'PASSED', 'Completed evidence has failed initialization audit')
        require(all((Path(plan['args']['save_dir'])/name).is_file() for name in ('args.yaml','results.csv','results.png')),
                'Completed training evidence lacks args/CSV/curve')
        for split in ('val','test'):
            names = {p.name for p in getattr(options,split).rglob('*.png')}
            require('confusion_matrix.png' in names and 'confusion_matrix_normalized.png' in names,
                    'Completed evaluation lacks confusion matrices: '+split)
            require(any(name.endswith('PR_curve.png') for name in names), 'Completed evaluation lacks PR curve: '+split)
    metadata = dict(created=datetime.now(timezone.utc).isoformat(), variant=plan['variant'], runtime=runtime(),
                    training=state, evaluations=evaluations, source_fingerprints=plan['fingerprints'],
                    exclusions='weights, datasets, reference archives, per-image predictions, individual training batches',
                    evidence_complete=complete)
    extra = {'PACKAGE.json': (json.dumps(metadata,ensure_ascii=False,indent=2)+'\n').encode()}
    manifest = [dict(path=name, bytes=path.stat().st_size, sha256=sha256(path)) for name,path in sorted(files.items())]
    manifest += [dict(path=name,bytes=len(value),sha256=hashlib.sha256(value).hexdigest()) for name,value in extra.items()]
    extra['MANIFEST.json'] = (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream, tarfile.open(fileobj=stream,mode='w|gz') as archive:
        for name,path in sorted(files.items()):
            archive.add(path,arcname=name,recursive=False)
        for name,value in extra.items():
            entry=tarfile.TarInfo(name); entry.size=len(value)
            archive.addfile(entry,io.BytesIO(value))
    verify_archive(destination)
    with Path(str(destination)+'.sha256').open('x',encoding='utf-8') as stream:
        stream.write(sha256(destination)+'  '+destination.name+'\n')
    print(json.dumps(dict(package=str(destination),sha256=sha256(destination),members=len(manifest)+1,
                          training=state['status'],evaluations=evaluations),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='mode',required=True)
    for mode in ('val','test'):
        p=commands.add_parser(mode)
        for name in ('plan','launch','output'):
            p.add_argument('--'+name,type=Path,required=True)
        p.add_argument('--device',default='0')
        p.add_argument('--val-report',type=Path)
    p=commands.add_parser('pack')
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',type=Path,required=True,help='Preflight report directory; excludes diagnostic checkpoints')
    p.add_argument('--initialization-report',type=Path,required=True)
    for name in ('launch','val','test'):
        p.add_argument('--'+name,type=Path)
    options=parser.parse_args(); torch.set_num_threads(4)
    package(options) if options.mode=='pack' else evaluate(options)


if __name__=='__main__':
    main()
