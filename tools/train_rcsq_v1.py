"""RCS-Q plan/start/resume lifecycle. A plan never starts training; start reruns live server gates."""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import traceback

import torch
from init_rcsq_v1 import (ROOT, MODEL_DIR, VARIANTS, SOURCE_SHA256, build_training_model, code_fingerprint,
                          controlled_models, git, initialized_path, is_added, require, runtime, sha256,
                          tensor_digest, verify_model, write_json)
from ultralytics import RTDETR
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from c19_lif_v1_data import dataset_inventory
from train_c19_lif_v1 import ensure_amp_resources, process_token  # generic staging/PID checks, no training dispatch

DEFAULT_VARIANT = 'cbr_lif_rcsq_v1'


def recipe(authority, variant, init, data, project):
    """Inherit every field/type; only model, output identity and explicit data path may vary."""
    source = YAML.load(authority)
    reference = YAML.load(ROOT / 'docs/c19_lif_v1/c2_args.yaml')
    identities = {'model', 'name', 'save_dir', 'project', 'data'}
    require(set(source) == set(reference), 'Authoritative full recipe fields differ from verified C2 archive')
    require(all(type(source[k]) is type(reference[k]) and source[k] == reference[k]
                for k in source if k not in identities), 'Authoritative recipe differs outside allowed identity/path fields')
    target = dict(source)
    target.update(model=str(Path(init).resolve()), data=str(Path(data).resolve()), project=str(Path(project).resolve()),
                  name=VARIANTS[variant][2], save_dir=str(Path(project).resolve() / VARIANTS[variant][2]))
    require(target['exist_ok'] is False and target['resume'] is False, 'Fresh formal run must not overwrite/resume')
    rows = [dict(field=k, authoritative=source[k], planned=target[k], changed=source[k] != target[k],
                 reason=('experiment identity/path mapping' if k in identities and source[k] != target[k] else 'inherited'))
            for k in sorted(source)]
    require({row['field'] for row in rows if row['changed']} <= identities, 'Unexpected formal recipe difference')
    return target, rows


def verify_data(path):
    actual = YAML.load(path)
    archived = YAML.load(ROOT / 'docs/c19_lif_v1/c2_data.yaml')
    require({k: v for k, v in actual.items() if k != 'path'} == {k: v for k, v in archived.items() if k != 'path'},
            'Dataset splits/classes changed')
    resolved = check_det_dataset(str(path), autodownload=False)
    require(resolved['nc'] == 1 and resolved['names'] == {0: 'crack'}, 'Crack nc/classes changed')
    inventory = dataset_inventory(Path(resolved['path']))
    expected = json.loads((ROOT / 'docs/c19_lif_v1/summary.json').read_text(encoding='utf-8'))['dataset']
    require(inventory == expected, 'Train/val/test path or label fingerprints differ from verified original experiment')
    return resolved, inventory


def verify_data_config(path):
    resolved, inventory = verify_data(path)
    return dict(status='PASSED', config=YAML.load(path), resolved=resolved, inventory=inventory)


def verify_init(path, source, variant, expected_code=None):
    require(sha256(source) == SOURCE_SHA256, 'Fixed source hash changed')
    ckpt = torch_load(path, map_location='cpu')
    provenance = ckpt.get('rcsq_v1_provenance', {})
    require(ckpt.get('epoch') == -1 and ckpt.get('optimizer') is None and ckpt.get('ema') is None and
            provenance.get('formal_training_updates') == 0 and provenance.get('source_sha256') == SOURCE_SHA256 and
            provenance.get('variant') == variant, 'Initialization contains trained or foreign state')
    require(provenance.get('code_fingerprint') == (expected_code or code_fingerprint()['sha256']), 'Initialization source code changed; regenerate in a fresh init directory')
    actual = ckpt['model'].float()
    verify_model(actual, variant, zero=True)
    require(provenance.get('state_sha256') == {k: tensor_digest(v) for k, v in actual.state_dict().items()}, 'Initialization state differs from its recorded hashes')
    return actual


def plan(source, init_dir, authority, data, project, output, variant=DEFAULT_VARIANT, main=None):
    output = Path(output).resolve()
    require(not output.exists(), 'Existing plan directory preserved: ' + str(output))
    init = initialized_path(init_dir, variant).resolve()
    verify_init(init, source, variant)
    args, diff = recipe(authority, variant, init, data, project)
    require(not Path(args['save_dir']).exists(), 'Existing formal run preserved: ' + args['save_dir'])
    _, inventory = verify_data(data)
    output.mkdir(parents=True, exist_ok=False)
    YAML.save(output / 'train_args.yaml', args)
    shutil.copyfile(authority, output / 'authoritative_args.yaml')
    shutil.copyfile(data, output / 'data.yaml')
    write_json(output / 'parameter_diff.json', diff)
    write_json(output / 'dataset_inventory.json', inventory)
    main = Path(main or os.environ.get('RCSQ_V1_MAIN', '/root/autodl-tmp/projects/Crack_RTDETR')).resolve()
    result = dict(schema='rcsq_v1_plan', variant=variant, args=args, output=str(output), main=str(main), source=str(Path(source).resolve()),
                  source_sha256=sha256(source), init_dir=str(Path(init_dir).resolve()), init=str(init), init_sha256=sha256(init),
                  authority=str(Path(authority).resolve()), authority_sha256=sha256(authority),
                  data_sha256=sha256(data), data_inventory=inventory, code=code_fingerprint(), runtime=runtime(),
                  formal_training='NOT_STARTED', final_test='NOT_RUN', created=datetime.now(timezone.utc).isoformat())
    write_json(output / 'plan.json', result)
    return result


def load_plan(path):
    p = json.loads(Path(path).read_text(encoding='utf-8'))
    require(p.get('schema') == 'rcsq_v1_plan' and p.get('variant') in VARIANTS, 'Wrong plan')
    require(p['code']['sha256'] == code_fingerprint()['sha256'], 'Code differs from plan')
    require(p['runtime']['commit'] == runtime()['commit'], 'Commit differs from plan; regenerate plan after delivery')
    require(not git('status', '--porcelain', '--untracked-files=no'), 'Tracked changes must be reviewed/committed before formal training')
    require(sha256(p['source']) == p['source_sha256'] == SOURCE_SHA256, 'Source differs from plan')
    require(sha256(p['init']) == p['init_sha256'] and sha256(p['authority']) == p['authority_sha256'] and
            sha256(p['args']['data']) == p['data_sha256'], 'Initialization/recipe/data changed after plan')
    actual_args, _ = recipe(p['authority'], p['variant'], p['init'], p['args']['data'], p['args']['project'])
    require(actual_args == p['args'], 'Plan recipe was edited')
    _, inventory = verify_data(p['args']['data'])
    require(inventory == p['data_inventory'], 'Dataset differs from plan')
    return p


def optimizer_audit(model, optimizer):
    require(type(optimizer) is torch.optim.AdamW, 'Formal optimizer must remain AdamW')
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    added = {name: ids.count(id(p)) for name, p in model.named_parameters() if is_added(name) and p.requires_grad}
    require(added and all(count == 1 for count in added.values()), 'RCS-Q optimizer coverage must be exactly once')
    require(all(ids.count(id(p)) == 1 for p in model.parameters() if p.requires_grad), 'Common optimizer parameters missing/duplicated')
    return dict(optimizer='AdamW', rcsq_occurrences=added,
                groups=[dict(index=i, count=len(g['params']), lr=g['lr'], weight_decay=g.get('weight_decay')) for i, g in enumerate(optimizer.param_groups)])


def disable_oom_retry(trainer):
    trainer._oom_retries = 3  # native retry branch may otherwise silently halve the prescribed batch


def recover_failed_preflight(p, plan_path):
    """Preserve a failed gate and its dead reservation before an explicit fresh retry."""
    folder = Path(p['output']).resolve()
    launch = folder / 'launch'
    run = Path(p['args']['save_dir']).resolve()
    lock = run.with_name(run.name + '.rcsq_v1.lock')
    require(launch.is_dir() and not launch.is_symlink() and lock.is_dir() and not lock.is_symlink(), 'Missing/unsafe prior reservation')
    require(not run.exists() and not (launch / 'training_result.json').exists(), 'A training run exists; use resume, never retry-preflight')
    state = json.loads((launch / 'state.json').read_text(encoding='utf-8'))
    owner = json.loads((lock / 'owner.json').read_text(encoding='utf-8'))
    require(state.get('status') == 'PREFLIGHT_FAILED' and state.get('formal_training') == 'NOT_STARTED', 'Only failed, unstarted preflight can be retried')
    require(Path(owner['plan']).resolve() == Path(plan_path).resolve() and Path(owner['worktree']).resolve() == ROOT,
            'Reservation belongs to another plan/worktree')
    token = process_token(owner['pid'])
    require(token is None or token != owner.get('process_token'), 'Previous preflight owner is still alive')
    archived = folder / ('failed_preflight_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    # Both resolved sources are exact task paths; preserve whole evidence directories, never delete them.
    require(launch.resolve().parent == folder and lock.resolve().parent == run.parent and not archived.exists(), 'Recovery path changed')
    launch.rename(archived)
    lock.rename(archived / 'reservation')
    return archived


def start(plan_path, preflight_path, retry_preflight=False):
    p = load_plan(plan_path)
    require(torch.cuda.is_available() and os.environ.get('CONDA_DEFAULT_ENV') == 'rtdetr', 'Activate existing rtdetr CUDA environment')
    require(platform.python_version() == '3.10.13' and str(torch.__version__) == '2.1.2+cu121',
            'Expected successful server Python3.10.13 / torch2.1.2+cu121; record mismatch without upgrades')
    require(not Path(p['args']['save_dir']).exists(), 'Existing run preserved')
    # The old report is retained only as context; its JSON status is never an authorization token.
    require(Path(preflight_path).is_file(), 'Supply the server preflight report for audit context')
    prior_report = Path(preflight_path).read_bytes()
    launch = Path(p['output']) / 'launch'
    if retry_preflight:
        recover_failed_preflight(p, plan_path)
    require(not launch.exists(), 'Existing launch preserved: use --retry-preflight after a failed gate, or resume for a real checkpoint')
    run = Path(p['args']['save_dir'])
    lock = run.with_name(run.name + '.rcsq_v1.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.mkdir(exist_ok=False)
    write_json(lock / 'owner.json', dict(pid=os.getpid(), process_token=process_token(os.getpid()),
                                       plan=str(Path(plan_path).resolve()), worktree=str(ROOT), variant=p['variant']))
    launch.mkdir(exist_ok=False)
    (launch / 'prior_preflight.json').write_bytes(prior_report)
    write_json(launch / 'state.json', dict(status='LIVE_PREFLIGHT', pid=os.getpid(), formal_training='NOT_STARTED', final_test='NOT_RUN'))
    try:
        verify_init(p['init'], p['source'], p['variant'], p['code']['sha256'])
        # Reconstruct from the fixed source; an edited provenance document cannot bless altered tensors.
        models, _ = controlled_models(p['source'])
        loaded = RTDETR(p['init']).model
        require(all(torch.equal(v, loaded.state_dict()[k]) for k, v in models[p['variant']].state_dict().items()), 'Initialization is not the controlled fixed-source state')
        del models, loaded
        from preflight_rcsq_v1 import run_checks
        before = (code_fingerprint()['sha256'], sha256(p['source']), sha256(p['init']), sha256(p['args']['data']))
        fresh = run_checks(p['source'], p['init_dir'], launch / 'live_preflight', variant=p['variant'],
                           device='cuda', data=p['args']['data'], capacity=True, main=p['main'])
        require(fresh.get('status') == 'PASSED', 'Live server CUDA/AMP/B16-640 preflight did not pass; training not started')
        require(before == (code_fingerprint()['sha256'], sha256(p['source']), sha256(p['init']), sha256(p['args']['data'])), 'Inputs changed during live gate')
        verify_data(p['args']['data'])
        write_json(launch / 'gate.json', dict(status='PASSED', identities=before, fresh_run=True,
                                             report=fresh, completed=datetime.now(timezone.utc).isoformat()))
        run_training(p, launch)
    except BaseException as error:
        if not (launch / 'training_result.json').exists():
            write_json(launch / 'state.json', dict(status='PREFLIGHT_FAILED', error=repr(error), formal_training='NOT_STARTED', final_test='NOT_RUN'))
        raise


def resume(plan_path, checkpoint):
    p = load_plan(plan_path)
    require(torch.cuda.is_available() and os.environ.get('CONDA_DEFAULT_ENV') == 'rtdetr' and
            platform.python_version() == '3.10.13' and str(torch.__version__) == '2.1.2+cu121', 'Resume requires unchanged server environment')
    launch = Path(p['output']) / 'launch'
    with active_run_guard(p, launch, resumed=True):
        return _resume_checked(p, checkpoint)


def _resume_checked(p, checkpoint):
    launch = Path(p['output']) / 'launch'
    gate = json.loads((launch / 'gate.json').read_text(encoding='utf-8'))
    require(gate.get('status') == 'PASSED' and gate.get('fresh_run') is True and
            gate['identities'] == [p['code']['sha256'], p['source_sha256'], p['init_sha256'], p['data_sha256']], 'Original live gate identities changed')
    checkpoint = Path(checkpoint).resolve()
    require(checkpoint == (Path(p['args']['save_dir']) / 'weights/last.pt').resolve(), 'Resume only this run last.pt')
    ckpt = torch_load(checkpoint, map_location='cpu')
    require(0 <= ckpt.get('epoch', -1) < p['args']['epochs'] - 1 and ckpt.get('optimizer') is not None, 'Checkpoint is completed/stripped or cannot resume')
    saved_args = ckpt['train_args']
    require(all(saved_args.get(k) == v for k, v in p['args'].items() if k not in {'model', 'resume'}), 'Resume recipe differs')
    restored = RTDETR(str(checkpoint)).model
    verify_model(restored, p['variant'], zero=False)
    # No zero-initialization checks or reinitialization occur on the learned checkpoint.
    resume_dir = launch / ('resume_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    resume_dir.mkdir(exist_ok=False)
    write_json(resume_dir / 'checkpoint.json', dict(path=str(checkpoint), sha256=sha256(checkpoint), epoch=ckpt['epoch'],
               rcsq_sha256={k: tensor_digest(v) for k, v in restored.state_dict().items() if is_added(k)}))
    _run_training(p, resume_dir, checkpoint=checkpoint)


@contextmanager
def active_run_guard(p, launch, resumed=False):
    """Hold a kernel lock across training/resume; a second process cannot share the run."""
    import fcntl  # formal dispatch is on the existing Linux server
    run = Path(p['args']['save_dir'])
    lock = run.with_name(run.name + '.rcsq_v1.active.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another active RCS-Q training/resume process owns this run') from error
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps(dict(pid=os.getpid(), launch=str(launch), resume=resumed)))
        stream.flush()
        yield


def run_training(p, launch, checkpoint=None):
    with active_run_guard(p, launch, resumed=checkpoint is not None):
        return _run_training(p, launch, checkpoint)


def _run_training(p, launch, checkpoint=None):
    resumed = checkpoint is not None
    variant = p['variant']
    state = dict(status='STARTING', formal_training='STARTING', final_test='NOT_RUN', resume=resumed)
    write_json(launch / 'state.json', state)
    model = RTDETR(str(checkpoint) if resumed else p['init'])
    source_state = {k: v.detach().clone() for k, v in model.model.state_dict().items() if is_added(k)}

    class RecordingTrainer(RTDETRTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            result, audit = build_training_model(cfg, weights, self.data, variant, zero=not resumed)
            write_json(launch / 'nc1_loading.json', audit)
            return result

    def setup(trainer):
        verify_model(trainer.model, variant, zero=not resumed)
        require(bool(trainer.amp), 'Native AMP check disabled AMP; prescribed recipe cannot change')
        actual = vars(trainer.args)
        ignored = {'model', 'resume'} if resumed else set()
        diff = {k: [v, actual.get(k)] for k, v in p['args'].items() if k not in ignored and
                (type(actual.get(k)) is not type(v) or actual.get(k) != v)}
        require(not diff, 'Native Trainer changed formal recipe: ' + str(diff))
        require(all(torch.equal(v.cpu(), trainer.model.state_dict()[k].cpu()) for k, v in source_state.items()),
                'RCS-Q parameters changed before first optimizer update')
        YAML.save(launch / 'actual_train_args.yaml', actual)
        write_json(launch / 'optimizer.json', optimizer_audit(trainer.model, trainer.optimizer))
        write_json(launch / 'state.json', dict(status='RUNNING', formal_training='RUNNING', final_test='NOT_RUN', resume=resumed))

    model.add_callback('on_train_start', setup)
    model.add_callback('on_train_batch_start', disable_oom_retry)
    result = dict(status='FAILED', formal_training='FAILED', final_test='NOT_RUN', resume=resumed,
                  started=datetime.now(timezone.utc).isoformat(), code=p['code']['sha256'])
    try:
        args = dict(p['args'])
        if resumed:
            args.update(model=str(checkpoint), resume=str(checkpoint))
        model.train(trainer=RecordingTrainer, **args)
        trainer = model.trainer
        epochs = int(trainer.epoch) + 1
        require(trainer.best.is_file() and trainer.last.is_file() and trainer.csv.is_file(), 'Training exited without expected artifacts')
        status = 'COMPLETED_200_EPOCHS' if epochs == p['args']['epochs'] else (
            'PATIENCE_EARLY_STOP' if epochs < p['args']['epochs'] and trainer.stopper.possible_stop else 'STOPPED_OTHER')
        result.update(status=status, formal_training=status, epochs_completed=epochs,
                      best_sha256=sha256(trainer.best), last_sha256=sha256(trainer.last))
    except KeyboardInterrupt:
        result.update(status='INTERRUPTED', formal_training='INTERRUPTED')
        raise
    except BaseException as error:
        result.update(status='OOM' if 'out of memory' in str(error).lower() else 'FAILED', error=repr(error), traceback=traceback.format_exc())
        result['formal_training'] = result['status']
        raise
    finally:
        result['finished'] = datetime.now(timezone.utc).isoformat()
        write_json(launch / 'training_result.json', result)
        write_json(launch / 'state.json', result)
        if resumed:
            write_json(Path(p['output']) / 'launch/latest_training_result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    create = sub.add_parser('plan', help='Write the complete recipe and fingerprints; no training')
    create.add_argument('--variant', choices=list(VARIANTS), default=DEFAULT_VARIANT)
    for name in ('source', 'init-dir', 'c2-args', 'data', 'project', 'output'):
        create.add_argument('--' + name, type=Path, required=True)
    create.add_argument('--main', type=Path, default=Path(os.environ.get('RCSQ_V1_MAIN', '/root/autodl-tmp/projects/Crack_RTDETR')))
    launch = sub.add_parser('start', help='Rerun live server gates, then explicitly start formal training')
    launch.add_argument('--plan', type=Path, required=True)
    launch.add_argument('--preflight', type=Path, required=True)
    launch.add_argument('--retry-preflight', action='store_true', help='Preserve a previous failed/unstarted gate, then rerun it from scratch')
    again = sub.add_parser('resume', help='Resume this run last.pt without initializing learned RCS-Q again')
    again.add_argument('--plan', type=Path, required=True)
    again.add_argument('--checkpoint', type=Path, required=True)
    resources = sub.add_parser('amp-resources', help='Stage existing native AMP-check assets')
    resources.add_argument('--main', type=Path, required=True)
    resources.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == 'plan':
        result = plan(args.source, args.init_dir, args.c2_args, args.data, args.project, args.output, args.variant, args.main)
        print(json.dumps(dict(plan=str(Path(args.output) / 'plan.json'), variant=result['variant'], formal_training='NOT_STARTED'), indent=2))
    elif args.mode == 'start':
        start(args.plan, args.preflight, args.retry_preflight)
    elif args.mode == 'resume':
        resume(args.plan, args.checkpoint)
    else:
        ensure_amp_resources(args.main, args.report)


if __name__ == '__main__':
    main()
