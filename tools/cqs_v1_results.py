"""Explicit val/test, bounded train opportunity probe, and verifiable LIGHT packaging."""
from __future__ import annotations

from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from cqs_v1_common import *
from cqs_v1_training import CQSValidator, selection_summary
# The actual corrected sorting/masking implementation is reused, not just its name.
from c19_lif_v1_results import EVAL, POLICY, postprocess, image_record, verify_archive


def evaluate(split):
    from cqs_v1 import active_processes
    require(split in ('val','test'),'Explicit val/test only')
    verify_delivery()
    require(not active_processes(),'Formal run is active')
    state=read_json(OUT/'state.json')
    require(state['status']=='COMPLETED','Formal training must complete before independent evaluation')
    prepared=read_json(OUT/'prepare.json')
    require(fingerprint(prepared)['sha256']==prepared['fingerprint']['sha256'],'Source/environment/data changed since prepare')
    weights=RUN/'weights/best.pt'
    require(weights.is_file(),'This run best.pt is missing')
    data=Path(prepared['data'])
    identity=dict(weights_sha256=sha256(weights),data=inventory(data),source=source_hashes(),config=CQS_CONFIG,settings=EVAL)
    ident=digest_json(identity)
    lock_path=OUT/'selection_lock.json'
    if split=='test':
        require(lock_path.is_file(),'Test requires completed val and a locked checkpoint')
        require(read_json(lock_path)['identity']==ident,'Test weight/data/code differs from val lock')
    if lock_path.exists():
        require(read_json(lock_path)['identity']==ident,'A different best is already locked; do not reselect using test')
    folder=OUT/f'evaluation_{split}'
    if folder.exists():
        prior=read_json(folder/'metrics.json')
        require(prior.get('status')=='completed' and prior['identity']==ident,'Existing incomplete/different evaluation preserved')
        print(json.dumps(prior,ensure_ascii=False,indent=2))
        return prior
    folder.mkdir(parents=True,exist_ok=False)
    settings=dict(EVAL,data=str(data),split=split,device='0',plots=True,save_json=False,save_txt=False,
                  project=str(folder),name='plots',exist_ok=False)
    report=dict(status='failed',split=split,identity=ident,identity_details=identity,checkpoint=str(weights),
                checkpoint_sha256=sha256(weights),policy=POLICY,settings=settings,runtime=runtime(),
                evidence_scope='full_split',boxes='final original CBR refinement',
                precision_recall_policy='each model own maximum-F1 working point',CQS='enabled beta=0.5')
    counts=dict(images=0,ground_truth=0,predictions=0,metric_predictions=0,historical_mask_affected_images=0)
    seen=set()
    try:
        wrapper=RTDETR(str(weights));old=wrapper.model.float()
        verify_model(old)
        rebuilt=build()
        rebuilt.load_state_dict(old.state_dict(),strict=True)
        for attr in ('args','names','nc','task','pt_path','stride'):
            if hasattr(old,attr):
                setattr(rebuilt,attr,deepcopy(getattr(old,attr)))
        wrapper.model=rebuilt.eval()
        stream_path=folder/'predictions_gt.jsonl.gz'
        with gzip.open(stream_path,'xt',encoding='utf-8') as stream:
            class StreamingValidator(CQSValidator):
                def init_metrics(self, model):
                    super().init_metrics(model)
                    require(not self.training and not self.args.half,'Independent FP32 only')
                    require(all(type(getattr(self.args,k)) is type(v) and getattr(self.args,k)==v for k,v in EVAL.items()),'Actual eval settings changed')
                    report['actual_settings']=vars(self.args).copy()
                    files=[Path(p).resolve().relative_to(Path(self.data['path']).resolve()).as_posix() for p in self.dataloader.dataset.im_files]
                    require(files and len(files)==len(set(files)),'Empty/duplicate evaluation split')
                    self.expected_images=set(files)
                def postprocess(self,preds):
                    chosen,affected=postprocess(preds,self.args.imgsz,self.args.conf)
                    full,_=postprocess(preds,self.args.imgsz,-float('inf'))
                    counts['historical_mask_affected_images']+=affected
                    for item,all_queries in zip(chosen,full):
                        require(len(all_queries['conf'])==300,'Ordinary output count changed')
                        item['_all_queries']=all_queries
                    return chosen
                def update_metrics(self,preds,batch):
                    for i,pred in enumerate(preds):
                        row=image_record(pred.pop('_all_queries'),self._prepare_batch(i,batch),self.data['path'],self.args.conf)
                        require(row['image'] not in seen,'Duplicate evaluated image')
                        seen.add(row['image']);counts['images']+=1
                        counts['ground_truth']+=len(row['ground_truth']);counts['predictions']+=len(row['predictions'])
                        counts['metric_predictions']+=len(pred['conf'])
                        stream.write(json.dumps(row,allow_nan=False)+'\n')
                    return super().update_metrics(preds,batch)
                def finalize_metrics(self):
                    require(seen==self.expected_images and counts['ground_truth']>0,'Empty GT/incomplete split')
                    result=super().finalize_metrics()
                    report['cqs_calls']=self.cqs_evidence
                    return result
            metrics=wrapper.val(validator=StreamingValidator,**settings)
        expected=identity['data']['splits'][split]
        require(counts['images']==expected['images'] and counts['ground_truth']==expected['instances'],'Full split count mismatch')
        ap=np.asarray(metrics.box.all_ap)
        require(ap.shape==(1,10) and np.isfinite(ap).all(),'Invalid AP')
        require(sha256(weights)==identity['weights_sha256'],'Weight changed during eval')
        report.update(status='completed',mAP50_95=float(metrics.box.map),AP50=float(metrics.box.map50),AP75=float(ap[:,5].mean()),
                      precision=float(metrics.box.mp),recall=float(metrics.box.mr),ap_by_class=ap.tolist(),
                      predictions_gt_sha256=sha256(stream_path),**counts)
        if split=='val':
            write_json(lock_path,dict(identity=ident,weights=str(weights),weights_sha256=sha256(weights),val_report=str(folder/'metrics.json')))
    except BaseException as error:
        import traceback
        report.update(error=repr(error),traceback=traceback.format_exc(),**counts)
        raise
    finally:
        write_json(folder/'metrics.json',report)
    return report


def probe(parent_best=None, data=None, count=64, device=None):
    require(1<=count<=64,'Probe is limited to 64 train images')
    parent_best=Path(parent_best) if parent_best else MAIN/'runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt'
    data=Path(data) if data else MAIN/'configs/crack_autodl.yaml'
    report=dict(status='PENDING',scope='read-only train candidate coverage, not Recall/AP',parent_best=str(parent_best),
                formal_training='NOT_RUN',test='NOT_RUN',config=CQS_CONFIG)
    if not parent_best.is_file() or not data.is_file():
        report['reason']='Missing trained mother best or data; random initialization is not substituted'
        write_json(OUT/'probe.json',report);return report
    import cv2
    from ultralytics.utils.ops import xywh2xyxy
    from ultralytics.utils.metrics import box_iou
    parent=RTDETR(str(parent_best)).model.float()
    require(type(parent.model[-1]) is RTDETRDecoderCBR,'Opportunity source must be original CBR+LIF mother')
    from init_c19_lif_v1 import verify_model as verify_parent
    verify_parent(parent)
    model=build();model.load_state_dict(parent.state_dict(),strict=True);del parent
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device).eval()
    cfg=YAML.load(data);base=Path(cfg['path']).resolve();train=(base/cfg['train']).resolve()
    require(train==(base/'images/train').resolve(),'Probe only accepts train split')
    files=sorted(p for p in train.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'})[:count]
    require(files,'Empty probe train split')
    records=[];best={k:[] for k in ('native300','cqs300','pool600')};extra=[]
    for p in files:
        image=cv2.imread(str(p));require(image is not None,f'Cannot read {p}')
        label=base/'labels/train'/p.relative_to(train).with_suffix('.txt')
        gt=[list(map(float,l.split()))[1:] for l in label.read_text().splitlines() if l.strip()]
        require(gt,'Probe image has no GT')
        x=torch.from_numpy(cv2.resize(image,(640,640),interpolation=cv2.INTER_LINEAR)[:,:,::-1].copy()).permute(2,0,1)[None].float().to(device)/255
        with torch.no_grad(),model.model[-1].capture_selection(full=True) as rows:
            model(x)
        row=rows[0];boxes=row['boxes'][0];selected=row['selected'][0]
        iou=box_iou(xywh2xyxy(boxes),xywh2xyxy(torch.tensor(gt,device=device)))
        for name,values in (('native300',iou[:300]),('cqs300',iou[selected]),('pool600',iou)):
            best[name].extend(values.max(0).values.cpu().tolist())
        extra.append(int(row['extra_count'][0]))
        records.append(dict(image=p.relative_to(base).as_posix(),image_sha256=sha256(p),label_sha256=sha256(label),
                            gt=len(gt),selection=selection_summary(row)))
    arrays={k:np.asarray(v) for k,v in best.items()};coverage={}
    for threshold in (.3,.5,.7):
        old=arrays['native300']>=threshold
        coverage[str(threshold)]={k:dict(candidate_coverage=float((v>=threshold).mean()),
            newly_covered=int(((v>=threshold)&~old).sum()),lost_coverage=int(((v<threshold)&old).sum())) for k,v in arrays.items()}
    supported=coverage['0.5']['pool600']['newly_covered']>0 and coverage['0.5']['cqs300']['newly_covered']>=coverage['0.5']['cqs300']['lost_coverage']
    report.update(status='COMPLETED',runtime=runtime(),parent_best_sha256=sha256(parent_best),images=len(files),ground_truth=len(arrays['native300']),
                  preprocessing='fixed sorted train list; RGB stretch640 cv2.INTER_LINEAR /255; no random augmentation',
                  coverage=coverage,mean_best_IoU={k:float(v.mean()) for k,v in arrays.items()},extra_count_mean=float(np.mean(extra)),
                  interpretation='Opportunity diagnostic supports further experiment, not an AP claim' if supported else
                  'This opportunity diagnostic does not support the mechanism; fixed 240/60, pool600 and beta0.5 unchanged',
                  files=records)
    write_json(OUT/'probe.json',report)
    return report


def package(destination=None):
    from cqs_v1 import active_processes
    require(not active_processes(),'Package only after this experiment worker is inactive')
    destination=Path(destination) if destination else MAIN/'downloads/cqs_v1'/f'cqs_v1_LIGHT_{utc()}.tar.gz'
    require(not destination.exists(),'Existing LIGHT archive preserved')
    destination.parent.mkdir(parents=True,exist_ok=True)
    entries={}
    def add(path,arc):
        require(not path.is_symlink(),'No symbolic links in LIGHT')
        if path.is_file():
            entries[arc]=path.read_bytes()
    # Runtime source plus docs, including research identity; weights/datasets are excluded.
    for name in source_hashes():
        add(ROOT/name,'source/'+name)
    for p in sorted((ROOT/'docs/cqs_v1').glob('*')):
        if p.is_file():add(p,'source/docs/cqs_v1/'+p.name)
    for p in sorted(OUT.rglob('*')):
        rel=p.relative_to(OUT)
        if any(part.startswith('ops_fixture_') for part in rel.parts):
            continue  # operational fixture PASS records are not experiment preflight evidence
        if not p.is_file() or p.suffix not in ('.json','.yaml','.md','.csv','.jsonl','.txt') or any(x in rel.parts for x in ('data','isolated')):
            continue
        if p.name.endswith('_manifest.json'):
            # Store their identities in prepare; full file lists are still small enough for LIGHT.
            pass
        if p.name=='selection_events.jsonl':
            lines=p.read_text(encoding='utf-8').splitlines()
            selected=lines[:16]+lines[-16:] if len(lines)>32 else lines
            entries['outputs/'+rel.as_posix()]=('\n'.join(selected)+'\n').encode()
        elif p.stat().st_size<=8*1024*1024:
            add(p,'outputs/'+rel.as_posix())
    for p in sorted(OUT.glob('console_*.log')):
        with p.open('rb') as stream:
            first=stream.read(8192);stream.seek(max(8192,p.stat().st_size-32768));last=stream.read(32768)
        entries['logs/'+p.name+'.summary.txt']=first+b'\n[bounded middle omitted]\n'+last
    for name in ('args.yaml','results.csv'):
        add(RUN/name,'training/'+name)
    weights={name:dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p)) for name,p in
             [('controlled_init',INIT),('best',RUN/'weights/best.pt'),('last',RUN/'weights/last.pt')] if p.is_file()}
    meta=dict(created=utc(),source_sha=git('rev-parse','HEAD'),base_sha=BASE_SHA,branch=BRANCH,
              runtime=runtime(),weights=weights,formal_test='COMPLETED' if (OUT/'evaluation_test/metrics.json').is_file() and
              read_json(OUT/'evaluation_test/metrics.json').get('status')=='completed' else 'NOT_RUN',
              missing_mechanism_events=not (OUT/'selection_events.jsonl').is_file(),
              missing_mechanism_summary=not (OUT/'selection_epochs.jsonl').is_file(),
              policy='LIGHT: all available epoch summaries and bounded representative events; no datasets or large .pt')
    entries['metadata.json']=(json.dumps(meta,ensure_ascii=False,indent=2)+'\n').encode()
    entries['git/source.patch']=subprocess.check_output(['git','diff','--binary',BASE_SHA,'HEAD'],cwd=ROOT)
    entries['git/working.patch']=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)
    entries['README.txt']=('CQS v1 LIGHT. Read metadata.json and outputs reports. Missing training/eval is NOT_RUN.\n'
                           'Candidate coverage is not detector AP. Large weights and original logs remain on server.\n').encode()
    manifest=[dict(path=k,bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in sorted(entries.items())]
    entries['MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    with destination.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:gz') as archive:
        for name,value in sorted(entries.items()):
            member=tarfile.TarInfo(name);member.size=len(value);archive.addfile(member,io.BytesIO(value))
    verify_archive(destination)
    result=dict(status='PASS',archive=str(destination.resolve()),bytes=destination.stat().st_size,sha256=sha256(destination),members=len(entries))
    write_json(destination.with_suffix(destination.suffix+'.verification.json'),result)
    write_json(OUT/'last_package.json',result)
    (OUT/'FILEZILLA_PATH.txt').write_text(str(destination.resolve())+'\n',encoding='utf-8')
    print(str(destination.resolve()))
    return result
