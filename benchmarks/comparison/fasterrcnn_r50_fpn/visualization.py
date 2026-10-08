"""Recoverable gradient-capable model; inference is a separate caller choice."""
from pathlib import Path
from support import HERE,read_json,sha256,write_json

def load_for_visualization(config,checkpoint,device='cuda:0'):
    from configuration import frozen_config
    from data_adapter import letterbox,inverse_boxes
    from engine import load_checkpoint,validate_checkpoint
    from model import build_model
    import torch,numpy as np
    config=Path(config); run=config if config.is_dir() else config.parent
    cfg=frozen_config(run); identity=read_json(run/'identity.json'); checkpoint=Path(checkpoint).absolute()
    index=read_json(run/'checkpoints/index.json')
    if checkpoint!=run.absolute()/'checkpoints/best.pt' or sha256(checkpoint)!=index['best_sha256']:
        raise ValueError('Visualization must restore this run selected best')
    saved=load_checkpoint(checkpoint); validate_checkpoint(saved,identity,cfg)
    model,_=build_model(cfg); model.load_state_dict(saved['model'],strict=True); model.to(device).float().eval()
    def preprocess(rgb):
        image,geometry=letterbox(rgb,cfg['imgsz'])
        tensor=torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float()/255
        return tensor.to(device),geometry
    return {'model':model,'preprocess_rgb':preprocess,'inverse_boxes':inverse_boxes,
        'class_map':{'background':0,'crack_model':1,'crack_public':1,'crack_yolo_source':0},
        'config':cfg,'checkpoint_sha256':index['best_sha256'],
        'gradient_policy':'no global inference_mode/no_grad; callers choose torch.no_grad for detection or enable_grad for Grad-CAM++',
        'target_layers':[n for n,_ in model.named_modules() if n.startswith(('backbone.body.layer','backbone.fpn'))]}

def write_handoff(run):
    run=Path(run); cfg=read_json(run/'identity.json')['recipe']; training=read_json(run/'train_status.json')
    value={'schema_version':1,'model':'Faster R-CNN (ResNet-50-FPN)','model_state':'model','ema':False,
        'project':str(HERE.parents[2]),'python':read_json(run/'environment.json')['python'],
        'code_commit':read_json(run/'identity.json')['model_code_sha'],'code_path':str(HERE),
        'config':str(run/'resolved_config.yaml'),'best_checkpoint':str(run/'checkpoints/best.pt'),
        'best_sha256':training['best_sha256'],'best_epoch':training['best_epoch'],
        'data_root':read_json(run/'manifest.json')['data_root'],'inference':{k:cfg[k] for k in ('imgsz','eval_batch_size','eval_workers','box_score_thresh','box_nms_thresh','box_detections_per_img')},
        'results':{split:{'predictions':str(run/'predictions'/(split+'.jsonl.gz')),'metrics':read_json(run/'metrics'/(split+'_complete.json'))['metrics_path']} for split in ('val','test')},
        'restore_interface':str(HERE/'visualization.py')+':load_for_visualization(config_run_dir, checkpoint_best_path, device)',
        'class_map':{'background':0,'crack':1,'public':1},
        'observed_backbone_fpn_shapes':read_json(run/'visualization_layers.json') if (run/'visualization_layers.json').exists() else {'status':'not_probed'},
        'verification':{'host':'server' if __import__('os').name!='nt' and read_json(run/'run_id.json')['scope']=='FORMAL' else 'local smoke',
                        'scope':read_json(run/'run_id.json')['scope'],'formal_server_verified':read_json(run/'run_id.json')['scope']=='FORMAL' and training['status']=='completed'},
        'gradcam_implementation':'handoff only; no Grad-CAM++ or automatic paper image selection'}
    write_json(run/'visualization_handoff.json',value); return value
