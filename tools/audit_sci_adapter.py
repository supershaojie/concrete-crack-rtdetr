"""Verify successful source preservation, old combination graphs, and typed C2 recipe."""
from __future__ import annotations
import argparse
import ast
import hashlib
import subprocess
from sci_adapter import *
from train_sci_adapter import recipe

C17='0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139'
C19='025997e3c51eaf6933534308a95da6ebf97bff53'
C24='f6e9dfda765046ae7691302cf5ec89d3f76cec5d'
C25='ac32e223a509981ee9e61be66a352f2b80c5bfd1'
PREFIX='ultralytics-main/ultralytics/'


def historical(commit,path):
    return subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','show',f'{commit}:{path}'],cwd=ROOT)


def run():
    import yaml
    files={}
    protected={
        C17:['nn/modules/cscef_v5.py','nn/modules/cscef_v51.py','cfg/models/rt-detr/rtdetr-resnet18-lite-cscef-v51.yaml'],
        C19:['nn/modules/cbr.py','cfg/models/rt-detr/rtdetr-resnet18-lite-cbr.yaml'],
        C24:['nn/modules/scca_aifi.py','cfg/models/rt-detr/rtdetr-resnet18-lite-scca.yaml'],
        BASE_COMMIT:['nn/modules/head.py','nn/modules/conv.py','nn/modules/block.py','nn/modules/utils.py',
                     'models/rtdetr/train.py','models/rtdetr/val.py','models/utils/loss.py','models/utils/ops.py',
                     'utils/loss.py','utils/tal.py','cfg/models/rt-detr/rtdetr-resnet18-lite.yaml']}
    for commit,paths in protected.items():
        for path in paths:
            name=PREFIX+path;old=historical(commit,name).replace(b'\r\n',b'\n');new=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
            require(old==new,f'Protected source changed: {name}')
            files[name]=dict(commit=commit,canonical_sha256=hashlib.sha256(new).hexdigest(),exact=True)
    # Original C2 execution methods and defaults survive the parser's additive class registration.
    oldtree=ast.parse(historical(BASE_COMMIT,PREFIX+'nn/tasks.py'));newtree=ast.parse((ROOT/PREFIX/'nn/tasks.py').read_text())
    def classes(tree):return {n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
    for name in ('BaseModel','RTDETRDetectionModel'):
        require(ast.dump(classes(oldtree)[name])==ast.dump(classes(newtree)[name]),f'C2 model execution changed: {name}')
    # Transformer has the existing optional C19 final-query return. v2 must not touch it at all.
    require(historical(PARENT_COMMIT,PREFIX+'nn/modules/transformer.py').replace(b'\r\n',b'\n')==
            (ROOT/PREFIX/'nn/modules/transformer.py').read_bytes().replace(b'\r\n',b'\n'),'Transformer changed in v2')
    combinations={}
    for label,commit,file in (
        ('c20','81d5f0175e0168e022f29feb1ceb54de8a52deb5','rtdetr-resnet18-lite-cscef-cbr.yaml'),
        ('c25',C25,'rtdetr-resnet18-lite-cscef-scca.yaml'),
        ('c26','beedcfa307e250fb2de47587097c51f9c141123b','rtdetr-resnet18-lite-cbr-scca.yaml')):
        cfg=yaml.safe_load(historical(commit,PREFIX+'cfg/models/rt-detr/'+file))
        combinations[label]=dict(commit=commit,graph=rows(cfg))
        if label=='c25':
            replay=YAML.load(MODEL_DIR/LEGACY['c25'])
            require(all(cfg[k]==replay[k] for k in ('head','backbone','scales','nc')),'Replay differs from original C25')
    cfgs={};recipes={}
    for variant in VARIANTS:
        cfgs[variant]=topology(build(variant),variant)
        args,diff=recipe(ROOT/'docs/sci_adapter/c2_args.yaml',variant,ROOT/'weights'/f'{variant}_controlled_init.pt')
        require(len(diff)==109,'C2 recipe must have all 109 fields')
        recipes[variant]=diff
    return dict(status='passed',runtime=runtime(),protected_files=files,C2_execution_AST_exact=True,
                transformer_unchanged_from_parent=True,old_combinations=combinations,old_C25_replay_exact=True,
                new_topology=cfgs,recipe_all_109_fields=recipes)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);a=p.parse_args()
    torch.set_num_threads(4);write_json(a.report,run())
