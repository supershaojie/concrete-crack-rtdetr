"""Compare the v5 detection adapter to the unmodified pinned official CutMix."""
import ast
from pathlib import Path
from typing import Any
from unittest.mock import patch
import numpy as np
from b19_augment import bbox_ioa, detection_cutmix


class Instances:
    def __init__(self, boxes):
        self.bboxes=boxes.copy()
        self.segments=[]
    def __getitem__(self, index):
        return Instances(self.bboxes[index])
    def convert_bbox(self, fmt):
        assert fmt=='xyxy'
    def denormalize(self, w, h):
        pass  # reference inputs are already absolute xyxy after perspective
    def add_padding(self, x, y):
        self.bboxes+=np.array([x,y,x,y],dtype=np.float32)
    def clip(self, w, h):
        self.bboxes[:,[0,2]]=np.clip(self.bboxes[:,[0,2]],0,w)
        self.bboxes[:,[1,3]]=np.clip(self.bboxes[:,[1,3]],0,h)
    @staticmethod
    def concatenate(items, axis):
        return Instances(np.concatenate([x.bboxes for x in items],axis=axis))


class BaseMixTransform:
    def __init__(self, **kwargs):
        pass


def reference_class():
    source=Path(__file__).parent/'vendor/cutmix_reference.txt'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    cut=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CutMix')
    namespace={'np':np,'Any':Any,'Instances':Instances,'bbox_ioa':bbox_ioa,'BaseMixTransform':BaseMixTransform}
    ioa_source=source.with_name('bbox_ioa_reference.txt')
    exec(compile(ioa_source.read_text(encoding='utf-8'),str(ioa_source),'exec'),namespace)
    exec(compile(ast.Module(body=[cut],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['CutMix']


def verify_reference():
    cls=reference_class()
    matched=0
    outcomes={}
    for seed in range(100):
        image=np.zeros((64,64,3),dtype=np.uint8)
        donor=np.full_like(image,225)
        base=np.array([[0,2,2,10,10]],dtype=np.float32) if seed%3 else np.zeros((0,5),dtype=np.float32)
        target=np.array([[0,10,10,50,50],[0,52,52,63,63]],dtype=np.float32)
        labels={'img':image.copy(),'cls':base[:,:1].copy(),'instances':Instances(base[:,1:]),
                'mix_labels':[{'img':donor.copy(),'cls':target[:,:1].copy(),'instances':Instances(target[:,1:])}]}
        np.random.seed(seed)
        result=cls(dataset=None)._mix_transform(labels)
        np.random.seed(seed)
        actual, boxes, outcome=detection_cutmix(image.copy(),base.copy(),donor,target)
        expected=np.concatenate([result['cls'],result['instances'].bboxes],axis=1)
        assert np.array_equal(actual,result['img']) and np.array_equal(boxes,expected),seed
        outcomes[outcome]=outcomes.get(outcome,0)+1
        matched+=1
    assert outcomes.get('applied',0)>0
    return {'seeds':matched,'pixels_and_absolute_labels_match_fixed_source':True,
            'official_bbox_ioa_reference_used':True,'outcomes':outcomes}


def geometry_cases():
    image=np.zeros((64,64,3),dtype=np.uint8)
    donor=np.full_like(image,200)
    base=np.array([[0,1,1,8,8]],dtype=np.float32)
    targets=np.array([[0,25,25,55,55],[0,45,45,63,63]],dtype=np.float32)
    with patch('b19_augment.rand_bbox',return_value=(30,30,50,50)):
        actual,labels,outcome=detection_cutmix(image.copy(),base,donor,targets)
    assert outcome=='applied'
    assert labels.tolist()==[[0,1,1,8,8],[0,30,30,50,50]]
    assert np.all(actual[30:50,30:50]==200) and not actual[:30].any()
    with patch('b19_augment.rand_bbox',return_value=(1,1,8,8)):
        _,labels,blocked=detection_cutmix(image.copy(),base,donor,targets)
    with patch('b19_augment.rand_bbox',return_value=(10,10,15,15)):
        _,labels,empty=detection_cutmix(image.copy(),base,donor,targets)
    assert blocked=='no_free_area' and empty=='no_donor'
    return {'base_boxes_preserved':True,'donor_IOA_threshold':.1,'donor_clipped_to_patch':True,
            'no_base_overlap_skip':blocked,'no_donor_skip':empty,'classification_soft_labels':False}


if __name__=='__main__':
    print(verify_reference())
    print(geometry_cases())
