"""Official M grouping with project LR, complete coverage and update-based schedule."""
from __future__ import annotations
import math
import torch

def build_optimizer(model,cfg):
    groups={}; names=[]
    for name,param in model.named_parameters():
        if not param.requires_grad: continue
        backbone=name.startswith('backbone.')
        # Match official M name-based norm/bn grouping; bias uses zero decay throughout.
        no_decay=any(x in name for x in ('norm','bn','bias'))
        role=('backbone' if backbone else 'encoder_decoder')+('_zero_decay' if no_decay else '_decay')
        g=groups.setdefault(role,{'params':[],'names':[],'lr':cfg['backbone_lr'] if backbone else cfg['lr0'],
            'weight_decay':0. if no_decay else cfg['weight_decay'],'role':role})
        g['params'].append(param); g['names'].append(name); names.append(name)
    expected={n for n,p in model.named_parameters() if p.requires_grad}
    if len(names)!=len(set(names)) or set(names)!=expected: raise ValueError('Duplicate/missing optimizer parameter')
    audit=[{k:v for k,v in g.items() if k!='params'} | {'parameter_tensors':len(g['params']),
        'parameter_elements':sum(p.numel() for p in g['params'])} for g in groups.values()]
    # Names kept in receipts, not optimizer state.
    opt=torch.optim.AdamW([{k:v for k,v in g.items() if k!='names'} for g in groups.values()],betas=tuple(cfg['betas']))
    return opt,audit

class UpdateSchedule:
    def __init__(self,optimizer,total_updates,warmup_updates,start_factor,lrf):
        if not 1<=total_updates or not 0<=warmup_updates<total_updates: raise ValueError('Invalid actual loader schedule')
        self.optimizer=optimizer; self.total=total_updates; self.warmup=warmup_updates
        self.start_factor=start_factor; self.lrf=lrf; self.targets=[g['lr'] for g in optimizer.param_groups]; self.updates=0
    def factor(self,index):
        if not 0<=index<self.total: raise ValueError('Schedule update outside plan')
        if self.warmup and index<self.warmup:
            return self.start_factor+(1-self.start_factor)*index/max(1,self.warmup-1)
        progress=(index-self.warmup)/max(1,self.total-self.warmup-1)
        return self.lrf+(1-self.lrf)*(1+math.cos(math.pi*progress))/2
    def apply(self):
        factor=self.factor(self.updates)
        for target,g in zip(self.targets,self.optimizer.param_groups): g['lr']=target*factor
        return [g['lr'] for g in self.optimizer.param_groups]
    def advance(self): self.updates+=1
    def state_dict(self):
        return {k:getattr(self,k) for k in ('total','warmup','start_factor','lrf','targets','updates')}
    def load_state_dict(self,state):
        for k in ('total','warmup','start_factor','lrf','targets'):
            if state[k]!=getattr(self,k): raise ValueError('Resume schedule differs: '+k)
        if not 0<=state['updates']<=self.total: raise ValueError('Invalid saved update count')
        self.updates=state['updates']
