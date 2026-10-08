"""Plots contain actual CSV observations and the public evaluator's raw curves."""
from pathlib import Path
import csv
import numpy as np
from support import ROOT,read_json

def pyplot():
    import os
    os.environ['MPLCONFIGDIR']=str(ROOT/'.runtime/dfine-m-configurable/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt

def plot_training(run):
    run=Path(run); source=run/'train/results.csv'
    if not source.exists(): raise ValueError('Missing actual training CSV')
    with source.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError('Empty actual training CSV')
    plt=pyplot(); fig,axes=plt.subplots(3,3,figsize=(18,13),layout='constrained')
    x=[int(r['epoch']) for r in rows]; keys=list(rows[0])
    def series(ax,selected,title):
        for key in selected:
            values=[float(r[key]) if r.get(key) else np.nan for r in rows]
            ax.plot(x,values,label=key,linewidth=1.2)
        ax.set_xlabel('Completed epoch'); ax.set_title(title); ax.grid(alpha=.2)
        if selected: ax.legend(fontsize=6,ncol=2 if len(selected)>4 else 1)
    series(axes[0,0],['loss_total'],'Native total weighted loss')
    for base,ax in zip(('vfl','bbox','giou','fgl','ddf'),axes.ravel()[1:6]):
        prefix='loss_'+base
        series(ax,[k for k in keys if k==prefix or k.startswith(prefix+'_')],
            'Native '+base.upper()+' (final / auxiliary / denoising)')
    series(axes[2,0],[k for k in keys if k.startswith('lr_')],'Actual optimizer group LR')
    selected='EMA' if read_json(run/'identity.json')['ema'] else 'model'
    series(axes[2,1],[k for k in keys if k.startswith('val_')],'FP32 public '+selected+' val [0,1]')
    axes[2,1].set_ylim(0,1)
    series(axes[2,2],['ema_updates','optimizer_updates'],'Continuous actual updates')
    scope=read_json(run/'run_id.json')['scope']
    fig.suptitle('Actual epoch observations'+(' - SYNTHETIC SMOKE ONLY' if scope=='SMOKE_ONLY' else ''),fontsize=14)
    fig.savefig(run/'train/results.png',dpi=140); plt.close(fig)


def plot_curves(run,split):
    run=Path(run); raw=read_json(run/f'curves/{split}_raw.json'); plt=pyplot()
    x=np.asarray(raw.get('confidence',[]))
    for name,key in [('P','p_curve'),('R','r_curve'),('F1','f1_curve')]:
        fig,ax=plt.subplots(figsize=(7,5))
        values=np.asarray(raw.get(key,[]))
        if values.size: ax.plot(x,values[0],label='crack')
        ax.set(xlabel='Confidence',ylabel=name,xlim=(0,1),ylim=(0,1),title=f'{split.upper()} {name}, IoU=0.50')
        ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(run/f'curves/{split}_{name}_confidence.png',dpi=150); plt.close(fig)
    values=np.asarray(raw.get('pr_precision',[])); fig,ax=plt.subplots(figsize=(7,5))
    if values.size: ax.plot(x,values[0],label='crack')
    ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1),title=split.upper()+' PR, IoU=0.50')
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(run/f'curves/{split}_PR.png',dpi=150); plt.close(fig)
