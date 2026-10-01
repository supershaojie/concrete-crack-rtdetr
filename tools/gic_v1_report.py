"""Render a small offline metrics page from recorded results; never runs inference."""
from html import escape
from gic_v1_common import OUT,read_json


def render():
    rows=[];details=[]
    for split in ('val','test'):
        r=read_json(OUT/f'evaluation_{split}.json',{})
        if r.get('status')!='COMPLETE':
            rows.append(f'<tr><th>{split}</th><td colspan="10">NOT_RUN / incomplete</td></tr>');continue
        cells=[split,'all / crack',str(r['images']),str(r['ground_truths'])]
        cells += [f'{100*r[k]:.6f}%' for k in ('precision','recall','F1','AP50','AP75','mAP50_95')]
        cells += [f'{r["mother_delta_pp"]:+.6f} pp']
        rows.append('<tr>'+''.join('<td>'+escape(v)+'</td>' for v in cells)+'</tr>')
        details.append('<p>'+escape(f"{split}: {r['identity']['protocol']} · exit={r['exit_code']} · best={r['best']}")+
                       '<br>SHA256: '+escape(r['best_sha256'])+'<br>AP50:0.05:0.95: '+
                       ', '.join(f'{100*x:.6f}%' for x in r['ap_by_iou'][0])+'</p>')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>GIC v1 metrics</title>
<style>body{font:16px/1.6 system-ui;max-width:1400px;margin:40px auto;padding:20px;color:#172333}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:9px;border-bottom:1px solid #cad5df;text-align:right}th{background:#edf3fa}p{overflow-wrap:anywhere}</style>
<h1>GIC v1 · Independent FP32</h1><p>640 / B16 · P = Precision。原始未舍入数值保存在 evaluation_val/test.json。阈值分析仅由 val 选阈值。</p>
<table><tr>'''+''.join('<th>'+s+'</th>' for s in ['split','class','images','GT','Precision','Recall','F1','AP50','AP75','mAP50–95','vs mother'])+'</tr>'+''.join(rows)+'</table>'+''.join(details)+'''
<p>GIC 新正项 loss 变小不能证明检测效果改善。训练阶段 AMP 验证成绩不作为独立 FP32 成绩。</p></html>'''
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'metrics.html').write_text(html,encoding='utf-8')
