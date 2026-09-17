"""Recalculate metrics, uncertainty, error inventory, and integrity checks offline."""
import csv,json,collections
from pathlib import Path
import numpy as np
from experiment import metrics,LABELS,MODELS
ROOT=Path(__file__).resolve().parents[1]
def main():
    rows=[json.loads(s) for s in (ROOT/'results/predictions.jsonl').read_text().splitlines()]
    manifest=list(csv.DictReader((ROOT/'manifests/experiment.csv').open()));byid={r['id']:r for r in manifest}
    assert len({(r['stage'],r['model'],r['mode'],r['id']) for r in rows})==len(rows)
    for r in rows:
        assert r['truth']==byid[r['id']]['label'] and r['pixel_sha256']==byid[r['id']]['pixel_sha256']
    stages=['support','validation','test','application']
    for i,a in enumerate(stages):
        for b in stages[i+1:]:
            assert not ({r['source_group'] for r in manifest if r['split']==a}&{r['source_group'] for r in manifest if r['split']==b})
    summary=[];rng=np.random.default_rng(17092026)
    for (stage,model,mode),group in __import__('itertools').groupby(sorted(rows,key=lambda r:(r['stage'],r['model'],r['mode'])),lambda r:(r['stage'],r['model'],r['mode'])):
        sample=list(group);m=metrics(sample);groups=collections.defaultdict(list)
        for r in sample:groups[byid[r['id']]['source_group']].append(r)
        names=sorted(groups);scores=[]
        if stage!='validation':
            for _ in range(2000):
                boot=[r for name in rng.choice(names,len(names),replace=True) for r in groups[name]]
                if len({r['truth'] for r in boot})==4:scores.append(metrics(boot)['macro_f1'])
        m.update(stage=stage,model=model,mode=mode,source_groups=len(names),bootstrap_valid_draws=len(scores),macro_f1_ci95=list(map(float,np.percentile(scores,[2.5,97.5]))) if scores else None,transport_errors=sum(bool(r.get('error')) for r in sample),malformed_responses=sum(not r.get('error') and r.get('prediction') is None for r in sample))
        summary.append(m)
    (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2))
    fields=['stage','model','mode','n','accuracy','macro_f1','invalid_rate','defect_miss_rate','cost_usd','cost_per_1000_usd','latency_mean_s','latency_p95_s']
    with (ROOT/'results/metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(summary)
    with (ROOT/'results/errors.csv').open('w',newline='') as f:
        fields=['stage','model','mode','id','truth','prediction','path'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows([r for r in rows if r['truth']!=r.get('prediction')])
    comparison=[s for s in summary if s['stage']=='test'];delta={m:next(s['macro_f1'] for s in comparison if s['model']==m and s['mode']=='four-shot')-next(s['macro_f1'] for s in comparison if s['model']==m and s['mode']=='zero-shot') for m in MODELS}
    ex=[json.loads(s) for s in (ROOT/'results/excluded_provider_attempts.jsonl').read_text().splitlines()]
    retried=[a for r in rows for a in r.get('retry_attempts',[])]
    totals={'final_predictions':len(rows),'excluded_attempts':len(ex),'embedded_retry_attempts':len(retried),'recorded_cost_usd':sum(r.get('cost_usd') or 0 for r in rows+ex+retried),'final_cost_missing':sum(r.get('cost_usd') is None for r in rows),'few_shot_delta_macro_f1':delta,'mean_delta':float(np.mean(list(delta.values()))),'final_providers':dict(collections.Counter(r.get('provider','unknown') for r in rows))}
    (ROOT/'results/totals.json').write_text(json.dumps(totals,indent=2));print(json.dumps(totals,indent=2))
    print('Integrity checks passed')
if __name__=='__main__':main()
