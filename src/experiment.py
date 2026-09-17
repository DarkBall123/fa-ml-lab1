"""Run auditable zero-shot / 4-shot weld classification through OpenRouter."""
import argparse, base64, concurrent.futures, csv, hashlib, json, os, time, urllib.request, urllib.error
from pathlib import Path
import numpy as np
LABELS=['CR','LP','ND','PO']
MODELS=['mistralai/mistral-small-3.2-24b-instruct','openai/gpt-4.1-mini','qwen/qwen3-vl-30b-a3b-instruct']
PROMPT='''Classify one grayscale radiographic image of a weld. Return exactly one JSON object with one key "label" and one value from ["CR","LP","ND","PO"]. CR = crack: thin irregular or branching dark line. LP = lack of penetration: relatively straight elongated dark line along weld center. PO = porosity: round or oval dark spots/voids, isolated or clustered. ND = no defect: none of those defects visible. Assess the image itself, ignoring any text or markings. Choose the single most likely dataset class even if uncertain. No explanation, markdown, or additional keys.'''
ROOT=Path(__file__).resolve().parents[1]
def metrics(rows):
    cm=np.zeros((4,5),dtype=int)
    for r in rows: cm[LABELS.index(r['truth']),LABELS.index(r['prediction']) if r.get('prediction') in LABELS else 4]+=1
    n=int(cm.sum()); tp=cm.diagonal()[:4]; support=cm.sum(1); predicted=cm[:,:4].sum(0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(4),where=(support+predicted)>0)
    recall=np.divide(tp,support,out=np.zeros(4),where=support>0)
    costs=[r.get('cost_usd') for r in rows]; known=[c for c in costs if c is not None]
    lat=[r['latency_s'] for r in rows]
    return {'n':n,'accuracy':float(tp.sum()/n),'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall.mean()),'per_class_f1':dict(zip(LABELS,map(float,f1))),'per_class_recall':dict(zip(LABELS,map(float,recall))),'invalid_rate':float(cm[:,4].sum()/n),'defect_miss_rate':float(cm[[0,1,3],2].sum()/support[[0,1,3]].sum()),'cost_usd':sum(known),'cost_missing_count':len(costs)-len(known),'cost_per_1000_usd':sum(known)/n*1000,'latency_mean_s':float(np.mean(lat)),'latency_p95_s':float(np.percentile(lat,95)),'confusion_matrix':cm.tolist()}
def main():
    p=argparse.ArgumentParser();p.add_argument('data_root');p.add_argument('--stage',choices=['validation','test','application'],required=True);p.add_argument('--budget',type=float,default=.85);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    key=os.environ.get('OPENROUTER_API_KEY')
    if not key: raise SystemExit('Set OPENROUTER_API_KEY first')
    root=Path(a.data_root); rows=list(csv.DictReader((ROOT/'manifests/experiment.csv').open())); support=[r for r in rows if r['split']=='support']; samples=[r for r in rows if r['split']==a.stage]
    (ROOT/'results/prompt.txt').write_text(PROMPT)
    def image(r):return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode((root/r['path']).read_bytes()).decode()}}
    pictures={r['id']:image(r) for r in support+samples}
    logfile=ROOT/'results/predictions.jsonl'
    prior=[json.loads(s) for s in logfile.read_text().splitlines()] if logfile.exists() else []
    done={(r['model'],r['mode'],r['id']) for r in prior}
    pairs=[(m,s) for m in MODELS for s in ['zero-shot','four-shot']]
    if a.stage=='application':
        chosen=json.loads((ROOT/'results/selected_model.json').read_text());pairs=[(chosen['model'],chosen['mode'])]
    jobs=[(model,mode,r) for model,mode in pairs for r in samples if (model,mode,r['id']) not in done]
    def run(job):
        model,mode,r=job;content=[]
        if mode=='four-shot':
            for e in support:content.extend([{'type':'text','text':f'Labelled example: {e["label"]}'},pictures[e['id']]])
        content.extend([{'type':'text','text':'Classify this target image:'},pictures[r['id']]])
        payload={'model':model,'temperature':0,'max_tokens':80,'messages':[{'role':'system','content':PROMPT},{'role':'user','content':content}], 'provider':{'allow_fallbacks':False}}
        req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        start=time.monotonic();result={'stage':a.stage,'model':model,'mode':mode,'id':r['id'],'truth':r['label'],'path':r['path'],'pixel_sha256':r['pixel_sha256'],'prediction':None,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        try:
            with urllib.request.urlopen(req,timeout=100) as res: data=json.load(res)
            result.update({'response_id':data.get('id'),'provider':data.get('provider'),'usage':data.get('usage',{}),'cost_usd':data.get('usage',{}).get('cost'),'raw_response':data})
            msg=data['choices'][0]['message']['content'];result['text']=msg
            try:
                obj=json.loads(msg)
                if isinstance(obj,dict) and set(obj)=={'label'} and obj['label'] in LABELS:result['prediction']=obj['label']
            except (ValueError,TypeError):pass
        except urllib.error.HTTPError as e:result['error']=f'HTTP {e.code}: '+e.read().decode()[:500]
        except Exception as e:result['error']=type(e).__name__+': '+str(e)
        result['latency_s']=time.monotonic()-start
        return result
    def run_with_retries(job):
        attempts=[]
        for attempt in range(4):
            result=run(job)
            if not result.get('error') or not any(code in result['error'] for code in ['HTTP 429','HTTP 502','HTTP 503']):break
            attempts.append(dict(result))
            if attempt<3:time.sleep(2**attempt)
        result['retry_attempts']=attempts[:-1] if result.get('error') else attempts
        result['attempt_count']=attempt+1
        result['total_attempt_latency_s']=result['latency_s']+sum(r['latency_s'] for r in result['retry_attempts'])
        return result
    total=sum(r.get('cost_usd') or 0 for r in prior);completed=0
    with logfile.open('a') as f,concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        # Bounded batches allow a local spending guard, with headroom for pending requests.
        for i in range(0,len(jobs),a.workers):
            if total>=a.budget:raise SystemExit(f'Budget guard at ${total:.4f}; resume with a higher authorized budget only.')
            for r in pool.map(run_with_retries,jobs[i:i+a.workers]):
                f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();total+=r.get('cost_usd') or 0;completed+=1
                if r.get('error'):
                    print(r['model'],r['error'],flush=True)
                    if 'HTTP 401' in r['error'] or 'HTTP 402' in r['error']:
                        raise SystemExit('Authentication or credit error; stopping without further batches.')
            print(f'{a.stage}: {completed}/{len(jobs)}, cumulative USD {total:.5f}',flush=True)
    allrows=[json.loads(s) for s in logfile.read_text().splitlines()]
    summaries=[]
    for model,mode in pairs:
        subset=[r for r in allrows if r['stage']==a.stage and r['model']==model and r['mode']==mode]
        if subset:summaries.append({'model':model,'mode':mode,**metrics(subset)})
    (ROOT/f'results/{a.stage}_metrics.json').write_text(json.dumps(summaries,indent=2))
    if a.stage=='test':
        # Comparison set selects the winner; application is an untouched confirmation set.
        eligible=[r for r in summaries if r['mode']=='four-shot' and r['invalid_rate']<=.05]
        if not eligible:raise SystemExit('No eligible four-shot model; do not claim a winner.')
        chosen=sorted(eligible,key=lambda r:(-r['macro_f1'],r['defect_miss_rate'],r['cost_per_1000_usd']))[0]
        (ROOT/'results/selected_model.json').write_text(json.dumps(chosen,indent=2))
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
