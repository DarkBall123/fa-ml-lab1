"""Audit original RIAWELC and freeze mutually distinct experimental subsets."""
import argparse, collections, csv, hashlib, json, random
from pathlib import Path
from PIL import Image
import numpy as np

LABELS=['CR','LP','ND','PO']
def dhash(im):
    a=np.asarray(im.resize((9,8), Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(a[:,1:]>a[:,:-1]).tobytes(),'big')
def main():
    p=argparse.ArgumentParser(); p.add_argument('data_root'); p.add_argument('--out',default=str(Path(__file__).resolve().parents[1])); args=p.parse_args()
    root=Path(args.data_root).resolve(); out=Path(args.out); (out/'manifests').mkdir(parents=True,exist_ok=True)
    rows=[]; failures=[]; groups=collections.defaultdict(list)
    for path in sorted(root.rglob('*.png')):
        parts=path.relative_to(root).parts
        label=next(({'Difetto1':'CR','Difetto2':'PO','Difetto4':'LP','NoDifetto':'ND'}[s] for s in parts if s in {'Difetto1','Difetto2','Difetto4','NoDifetto'}),None)
        if label is None:
            label=next((s for s in LABELS if path.name.upper().startswith(s)),None)
        if label is None: continue
        try:
            with Image.open(path) as im:
                gray=im.convert('L'); pixels=gray.tobytes(); h=hashlib.sha256(pixels).hexdigest()
                row={'path':path.relative_to(root).as_posix(),'source_group':path.name.split('_Img')[0],'label':label,'width':im.width,'height':im.height,'mode':im.mode,'pixel_sha256':h,'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'dhash':str(dhash(gray)),'original_split':next((s for s in parts if 'train' in s.lower() or 'test' in s.lower() or 'valid' in s.lower()),'unknown')}
            rows.append(row); groups[h].append(row)
        except Exception as e: failures.append({'path':str(path),'error':str(e)})
    if not rows: raise RuntimeError('No labelled PNGs found')
    conflicts={h for h,g in groups.items() if len({r['label'] for r in g})>1}
    unique=[g[0] for h,g in groups.items() if h not in conflicts]
    rng=random.Random(17092026); rng.shuffle(unique)
    selected=[]; selected_hashes=[]; rejected=0
    source_groups=sorted({r['source_group'] for r in unique})
    for attempt in range(1000):
        rng.shuffle(source_groups); k=len(source_groups)
        assignment={g:('support' if i<int(k*.2) else 'validation' if i<int(k*.4) else 'test' if i<int(k*.7) else 'application') for i,g in enumerate(source_groups)}
        if all(sum(r['label']==c and assignment[r['source_group']]==sp for r in unique)>=n*2 for sp,n in [('support',1),('validation',4),('test',20),('application',40)] for c in LABELS):break
    else:raise RuntimeError('Could not form class-covered source-disjoint splits')
    pools={(sp,c):[r for r in unique if r['label']==c and assignment[r['source_group']]==sp] for sp in ['support','validation','test','application'] for c in LABELS}
    (out/'manifests/source_group_split.json').write_text(json.dumps(assignment,indent=2))
    # Each class: 1 training exemplar, 4 validation, 20 test, 40 application.
    targets=[('support',1),('validation',4),('test',20),('application',40)]
    for split,n in targets:
        for label in LABELS:
            accepted=0
            while accepted<n:
                if not pools[(split,label)]: raise RuntimeError(f'Not enough distinct images: {label}')
                r=pools[(split,label)].pop(); h=int(r['dhash'])
                if any((h^x).bit_count()<=4 for x in selected_hashes): rejected+=1; continue
                r=dict(r,split=split,id=f'{split}_{label}_{accepted+1:03d}')
                selected.append(r); selected_hashes.append(h); accepted+=1
    def savecsv(path, records):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    savecsv(out/'manifests/full_inventory.csv',rows); savecsv(out/'manifests/experiment.csv',selected)
    cross=[g for g in groups.values() if len({r['original_split'] for r in g})>1]
    audit={'source':'https://github.com/stefyste/RIAWELC','commit':'d53d3daf28ed3901f9db89764c030d19ced7696d','seed':17092026,'source_group_count':len(source_groups),'source_group_split_counts':dict(collections.Counter(assignment.values())),'label_mapping':{'Difetto1':'CR','Difetto2':'PO','Difetto4':'LP','NoDifetto':'ND'},'total_files':len(rows),'classes':dict(collections.Counter(r['label'] for r in rows)),'original_splits':dict(collections.Counter(r['original_split'] for r in rows)),'unique_pixel_hashes':len(groups),'duplicate_excess':len(rows)-len(groups),'conflicting_hash_groups':len(conflicts),'cross_original_split_duplicate_groups':len(cross),'decode_failures':failures,'sizes':dict(collections.Counter(f"{r['width']}x{r['height']}" for r in rows)),'modes':dict(collections.Counter(r['mode'] for r in rows)),'selected_count':len(selected),'near_duplicate_candidates_rejected':rejected,'near_duplicate_method':'64-bit dHash Hamming distance <=4, exclusion among all selected objects','min_selected_hamming':min((a^b).bit_count() for i,a in enumerate(selected_hashes) for b in selected_hashes[i+1:]),'experiment_splits':dict(collections.Counter(r['split'] for r in selected)),'unique_classes':dict(collections.Counter(r['label'] for r in unique))}
    (out/'results/data_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)); print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
