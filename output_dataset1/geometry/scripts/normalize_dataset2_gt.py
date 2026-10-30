"""Normalize paired geometry GT using unique selectors or verified DOM identities.

DOM metadata is used ONLY to disambiguate GT endpoints, never predictions.
"""
import json
from collections import Counter,defaultdict
from pathlib import Path
from geometry.scripts.detect_dataset2 import ROOT,save,sha
OUT=ROOT/'geometry/results/dataset2_from502_detection_v1'

def normalize(gt,dom):
    groups={s:defaultdict(list) for s in ('before','after')}
    for side in groups:
        for i,b in enumerate(gt[side+'_bboxes']):
            if 'geometry' in b.get('type',[]):groups[side][b['selector']].append((i,b))
    boxes=[];unresolved=[]
    def add(sel,b,a,method,identity=None):
        bi,bb=b;ai,ab=a
        x=min(bb['x'],ab['x']);y=min(bb['y'],ab['y']);x2=max(bb['x']+bb['w'],ab['x']+ab['w']);y2=max(bb['y']+bb['h'],ab['y']+ab['h'])
        boxes.append({'x':x,'y':y,'w':x2-x,'h':y2-y,'selector':sel,'type':sorted(set(bb['type'])|set(ab['type'])),'fallback':bool(bb.get('fallback') or ab.get('fallback')),'provenance':{'before_index':bi,'after_index':ai,'method':method,'dom_identity':identity,'before_bbox':bb,'after_bbox':ab}})
    for sel in sorted(set(groups['before'])|set(groups['after'])):
        b=groups['before'][sel];a=groups['after'][sel]
        if len(b)==len(a)==1:
            add(sel,b[0],a[0],'unique_selector');continue
        candidates={};counts={s:Counter(e.get('selector') for e in dom[s]) for s in dom}
        for side,arr in [('before',b),('after',a)]:
            mapping={}
            for idx,box in arr:
                ids=set()
                for e in dom[side]:
                    ident=e.get('selector');eb=e.get('bbox',{})
                    if ident and counts[side][ident]==1 and all(isinstance(eb.get(k),(int,float)) and abs(eb[k]-box[v])<1.0 for k,v in [('x','x'),('y','y'),('width','w'),('height','h')]):ids.add(ident)
                mapping[idx]=ids
            candidates[side]=mapping
        possible=[]
        for bi,bb in b:
            for ai,ab in a:
                ids=candidates['before'][bi]&candidates['after'][ai]
                if len(ids)==1:possible.append((bi,ai,next(iter(ids))))
        bc=Counter(x[0] for x in possible);ac=Counter(x[1] for x in possible)
        usedb=set();useda=set()
        for bi,ai,ident in possible:
            if bc[bi]==ac[ai]==1:
                add(sel,next(x for x in b if x[0]==bi),next(x for x in a if x[0]==ai),'unique_dom_selector_and_subpixel_bbox',ident);usedb.add(bi);useda.add(ai)
        rem_b=[i for i,_ in b if i not in usedb];rem_a=[i for i,_ in a if i not in useda]
        if rem_b or rem_a:unresolved.append({'selector':sel,'before_indices':rem_b,'after_indices':rem_a,'reason':'No unambiguous stable DOM identity; no order/spatial guess applied'})
    return {'pair_id':gt['pair_id'],'bboxes':boxes,'unresolved':unresolved,'fully_resolved':not unresolved,'schema_note':'Derived DOM endpoint envelopes; historical pixel-diff GT equivalence NOT established.'}

def main():
    OUT.mkdir(exist_ok=True);rows=[]
    for folder in sorted((ROOT/'dataset2').iterdir(),key=lambda p:int(p.name.split('_')[0])):
        pid=int(folder.name.split('_')[0])
        if pid<502:continue
        gp=ROOT/'groundtruth/groundtruth'/f'{pid:03d}.json';gt=json.loads(gp.read_text(encoding='utf-8'))
        if not any('geometry' in b.get('type',[]) for side in ('before','after') for b in gt[side+'_bboxes']):continue
        dom={s:json.loads((folder/f'elements_{s}.json').read_text(encoding='utf-8')) for s in ('before','after')}
        result=normalize(gt,dom);result['source']={'gt_path':str(gp),'gt_sha256':sha(gp),'dom_hashes':{s:sha(folder/f'elements_{s}.json') for s in dom}}
        path=OUT/'normalized_gt'/f'{pid:03d}.json';save(path,result)
        rows.append({'pair_id':str(pid),'folder':str(folder),'gt_path':str(path),'gt_count':len(result['bboxes']),'fully_resolved':result['fully_resolved'],'unresolved':result['unresolved']})
    summary={'samples':len(rows),'fully_resolved_samples':sum(r['fully_resolved'] for r in rows),'partial_samples':sum(not r['fully_resolved'] for r in rows),'normalized_geometry_gt':sum(r['gt_count'] for r in rows),'unresolved_selector_groups':sum(len(r['unresolved']) for r in rows),'methods':dict(Counter(b['provenance']['method'] for r in rows for b in json.loads(Path(r['gt_path']).read_text(encoding='utf-8'))['bboxes'])),'historical_pixel_diff_equivalence':False}
    save(OUT/'gt_manifest.json',{'samples':rows,'summary':summary});print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
