"""Build a GT-only viewer for all 505 dataset2 samples, filtering geometry only."""
import json
import os
from pathlib import Path
from PIL import Image
from geometry.scripts.detect_dataset2 import ROOT,save,sha
OUT=ROOT/'geometry/results/dataset2_gt_review'

def main():
    OUT.mkdir(exist_ok=True);samples=[];inputs=[]
    for folder in sorted((ROOT/'dataset2').iterdir(),key=lambda p:int(p.name.split('_')[0])):
        if not folder.is_dir():continue
        pid=folder.name.split('_')[0];gp=ROOT/'groundtruth/groundtruth'/f'{int(pid):03d}.json'
        gt=json.loads(gp.read_text(encoding='utf-8'));assert int(gt['pair_id'])==int(pid)
        inputs.append({'path':str(gp),'sha256':sha(gp)})
        sample={'id':pid,'name':folder.name,'sides':{}}
        for side,imgname in [('before','baseline.png'),('after','inject.png')]:
            image=folder/imgname
            with Image.open(image) as im:width,height=im.size
            boxes=[]
            for i,b in enumerate(gt[side+'_bboxes']):
                if 'geometry' not in b.get('type',[]):continue
                boxes.append(dict(b,original_index=i,outside_viewport=b['x']<0 or b['y']<0 or b['x']+b['w']>width or b['y']+b['h']>height))
            sample['sides'][side]={'image':os.path.relpath(image,OUT),'width':width,'height':height,'boxes':boxes}
        sample['count']=sum(len(s['boxes']) for s in sample['sides'].values());samples.append(sample)
    summary={'samples':len(samples),'geometry_samples':sum(s['count']>0 for s in samples),'geometry_before':sum(len(s['sides']['before']['boxes']) for s in samples),'geometry_after':sum(len(s['sides']['after']['boxes']) for s in samples),'source':'original before_bboxes / after_bboxes, geometry filter only, no inferred pairing or merged envelopes'}
    save(OUT/'summary.json',summary);save(OUT/'manifest.json',{'inputs':inputs,'summary':summary})
    data=json.dumps(samples,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    template=(ROOT/'geometry/scripts/dataset2_gt_review.html').read_text(encoding='utf-8')
    (OUT/'index.html').write_text(template.replace('/*SAMPLES*/[]',data), encoding='utf-8')
    assert all(sha(Path(e['path']))==e['sha256'] for e in inputs)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
