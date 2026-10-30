"""Review-only catalog of ALL metadata content samples; never used by inference/scoring.

Uses saved HTML/DOM snapshots and selectors to crop declared targets. If no target
bbox can be resolved, uses screenshot-difference context, explicitly marked.
No new GT or predictions are created.
"""
import argparse
import base64
import csv
import hashlib
import json
import re
from dataclasses import dataclass,field
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from PIL import Image,ImageChops,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
VOID={'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}

@dataclass(eq=False)
class Element:
    tag:str
    attrs:dict
    parent:object=None
    children:list=field(default_factory=list)
    text:list=field(default_factory=list)

class SavedDOM(HTMLParser):
    def __init__(self,html):
        super().__init__(convert_charrefs=True);self.root=Element('document',{});self.stack=[self.root];self.nodes=[];self.feed(html)
    def handle_starttag(self,tag,attrs):
        node=Element(tag,dict(attrs),self.stack[-1]);node.parent.children.append(node);node.parent.text.append(node);self.nodes.append(node)
        if tag not in VOID:self.stack.append(node)
    def handle_startendtag(self,tag,attrs):
        self.handle_starttag(tag,attrs)
        if self.stack[-1].tag==tag:self.stack.pop()
    def handle_endtag(self,tag):
        for i in range(len(self.stack)-1,0,-1):
            if self.stack[i].tag==tag:self.stack=self.stack[:i];break
    def handle_data(self,data):self.stack[-1].text.append(data)

def tokens(selector):
    selector=selector.strip();result=[];buf='';depth=0;quote=None;relation=None
    for ch in selector:
        if quote:
            buf+=ch
            if ch==quote:quote=None
            continue
        if ch in ['"',"'"] and depth:quote=ch;buf+=ch;continue
        if ch in '[(':depth+=1
        if ch in '])':depth-=1
        if depth==0 and (ch.isspace() or ch=='>'):
            if buf:result.append((buf,relation));buf='';relation=' '
            if ch=='>':relation='>'
        else:buf+=ch
    if buf:result.append((buf,relation))
    return result

def simple_match(node,part):
    if part=='*':return True
    for name,op,value in re.findall(r'\[([\w-]+)\s*(\*=|=)?\s*([^\]]*)\]',part):
        if name not in node.attrs:return False
        value=value.strip().strip('"\'');actual=node.attrs.get(name) or ''
        if op=='=' and actual!=value:return False
        if op=='*=' and value not in actual:return False
    part=re.sub(r'\[[^\]]*\]','',part)
    for pseudo,n in re.findall(r':(nth-of-type|nth-child)\((\d+)\)',part):
        siblings=node.parent.children if node.parent else []
        if pseudo=='nth-of-type':siblings=[s for s in siblings if s.tag==node.tag]
        if node not in siblings or siblings.index(node)+1!=int(n):return False
    part=re.sub(r':(?:nth-of-type|nth-child)\(\d+\)','',part)
    for pseudo in ['first-child','last-child','first-of-type','last-of-type']:
        if ':'+pseudo in part:
            siblings=node.parent.children if node.parent else []
            if pseudo.endswith('of-type'):siblings=[s for s in siblings if s.tag==node.tag]
            if not siblings or node is not siblings[-1 if pseudo.startswith('last') else 0]:return False
            part=part.replace(':'+pseudo,'')
    if any(ch in part for ch in [':','[',']','+','~',',','\\']):return False
    tag=re.match(r'^[\w*-]+',part)
    if tag and tag.group() not in ['*',node.tag]:return False
    ids=re.findall(r'#([\w-]+)',part)
    if ids and node.attrs.get('id') not in ids:return False
    classes=set((node.attrs.get('class') or '').split())
    return all(c in classes for c in re.findall(r'\.([\w-]+)',part))

def css_match(node,selector):
    parts=tokens(selector)
    def at(current,index):
        if current is None or not simple_match(current,parts[index][0]):return False
        if index==0:return True
        relation=parts[index][1]
        if relation=='>':return at(current.parent,index-1)
        parent=current.parent
        while parent:
            if at(parent,index-1):return True
            parent=parent.parent
        return False
    return bool(parts) and at(node,len(parts)-1)

def visible_text(node):
    if node.tag in ['script','style','noscript']:return ''
    return re.sub(r'\s+',' ',''.join(visible_text(t) if isinstance(t,Element) else t for t in node.text)).strip()

def snapshot_targets(folder,side,selector):
    dom=SavedDOM((folder/f'page_{side}.html').read_text(errors='replace', encoding='utf-8'))
    records=json.loads((folder/f'elements_{side}.json').read_text(encoding='utf-8'));by_tag={}
    for e in records:by_tag.setdefault(e.get('tag'),[]).append(e)
    selected=[n for n in dom.nodes if css_match(n,selector)];result=[]
    for n in selected:
        possible=[]
        for e in by_tag.get(n.tag,[]):
            s=e.get('selector')
            if s and css_match(n,s):possible.append(e)
        # Multiple nodes can share a nonunique snapshot selector. Do not invent
        # which bbox belongs to this node: preserve only unique resolved entries.
        exact=[e for e in possible if e.get('id') and e.get('id')==n.attrs.get('id')]
        entries=exact or possible
        unique={json.dumps(e.get('bbox'),sort_keys=True):e for e in entries if e.get('bbox')}
        bbox=None;record_selector=None
        if len(unique)==1:
            entry=next(iter(unique.values()));b=entry['bbox'];bbox=[b['x'],b['y'],b['x']+b['width'],b['y']+b['height']];record_selector=entry.get('selector')
        attrs={key:n.attrs[key] for key in ['placeholder','value','alt','src','title'] if key in n.attrs}
        result.append({'tag':n.tag,'text':visible_text(n),'attributes':attrs,'bbox':bbox,'snapshot_selector':record_selector,
                       'bbox_status':'resolved' if bbox else 'missing_or_ambiguous_snapshot_selector'})
    return result

def signature(t):return (t['text'],json.dumps(t['attributes'],sort_keys=True))
def enclosure(boxes):return [min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inline-path',type=Path);args=ap.parse_args()
    out=ROOT/'content/results/catalog_97';out.mkdir(parents=True,exist_ok=True);samples=[];sources=[];all_folders=[]
    for folder in sorted((ROOT.parent/'dataset1').iterdir()):
        mp=folder/'metadata.json'
        if not folder.is_dir() or not mp.is_file():continue
        all_folders.append(folder.name);meta=json.loads(mp.read_text(encoding='utf-8'));changes=[c for c in meta.get('changes',[]) if 'content' in c.get('type',[])]
        if not changes:continue
        pid=folder.name.split('_')[0];operations=[]
        for change in changes:
            sels=change['target_selector'];sels=sels if isinstance(sels,list) else [sels]
            for selector in sels:
                b=snapshot_targets(folder,'before',selector.strip());a=snapshot_targets(folder,'after',selector.strip())
                changed=[]
                for i in range(max(len(b),len(a))):
                    x=b[i] if i<len(b) else None;y=a[i] if i<len(a) else None
                    if x is None or y is None or signature(x)!=signature(y):changed.append(i)
                operations.append({'selector':selector.strip(),'description':change['description'],'labels':change['type'],'before':b,'after':a,'changed_target_indices':changed,
                                   'association':'same selector result order for review only; not inferred correspondence'})
        images={side:Image.open(folder/name).convert('RGB') for side,name in [('before','baseline.png'),('after','inject.png')]}
        # Highlight changed selector instances where resolved; otherwise declared
        # instances, then screenshot difference (which may include other labels).
        rois={};kinds={}
        for side in ['before','after']:
            boxes=[]
            for op in operations:
                ids=op['changed_target_indices'] or list(range(len(op[side])))
                boxes.extend(op[side][i]['bbox'] for i in ids if i<len(op[side]) and op[side][i]['bbox'])
            im=images[side];visible=[b for b in boxes if b[2]>0 and b[3]>0 and b[0]<im.width and b[1]<im.height and b[2]>b[0] and b[3]>b[1]]
            if visible:
                b=enclosure(visible);pad=24;rois[side]=[max(0,b[0]-pad),max(0,b[1]-pad),min(im.width,b[2]+pad),min(im.height,b[3]+pad)];kinds[side]='DOM target crop'
            else:
                if images['before'].size==images['after'].size:
                    delta=ImageChops.difference(images['before'],images['after']).convert('L').point(lambda x:255 if x>35 else 0).getbbox()
                else:delta=None
                b=delta or [0,0,im.width,im.height];rois[side]=[max(0,b[0]-16),max(0,b[1]-16),min(im.width,b[2]+16),min(im.height,b[3]+16)];kinds[side]='Screenshot diff context; target bbox unresolved'
        crops={}
        dest=out/pid;dest.mkdir(exist_ok=True)
        for side in ['before','after']:
            crop=images[side].crop(tuple(rois[side]));crop.thumbnail((520,240));crop.save(dest/f'{side}.webp',quality=60);crops[side]=crop
        gt=ROOT/'groundtruth'/f'{pid}.json';gt_data=json.loads(gt.read_text(encoding='utf-8')) if gt.exists() else None
        gt_count=sum('content' in g['type'] for g in gt_data['bboxes']) if gt_data else 0
        evaluation=ROOT/'content/results/organized_v1'/pid/'evaluation.json'
        ev=json.loads(evaluation.read_text(encoding='utf-8')) if evaluation.exists() else None
        sample={'id':pid,'name':folder.name,'domain':meta.get('environment',{}).get('url',''),'operations':operations,'other_labels':sorted({label for c in meta.get('changes',[]) for label in c.get('type',[]) if label!='content'}),
                'gt_count':gt_count,'evaluated':bool(ev),'covered':ev['covered'] if ev else None,'roi':rois,'roi_kind':kinds,
                'images':{side:str((dest/f'{side}.webp').relative_to(out)) for side in ['before','after']},'source_images':{side:str((folder/name).resolve()) for side,name in [('before','baseline.png'),('after','inject.png')]}}
        samples.append(sample)
        for name in ['metadata.json','page_before.html','page_after.html','elements_before.json','elements_after.json','baseline.png','inject.png']:
            p=folder/name;sources.append({'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest={'dataset_samples':len(all_folders),'content_samples':len(samples),'evaluated_samples':sum(s['evaluated'] for s in samples),'samples_without_existing_content_gt':sum(s['gt_count']==0 for s in samples),
              'purpose':'metadata/DOM/screenshot review catalog, NOT predictions or new GT','samples':samples,'sources':sources}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    with (out/'samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','name','description','selector','gt_count','evaluated','other_labels','before_roi_kind','after_roi_kind']);w.writeheader()
        for s in samples:w.writerow({'id':s['id'],'name':s['name'],'description':' | '.join(o['description'] for o in s['operations']),'selector':' | '.join(o['selector'] for o in s['operations']),'gt_count':s['gt_count'],'evaluated':s['evaluated'],'other_labels':','.join(s['other_labels']),'before_roi_kind':s['roi_kind']['before'],'after_roi_kind':s['roi_kind']['after']})
    template=(ROOT/'content/scripts/content_catalog_fragment.html').read_text(encoding='utf-8')
    if args.inline_path:
        inline=json.loads(json.dumps(samples))
        for width,quality in [(440,40),(380,30),(320,25),(280,20)]:
            for s in inline:
                for side in ['before','after']:
                    im=Image.open(out/s['images'][side]);im.thumbnail((width,220));buf=BytesIO();im.save(buf,format='WEBP',quality=quality)
                    s['images'][side]='data:image/webp;base64,'+base64.b64encode(buf.getvalue()).decode()
            fragment=template.replace('__CONTENT_CATALOG_DATA__',json.dumps(inline,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
            if len(fragment.encode())<990000:break
            inline=json.loads(json.dumps(samples))
        if len(fragment.encode())>=1000000:raise RuntimeError('Inline catalog exceeds 1 MB')
        args.inline_path.write_text(fragment, encoding='utf-8')
        print('Inline bytes:',len(fragment.encode()))
    # Project copy keeps higher-quality local image files, with the same review UI.
    payload=json.dumps(samples,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    standalone=template.replace('__CONTENT_CATALOG_DATA__',payload)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>97 content changes</title><style>:root{--foreground:#192632;--background:#fff;--border:#ccd4dd;--muted-foreground:#52616e;--primary:#243e54;--primary-foreground:#fff}body{font:15px system-ui;max-width:1000px;margin:24px auto;padding:16px}.btn{padding:8px;border:1px solid var(--border);background:transparent;color:inherit;border-radius:6px;cursor:pointer}.btn[aria-pressed="true"]{background:var(--primary);color:var(--primary-foreground)}.text-small{font-size:12px}.text-muted{color:var(--muted-foreground)}</style>'+standalone, encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['samples','sources']},ensure_ascii=False))

if __name__=='__main__':main()
