"""Reproducible GT/detector and matching contact sheets. Review only; never inference."""
import argparse
import json
import sys
from pathlib import Path
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from content.pipeline import ROOT,read
from shared.regions import area,intersect


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-root',type=Path,default=ROOT/'content/results/organized_v1');ap.add_argument('--split',choices=['dev','all'],default='dev');args=ap.parse_args();out=args.output_root.resolve()
    if not out.is_relative_to(ROOT/'content/results'):raise ValueError('Unsafe output directory')
    manifest=read(out/'data_manifest.json');target=out/'visual_examples';target.mkdir(exist_ok=True)
    for s in manifest['samples']:
        if args.split=='dev' and s['split']!='dev':continue
        pid=s['pair_id'];gs=[(i,g) for i,g in enumerate(read(s['sources']['groundtruth'])['bboxes']) if 'content' in g['type']];rows=[]
        r=read(out/pid/'inference.json') if (out/pid/'inference.json').exists() else None
        for gi,g in gs:
            bb=[g['x'],g['y'],g['x']+g['w'],g['y']+g['h']];pad=30
            region=[max(0,bb[0]-pad),max(0,bb[1]-pad),bb[2]+pad,bb[3]+pad]
            row=Image.new('RGB',(1400,270),'#f3f4f5');d=ImageDraw.Draw(row)
            d.text((8,5),f"{pid} GT original_index={gi} {g['selector']} {g['type']} fallback={g.get('fallback',False)}",fill='black')
            for j,side in enumerate(['before','after']):
                im=Image.open(s['sources'][side+'_image']).convert('RGB');det=read(s['sources'][side])['elements'];draw=ImageDraw.Draw(im)
                for e in det:
                    if area(intersect(e['bbox'],region))>0:
                        draw.rectangle(e['bbox'],outline='#999999',width=1);draw.text((e['bbox'][0],e['bbox'][1]),e['id'].replace('fused_',''),fill='#676767')
                draw.rectangle(bb,outline='#e02020',width=2)
                crop=im.crop(tuple(region));crop.thumbnail((690,220));row.paste(crop,(700*j+5,35))
            rows.append(row)
        canvas=Image.new('RGB',(1400,270*len(rows)),'white')
        for i,row in enumerate(rows):canvas.paste(row,(0,270*i))
        canvas.save(target/f'{pid}_gt_detectors.jpg')
        if not r:continue
        pairs=[]
        for state in ['matched_pairs','uncertain_pairs']:
            for m in r[state]:
                if any(any(area(intersect(m[side+'_bbox'],[g['x'],g['y'],g['x']+g['w'],g['y']+g['h']]))/max(area(m[side+'_bbox']),1)>.5 for side in ['before','after']) for _,g in gs):pairs.append((state,m))
        for offset in range(0,len(pairs),8):
            sheet=Image.new('RGB',(1400,160*len(pairs[offset:offset+8])),'#f3f4f5');draw=ImageDraw.Draw(sheet)
            for i,(state,m) in enumerate(pairs[offset:offset+8]):
                draw.text((5,i*160+4),f"{state}: {m['before_id']} -> {m['after_id']}; content={m['content']['state']} {m['content']['kind']}; gap={m['evidence']['cost_gap_ban']}",fill='black')
                for j,side in enumerate(['before','after']):
                    im=Image.open(s['sources'][side+'_image']).convert('RGB');b=m[side+'_bbox'];dr=ImageDraw.Draw(im);dr.rectangle(b,outline='green' if state=='matched_pairs' else 'orange',width=1)
                    crop=im.crop((max(0,b[0]-8),max(0,b[1]-8),min(im.width,b[2]+8),min(im.height,b[3]+8)));crop.thumbnail((690,130));sheet.paste(crop,(j*700,i*160+25))
            sheet.save(target/f'{pid}_matching_{offset//8}.jpg')
    print(target)

if __name__=='__main__':main()
