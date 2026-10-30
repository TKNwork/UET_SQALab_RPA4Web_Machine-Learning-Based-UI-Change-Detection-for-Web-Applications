"""Review all missed canonical geometry GT touched by changed predictions/groups."""
import csv
import gzip
import json
import os
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw
from geometry import pipeline
from geometry.scripts.analyze_dataset2_failures import metrics,passes
from geometry.scripts.detect_dataset2 import ROOT,save,sha
OUT=ROOT/'geometry/results/all_geometry_low_overlap_review_v1'
GT=ROOT.parent/'dataset1/thesis-dataset-main/groundtruth'

def main():
    OUT.mkdir(exist_ok=True);cases=[];summary={};protected=[]
    sources=[('first',ROOT/'geometry/results/group_coverage_v1',ROOT/'geometry/results'),('second',ROOT/'geometry/results/dataset2_same_gt_protocol_v1',ROOT/'geometry/results/dataset2_from502_detection_v1')]
    for dataset,evroot,cache in sources:
        ids=json.loads((ROOT/'geometry/results/matching_manifest.json').read_text(encoding='utf-8'))['pair_ids'] if dataset=='first' else [r['pair_id'] for r in json.loads((evroot/'data_manifest.json').read_text(encoding='utf-8'))['samples'] if r['geometry_count']]
        missed=0;failed=0;total_gt=0;dataset_cases=[]
        for pid in ids:
            evalpath=evroot/pid/'eval_edges8_union_support.json';group_path=(evroot if dataset=='first' else cache)/pid/'groups_edges8.json';p=cache/pid/('matching_result_full.json' if dataset=='first' else 'matching_result_full.json.gz');gp=GT/f'{int(pid):03d}.json'
            for path in (evalpath,group_path,p,gp):protected.append({'path':str(path),'sha256':sha(path)})
            ev=json.loads(evalpath.read_text(encoding='utf-8'));groups=json.loads(group_path.read_text(encoding='utf-8'));gt=json.loads(gp.read_text(encoding='utf-8'))
            total_gt+=ev['gt_count'];failed+=not ev['pass'];missed+=len(ev['missed_gt'])
            if ev['pass']:continue
            r=json.loads(p.read_text(encoding='utf-8')) if dataset=='first' else json.loads(gzip.open(p,'rt').read())
            changed=[m for m in r['matched_pairs'] if pipeline.classify_change(m['delta'],'edges8')]
            assert groups['selected_predictions']==len(changed)
            for cov in ev['gt_coverage']:
                if cov['covered']:continue
                g=gt['bboxes'][cov['original_gt_index']];assert 'geometry' in g['type'];b=[g['x'],g['y'],g['x']+g['w'],g['y']+g['h']]
                candidates=[]
                for i,m in enumerate(changed):
                    region=pipeline.prediction_envelope(m['before_bbox'],m['after_bbox']);value=metrics([region],b)
                    assert not passes(value)
                    if value['intersection']<=0:continue
                    candidates.append({'kind':'single','id':f'prediction_{i}','prediction_indices':[i],'before':[m['before_bbox']],'after':[m['after_bbox']],'regions':[region],'before_ids':[m['before_id']],'after_ids':[m['after_id']],'delta':m['delta'],**value})
                for group in groups['groups']:
                    value=metrics(group['member_regions'],b);assert not passes(value)
                    if value['intersection']<=0:continue
                    candidates.append({'kind':'group','id':group['group_id'],'group_kind':group['kind'],'prediction_indices':group['member_prediction_indices'],'before':[m['before_bbox'] for m in group['members']],'after':[m['after_bbox'] for m in group['members']],'regions':group['member_regions'],'before_ids':[m['before_id'] for m in group['members']],'after_ids':[m['after_id'] for m in group['members']],**value})
                if not candidates:continue
                candidates.sort(key=lambda c:(max(c['iou']/.3,c['ccr']/.5),c['iou']),reverse=True)
                paths={side:Path(r['image_paths'][side]) for side in ('before','after')}
                assert all(p.exists() for p in paths.values())
                sizes={}
                for side,path in paths.items():
                    with Image.open(path) as im:sizes[side]=list(im.size)
                case={'case_id':f'{pid}-GT{cov["original_gt_index"]+1}','dataset':dataset,'pair_id':pid,'original_gt_index':cov['original_gt_index'],'gt':g,'candidates':candidates,'before':os.path.relpath(paths['before'],OUT),'after':os.path.relpath(paths['after'],OUT),'image_sizes':sizes,'candidate_count':len(candidates)}
                cases.append(case);dataset_cases.append(case)
        summary[dataset]={'geometry_samples_evaluated':len(ids),'geometry_gt_evaluated':total_gt,'failed_samples_total':failed,'missed_gt_total':missed,'affected_samples':len({c['pair_id'] for c in dataset_cases}),'affected_gt':len(dataset_cases),'pair_ids':sorted({c['pair_id'] for c in dataset_cases},key=int)}
    summary['all']={k:summary['first'][k]+summary['second'][k] for k in ('geometry_samples_evaluated','geometry_gt_evaluated','failed_samples_total','missed_gt_total','affected_samples','affected_gt')}
    summary['scope']={'sample_ids':'000–500 and 502–1005','sample_count':1005,'excluded_id':'501 (no geometry in canonical GT)','images_per_sample':2,'gt_root':str(GT),'classifier':'edges8','mode':'union_support','condition':'GT missed; at least one changed matched-pair envelope or frozen group union has positive intersection; every such candidate has IoU<0.30 AND CCR<0.50','not_a_semantic_correctness_label':True}
    assert summary['all']['affected_samples']==63 and summary['all']['affected_gt']==79
    assert summary['second']['affected_gt']==46
    assert all(sha(Path(e['path']))==e['sha256'] for e in protected)
    for name,obj in [('summary.json',summary),('cases.json',cases),('validation.json',{'all_296_geometry_samples_scanned':True,'all_158_missed_gt_scanned':True,'unique_case_keys':len({c['case_id'] for c in cases}),'all_79_have_positive_intersection_and_no_passing_candidate':True,'source_files_verified_unchanged':len(protected)})]:save(OUT/name,obj)
    with (OUT/'cases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','pair_id','original_gt_index','selector','candidate_count','best_kind','best_iou','best_ccr']);writer.writeheader()
        for c in cases:
            p=c['candidates'][0];writer.writerow({'dataset':c['dataset'],'pair_id':c['pair_id'],'original_gt_index':c['original_gt_index'],'selector':c['gt']['selector'],'candidate_count':len(c['candidates']),'best_kind':p['kind'],'best_iou':p['iou'],'best_ccr':p['ccr']})
    template=(ROOT/'geometry/scripts/all_low_overlap_review.html').read_text(encoding='utf-8');(OUT/'index.html').write_text(template.replace('/*CASES*/[]',json.dumps(cases,ensure_ascii=False).replace('<','\\u003c')), encoding='utf-8')
    lines=['# Prediction thay đổi giao GT nhưng chưa đạt IoU/CCR — hai tập','','Phạm vi 1005 mẫu: ID 000–500 và 502–1005 (mỗi mẫu là một cặp ảnh). ID 501 ngoài phạm vi đã chọn và không có geometry theo GT chuẩn. Chỉ đánh giá geometry: 296 mẫu / 691 vùng GT; cùng GT chuẩn và evaluator edges8 + union_support.','','| Tập | Mẫu thuộc điều kiện | GT thuộc điều kiện | Tổng GT missed |','|---|---:|---:|---:|']
    for k,label in [('first','Tập đầu'),('second','Tập sau'),('all','Tổng')]:s=summary[k];lines.append(f"| {label} | {s['affected_samples']} | {s['affected_gt']} | {s['missed_gt_total']} |")
    lines+=['','Điều kiện: GT chưa được phủ; có envelope prediction thay đổi hoặc union nhóm giao GT >0, nhưng mọi ứng viên đều IoU<0,30 VÀ CCR<0,50. Một mẫu có thể có nhiều GT missed; chỉ cần một GT thuộc điều kiện là mẫu được đếm một lần. 79/158 GT missed = 50%.','','Giao GT không xác nhận correspondence đúng. Không tự động kết luận chỉ cần nới ngưỡng hoặc làm bbox lớn hơn. Viewer lưu mọi ứng viên giao GT, sắp theo mức gần đạt ngưỡng nhất: max(IoU/0,30, CCR/0,50). Nhóm được vẽ từng envelope thành viên; không tô bbox bao nhóm như thể có bằng chứng ở khoảng trống.','','## ID mẫu','','Tập đầu: '+', '.join(summary['first']['pair_ids']),'','Tập sau: '+', '.join(summary['second']['pair_ids']),'','## Dữ liệu và tái lập','','`cases.json`: GT gốc, ảnh, tất cả ứng viên giao GT, thành viên, ID và IoU/CCR. `cases.csv`: bảng gọn. `index.html`: viewer.','', '`python3 -m geometry.scripts.build_all_low_overlap_review`']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    contacts(cases)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def contacts(cases):
    for page,start in enumerate(range(0,len(cases),18),1):
        subset=cases[start:start+18];canvas=Image.new('RGB',(1500,230*((len(subset)+2)//3)),'#ededed');d=ImageDraw.Draw(canvas)
        for j,c in enumerate(subset):
            x=(j%3)*500;y=(j//3)*230;g=c['gt'];p=c['candidates'][0]
            d.text((x+6,y+4),f"{c['case_id']} | IoU {p['iou']:.2f} CCR {p['ccr']:.2f}",fill='black')
            crop=[int(g['x']-40),int(g['y']-40),int(g['x']+g['w']+40),int(g['y']+g['h']+40)]
            for k,side in enumerate(('before','after')):
                im=Image.open(OUT/c[side]).convert('RGB');draw=ImageDraw.Draw(im)
                for r in p[side]:draw.rectangle(r,outline='blue',width=3)
                for r in p['regions']:draw.rectangle(r,outline='orange',width=2)
                draw.rectangle([g['x'],g['y'],g['x']+g['w'],g['y']+g['h']],outline='red',width=3)
                im=im.crop(crop);im.thumbnail((242,195));canvas.paste(im,(x+4+k*248,y+28))
        canvas.save(OUT/f'overview_{page}.jpg',quality=88)
if __name__=='__main__':main()
