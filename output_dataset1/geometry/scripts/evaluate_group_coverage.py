"""Two-phase experiment: build all groups without GT, then evaluate frozen groups."""
import csv
import hashlib
import json
from shared.paths import resolve_recorded_path
import sys
import time
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from geometry import pipeline
from geometry.prediction_groups import GROUP_CONFIG,build_prediction_groups

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'geometry/results'
OUT=SOURCE/'group_coverage_organized_v1'
MODES=('union_support','structural_proxy')

def main():
    start=time.perf_counter();ids=json.loads((SOURCE/'matching_manifest.json').read_text(encoding='utf-8'))['pair_ids']
    OUT.mkdir(exist_ok=True)
    (OUT/'group_config.json').write_text(json.dumps(GROUP_CONFIG,indent=2), encoding='utf-8')
    (OUT/'run_status.json').write_text(json.dumps({'status':'grouping_without_gt','requested_pairs':ids}), encoding='utf-8')
    frozen=[]
    # Phase 1 never opens GT files. Group artifacts are committed before scoring.
    for pid in ids:
        source=SOURCE/pid/'matching_result_full.json';r=json.loads(source.read_text(encoding='utf-8'))
        frozen.append({'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
        with Image.open(r['image_paths']['before']) as image:size=image.size
        dest=OUT/pid;dest.mkdir(exist_ok=True)
        for clf in pipeline.CLASSIFIERS:
            selected=[m for m in r['matched_pairs'] if pipeline.classify_change(m['delta'],clf)]
            groups=build_prediction_groups(selected,r['matched_pairs'],size)
            (dest/f'groups_{clf}.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2), encoding='utf-8')
    print(f'Groups frozen for {len(ids)} samples before GT evaluation',flush=True)
    (OUT/'run_status.json').write_text(json.dumps({'status':'evaluating_frozen_groups','requested_pairs':ids}), encoding='utf-8')
    rows=[];rescues=[];failures={mode:[] for mode in MODES}
    for pid in ids:
        r=json.loads((SOURCE/pid/'matching_result_full.json').read_text(encoding='utf-8'));dest=OUT/pid;gt=ROOT/'geometry/pairs'/pid/'groundtruth.json'
        frozen.append({'path':str(gt),'sha256':hashlib.sha256(gt.read_bytes()).hexdigest()})
        for clf in pipeline.CLASSIFIERS:
            groups=json.loads((dest/f'groups_{clf}.json').read_text(encoding='utf-8'))
            baseline=pipeline.evaluate(r['matched_pairs'],gt,clf)
            (dest/f'eval_{clf}_single_envelope.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2), encoding='utf-8')
            rows.append({'pair_id':pid,'classifier':clf,'mode':'single_envelope','gt_count':baseline['gt_count'],'covered':baseline['covered'],'pass':baseline['pass'],'predictions':baseline['predictions'],'groups':0,'group_rescued':0,'unverified':baseline['unverified']})
            for mode in MODES:
                ev=pipeline.evaluate_grouped(r['matched_pairs'],gt,clf,groups,mode)
                assert ev['covered']>=baseline['covered'] and 0<=ev['unverified']<=ev['predictions']
                (dest/f'eval_{clf}_{mode}.json').write_text(json.dumps(ev,ensure_ascii=False,indent=2), encoding='utf-8')
                rows.append({'pair_id':pid,'classifier':clf,'mode':mode,'gt_count':ev['gt_count'],'covered':ev['covered'],'pass':ev['pass'],'predictions':ev['predictions'],'groups':len(groups['groups']),'group_rescued':len(ev['group_rescued_gt']),'unverified':ev['unverified']})
                if clf=='edges8':
                    if not ev['pass']:failures[mode].append({'pair_id':pid,'covered':ev['covered'],'gt_count':ev['gt_count'],'missed_gt':ev['missed_gt']})
                    for gi in ev['group_rescued_gt']:
                        links=[l for l in ev['group_links'] if l['gt_index']==gi]
                        rescues.append({'pair_id':pid,'gt_index':gi,'mode':mode,'groups':[{**l,'group':next(g for g in groups['groups'] if g['group_id']==l['group_id'])} for l in links],'human_correct':None})
    with (OUT/'evaluation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for clf in pipeline.CLASSIFIERS:
        summary[clf]={}
        for mode in ('single_envelope',*MODES):
            rs=[r for r in rows if r['classifier']==clf and r['mode']==mode]
            summary[clf][mode]={k:sum(r[k] for r in rs) for k in ('gt_count','covered','predictions','groups','group_rescued','unverified')}
            summary[clf][mode].update(samples=len(rs),pass_samples=sum(r['pass'] for r in rs))
    for entry in frozen:assert hashlib.sha256(resolve_recorded_path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
    for name,obj in [('summary.json',summary),('rescued_gt_review.json',rescues),('failure_samples.json',failures)]:
        (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2), encoding='utf-8')
    manifest={'pair_ids':ids,'group_config':GROUP_CONFIG,'inputs':frozen,'input_hashes_verified':True,'grouping_reads_gt':False,'phases':['group_all_without_gt','evaluate_frozen_groups'],'matcher_reused':True,'elapsed_seconds':time.perf_counter()-start,'group_code_sha256':hashlib.sha256((ROOT/'geometry/prediction_groups.py').read_bytes()).hexdigest(),'evaluator_code_sha256':hashlib.sha256((ROOT/'geometry/pipeline.py').read_bytes()).hexdigest(),'summary':summary}
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2), encoding='utf-8')
    (OUT/'run_status.json').write_text(json.dumps({'status':'complete','completed_pairs':ids}), encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
