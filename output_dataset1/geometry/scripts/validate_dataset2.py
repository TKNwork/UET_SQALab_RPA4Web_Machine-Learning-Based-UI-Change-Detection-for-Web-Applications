"""Check final receipts, scoring accounting and immutable source hashes."""
import gzip
import json
from pathlib import Path
from collections import Counter
from geometry.scripts.detect_dataset2 import ROOT,save,sha
DET=ROOT/'dataset2_detection'
OUT=ROOT/'geometry/results/dataset2_from502_detection_v1'

def main():
    manifest=json.loads((DET/'manifest.json').read_text(encoding='utf-8'));issues=[];totals=Counter()
    for row in manifest['images']:
        totals[row['status']]+=1
        if row['status']!='complete':issues.append(row);continue
        p=Path(row['output']);receipt=DET/row['side']/'receipts'/f"{p.name.removesuffix('_full_hierarchy.json')}.json";rec=json.loads(receipt.read_text(encoding='utf-8'));raw=Path(row['ocr_raw'])
        assert sha(Path(row['image']))==rec['image_sha256']
        assert sha(p)==rec['output_sha256']
        assert sha(raw)==rec['ocr_raw_sha256']
        with gzip.open(raw,'rt') as f:ocr=json.load(f)
        assert not ocr['responses'][0].get('error')
        d=json.loads(p.read_text(encoding='utf-8'));assert d['total_elements']==len(d['elements'])
    protected=json.loads((ROOT/'geometry/results/dataset2_from502_v1/protected_inputs.json').read_text(encoding='utf-8'))
    changed=[r['path'] for r in protected if not Path(r['path']).exists() or sha(Path(r['path']))!=r['sha256']]
    assert not changed
    all_results=json.loads((OUT/'all_results.json').read_text(encoding='utf-8'))
    for row in all_results['samples']:
        assert int(row['pair_id'])>=502
        if not row['fully_resolved']:assert row.get('pass_') is not True
        if row['status']=='not_scored_unresolved_gt':assert row['gt_count']==0
        else:
            e=json.loads((OUT/row['pair_id']/'eval_edges8_union_support.json').read_text(encoding='utf-8'))
            assert e['gt_count']==row['gt_count']>0 and e['covered']==row['covered']
            assert len(e['missed_gt'])+e['covered']==e['gt_count']
    result={'protected_input_files':len(protected),'source_and_old_result_files_unchanged':not changed,'detector_images_requested':manifest['total_images'],'detector_status_counts':dict(totals),'detector_receipts_and_raw_ocr_verified':True,'geometry_accounting_verified':True,'partial_gt_never_counted_as_full_pass':True,'human_semantic_review_complete':False,'historical_detector_hash_available':False,'historical_pixel_diff_gt_equivalence_established':False}
    save(OUT/'validation.json',result)
    lines=['# Dataset2 detector','','Current dataset1 combine entrypoint reused without changes to inference/fusion functions. Raw OCR cached with gzip; annotated detector PNGs omitted to save disk (geometry viewer uses source images). Historical upstream version hash unavailable.','','```json',json.dumps(result,indent=2),'```','','Failed or blocked images:','','```json',json.dumps(issues,ensure_ascii=False,indent=2),'```','','Resume: `python3 -m geometry.scripts.detect_dataset2`','','Geometry report: `../geometry/results/dataset2_from502_detection_v1/REPORT.md`']
    (DET/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    final_manifest=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    final_manifest['driver_code_hashes']={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'geometry/scripts/detect_dataset2.py',ROOT/'geometry/scripts/normalize_dataset2_gt.py',ROOT/'geometry/scripts/evaluate_dataset2.py',ROOT/'geometry/scripts/validate_dataset2.py']}
    final_manifest['detector_config_sha256']=sha(DET/'config.json');save(OUT/'run_manifest.json',final_manifest)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
