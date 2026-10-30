"""Generate current project inventory and Python symbol map; never reads GT semantics."""
import ast
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
DOCS=ROOT/'docs'
EXCLUDED={'node_modules','dist','__pycache__','.git'}
CORE={
 'geometry/match_geometry.py':('Entry point CLI','user','new_pipeline.main'),
 'geometry/pipeline.py':('Matcher/classifier/evaluator hiện hành','detection + images + config; GT chỉ evaluator','matching cache + coverage'),
 'geometry/prediction_groups.py':('GT-independent grouping + geometry union','changed matched pairs + matched parents + image size','groups'),
 'README.md':('Hướng dẫn chạy hiện hành','maintainer','user'),
 'geometry/scripts/evaluate_group_coverage.py':('Tạo nhóm toàn tập trước GT rồi chấm 3 mode','matching cache + GT sau grouping','group_coverage_v1'),
 'geometry/scripts/build_group_report.py':('Báo cáo + publish viewer + group review','group_coverage_v1 + matching cache','REPORT.md + public'),
 'geometry/scripts/build_gt_review.py':('Sinh bộ xem toàn bộ GT','groundtruth + matching_manifest + images','public/gt + inline GT'),
 'geometry/scripts/group_review_page.html':('Template trang group review','rescued GT/member JSON','public/group-review'),
 'geometry/scripts/gt_review_fragment.html':('Template bộ xem GT','GT samples/image JSON','public/gt + inline GT'),
 'geometry/scripts/check_gt_review.mjs':('Kiểm tra logic GT viewer bằng mock DOM','public/gt/index.html','PASS/error'),
 'scripts/build_project_inventory.py':('Sinh inventory/cây thư mục/symbol map','project paths + AST','docs'),
 'image_full_hierarchy.json':('Ví dụ detection upstream, không phải input CLI trực tiếp','upstream detector','manual reference'),
 'image_full_hierarchy.png':('Ảnh ví dụ upstream, không phải input CLI trực tiếp','upstream detector','manual reference'),
}

def classify(path):
    rel=path.relative_to(ROOT).as_posix();parts=path.relative_to(ROOT).parts
    if rel in CORE:return CORE[rel]
    if parts[0] in ('baseline','inject'):return ('Dữ liệu upstream Before/After, giữ nguyên','upstream detector','data provenance/reference')
    if parts[0]=='groundtruth':return ('GT gốc XYWH/type/selector/fallback','dataset generator','GT review / cross-check geometry_pairs')
    if rel.startswith('geometry/pairs/'):return ('Input subset geometry: detection/GT/manifest','dataset preparation','matcher/evaluator')
    if rel.startswith('geometry/study_first5/'):return ('Dữ liệu fixture 5 mẫu phát triển','subset copy','manual data checks')
    if rel.startswith('geometry/results/'):
        if len(parts)==4 and path.name=='matching_result_full.json':return ('Correspondence cache hiện hành','new_pipeline.process_pair','group evaluation / viewer')
        if path.name=='matching_manifest.json':return ('ID/provenance cache, không có summary chấm cũ','CLI/cache metadata','evaluate_group_coverage / GT review')
        name=path.name
        if name.startswith('groups_'):return ('Prediction groups/member provenance độc lập GT','build_prediction_groups','evaluate_grouped / group review')
        if name.startswith('eval_'):return ('Coverage detail hiện hành cho mode/classifier','evaluate / evaluate_grouped','report / viewer')
        return ('Tổng hợp/provenance/review run nhóm hiện hành','evaluate_group_coverage / build_group_report','user / viewer')
    if 'tests' in parts:return ('Test có ý nghĩa cho pipeline/nhóm/CLI','maintainer','unittest')
    if rel.startswith('content/results/'):return ('Content inference, evaluation, review and provenance','content.pipeline / reports','user / content viewer')
    if parts[0]=='content':return ('Content pipeline, scripts and docs','maintainer','content inference / evaluator')
    if parts[0]=='shared':return ('Shared normalization, assignment, rectangle union, path relocation','maintainer','geometry and content')
    if parts[0]=='docs':return ('Tài liệu/cấu trúc/cleanup audit','maintainer/inventory generator','user')
    if rel.startswith('geometry/report_viewer/public/data/geometry-v3/'):
        return ('Bản xuất viewer hiện hành, không dùng scoring','build_group_report','React viewer')
    if rel.startswith('geometry/report_viewer/public/gt/'):return ('GT viewer được sinh','build_gt_review','browser/manual review')
    if rel.startswith('geometry/report_viewer/public/group-review/'):return ('Group review được sinh','build_group_report','browser/manual review')
    if rel.startswith('geometry/report_viewer/src/'):return ('Mã/kiểu/style UI hiện hành','maintainer','Vite/React')
    if rel.startswith('geometry/report_viewer/'):return ('Cấu hình/dependency/static asset viewer','maintainer/npm','Vite/React')
    return ('Project file','maintainer','project')

def main():
    DOCS.mkdir(exist_ok=True)
    code={}
    for path in sorted(ROOT.rglob('*.py')):
        if any(part in EXCLUDED for part in path.relative_to(ROOT).parts) or 'migrations' in path.relative_to(ROOT).parts:continue
        source=path.read_text(encoding='utf-8');tree=ast.parse(source)
        imports=[ast.get_source_segment(source,n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
        symbols=[]
        for n in ast.walk(tree):
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                args=[a.arg for a in (*n.args.posonlyargs,*n.args.args,*n.args.kwonlyargs)]
                symbols.append({'kind':'function','name':n.name,'line':n.lineno,'arguments':args,'returns':ast.unparse(n.returns) if n.returns else None,'doc':ast.get_docstring(n)})
            elif isinstance(n,ast.ClassDef):symbols.append({'kind':'class','name':n.name,'line':n.lineno,'bases':[ast.unparse(b) for b in n.bases],'doc':ast.get_docstring(n)})
        code[path.relative_to(ROOT).as_posix()]={'imports':imports,'symbols':sorted(symbols,key=lambda s:s['line'])}
    (DOCS/'code_structure.json').write_text(json.dumps(code,ensure_ascii=False,indent=2), encoding='utf-8')
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and not any(part in EXCLUDED for part in p.relative_to(ROOT).parts)]
    rows=[]
    for p in files:
        role,producer,consumer=classify(p)
        rows.append({'path':p.relative_to(ROOT).as_posix(),'extension':p.suffix,'bytes':p.stat().st_size,'role':role,'producer':producer,'consumer':consumer})
    with (DOCS/'file_inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','extension','bytes','role','producer','consumer']);w.writeheader();w.writerows(rows)
    lines=['Current project file tree (node_modules/dist/__pycache__/.git excluded):']
    for row in rows:lines.append(row['path'])
    (DOCS/'DIRECTORY_TREE.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Inventory files:',len(rows),'Python modules:',len(code))

if __name__=='__main__':main()
