# Chạy từng nghiên cứu

Các ví dụ PowerShell bên dưới chạy từ root repository. Ubuntu dùng `.venv/bin/python` thay `.\.venv\Scripts\python.exe`; không cần đổi các argument. Python 3.12 được kiểm chứng. Matching dùng CPU; không cần cài PyTorch, Gemini SDK hoặc Node.js.

## Geometry: dữ liệu → correspondence → change → groups → scoring

Chuẩn bị dependency `requirements/matching.txt` và khôi phục gói mẫu theo [ASSETS.md](ASSETS.md). Cặp input phải có `before.json`, `after.json`, `groundtruth.json`; ảnh lấy từ `dataset1/<ID>_*`. GT chỉ dùng selection/scoring, không đưa vào matching.

```powershell
.\.venv\Scripts\python.exe run.py geometry --help
.\.venv\Scripts\python.exe run.py geometry --pair-ids 000 --output-root outputs/geometry-first-run
```

Xem JSON, báo cáo, hình và trạng thái trong `outputs/geometry-first-run`. Dùng tên output mới cho mỗi lần nghiên cứu, giữ kết quả cũ. Khi dùng input ngoài repository, đổi `geometry_pairs` và `dataset1` trong `config.local.json`, hoặc truyền `--input-root` và `--image-root` trực tiếp.

`--all-geometry` chỉ chạy sau khi đã restore đầy đủ ảnh/cặp; gói mẫu không đủ tài nguyên toàn tập. `--ablations` là thí nghiệm bổ sung, không chạy mặc định.

Các script `geometry/scripts/evaluate_dataset2_*`, `experiment_hierarchical_geometry`, `audit_coarse_gt_events` và báo cáo group là những nghiên cứu riêng. Chúng phụ thuộc dataset2, GT canonical/native, cache và run cụ thể; xem README geometry và hằng số ROOT/DATA/GT/SOURCE/OUT của script trước khi dùng. Không dùng các lệnh chấm cache lịch sử như một quick start từ source clone sạch. Schema/metric/cohort giữa các nghiên cứu khác nhau, không thay thế cho nhau.

## Content: survey → dev → review → freeze → all → report

Chuẩn bị original GT, rồi đặt `groundtruth` trong `config.local.json` thành `artifacts/content_groundtruth` hoặc dùng biến môi trường dưới đây. Nên giữ setting này cho toàn bộ phiên làm việc.

```powershell
.\.venv\Scripts\python.exe tools/prepare_content_gt.py
$env:RPA_GROUNDTRUTH = "artifacts/content_groundtruth"
.\.venv\Scripts\python.exe run.py content survey --output-root output_dataset1/content/results/windows-first-run
.\.venv\Scripts\python.exe run.py content run --split dev --output-root output_dataset1/content/results/windows-first-run
.\.venv\Scripts\python.exe run.py content report --output-root output_dataset1/content/results/windows-first-run
```

Trên Ubuntu, biến môi trường tương ứng là:

```bash
export RPA_GROUNDTRUTH=artifacts/content_groundtruth
```

Mở `output_dataset1/content/results/windows-first-run/index.html` trong browser; từng mẫu có `review.html`, `inference.json`, `evaluation.json` và ảnh. Review các mẫu dev, ghi nhận nhận xét vào `development_review.md` trong run; sau đó mới chạy:

```powershell
.\.venv\Scripts\python.exe run.py content freeze --output-root output_dataset1/content/results/windows-first-run
.\.venv\Scripts\python.exe run.py content run --split all --output-root output_dataset1/content/results/windows-first-run
.\.venv\Scripts\python.exe run.py content report --output-root output_dataset1/content/results/windows-first-run
```

Freeze kiểm tra hash code/config và bản review, không nên bỏ qua. Code đã thay đổi để hỗ trợ portability sẽ không khớp hash run lịch sử; tạo run mới. Không dùng `run.py` để tự động freeze khi chưa review. Chỉ có 6 mẫu/14 GT ở giao thức này; dev có 5 mẫu, phần còn lại 1 mẫu. Coverage không phải precision/F1 hay kiểm thử độc lập.

Catalog DOM đầy đủ (`content/scripts/build_content_catalog.py`) cần HTML/metadata và toàn bộ dataset, không nằm trong gói mẫu.

## Perception: screenshot → OCR + YOLO + CV → hierarchy

Cài thêm dependency khi thực sự cần detector. Với CPU, trong môi trường Python đã tạo:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements/perception.txt
.\.venv\Scripts\python.exe run.py doctor --flow perception
```

Lệnh CPU là cấu hình khởi đầu. Nếu dùng GPU, chọn cặp torch/torchvision và CUDA phù hợp máy theo hướng dẫn PyTorch trước khi cài perception; không copy virtualenv Ubuntu sang Windows. Dependency perception đã ghi theo runtime local; chưa kiểm tra cài mới và inference trên Windows, vì vậy chưa có lock transitive hoặc cam kết tái lập byte-for-byte cho detector.

Copy `yl26best.pt` vào vị trí chuẩn hoặc đặt `yolo_model` trong config local. Thiết lập Google Cloud Vision API key trong shell của bạn. Không dán key vào README/script.

```powershell
$env:GOOGLE_VISION_API_KEY = "YOUR_OWN_KEY"
.\.venv\Scripts\python.exe run.py perception --image dataset1/000_wikipedia_org/baseline.png --output outputs/perception-first-run --visualize
```

Lệnh inference gọi Google Vision, có thể phát sinh phí và gửi ảnh tới dịch vụ đó. Nó không thuộc smoke test offline. Output có hierarchy JSON và ảnh debug nếu bật visualize. Không cần Gemini key cho perception cơ bản.

`run_dataset1_combine_pipeline.py` và `run_full_dataset_pipeline.py` là batch runners nghiên cứu, không chạy tự động khi cài. Dataset1 batch hỗ trợ `RPA_DATASET1`, `RPA_YOLO_MODEL`, `RPA_DETECTION_OUTPUT`; output mặc định vẫn giữ layout cũ. Full batch còn phụ thuộc bộ pixelweb và labels trong `data/yolo_pixelweb_dataset`. Các `combine/<strategy>/run_pipeline.py` giữ vai trò so sánh phương pháp; không xóa chỉ vì không được launcher gọi.

## Detection mAP / COCO

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/evaluation.txt
.\.venv\Scripts\python.exe -X utf8 "OCR + Yolo26/eval_groundtruth/run_evaluation.py"
```

Lệnh dùng labels pixelweb theo loader hiện có và hierarchy JSON tại `combine/full_pipeline/results/json`, ghi `eval_groundtruth/results/evaluation_summary.json`. Cần khôi phục inputs đầy đủ trước; script không có tùy chọn giới hạn một mẫu, không dùng làm smoke. `pycocotools` là dependency tùy chọn, chưa kiểm chứng cài đặt Windows trong lần chuẩn bị này.

## WebClick và Gemini grounding

Luồng độc lập: WebClick screenshots/instructions → perception → `build_gemini_inputs.py` → manifest/ảnh annotate → `run_gemini_test.py` hoặc flash-only → prediction JSON. Không thay model nghiên cứu mặc định; khả năng truy cập model phải kiểm tra bằng tài khoản của bạn. Manifest và đường dẫn input/output cụ thể xem trong script.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/gemini.txt
.\.venv\Scripts\python.exe run.py doctor --flow gemini
Set-Location "OCR + Yolo26"
..\.venv\Scripts\python.exe -X utf8 scripts/run_gemini_test.py --help
# Sau khi đã restore/chuẩn bị webclick_vlm_input/manifest.jsonl:
$env:GEMINI_API_KEY = "YOUR_OWN_KEY"
..\.venv\Scripts\python.exe -X utf8 scripts/run_gemini_test.py --start_id 26 --end_id 26 --limit 1
Set-Location ..
```

Nếu dùng proxy đã được cấp, đặt `GEMINI_BASE_URL` trong shell; không commit địa chỉ có credentials. `verify_runtime_and_single_sample.py` có gọi API thật, không phải doctor offline. Flash-only là baseline nghiên cứu riêng và không tự được chạy trong quy trình cài.

Chỉ khi muốn tải WebClick parquet mới cài `requirements/webclick.txt` và chạy `download_webclick.py` từ thư mục `OCR + Yolo26`. Script tải nhiều phần và không có chế độ mẫu nhỏ; chưa chạy tải trong phiên chuẩn bị.

## Viewer geometry tùy chọn

Cài Node.js **24**, rồi:

```powershell
Set-Location output_dataset1/geometry/report_viewer
npm ci
npm run build
npm run dev -- --host 127.0.0.1
```

Mở địa chỉ localhost do Vite in ra. `npm ci` dùng package-lock đã có; không copy node_modules Ubuntu sang Windows. Source clone không có `public/data`, `/gt`, `/group-review`; phải restore bản copy viewer từ Ubuntu, hoặc tạo một run mới và công cụ publish tương ứng. Viewer rỗng sẽ thiếu manifest, không có kết quả nghiên cứu sẵn. Các publisher lịch sử kiểm tra hash và tên run cố định; xem README viewer trước khi tái tạo, không sửa hash để vượt kiểm tra.
