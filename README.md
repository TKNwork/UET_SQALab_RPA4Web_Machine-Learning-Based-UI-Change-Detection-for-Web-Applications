# Machine Learning Based UI Change Detection for Web Applications

Project nghiên cứu nhận diện thành phần giao diện và phát hiện thay đổi trước/sau trên web. Mã nguồn chạy bằng Python; viewer geometry dùng React/Vite tùy chọn.

Repository đích: https://github.com/TKNwork/UET_SQALab_RPA4Web_Machine-Learning-Based-UI-Change-Detection-for-Web-Applications

## Bắt đầu nhanh trên Windows (PowerShell)

Cài Git và Python **3.12 x64** trước. Dùng thư mục clone ngắn, ví dụ `C:\src`, để giảm nguy cơ đường dẫn dài. Các lệnh dưới đây không cần activate virtualenv hay thay ExecutionPolicy.

```powershell
git clone https://github.com/TKNwork/UET_SQALab_RPA4Web_Machine-Learning-Based-UI-Change-Detection-for-Web-Applications.git RPA4Web
Set-Location RPA4Web
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements/matching.txt
.\.venv\Scripts\python.exe run.py doctor
.\.venv\Scripts\python.exe run.py smoke
.\.venv\Scripts\python.exe run.py test
```

Smoke test dùng dữ liệu tổng hợp nhỏ, không cần dataset, weights, GPU hoặc API key. Doctor kiểm tra thư viện và báo riêng tài nguyên còn thiếu; thêm `--require-assets` nếu muốn tài nguyên thiếu làm lệnh thất bại. Việc chạy trên dữ liệu nghiên cứu cần khôi phục tài nguyên theo [ASSETS.md](docs/ASSETS.md).

Repository đã được truy cập bằng Git ngày 30/10/2026 và xác nhận trống trước khi chuẩn bị commit đầu tiên. Nếu Git báo không tìm thấy repo trên máy khác, kiểm tra quyền truy cập/đăng nhập trước.

## Ubuntu

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/matching.txt
.venv/bin/python run.py doctor
.venv/bin/python run.py smoke
.venv/bin/python run.py test
```

Nếu bản Ubuntu chưa có module venv, cài gói `python3.12-venv` bằng package manager. Các lệnh nghiên cứu bên dưới giống Windows khi thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python`.

## Cấu trúc

```text
run.py                         Lệnh chạy chung, đường dẫn tính từ root repository
requirements/                  Dependency riêng theo nhu cầu
tools/                         Doctor, smoke, đóng gói source và tài nguyên mẫu
docs/                          Hướng dẫn vận hành, tài nguyên và báo cáo chuẩn bị
OCR + Yolo26/
  run_perception.py            Ảnh → OCR/YOLO/CV → hierarchy JSON
  combine/, cv/, ocr/, yolo/   Detector và các nghiên cứu kết hợp
  gemini/, scripts/            WebClick/VLM grounding và công cụ phụ trợ
  eval_groundtruth/            COCO/mAP của detection
output_dataset1/
  geometry/                    Matching, geometry evidence, grouping và đánh giá
  content/                     Matching và bằng chứng thay đổi nội dung
  shared/                      Chuẩn hóa, assignment, vùng và đường dẫn dùng chung
  geometry/report_viewer/      Viewer React tùy chọn
  docs/                        Tài liệu thiết kế nghiên cứu trước đây
prepare_geometry_pairs.py      Tiện ích tạo cặp detection; không tự tạo GT
```

Dataset, model, results, các bản snapshot lịch sử và môi trường local được giữ ngoài gói mã nguồn. Không coi file bị `.gitignore` là file đã bị xóa. Xem [báo cáo chuẩn bị](docs/PREPARATION_REPORT.md).

## Chọn luồng nghiên cứu

| Luồng | Dependency | Input | Lệnh chính | Output |
|---|---|---|---|---|
| Geometry | `matching.txt` | Cặp JSON, ảnh before/after, GT để chấm | `run.py geometry` | Thư mục `--output-root` riêng |
| Content | `matching.txt` | Detection, ảnh, bộ GT content gốc | `run.py content` | `output_dataset1/content/results/<run>` |
| Perception | `perception.txt` + PyTorch | Screenshot, weights YOLO, Cloud Vision key | `run.py perception` | Thư mục `--output` |
| Detection mAP | `evaluation.txt` | Detection + nhãn dataset tương ứng | `eval_groundtruth/run_evaluation.py` | `eval_groundtruth/results` |
| WebClick/Gemini | `gemini.txt`, thêm `webclick.txt` nếu tải parquet | Screenshot, instruction, perception và API key | Script trong `OCR + Yolo26/scripts` | Theo từng script |
| Geometry viewer | Node.js 24 + `npm ci` | Bản copy kết quả trong `public/` | `npm run dev` | Web trên máy local |

Hướng dẫn input, output, lệnh chạy và giới hạn từng luồng: [WORKFLOWS.md](docs/WORKFLOWS.md). Windows checklist: [WINDOWS.md](docs/WINDOWS.md).

## Cấu hình đường dẫn

```powershell
Copy-Item config.example.json config.local.json
```

Sửa `config.local.json` nếu tài nguyên nằm ngoài repository. Dùng `/` trong JSON, ví dụ `D:/RPA-data/dataset1`. Đường dẫn tương đối luôn tính từ repository root khi chạy qua `run.py`; đường dẫn tuyệt đối được giữ nguyên. `--config` đặt trước tên luồng:

```powershell
.\.venv\Scripts\python.exe run.py --config config.local.json doctor --require-assets
```

Cấu hình local có ưu tiên cao nhất, sau đó biến môi trường `RPA_DATASET1`, `RPA_BASELINE`, `RPA_INJECT`, `RPA_GROUNDTRUTH`, `RPA_YOLO_MODEL`, cuối cùng là mặc định. Các script nghiên cứu lịch sử chạy trực tiếp vẫn có thể yêu cầu layout gốc; launcher bao phủ các entrypoint chính, không viết lại các artifact lịch sử.

Key dùng biến môi trường theo `.env.example`; file `.env.example` chỉ mô tả và không tự được load. Không commit key thật, proxy cá nhân hoặc `config.local.json`.

## Chuẩn bị để review/publish

```powershell
.\.venv\Scripts\python.exe tools/source_bundle.py --report artifacts/source-audit.json --output artifacts/source-review.zip
```

Công cụ dùng chính quy tắc Git ignore, kiểm tra tên file Windows, trùng tên theo chữ hoa/thường, đường dẫn dài, file lớn và một số mẫu secret phổ biến. Việc quét này không thay thế review nội dung. Không ghi vào `.git` local, không stage, commit hoặc push.

Clone repository đích sang checkout riêng rồi giải nén source đã review vào đó để so sánh với nội dung remote. Không khởi tạo hoặc ghi đè `.git` placeholder trong workspace này. Quy trình đồng bộ an toàn nằm trong [PREPARATION_REPORT.md](docs/PREPARATION_REPORT.md).

## Tình trạng kiểm chứng

Đã kiểm tra cài mới matching trên Ubuntu/Python 3.12, unit tests, smoke offline và mẫu geometry/content nhỏ. Có wheel Windows x64/Python 3.12 cho ba dependency matching. Chưa chạy trực tiếp trên Windows; chưa chạy API trả phí hoặc tái lập toàn bộ dataset. Workflow GitHub Actions có cả Ubuntu và Windows sẽ chạy sau khi bạn đưa thay đổi lên GitHub.
