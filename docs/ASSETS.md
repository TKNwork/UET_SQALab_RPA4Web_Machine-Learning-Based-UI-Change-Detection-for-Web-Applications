# Tài nguyên ngoài Git

Mã nguồn được lưu bằng Git thông thường. Dataset, weights và kết quả nghiên cứu được chuyển riêng; chưa dùng Git LFS vì chưa xác định quyền phân phối, quota hoặc nơi lưu trữ lâu dài. Không đẩy zip dữ liệu vào repository. Trước khi chia sẻ, cần kiểm tra quyền phân phối dataset/model.

Không có URL tải đã xác minh cho dataset nội bộ, model `yl26best.pt` và kết quả lịch sử. Người clone mới phải lấy chúng từ máy Ubuntu hiện tại hoặc kho tài nguyên do chủ project cung cấp. Clone mã nguồn chỉ đủ chạy smoke/test tổng hợp.

| Tài nguyên | Vị trí trên Ubuntu và sau khi restore | Luồng sử dụng |
|---|---|---|
| Screenshot dataset1 | `dataset1/<ID>_*/baseline.png`, `inject.png` | Geometry, content |
| DOM/metadata dataset1 | Các JSON/HTML cạnh ảnh trong `dataset1` | Catalog/survey nội dung nâng cao |
| Detection gốc | `output_dataset1/{baseline,inject}/json/*_full_hierarchy.json` | Content, chuẩn bị cặp |
| Cặp geometry | `output_dataset1/geometry/pairs/<ID>/{before,after,groundtruth}.json` | Matching và chấm geometry |
| GT content gốc | Sinh bản copy tại `artifacts/content_groundtruth/*.json` từ GT của 131 cặp | Content 6 mẫu/14 GT |
| GT hai phía | `output_dataset1/groundtruth/groundtruth` | Nghiên cứu native GT, không thay GT content |
| GT canonical toàn tập | `dataset1/thesis-dataset-main/groundtruth` | Nghiên cứu same-GT; khác cohort content gốc |
| Dataset2 | `output_dataset1/dataset2`, `dataset2_detection` | Nghiên cứu dataset2 |
| YOLO weights | `OCR + Yolo26/yolo/models/yl26best.pt` (~42 MiB) | Perception |
| WebClick/parquet | `OCR + Yolo26/data/webclick` | VLM grounding |
| Kết quả/cache thực nghiệm | Các `results/` và `yolo/analysis/` | Tái lập báo cáo, viewer lịch sử |
| Dữ liệu viewer | `output_dataset1/geometry/report_viewer/public/{data,gt,group-review}` | Viewer React |

## Gói mẫu chuyển máy

Trên Ubuntu, từ repository root:

```bash
python3 tools/asset_bundle.py --output artifacts/windows-sample-assets.zip
# Tùy chọn: tạo gói riêng có thêm weights, không bao giờ kèm API key
python3 tools/asset_bundle.py --include-model --output artifacts/windows-sample-assets-with-model.zip
```

Gói mẫu gồm ảnh/detection của ID 000,187,418,437,451,489,492; cặp JSON của các ID này; và **131 file GT gốc nguyên byte** để khôi phục cohort content. Gói không có đầy đủ dataset, DOM, cache hay kết quả lịch sử. Gói đã tạo không kèm model khoảng 6 MB. Chuyển zip bằng ổ đĩa hoặc kênh lưu trữ riêng của bạn.

Trên Windows, đặt zip cạnh thư mục clone, rồi:

```powershell
Expand-Archive ..\windows-sample-assets.zip -DestinationPath .
.\.venv\Scripts\python.exe tools/asset_bundle.py --verify
.\.venv\Scripts\python.exe tools/prepare_content_gt.py
```

Không dùng `-Force` khi restore vào checkout có dữ liệu cũ. Manifest chứa SHA-256 từng tài nguyên để phát hiện thiếu hoặc thay đổi byte. Trong `config.local.json`, đặt `groundtruth` thành `artifacts/content_groundtruth`. Nếu không có config local, có thể đặt biến môi trường:

```powershell
$env:RPA_GROUNDTRUTH = "artifacts/content_groundtruth"
```

`prepare_content_gt.py` từ chối ghi đè GT khác nội dung và kiểm tra cohort 6 mẫu/14 bbox. Công cụ chỉ sao chép, không chuyển schema hoặc sinh nhãn mới. Không tự dùng bộ GT toàn tập thay thế: nó thay đổi cohort và điều kiện nghiên cứu.

## Khôi phục toàn bộ nghiên cứu

Copy nguyên các thư mục dữ liệu, kết quả và viewer tương ứng ở bảng trên, giữ layout tương đối. Bản copy full cần xử lý symlink Ubuntu thành file thực khi chuyển sang Windows; không chỉ copy symlink bị đứt. Các manifest kết quả cũ có thể chứa absolute path Ubuntu; resolver dùng chung hỗ trợ rebase đường dẫn đã biết nhưng một số script lịch sử vẫn dùng layout/run cố định. Kiểm tra từng script trước khi chạy lại; không sửa artifact/hash lịch sử để giả lập một lần chạy mới.

`download_webclick.py` có URL Hugging Face trong mã nguồn; script tải nhiều parquet, cần network và `webclick.txt`. URL và quyền truy cập chưa được kiểm chứng trong lần chuẩn bị này; không chạy tự động trong smoke test.
