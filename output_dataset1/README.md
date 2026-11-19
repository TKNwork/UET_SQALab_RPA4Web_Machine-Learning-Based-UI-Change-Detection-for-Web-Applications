# UI change matching — Geometry và Content

Hướng dẫn clone/cài trên Windows và Ubuntu hiện hành: [README gốc](../README.md), [các luồng chạy](../docs/WORKFLOWS.md), [khôi phục tài nguyên](../docs/ASSETS.md). Những lệnh chấm cache dưới đây yêu cầu khôi phục dữ liệu/kết quả lịch sử; không có sẵn trong source clone sạch.

Project tách thành hai nhánh độc lập. Chạy các lệnh dưới đây từ thư mục gốc `/home/tung/Desktop/RPA/output_dataset1`.

```text
geometry/        Pipeline, input subset, kết quả, viewer, scripts, tests của geometry
content/         Pipeline, kết quả, viewer HTML, scripts, tests của content
shared/          Normalization, duplicate/OCR evidence, Hungarian assignment, rectangle union
baseline/        Detection/ảnh upstream before — dữ liệu chung, giữ nguyên
inject/          Detection/ảnh upstream after — dữ liệu chung, giữ nguyên
groundtruth/     GT gốc XYWH/type — dữ liệu chung, chỉ survey/evaluation/review
scripts/         Công cụ bảo trì chung: build_project_inventory.py
docs/            Sơ đồ toàn project, inventory và audit di chuyển
```

[Hướng dẫn geometry](geometry/README.md) · [Hướng dẫn content](content/README.md) · [Cấu trúc và sơ đồ](docs/PROJECT_STRUCTURE.md).

## Geometry

```bash
python3 -m geometry.scripts.evaluate_group_coverage
python3 -m geometry.scripts.build_group_report
# Thêm --publish khi muốn cập nhật bản copy trong viewer
python3 -m geometry.scripts.build_gt_review
python3 -m geometry.match_geometry --input-root geometry/pairs --output-root geometry/results/full_run --image-root ../dataset1 --all-geometry --coverage-mode group_union
```

Cache đã có ở `geometry/results/<ID>/matching_result_full.json`. Run lịch sử giữ nguyên ở `geometry/results/group_coverage_v1/`; lần kiểm tra sau khi tách ghi riêng vào `group_coverage_organized_v1/`. Kết quả lịch sử: union_support phủ 220/270 GT trên 131 mẫu. Chưa có review thủ công đầy đủ; coverage không phải độ chính xác matching.

## Content

```bash
python3 -m content.pipeline survey
python3 -m content.pipeline run --split dev
python3 -m content.pipeline report
# Xem kết quả 5 mẫu, ghi development_review.md trước khi freeze
python3 -m content.pipeline freeze
python3 -m content.pipeline run --split all
python3 -m content.pipeline report
python3 -m content.scripts.survey_content --split all
```

Output mặc định: `content/results/organized_v1/`; kết quả trước khi tách giữ nguyên ở `content/results/v1/`. Viewer riêng: `index.html` trong từng run, mở trực tiếp bằng browser. Metadata toàn dataset có **97/501 mẫu content**. Đánh giá hiện có chỉ bao **6 mẫu / 14 GT** trong thư mục groundtruth; cấu hình phủ 4/14, 2/6 mẫu đủ GT. 91 mẫu còn lại chưa được chấm GT. Catalog toàn tập tại `content/results/catalog_97/index.html`. Xem báo cáo để phân biệt 5 mẫu phát triển, mẫu remaining và hạn chế nhãn.

## Kiểm tra

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 -m shared.scripts.check_migration
node geometry/scripts/check_gt_review.mjs
python3 scripts/build_project_inventory.py
```

Tách thư mục không thay dữ liệu nguồn hoặc các kết quả cũ. Manifest cũ giữ path/hash lịch sử; `shared/paths.py` giải quyết prefix đã chuyển, audit tại `docs/migrations/2026-10-21/`. Các code snapshot trước khi tách phục vụ xác minh hash, không phải pipeline đang dùng.

## So sánh geometry và content trên cùng một cặp

Chạy từ `output_dataset1` với Python của môi trường đã cài dependency geometry và content.

```text
python -m compare_changes --before BEFORE_full_hierarchy.json --after AFTER_full_hierarchy.json --before-image baseline.png --after-image inject.png --output comparison.json --visualization comparison.png
```

Kết quả JSON giữ đầu ra geometry và content riêng, cùng danh sách `events` ghép theo cặp ID before/after. Geometry dùng `edges8`. Content dùng profile `hierarchy-ocr-text-pairs-v1`, gồm cặp OCR đổi chữ có bằng chứng pixel. Hai ảnh là đầu vào bắt buộc để giữ bằng chứng ảnh của hai baseline. Bỏ `--visualization` để chỉ xuất JSON.

Trên hình before/after, khung xanh liền và nhãn G đánh dấu geometry. Khung cam đứt và nhãn C đánh dấu content. Các nhãn cùng số trỏ tới cùng sự kiện trong JSON.

Thêm `--reuse-geometry` để thử lấy cặp OCR từ Geometry. Chế độ thử nghiệm yêu cầu mỗi phía có đúng một thành viên OCR, cặp không xung đột với cặp khác và IoU của hai bbox OCR ít nhất 0,5. Các cặp có cờ nghi vấn từ Geometry được chuyển sang matcher OCR, ngoại trừ cờ OCR có thể đổi chữ. Matcher OCR chỉ chạy assignment trên các node còn lại. Số cặp tái sử dụng và số node còn lại nằm trong `content.geometry_reuse`.

## Baseline Visibility

`compare_changes` xuất thêm `visibility_candidates` từ các node unmatched của Geometry, bỏ những node đã được Content ghép qua thành viên nhóm trùng lặp. Mỗi ứng viên giữ phía quan sát được, bbox và nhãn sơ bộ. Chạy từ `output_dataset1` để chấm 500 mẫu:

```text
python -m evaluate_visibility --before-root baseline_detection_first500/baseline/json --after-root inject_detection_first500/inject/json --images ../dataset1 --gt ../thesis-dataset-main/groundtruth --output ../artifacts/visibility_baseline
python -m render_visibility ../artifacts/visibility_baseline/scores.jsonl ../dataset1 ../artifacts/visibility_baseline/example.png
```

Bộ chấm lưu toàn bộ ứng viên trước khi đọc GT, rồi ghi coverage và ghép một-một theo IoU ≥ 0,30 hoặc CCR ≥ 0,50. GT có bbox chung, nên hình mẫu hiển thị cùng bbox GT ở cả hai phía. Nhãn V− và V+ biểu thị ứng viên từ Before và After.
