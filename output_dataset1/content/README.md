# Content

- `pipeline.py`: inventory → correspondence before/after → content evidence → prediction/nhóm → GT coverage. `infer()` không nhận GT, selector hoặc metadata thao tác inject.
- `scripts/build_content_report.py`: báo cáo Markdown và viewer HTML độc lập; bảng/click overlay detection, matching, uncertain, unmatched, prediction và GT.
- `scripts/survey_content.py`: contact sheet GT/detector và cặp before/after; chỉ review.
- `tests/test_content_pipeline.py`: OCR thay toàn bộ, OCR thiếu/noise, duplicate, text trùng, split/merge, crop jitter, brightness, geometry của phép chấm và GT-free inference.
- `results/v1/`: kết quả trước khi tổ chức thư mục, giữ nguyên byte và provenance cũ.
- `results/organized_v1/`: chạy lại cùng thuật toán/cấu hình sau tách, đường dẫn và code hash mới.
- `results/survey/`: crop khảo sát ban đầu.
- `results/catalog_97/`: bộ xem toàn bộ 97 mẫu content theo metadata, text/placeholder/thuộc tính và crop ảnh trước–sau.
- `scripts/build_content_catalog.py`: tạo catalog từ metadata/DOM/ảnh, chỉ dùng review; không đưa vào inference hoặc scoring.

```bash
# Từ gốc project
python3 -m content.pipeline survey
python3 -m content.pipeline run --split dev
python3 -m content.pipeline report
# Xem 5 mẫu và ghi results/organized_v1/development_review.md
python3 -m content.pipeline freeze
python3 -m content.pipeline run --split all
python3 -m content.pipeline report
python3 -m content.scripts.survey_content --split all
python3 -m unittest discover -s content/tests -v
```

Thêm `--output-root content/results/TEN_RUN` để tạo run riêng; không ghi đè v1 để cập nhật hash cũ. Output luôn nằm dưới content/results, không thể trỏ vào input hay geometry.

Mỗi ID có inference.json (detections, duplicate, candidate trace, cost/evidence/gap, matched/uncertain/unmatched, content state, predictions, groups), evaluation.json (GT, mọi link, vùng bỏ sót/lý do), before.png, after.png và review.html. Run có config.json, data_manifest.json + hashes, development_ids.json, frozen_config.json, CSV/summary/failures theo split, REPORT.md, DEV_REPORT.md, development_review.md, full_review.md, validation.json, visual_examples/ và index.html.

Toàn dataset có **97 mẫu content theo metadata (501 mẫu nguồn)**. Catalog review tại `results/catalog_97/index.html`, manifest và CSV ở cùng thư mục; không phải predictions hoặc GT mới. Chỉ **6 mẫu có GT content hiện hành, tổng 14 bbox**. Dev cố định 187,418,437,451,489: 2/12 (16.67%), đủ GT 1/5. Remaining 492: 2/2 vùng GT, đủ 1/1. Toàn bộ: 4/14 (28.57%), đủ 2/6. Remaining chỉ là một mẫu, metadata đã được khảo sát; sau khi chạy toàn bộ không còn là tập chưa quan sát. Run organized chỉ kiểm tra hồi quy cấu trúc, không là đánh giá độc lập.

Content trong dữ liệu là thay text DOM (heading, button, thương hiệu, nhãn điều hướng); OCR detector không đầy đủ. Pipeline có OCR+visual evidence và visual_content_candidate thử nghiệm cho trường hợp thiếu OCR. OCR giống nhau được ưu tiên unchanged; thay đổi ảnh đơn thuần chưa xác nhận ngữ nghĩa. Visual candidate có thể lẫn style/crop/correspondence sai.

GT dùng XYWH trong mặt phẳng viewport, không có endpoint identity. 187 có vùng vị trí cũ và mới, 451 có fallback không đổi và reflow, 492 có hai nút cùng selector. Chấm hợp endpoint: I/area(GT) ≥ .50 và I/area(support) ≥ .25, cho singleton hoặc nhóm local; không lấp khoảng trống chuyển động. 492 một Apple button di chuyển có thể phủ hai GT dù nút Google không đổi, nên 2/2 coverage không có nghĩa hai nút cùng thay content. Không tính precision/recall/F1; unmatched GT và prediction unverified cần review nguyên nhân.
