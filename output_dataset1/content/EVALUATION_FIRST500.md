# Nghiên cứu đánh giá content trên 500 mẫu đầu

Ngày audit: 2026-11-01. Phạm vi mặc định: ID **000–499**, gồm đúng 500 ID liên tiếp. Đây là quy ước vận hành cần đối chiếu manifest lần chạy detector; nếu lần chạy thực tế là 001–500 thì dùng `--start-id 1`. Không lấy “500 file tìm thấy đầu tiên” vì sẽ che mất ID thiếu. Tài liệu cũ ghi 501 mẫu nguồn, nên chưa thể coi cohort cũ và cohort mới là giống nhau.

## Trạng thái bằng chứng tại checkout hiện tại

Đã đọc `content/pipeline.py`, `content/README.md`, `tools/prepare_content_gt.py`, catalog, evaluator/report geometry và path resolver. Các thư mục dữ liệu trong cấu hình local hiện chưa có: `dataset1`, `output_dataset1/baseline/json`, `output_dataset1/inject/json`, `output_dataset1/groundtruth`. Thư mục `content/results` lịch sử cũng không có trước audit. Preflight mới kiểm tra 500/500 ID và báo thiếu nguồn, tìm thấy **0 file GT**. Điều này là thiếu tài nguyên tại máy, không phải 0 content trong dataset, cũng không phải pipeline có coverage 0%.

Số liệu dưới đây lấy từ README lịch sử và kiểm tra điều kiện cohort trong `prepare_content_gt.py`; **chưa tái tính từ artifact gốc**. Không có inference/evaluation cũ để đếm prediction, phân tích từng bbox missed hoặc xác minh ảnh. Không gán nguyên nhân lỗi cho từng mẫu khi chưa có bằng chứng.

| Tập được tài liệu cũ mô tả | Mẫu có GT content | Vùng được phủ / GT | Coverage | Mẫu phủ đủ |
|---|---:|---:|---:|---:|
| Development: 187, 418, 437, 451, 489 | 5 | 2/12 | 16.67% | 1/5 |
| Remaining: 492 | 1 | 2/2 | 100% | 1/1 |
| Tổng cohort cũ | 6 | 4/14 | 28.57% | 2/6 |

GT count kỳ vọng lịch sử: 187=2, 418=1, 437=1, 451=7, 489=1, 492=2. Cả sáu ID nằm trong 000–499, nhưng cần kiểm tra nguồn GT mới trước khi khẳng định vẫn chỉ có 14 vùng. Không ép hard-code cohort 6 mẫu như script copy GT cũ: kiểm kê toàn bộ 500 file mới phát hiện được annotation content ở mẫu ngoài geometry subset.

## Phân tích kết quả hiện tại

1. Có 10/14 GT chưa được phủ theo báo cáo cũ; chưa có bằng chứng để quy cả 10 vùng thành lỗi OCR. Cần tách detector, correspondence, classifier, localization và tính hợp lệ GT.
2. Mẫu 451 có 7/14 GT, chiếm 50% mẫu số micro coverage. Vì vậy phải báo cả micro coverage, coverage từng mẫu và tỷ lệ mẫu phủ đủ; không chỉ dùng một tỷ lệ tổng. Tài liệu cũ ghi có fallback không đổi và reflow ở 451, cần review ảnh trước khi diễn giải là lỗi content.
3. 187 có vùng vị trí cũ và mới. GT annotation không đồng nghĩa một sự kiện độc lập; không cộng hai vùng thành hai lần thay nội dung nếu chưa xác định endpoint identity.
4. 492 có hai nút cùng selector; theo ghi chú lịch sử, prediction của nút Apple di chuyển có thể phủ hai GT trong khi nút Google không đổi. 2/2 localization coverage không xác nhận cả hai nút đều đổi nội dung. Ưu tiên review tất cả link của mẫu này.
5. Catalog cũ ghi **97 mẫu có thao tác content theo metadata trên 501 mẫu**. Không dùng 97 làm mẫu số recall: thao tác inject, nội dung hiển thị thay đổi và bbox GT là ba đơn vị khác nhau. Chưa thể suy ra chính xác số mẫu metadata content trong 500 ID mới khi chưa có manifest.
6. Inference hiện ưu tiên unchanged khi OCR chuẩn hóa giống nhau. Nhánh visual không có OCR hai phía chỉ là `visual_content_candidate`; có thể nhầm style, crop hoặc correspondence. Phải báo riêng `ocr_and_visual` và visual candidate, kể cả khi đều phủ GT.
7. Remaining chỉ có một mẫu và đã được khảo sát. Run organized là hồi quy sau tổ chức mã. Cả hai không hỗ trợ tuyên bố khả năng tổng quát hóa trên held-out test.

## Protocol: GT → detection content → join

### 1. Chốt cohort và provenance

Lưu 500 ID kể cả ID thiếu; một baseline hierarchy JSON, một inject JSON, một thư mục ảnh và một GT cho mỗi ID. Kiểm tra cùng stem, kích thước viewport, hash source, config và code. Không thay bằng native `before_bboxes/after_bboxes` hoặc lấy GT geometry làm nguồn duy nhất. Thiếu/ambiguous nguồn thì dừng chấm toàn cohort và giữ preflight để truy vết.

Thư mục dữ liệu mặc định theo `config.local.json`. Script mới đọc file này trực tiếp; đường dẫn tương đối tính từ repository root. Chưa chạy detector OCR/YOLO lại vì mục tiêu là đánh giá detection đã có của lần chạy 500 mẫu.

### 2. Xử lý GT độc lập

Đọc schema flat `bboxes`; chọn `'content' in type`, giữ multi-label, `fallback`, mọi thuộc tính và `original_gt_index`. Đổi XYWH sang XYXY bằng `(x,y,x+w,y+h)`; không tự clip hoặc suy đoán scale. Tọa độ không hữu hạn, kích thước không dương, ngoài viewport hoặc schema khác phải được review. Không gộp hai endpoint thành envelope rồi coi là GT mới.

`normalized_gt.json` là bản trích chuẩn hóa để audit, không chỉnh sửa file annotation gốc. Tách các cột review: annotation hợp lệ, nội dung thực sự đổi, reflow/geometry kèm theo, fallback và endpoint trùng sự kiện. Các cột chưa review để trống; không tự gán false hoặc true.

### 3. Thu content từ detection mà không dùng GT

Tái sử dụng `content.pipeline.infer()` với bốn input: detection before/after, ảnh before/after. Luồng: normalize → duplicate groups → anchors → assignment → matched/uncertain/unmatched → OCR và visual evidence → changed predictions → local groups. Selector, DOM và metadata inject chỉ phục vụ review, không đưa vào inference hay quyết định prediction.

Chạy trên **toàn bộ 500 ID**, bao gồm mẫu không có GT content, để thấy khối lượng prediction chưa được kiểm chứng. Không ép mọi unmatched thành content change. Lưu toàn bộ trace và danh sách prediction; freeze/hash tất cả inference trước khi gọi evaluator đầu tiên, giống quy trình group geometry.

Có thể đọc cache content `<ID>/inference.json` khi `uses_groundtruth=false` và config khớp. Không dùng geometry matching cache thay content inference vì cost/matching khác. Cache cũ không có manifest liên kết input hash chỉ được ghi là provenance chưa kiểm chứng độc lập, dù file cache được hash ở lần audit mới.

### 4. Ghép GT–prediction bằng support thực

Giữ metric content hiện hành để tái lập baseline:

`intersection(union(before_bbox, after_bbox), GT) / area(GT) >= 0.50`

và

`intersection(union(before_bbox, after_bbox), GT) / area(union support) >= 0.25`.

Áp dụng singleton hoặc local group tạo trước scoring. Union tính diện tích hợp thật, không đếm chồng hai lần và không lấp khoảng trống chuyển động. Lưu **mọi link**, không chỉ best link. Correspondence before–after là one-to-one sau duplicate normalization, còn prediction–GT coverage là many-to-many.

Geometry dùng các protocol envelope/IoU/CCR khác nhau theo run. Chỉ kế thừa cách tổ chức thí nghiệm hai pha; không so 28.57% content với 220/270 hay 533/691 geometry như cùng metric hoặc cùng cohort.

### 5. Báo cáo và audit nguyên nhân

Artifacts script mới tạo:

| Artifact | Nội dung |
|---|---|
| `input_manifest.json`, `preflight.json` | Cohort, nguồn, hash, ID thiếu và lỗi schema |
| `config.json`, `prediction_manifest.json` | Config baseline và freeze toàn prediction |
| `<ID>/inference.json` | Detections, matching, evidence, predictions, groups |
| `<ID>/normalized_gt.json`, `evaluation.json` | GT nguyên chỉ số, mọi link, missed diagnostics |
| `samples.csv` | Coverage từng mẫu; mẫu không có GT không được tính PASS content |
| `gt_join.csv` | Một hàng mỗi GT; link singleton/group và giả thuyết bỏ sót |
| `prediction_review.csv` | Một hàng mỗi prediction, OCR before/after, kind, GT link, cột review |
| `summary.json`, `validation.json`, `run_status.json` | Tổng hợp, xác minh hash và trạng thái chạy |

Review thủ công theo thứ tự: toàn bộ 14 GT lịch sử; mọi link 492; toàn bộ 451; prediction phủ GT bằng visual candidate; prediction không link và mẫu metadata content ngoài cohort GT. Đối chiếu ảnh nguyên viewport, crop hai phía, detection boxes, matched/uncertain/unmatched và OCR text. Các hypothesis do evaluator xuất chỉ là gợi ý, có thể đồng thời đúng; không cộng như các nguyên nhân loại trừ nhau.

Gán nguyên nhân sau review: `detector_missing`, `wrong_correspondence`, `uncertain_or_unmatched`, `ocr_missing_or_noise`, `classifier_suppressed`, `support_threshold`, `annotation_unchanged_or_fallback`, `endpoint_ambiguity`, `unresolved`. Giữ riêng correctness ngữ nghĩa và correctness correspondence cho prediction được phủ GT.

Báo cáo chính: covered GT / content GT, số mẫu positive phủ đủ / số mẫu positive, coverage từng mẫu, OCR-supported coverage, số prediction theo kind, prediction chưa được kiểm chứng, prediction trên mẫu không có GT content. **Unverified không tự động là false positive; không có GT content không tự động là negative.** Chỉ tính precision/recall/F1 sau khi có annotation đủ và quy tắc event identity + matching một-một phù hợp. Nếu làm sensitivity ngưỡng/grouping, giữ baseline nguyên và ghi là development trên dữ liệu đã quan sát.

## Chạy trên Windows

Từ repository root, sau khi phục hồi dữ liệu theo `docs/ASSETS.md` hoặc chỉnh file config:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'output_dataset1')
# Preflight; luôn dùng output mới, không ghi đè run cũ.
.\.venv\Scripts\python.exe -m content.scripts.evaluate_first500 --output-root output_dataset1/content/results/first500_preflight_v2
# Chỉ inference từ detection đã có; không chạy lại detector.
.\.venv\Scripts\python.exe -m content.scripts.evaluate_first500 --execute --output-root output_dataset1/content/results/first500_content_v1
# Hoặc đọc cache content, vẫn yêu cầu input nguồn để kiểm tra:
.\.venv\Scripts\python.exe -m content.scripts.evaluate_first500 --execute --cache-root output_dataset1/content/results/organized_v1 --output-root output_dataset1/content/results/first500_cached_v1
```

Cache organized cũ theo README chỉ chứa 6 mẫu positive nên chưa đủ cache cho 500; lệnh cache sẽ báo thiếu, cần dùng fresh inference cho toàn cohort. Script hiện không tự dựng HTML viewer; dùng JSON/CSV để audit trước. Viewer/report cũ kỳ vọng manifest/split riêng, không chạy trực tiếp lên schema run mới.

## Điều kiện hoàn tất nghiên cứu thực nghiệm

Có đủ input 500 ID và manifest lần chạy detector; chạy freeze→score thành công; đối chiếu số GT mới với cohort cũ; review mọi GT covered/missed và ít nhất tất cả prediction được dùng để khẳng định ngữ nghĩa; giữ riêng kết quả localization và semantic correctness. Trước các điều kiện đó, kết luận hợp lệ hiện tại là baseline lịch sử có coverage thấp và có ambiguity, chưa phải kết quả tái đánh giá 500 mẫu.
