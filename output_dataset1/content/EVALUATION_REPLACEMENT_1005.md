# Content replacement: text, image và icon

## Định nghĩa vận hành

Content change là nội dung hiển thị của cùng một phần tử (văn bản, hình ảnh hoặc biểu tượng) bị thay thế không có chủ đích, trong khi thuộc tính trình bày như kiểu chữ, màu sắc, vị trí được giữ nguyên. Thay đổi nội dung kèm presentation là mixed; chỉ thay presentation không phải content replacement. Một phía không có OCR chưa chứng minh text đã bị thêm/xóa.

Tài liệu này thay thế phạm vi text-only trong `EVALUATION_TEXT_1005.md` cho yêu cầu hiện tại. Baseline cũ vẫn được giữ nguyên để truy vết. Phạm vi là 1005 ID liên tiếp, mặc định 000–1004; phải đối chiếu manifest detector trước khi khẳng định cohort chính thức. Nếu chạy giai đoạn đầu 500 mẫu thì dùng `--count 500`.

## Phần đã triển khai

`replacement_changes.py` cung cấp profile `content-replacement-v1`. Script `scripts/evaluate_first500.py --replacement` thực hiện preflight → inference độc lập GT → tạo ứng viên → freeze tất cả → ghép GT → xuất bảng audit. Tên script được giữ để tương thích, số ID do `--count` quyết định.

### Tạo ứng viên độc lập GT

- Tái sử dụng correspondence, duplicate normalization và ảnh/detection của content pipeline. Không truyền GT, review, selector hay metadata inject vào `select_replacements()`.
- Text hai phía khác nhau và có visual fraction ≥ .025 tạo evidence thay text. Lấy OCR `class_label` nguyên gốc khi duplicate group có nhãn duy nhất để giữ chữ hoa/thường; fallback OCR composite được ghi rõ trong trace. Chuẩn hóa NFC và khoảng trắng, giữ dấu câu và số.
- Không có OCR hai phía: changed fraction ≥ .12 và MAE ≥ 10 tạo **visual candidate**, chưa chứng minh image/icon replacement hoặc loại trừ style. Detector label image/photo/picture hay icon/symbol chỉ là type hint. Không suy đoán loại dựa trên kích thước bbox.
- Kiểm tra tâm và kích thước khung: dung sai tạm thời 3 px. Content evidence với geometry ổn định là `pure_candidate`; geometry thay đổi là `mixed_candidate`. Hai trường hợp không đủ OCR/pixel evidence giữ uncertain hoặc unchanged.
- Chỉ matched `pure_candidate` vào prediction localization chính. Uncertain correspondence, mixed và các matched không thành prediction vẫn nằm trong `candidate_review.csv` và inference. Unmatched giữ trong inference để nghiên cứu detector/matching miss.
- Mọi candidate đều **chưa xác nhận** font, màu, presentation, semantic correctness và tính không chủ đích. Các ngưỡng chỉ là cấu hình thử nghiệm, chưa được hiệu chỉnh trên dữ liệu thật; không gọi `pure_candidate` là pure change đã chứng minh.

### Audit GT tách khỏi prediction

Giữ GT flat `bboxes`, original index, bbox và multi-label. Không tự xem annotation chỉ có nhãn content là đã đạt định nghĩa mới; cũng không tự loại multi-label mà chưa xem ảnh.

`gt_review_template.json` có cấu trúc:

```json
{
  "187": {
    "0": {
      "content_type": "unknown",
      "content_replaced": null,
      "same_element": null,
      "presentation_unchanged": null,
      "unintentional_confirmed": null,
      "evidence": ""
    }
  }
}
```

`187` và `0` ở đây chỉ minh họa schema, không phải annotation đã review. Dùng template sinh từ GT thực. Điền `content_type` là text/image/icon sau khi xem ảnh; mỗi boolean có thể true/false/null. Review có kết luận phải ghi evidence: ảnh/crop cụ thể, đặc điểm presentation và log inject hoặc annotation xác nhận thay đổi không chủ đích. Không dùng thông tin inject để tạo prediction.

- `pure`: xác nhận đủ bốn điều kiện boolean, loại nội dung đã biết.
- `mixed`: xác nhận nội dung đổi nhưng presentation không giữ nguyên.
- `excluded`: không phải replacement, không cùng phần tử hoặc không thỏa tính không chủ đích.
- `needs_review`: còn thiếu kết luận; không vào mẫu số pure GT.

File review được hash và chỉ dùng sau inference để xác định tập GT chấm. GT gốc không bị sửa; chỉ số bbox được giữ nguyên trong cả phép chấm subset.

### Hai lớp kết quả

1. `evaluation.json`: coverage tạm thời của candidate trên GT content gốc, giữ protocol endpoint union, GT fraction ≥ .50 và support fraction ≥ .25, singleton/local group, many-to-many. Đây là localization, không phải kết quả semantic chính thức.
2. `reviewed_pure_gt_evaluation.json`: candidate localization trên GT đã xác nhận pure. Nếu chưa có pure GT được review thì coverage là null, không ghi 0% hay PASS.

`summary.json` báo cả hai lớp, eligibility counts, breakdown GT text/image/icon và candidate type hints. `confirmed_content_changes` để null; script hiện **không tự cộng cột review prediction thành precision/recall/F1**. Để kết luận detection đúng, review thêm từng prediction về nội dung thay thế, correspondence, presentation và tính không chủ đích; mẫu không có GT content không tự động là negative.

`candidate_review.csv` chứa cả matched và uncertain: bbox trước/sau, lý do, độ dịch chuyển, loại gợi ý, trạng thái mixed/pure candidate và các cột human review. `gt_join.csv` chứa mỗi GT, mọi link, eligibility, content type đã review và diagnostic hypotheses. `prediction_review.csv` chỉ chứa prediction được dùng cho coverage; không thay thế bảng đầy đủ candidate.

## Chạy thực nghiệm

Từ repository root sau khi phục hồi dữ liệu và chỉnh `config.local.json`:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'output_dataset1')
# Kiểm kê và tạo template GT, không chạy inference.
.\.venv\Scripts\python.exe -m content.scripts.evaluate_first500 --count 1005 --replacement --output-root output_dataset1/content/results/replacement_preflight_v2
# Tạo ứng viên và coverage ban đầu; pure GT chưa review thì để null.
.\.venv\Scripts\python.exe -m content.scripts.evaluate_first500 --count 1005 --replacement --execute --output-root output_dataset1/content/results/replacement_run_v2
# Sau review GT, chấm lại trong thư mục mới.
.\.venv\Scripts\python.exe -m content.scripts.evaluate_first500 --count 1005 --replacement --gt-review artifacts/content_gt_review.json --execute --output-root output_dataset1/content/results/replacement_reviewed_v1
```

Không dùng `--text-only` đồng thời. Cache chỉ nhận inference content-v1 gốc, không nhận kết quả đã project thành profile mới; cache chưa có provenance nguồn đầy đủ sẽ được đánh dấu chưa xác minh độc lập. Script không chạy lại detector OCR/YOLO, không thay geometry hoặc run cũ.

## Audit dữ liệu tại máy ngày 2026-11-01

Đã kiểm tra đường dẫn cấu hình, hướng dẫn ASSETS và tìm ảnh/GT/detection/archive trong workspace. Không có nguồn dữ liệu thực để thực hiện nghiên cứu. Tài liệu ASSETS ghi dữ liệu nằm ngoài Git và phải chuyển từ máy Ubuntu/kho của chủ project; chưa có URL nguồn được xác minh.

Lệnh `--count 1005 --replacement --execute` đã được thử: preflight dừng vì cả 1005 ID thiếu nguồn, available GT files = 0. Đây là lỗi tài nguyên, không phải 0 sự kiện hoặc coverage 0%. Không thể review GT, font/màu hoặc image/icon thật trước khi dữ liệu được phục hồi. Cần ảnh, detection baseline/inject, GT canonical và manifest phạm vi ID; GT copy từ geometry 131 cặp chỉ đủ cohort lịch sử, không đủ đánh giá toàn 1005.

Các kiểm thử tổng hợp xác minh giữ raw OCR case, không sửa baseline, chặn pure prediction khi motion lớn, không suy image/icon từ kích thước, không tính missing OCR là replacement, giữ GT original index, tách mixed/excluded/unknown, đòi evidence cho review và freeze trước scoring. Chúng xác minh code/protocol, không thay thế đánh giá thực nghiệm.
