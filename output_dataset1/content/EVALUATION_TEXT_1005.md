# Đánh giá text content change trên phạm vi 1005 mẫu

## Định nghĩa và phạm vi

Content change trong nghiên cứu này là **text nhìn thấy trong ảnh trước và sau khác nhau**. Bao gồm thay chữ, số, dấu câu, chữ hoa/thường và dấu tiếng Việt. Chỉ đổi vị trí, kích thước, màu, font hoặc ảnh trong khi text giữ nguyên không đủ điều kiện. Text thêm/xóa là sự kiện thuộc phạm vi nghiên cứu, nhưng OCR thiếu một phía chưa chứng minh được sự kiện đó.

Phạm vi kế thừa yêu cầu so sánh với geometry: 1005 mẫu. Quy ước lệnh hiện tại là ID 000–1004; cần đối chiếu manifest detector để xác nhận ID đầu/cuối. Nếu thực tế bắt đầu 001 thì truyền `--start-id 1`. Không chọn trước 296 mẫu geometry hoặc chỉ sáu mẫu content lịch sử; phải kiểm kê GT content trên toàn cohort.

## Luồng thực nghiệm

1. Kiểm tra đủ ảnh, detection before/after và GT flat `bboxes` cho mọi ID; lưu hash. Trích các bbox có nhãn content, giữ nguyên nhãn phụ, fallback và original index. Không suy diễn text GT từ metadata inject.
2. Dùng matching content độc lập GT trên detection/ảnh. Tạo profile riêng `visible-text-v1`, không sửa baseline `content-v1` hoặc geometry.
3. Thu text OCR hoặc OCR con trong vùng phần tử ở cả hai phía. Chuẩn hóa Unicode NFC và khoảng trắng; giữ chữ hoa/thường, dấu, số và dấu câu. Đây là khác biệt có chủ đích với canonical cũ vốn bỏ dấu câu và casefold.
4. Prediction được chọn khi correspondence chắc chắn, có text hai phía khác nhau và pixel evidence đạt ngưỡng hiện hành 0.025. Text giống nhau không trở thành content prediction dù có motion/style. Visual-only candidate bị loại khỏi tập text prediction. OCR khác nhưng thiếu pixel confirmation, missing OCR, uncertain/unmatched được giữ để review.
5. Freeze toàn bộ prediction trước scoring. Ghép với GT bằng endpoint union thật, giữ baseline `GT fraction >= .50 AND support fraction >= .25`, singleton/local groups và mọi link many-to-many. Không lấp khoảng trống di chuyển.
6. Xuất samples.csv, gt_join.csv, prediction_review.csv và summary. Báo tổng mẫu có GT content, tổng bbox, covered/missed, tỷ lệ mẫu phủ đủ và prediction chưa được kiểm chứng. Chấm riêng tính đúng text/correspondence sau review ảnh; coverage không tự xác nhận text đúng.

Hai loại hạn chế cần ghi rõ: profile hiện chưa xác nhận tự động text thêm/xóa vì không phân biệt chắc chắn với OCR miss; một chỉnh sửa nhỏ có pixel evidence vẫn có thể là OCR noise nên có quality flag để review. Không lấy việc thiếu prediction trong các trường hợp này làm bằng chứng ảnh không đổi.

## So với kết quả geometry

Cùng cách tổ chức: input manifest → detection → matching → change evidence → freeze → GT join → review missed. Đơn vị của content vẫn là bbox annotation content. Giữ ngưỡng baseline content để có thể đối chiếu lịch sử, không gọi đây là cùng metric với geometry nếu geometry run dùng envelope/IoU/CCR khác.

Kết quả 4/14 của sáu mẫu cũ dùng `content-v1`, trong đó cho phép visual-only candidate; **không chuyển sang làm kết quả của visible-text-v1**. Phải chạy lại projection/inference và scoring. Các số geometry 533/691 cũng không phải mẫu số content.

## Lệnh chạy

Từ repository root:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'output_dataset1')
.\.venv\Scripts\python.exe -m content.scripts.evaluate_first500 --count 1005 --text-only --output-root output_dataset1/content/results/text1005_preflight_v1
.\.venv\Scripts\python.exe -m content.scripts.evaluate_first500 --count 1005 --text-only --execute --output-root output_dataset1/content/results/text1005_run_v1
```

Tên module giữ để tương thích script 500 mẫu; `--count` điều khiển phạm vi. Dùng `--count 500` để chạy giai đoạn đầu theo yêu cầu trước. Cấu hình đường dẫn lấy từ `config.local.json` hoặc `--config`. Không chạy lại OCR/YOLO trong lệnh này; đọc detection đã lưu.

Tại lần kiểm tra 2026-11-01, các nguồn dữ liệu trong config chưa được phục hồi. Vì vậy mới có thể kiểm tra code bằng dữ liệu tổng hợp và xuất preflight; chưa thể cung cấp coverage text trên dữ liệu thật. Cần phục hồi dataset ảnh, detection baseline/inject và GT của cả 1005 mẫu; nếu nằm nơi khác, chỉ cần đổi các đường dẫn config.
