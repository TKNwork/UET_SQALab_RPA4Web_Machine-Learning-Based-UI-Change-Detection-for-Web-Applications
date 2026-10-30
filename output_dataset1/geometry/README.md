# Geometry

- `pipeline.py`: evidence cost cho geometry, classifier edges8/relative15, evaluator, CLI.
- `match_geometry.py`: entrypoint; chạy `python3 -m geometry.match_geometry` từ gốc project.
- `prediction_groups.py`: nhóm shared-parent/local-motion độc lập GT.
- `pairs/`: 131 input subset before/after/GT, chuyển nguyên từ geometry_pairs.
- `results/`: matching cache, kết quả và provenance; run cũ group_coverage_v1 được giữ nguyên.
- `study_first5/`: fixture 5 mẫu cũ.
- `scripts/`: evaluate_group_coverage, build_group_report, build_gt_review và template/check viewer.
- `tests/`: kiểm thử geometry.
- `report_viewer/`: React/Vite viewer geometry, GT và group review.
- `docs/PROJECT_STRUCTURE.md`: schema chi tiết từ tài liệu trước khi tách.

Các hàm normalization, duplicate, anchor, histogram, delta và thuật toán assignment chuyển vào `shared/matching.py`. `geometry.pipeline.solve_assignment()` truyền geometry cost vào bộ assignment chung. Chiến lược/threshold geometry không đổi.

```bash
# Chấm cache, ghi vào results/group_coverage_organized_v1, không đè group_coverage_v1
python3 -m geometry.scripts.evaluate_group_coverage
python3 -m geometry.scripts.build_group_report
# Thêm --publish khi muốn cập nhật bản copy trong viewer
# Chạy matching mới vào output riêng
python3 -m geometry.match_geometry --input-root geometry/pairs --output-root geometry/results/full_run --image-root ../dataset1 --pair-ids 000 004 012 016 021
python3 -m unittest discover -s geometry/tests -v
cd geometry/report_viewer
npm run build
npm run dev -- --host 127.0.0.1
```

Input GT chỉ dùng chọn cohort, scoring và review. Correspondence one-to-one; prediction–GT không one-to-one. Union support là mặc định; structural proxy còn thử nghiệm. Coverage lịch sử 220/270 không phải precision/recall/F1 của matching.

## Dataset2 — chấm đầy đủ GT gốc từ ID 502

```bash
python3 -m geometry.scripts.evaluate_dataset2_native_gt
```

Run hiện hành cho schema `before_bboxes`/`after_bboxes` ở `results/dataset2_native_gt_v1/`: 166 mẫu, 799 annotation geometry gốc. Dùng nguyên matching/group cache ở `results/dataset2_from502_detection_v1/`, không chạy lại detector. Chấm từng bbox GT gốc với prediction envelope hoặc union envelope của một nhóm, giữ IoU≥0.30 hoặc CCR≥0.50. Không cần suy đoán correspondence giữa hai danh sách GT. Score đã đối chiếu bằng evaluator gốc trên cùng annotation.

Đơn vị GT là annotation ở từng phía, không phải GT envelope hay số thay đổi độc lập; không so trực tiếp với 298 envelope của báo cáo một phần trước đây hoặc 270 GT pixel-diff tập cũ. `REPORT.md`, `index.html`, `previously_unresolved_31.json` và `validation.json` trong run mới chứa kết quả đầy đủ. Báo cáo cũ được giữ nguyên để truy vết. Test evaluator: `python3 -m unittest geometry.tests.test_native_gt -v`.

## Dataset2 — kết quả so sánh cùng GT chuẩn với tập đầu

Lần đánh giá dùng cùng nguồn GT `../dataset1/thesis-dataset-main/groundtruth` và schema `bboxes`: **165 mẫu / 421 GT**, tại `results/dataset2_same_gt_protocol_v1/`. Chạy lại: `python3 -m geometry.scripts.evaluate_dataset2_same_gt`. Nguồn GT hai phía và các run native/derived trước đây là phép chấm khác, không dùng thay kết quả so sánh này. ID 522 chỉ có style theo nguồn chuẩn nên không thuộc subset geometry. Xem `REPORT.md`, `GT_SOURCE_COMPARISON.md`, `validation.json` và `index.html` trong output mới.

## Thử nghiệm grouping nhiều cấp và vùng layout suy ra

`python3 -m geometry.scripts.experiment_hierarchical_geometry` tạo toàn bộ nhóm/vùng trước khi chấm 296 mẫu geometry của phạm vi 1005 mẫu. Output riêng: `results/hierarchical_geometry_v1/`. Nhánh hierarchy union vẫn 533/691; nhánh layout suy ra thử nghiệm đạt 543/691, cứu 10/79 GT mục tiêu. Layout extent không phải diện tích union thực tế, cần review correctness và false positive, đặc biệt 704/GT14, GT16; không thay baseline mặc định. `index.html` xem 79 GT; `rescued_gt_review.json`, `remaining_69_analysis.json`, `large_region_review_flags.json` lưu bằng chứng. GT, matching, classifier và ngưỡng overlap giữ nguyên. Cấu hình đã dùng là development trên dữ liệu đã quan sát, không phải kiểm thử held-out.

## Audit GT thô bằng bằng chứng ảnh và containment (thử nghiệm)

`python3 -m geometry.scripts.audit_coarse_gt_events` dựng motion features/nhóm độc lập GT, rồi audit đủ 158 GT missed trên 296 mẫu geometry. Output `results/coarse_gt_event_audit_v1/`. Điểm localization cũ giữ nguyên 533/691; metric bổ sung baseline OR nhóm ≥2 phần tử có bằng chứng dịch chuyển trên ảnh và union nằm ≥90% trong GT đạt 559/691, 224/296 PASS theo metric event. Đây không phải cùng metric IoU/CCR, không xác nhận GT sai, không chứng minh đủ mọi phần tử bên trong GT và chưa có precision trên vùng không thay đổi. Giữ hai kết quả tách biệt. `failure_audit.json` có chi tiết 158 GT, `no_changed_prediction_79_trace.json` truy vết 79 trường hợp thiếu matched-changed giao GT, `index.html` cho xem cả ứng viên uncertain/dưới classifier (không tính các ứng viên này vào event score). `python3 -m geometry.scripts.review_coarse_gt_audit` xây lại viewer/trace từ evidence đã lưu.
