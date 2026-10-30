# Viewer hiện hành

Chạy `npm run dev -- --host 127.0.0.1`; `npm run build` tạo bản production.

- `/`: chọn mẫu trong manifest, xem prediction đơn và số liệu coverage theo 3 chế độ; uncertain mặc định tắt.
- `/gt/`: đủ 131 mẫu/297 bbox ground truth, không sửa nhãn.
- `/group-review/`: GT được nhóm phủ thêm, thành viên và footprint/envelope của nhóm.

Nguồn dữ liệu tại `public/data/geometry-v3/`, do `python3 -m geometry.scripts.build_group_report` đồng bộ từ matching cache và kết quả group. `python3 -m geometry.scripts.build_gt_review` tạo trang GT. Không sửa trực tiếp bản copy trong public để thay đổi thuật toán hoặc điểm.

`src/pipelineAdapter.ts` chuyển schema matcher sang view model dùng bởi các component. Các component chỉ hiển thị/lọc; tính coverage diễn ra ở Python evaluator. `union_support` là mặc định; structural proxy là thử nghiệm cần review.
