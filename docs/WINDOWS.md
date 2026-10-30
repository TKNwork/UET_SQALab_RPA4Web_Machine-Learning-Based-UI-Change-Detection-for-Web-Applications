# Checklist clone và chạy trên Windows

1. Có quyền clone repository đích; Git đã truy cập thành công trong phiên chuẩn bị publish.
2. Cài Git, Python 3.12 x64; clone vào đường dẫn ngắn như `C:\src\RPA4Web`.
3. Tạo virtualenv mới và cài `requirements/matching.txt` theo README. Không mang `.venv` Ubuntu sang.
4. Chạy `run.py doctor`, `run.py smoke`, `run.py test`. Doctor chỉ bắt buộc thư viện; tài nguyên thiếu được báo riêng.
5. Chuyển gói tài nguyên mẫu từ Ubuntu, giải nén không ghi đè dữ liệu cũ; chạy `tools/asset_bundle.py --verify`.
6. Chuẩn bị GT content, cập nhật config local; chạy geometry ID 000 và content dev theo WORKFLOWS.
7. Kiểm tra JSON tiếng Việt/Unicode và mở report HTML. Nếu chạy trực tiếp script cũ, dùng `python -X utf8`.
8. Chỉ cài perception/PyTorch, Gemini SDK hoặc Node.js nếu cần luồng đó. Cấp key riêng bằng biến môi trường.
9. Chạy workflow `Portable matching` trên GitHub sau khi publish; kiểm tra cả Ubuntu và Windows.

Lệnh dùng trực tiếp `.venv\Scripts\python.exe`, nên không cần `Activate.ps1` và không yêu cầu nới ExecutionPolicy. Chuỗi có dấu cách phải đặt trong dấu nháy. Trong JSON nên dùng `D:/RPA-data/...`; nếu dùng backslash, phải escape thành `\\`.

Các path argument của launcher tính từ repository root, kể cả khi gọi `run.py` bằng đường dẫn tuyệt đối từ thư mục khác. Khi dùng module trực tiếp (`python -m geometry...` hoặc `python -m content...`), phải đứng trong `output_dataset1` như tài liệu nghiên cứu cũ.

Đã kiểm tra khả năng tải wheel Windows của dependency matching; đây không phải chạy test trên Windows. GPU, Google Vision, Gemini, pycocotools, toàn bộ batch runner và dữ liệu viewer trên Windows chưa được xác nhận trong phiên này.
