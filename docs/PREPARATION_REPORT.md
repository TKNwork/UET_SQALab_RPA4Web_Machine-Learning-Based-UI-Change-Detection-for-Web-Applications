# Báo cáo chuẩn bị GitHub / Windows — 30/10/2026

## Git và repository đích

Workspace `/home/tung/Desktop/RPA` có thư mục `.git` placeholder rỗng, không phải Git repository hợp lệ. `git status`, `git remote -v`, `git branch --show-current` đều báo lỗi not a git repository. Không có trạng thái tracked/untracked, remote hoặc branch để đối chiếu; không thể suy ra thay đổi chưa commit từ Git.

Đã thử đọc repository GitHub đích bằng web và GitHub REST API. API trả **HTTP 404**, nên chưa đọc được cây file, README hay .gitignore remote. Repo private hoặc URL/quyền truy cập có thể là nguyên nhân; chưa xác định nguyên nhân nào. Không có dữ liệu để khẳng định local khớp remote.

Không init Git trong workspace, không thay remote/branch, không stage, commit, push hoặc force push.

## Đã thay đổi

| Nhóm | Thay đổi | Mục đích |
|---|---|---|
| Cài đặt | `requirements/{matching,perception,gemini,evaluation,webclick}.txt` | Cài đúng dependency của từng luồng; matching không cần PyTorch/API SDK |
| Khởi chạy | `run.py`, `tools/doctor.py`, `tools/smoke.py` | Entrypoint chung, Python interpreter hiện tại, subprocess không qua shell, UTF-8 và đường dẫn từ repo root |
| Cấu hình | `config.example.json`, `.env.example` | Đường dẫn tài nguyên ngoài repo; key qua môi trường, không chứa bí mật |
| Path/encoding | `shared/paths.py`, content inventory, YOLO/OCR, batch runners, WebClick path display | Rebase provenance đã biết; cấu hình dữ liệu/weights; ưu tiên key môi trường; đọc/ghi text UTF-8 |
| GT | `tools/prepare_content_gt.py` | Copy nguyên byte 131 GT gốc từ geometry pairs, kiểm tra đúng cohort content 6 mẫu/14 bbox |
| Tiện ích tạo cặp | `prepare_geometry_pairs.py` | Bỏ absolute path Ubuntu, có CLI input/output; từ chối output không rỗng; không tự tạo GT |
| Legacy | Chuyển `test_matching_pipeline.py` sang `legacy/test_matching_pipeline.py` | Test dùng API cũ không còn trong mã hiện hành; giữ file để review, loại khỏi source export |
| Git hygiene | `.gitignore`, `.gitattributes`, `tools/source_bundle.py` | Loại môi trường, secrets, data/model/results; export source có inventory SHA-256 và kiểm tra tên Windows |
| Tài nguyên | `tools/asset_bundle.py`, gói mẫu trong `artifacts/` | Chuyển 7 mẫu và GT gốc có checksum; dữ liệu không đưa lên Git |
| CI | `.github/workflows/portable-smoke.yml` | Smoke/test offline trên Ubuntu và Windows khi được publish |
| Tài liệu | README root, ASSETS, WORKFLOWS, WINDOWS và báo cáo này | Quy trình clone, cài, restore, chạy, xem output |

Không chủ đích thay thuật toán, threshold, model nghiên cứu mặc định hay giao thức đánh giá. Thay đổi source làm thay đổi code hash, nên run lịch sử được giữ nguyên và chạy kiểm chứng vào output mới. Không rewrite manifest/hash để vượt kiểm tra provenance.

## Đã dọn / giữ lại

Đã xóa **29 thư mục cache/build sinh tự động**, tổng khoảng **97 MiB**: `__pycache__` ngoài môi trường/dependency và `geometry/report_viewer/dist`. Danh sách cụ thể tại `docs/cleanup_manifest.json`. Build viewer đã chạy thành công trước khi xóa dist để xác nhận có thể tạo lại.

Giữ `.venv` và `node_modules` hiện có để không làm hỏng môi trường làm việc; chúng bị ignore và không có trong source bundle. Giữ dataset1, dataset2, detection, ground truth, model, results, viewer public data và code snapshots lịch sử; chúng được phân phối riêng. Không xóa file nghiên cứu, nên không cần phục hồi bản nghiên cứu đã xóa. File legacy được di chuyển, không xóa nội dung.

Dung lượng file ngoài dependency/môi trường lúc khảo sát: dataset1 ~3,745 GiB; OCR + Yolo26 ~1,533 GiB; output_dataset1 ~2,447 GiB. Archive WebClick ~1,05 GiB và weights YOLO ~42 MiB là những tài nguyên không nên đưa thẳng vào source repo. Các số này không phải dung lượng source bundle sau dọn.

Các script nghiên cứu độc lập trong combine, geometry/scripts và Gemini giữ lại dù launcher không gọi tất cả. Chưa có bằng chứng đủ để xóa chúng. Các inventory, audit và snapshots lịch sử nằm ngoài source export; vẫn giữ local cho truy vết. Tài liệu thiết kế và code chạy hiện hành vẫn nằm trong source.

## Tài nguyên mới phục vụ review

- `artifacts/windows-sample-assets.zip`: gói ~6 MB, 173 file tài nguyên + manifest; không có API key, model hay toàn bộ dataset.
- `artifacts/content_groundtruth`: bản copy GT gốc; cùng byte với GT trong geometry pairs.
- `outputs/geometry-portable-check`: lần chạy mới chỉ ID 000; không ghi vào run lịch sử.
- `output_dataset1/content/results/portable_check`: survey/dev/report kiểm chứng mới.
- `artifacts/source-review.zip`, `artifacts/source-audit.json`: gói source và inventory để review. Tất cả `artifacts/` đều bị ignore.

## Kiểm chứng

- Trước chỉnh sửa: 76 test hiện có pass trên Ubuntu.
- Môi trường mới `/tmp/rpa-clean-venv`, không kế thừa system-site-packages: cài thành công matching với Python 3.12.3; smoke và 78 test pass (thêm hai test về rebase Linux/Windows và resource override).
- Tải thành công wheel Windows x64/Python 3.12 cho numpy 2.5.2, scipy 1.18.1, Pillow 10.2.0. Chưa thực thi các wheel đó trên Windows.
- Geometry ID 000 chạy từ môi trường sạch thành công. Content survey xác nhận 131 file GT, 6 mẫu content/14 GT; dev/report chạy riêng trong run mới.
- `npm run build` viewer thành công trong môi trường Node.js 24.18.1/dependency local. Chưa kiểm tra `npm ci` sạch trên Windows.
- Perception `--help` import thành công từ môi trường local hiện có; không gọi Cloud Vision, Gemini hoặc load/chạy inference model.
- Source export kiểm tra file lớn, các mẫu token phổ biến, symlink, tên không hợp lệ Windows, case collision và độ dài tương đối. Không thấy vấn đề trong tập source được chọn. Đây không phải quét bí mật đầy đủ mọi loại credential hoặc toàn bộ lịch sử Git.
- Đã giải nén source vào checkout tạm không có dataset, dùng virtualenv sạch và gọi launcher từ `/tmp`: doctor/smoke/78 test pass. Sau khi restore gói mẫu, xác minh đủ 173 checksum, copy GT nguyên byte, chạy geometry 000 và content survey/dev/report thành công. Gói source gồm khoảng 185 file/~1 MB trước nén, khoảng 372 KB sau nén.

Log kiểm chứng được giữ trong `artifacts/validation/`. Việc kiểm tra gói source đã giải nén được ghi thêm trong `artifacts/validation/package-check.txt`.

## Giới hạn còn lại

Chưa chạy trực tiếp trên Windows, GPU, API trả phí, full dataset, full fresh install perception/evaluation/Gemini hoặc viewer. Các script chấm run lịch sử vẫn yêu cầu dữ liệu và layout cụ thể; source clone sạch không có kết quả/viewer sẵn. Không tự động tải tài nguyên nội bộ vì chưa có URL/kho phân phối được xác minh. Sử dụng gói mẫu để chạy nhanh; copy bộ đầy đủ theo ASSETS để nghiên cứu toàn tập.

Legacy test, code snapshots, dữ liệu và kết quả bị ignore cần được chủ project quyết định lưu kho dài hạn ở đâu. Chưa tạo Git LFS hoặc upload dữ liệu. Chưa xác minh nội dung/giấy phép repo remote nên không tự tạo LICENSE hoặc ghi đè tài liệu remote.

## Đồng bộ an toàn sau review

1. Xác nhận URL và quyền truy cập repository đích. Nếu private, đăng nhập Git bằng phương thức của bạn; không gửi token vào chat.
2. Giữ workspace Ubuntu hiện tại và các artifact nghiên cứu. Clone repo đích sang thư mục **khác**; kiểm tra branch, status, README và .gitignore thật tại đó.
3. Tạo branch riêng trong checkout mới. Giải nén source-review vào thư mục tạm, so sánh với checkout trước khi copy; không ghi đè `.git`, source hoặc tài liệu remote khi chưa review khác biệt.
4. Áp dụng từng thay đổi phù hợp; giữ file remote chưa có trong local. Kiểm tra `git diff`, `git status`, danh sách file được stage và secrets. Ignore không tự gỡ file đã tracked ở remote.
5. Cài sạch, chạy doctor/smoke/test; review dependency, tài nguyên thiếu và báo cáo này. Chỉ commit/push khi bạn quyết định thực hiện bước publish.
6. Clone branch đã publish trên Windows, chạy checklist WINDOWS; kiểm tra cả hai job CI trước khi hợp nhất.

## Cập nhật khi người dùng yêu cầu commit/push

Ngày 30/10/2026, truy cập bằng Git thành công và clone xác nhận repository đích trống. Đã tạo checkout riêng tại `artifacts/github-publish` để chuẩn bị commit đầu tiên trên main. Các ghi nhận HTTP 404 và chưa có quyền truy cập ở phần khảo sát phía trên là trạng thái tại thời điểm khảo sát, đã được giải quyết khi dùng Git. Người dùng đã cho phép commit/push; vẫn không đưa dataset, model, secrets hoặc kết quả nghiên cứu lên repository.
