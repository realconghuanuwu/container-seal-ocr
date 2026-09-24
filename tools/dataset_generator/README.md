# 🤖 Gemini Vision AI-Assisted Dataset Generator & Review Tool

Thư mục này chứa toàn bộ mã nguồn phục vụ việc **tạo và gán nhãn tự động cho dataset OCR container seal bằng Gemini Vision** kết hợp với quy trình kiểm thử và review (human-in-the-loop).

Thư mục này được thiết kế độc lập và sẵn sàng để tách ra thành một repository riêng biệt nhằm phục vụ chuyên sâu cho việc làm dữ liệu (Data Engineering).

---

## 📂 Danh mục tệp tin

| Tệp | Mô tả |
|---|---|
| `batch_labeling_v3.py` | Pipeline gán nhãn hàng loạt bằng Gemini Multimodal Vision, calibration đánh giá độ chính xác, routing nhãn (auto-accepted / needs-review) và quota tracking. |
| `create_recognition_dataset_v3.py` | Tạo dataset v3 qua 2-stage recognition: PaddleOCR + Gemini 3.8 Flash, tự động phát hiện ảnh trùng (pHash). |
| `finalize_dataset_v3.py` | Tổng hợp, phân chia tập Train / Validation theo nhóm tiền tố hoặc nguồn ảnh và xuất file `train.txt`, `val.txt`, `manifest.csv`. |
| `review_tool_v2.py` | Giao diện web UI (FastAPI) để người dùng kiểm tra, chỉnh sửa và xác nhận các mẫu nhãn có độ tin cậy thấp hoặc Gemini và PaddleOCR không đồng thuận. |
| `create_recognition_dataset_v2.py` | Pipeline xử lý dataset recognition phiên bản v2 (crop detector, rotate). |
| `finalize_dataset_v2.py` | Finalize dataset recognition phiên bản v2. |
| `test_dataset_generator.py` | Bộ unit test kiểm thử parsing, calibration, quota và routing của pipeline. |

---

## 🚀 Hướng dẫn tách thành dự án riêng (Independent Repo)

Khi bạn muốn tách bộ công cụ này thành dự án riêng:

1. **Sao chép thư mục**:
   Sao chép toàn bộ nội dung trong `tools/dataset_generator/` sang một thư mục mới bên ngoài dự án:
   ```bash
   cp -r tools/dataset_generator /path/to/my-seal-dataset-pipeline
   cd /path/to/my-seal-dataset-pipeline
   git init
   ```

2. **Cài đặt thư viện phụ thuộc**:
   ```bash
   pip install opencv-python pillow numpy fastapi uvicorn requests
   ```

3. **Cấu hình môi trường**:
   - Nếu gọi trực tiếp Gemini qua Antigravity CLI / Gemini API, thiết lập biến môi trường:
     ```bash
     export GEMINI_API_KEY="your-api-key"
     ```

4. **Chạy Review Tool**:
   ```bash
   python review_tool_v2.py --work-dir /path/to/work_dir --port 8080
   ```
   Sau đó truy cập: `http://localhost:8080` để tiến hành review và gán nhãn thủ công các ca khó.
