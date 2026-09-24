# Quy Ước Hệ Thống & Định Danh Mô Hình (System & Model Conventions)

Hệ thống hoạt động trên **một môi trường thống nhất duy nhất**, không phân tách môi trường DEV / UAT / PROD phức tạp. Mọi cấu hình đều được quản lý trực tiếp qua file `.env`.

---

## 1. Quy ước định danh Mô hình SemVer

Toàn bộ mô hình và phiên bản dịch vụ tuân thủ quy ước SemVer rõ ràng:

| Thành phần | Định danh SemVer | Thư mục trong `models/` | Mô tả |
| :--- | :--- | :--- | :--- |
| **YOLO Seal Detector** | `det-v1.0.0` | `models/seal-det-v1.0.0/` | Trọng số ONNX phát hiện vùng seal (`best.onnx`) |
| **PaddleOCR Recognition (Base)** | `rec-v1.0.0` | `models/seal-rec-v1.0.0/` | Model nhận diện gốc (14,500 mẫu, 82.67% accuracy) |
| **PaddleOCR Recognition (Manual)** | `rec-v1.0.3` | `models/seal-rec-v1.0.3/` | Model fine-tuned từ 14,500 mẫu + 100 ảnh manual |

### Chuỗi Phiên bản Pipeline (`MODEL_VERSION`)
Được tạo thành bằng cách ghép mã detector và recognizer:
$$\text{seal-det-vX.Y.Z-rec-vA.B.C}$$

* Ví dụ phiên bản Base: `seal-det-v1.0.0-rec-v1.0.0`
* Ví dụ phiên bản Manual: `seal-det-v1.0.0-rec-v1.0.3`

---

## 2. Cấu hình Môi trường Duy nhất (`.env`)

Hệ thống chỉ sử dụng 1 file `.env` duy nhất (tạo từ `.env.example`):

```ini
MODEL_VERSION=seal-det-v1.0.0-rec-v1.0.3
SEAL_DETECTOR_MODEL=models/seal-det-v1.0.0/best.onnx
RECOGNITION_MODEL_DIR=models/seal-rec-v1.0.3
```

---

## 3. Khởi chạy Hệ thống

### Bằng Python trực tiếp:
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Bằng Docker:
```powershell
docker build -t container-seal-ocr .
docker run -d --rm --name container-seal-ocr -p 8000:7860 container-seal-ocr
```
