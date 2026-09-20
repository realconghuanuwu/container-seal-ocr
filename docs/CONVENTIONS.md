# Quy Ước Đặt Tên Hệ Thống (DEV · UAT · PROD Conventions)

Tài liệu này chuẩn hoá toàn bộ quy ước đặt tên cho hệ thống Container Seal OCR trên 3 môi trường: **DEV (Phát triển)**, **UAT (Kiểm thử / QA Staging)** và **PROD (Sản xuất chính thức)**.

---

## 1. Bảng Tổng Hợp Quy Chuẩn

| Hạng mục / Đối tượng | DEV (Phát triển & R&D) | UAT (Kiểm thử & QA Staging) | PROD (Sản xuất chính thức) |
| :--- | :--- | :--- | :--- |
| **Môi trường (`APP_ENV`)** | `dev` | `uat` | `prod` |
| **Mô hình Pipeline (`MODEL_VERSION`)** | `seal-ocr-det-v{X}-rec-v{Y}-dev` | `seal-ocr-det-v{X}-rec-v{Y}-uat` (hoặc `-rc{N}`) | `seal-ocr-det-v{X}-rec-v{Y}` |
| **Thư mục Rec Model (`models/`)** | `seal-ocr-rec-v{Y}-dev` (hoặc `exp-*`) | `seal-ocr-rec-v{Y}-uat` | `seal-ocr-rec-v{Y}` |
| **Tên hiển thị (Web / API)** | *Seal OCR Dev ({model_id})* | *Seal OCR UAT v{Y}* | *Seal OCR Production v{Y}* |
| **Docker Image Tag** | `seal-ocr:dev`<br>`seal-ocr:det-v1-rec-v1-dev` | `seal-ocr:uat`<br>`seal-ocr:det-v1-rec-v1-uat` | `seal-ocr:latest`<br>`seal-ocr:det-v1-rec-v1`<br>`seal-ocr:v1.0.0` |
| **Tên Container Docker** | `seal-ocr-dev` | `seal-ocr-uat` | `seal-ocr-prod` |
| **Cổng Host Port** | `8001:8000` | `8080:8000` | `8000:8000` |
| **Tệp cấu hình môi trường** | `.env` (`APP_ENV=dev`) | `.env` (`APP_ENV=uat`) | `.env` (`APP_ENV=prod`) |
| **Hạn mức Timeout (`OCR_TIMEOUT`)** | `15` giây (hỗ trợ profiling, debug) | `10` giây | `9` giây (chặt chẽ SLA) |
| **Hạn mức tải lên (`MAX_UPLOAD_MB`)** | `20` MB (kiểm tra ảnh RAW) | `10` MB | `10` MB |
| **Nhánh Git / Tag phát hành** | `develop`, `feature/*`, `experiment/*` | `release/vX.Y.Z`, tag `vX.Y.Z-rcN` | `master` / `main`, tag `vX.Y.Z` |
| **Tệp Benchmark Baseline** | Báo cáo tạm thời trong `reports/` | `baselines/*-uat.json` | `baselines/seal-ocr-det-v{X}-rec-v{Y}.json` |

---

## 2. Chi Tiết Quy Ước Theo Từng Hạng Mục

### 2.1. Quy ước định danh Mô hình (Model & Pipeline Versioning)
Mỗi pipeline hoàn chỉnh gồm 2 mô hình độc lập:
1. **Seal Detector (YOLO)**:
   - Luôn định danh là: `seal-detector-v{X}` (hiện tại: `seal-detector-v1`).
   - Tệp trọng số chuẩn ONNX: `models/seal-detector-v1/best.onnx`.
2. **Recognition Model (PaddleOCR)**:
   - **DEV**: `models/seal-ocr-rec-v{Y}-dev` hoặc `models/seal-ocr-rec-exp-{name}`.
   - **UAT**: `models/seal-ocr-rec-v{Y}-uat` (Mô hình ứng viên Release Candidate).
   - **PROD**: `models/seal-ocr-rec-v{Y}` (Ví dụ bản phát hành hiện tại: `models/seal-ocr-rec-v1`).
3. **Pipeline Version (`MODEL_VERSION`)**:
   - Khi chạy qua API hoặc kiểm tra `/health/ready`, version string phản ánh đúng detector + recognizer + môi trường:
     - DEV: `seal-ocr-det-v1-rec-v1-dev`
     - UAT: `seal-ocr-det-v1-rec-v1-uat`
     - PROD: `seal-ocr-det-v1-rec-v1`

---

### 2.2. Quy ước Docker & Triển Khai Container (Dùng chung 1 tệp .env)

#### DEV Environment:
```powershell
docker stop seal-ocr-dev 2>$null
docker run -d --rm --name seal-ocr-dev `
  -p 127.0.0.1:8001:8000 `
  --env-file .env `
  -e APP_ENV=dev `
  -e MODEL_VERSION=seal-ocr-det-v1-rec-v1-dev `
  -v "${PWD}/models/seal-detector-v1:/models/seal-detector-v1:ro" `
  -v "${PWD}/models/seal-ocr-rec-v1:/models/seal-ocr-rec-v1:ro" `
  seal-ocr:dev
```

#### UAT Environment:
```powershell
docker stop seal-ocr-uat 2>$null
docker run -d --rm --name seal-ocr-uat `
  -p 127.0.0.1:8080:8000 `
  --env-file .env `
  -e APP_ENV=uat `
  -e MODEL_VERSION=seal-ocr-det-v1-rec-v1-uat `
  -v "${PWD}/models/seal-detector-v1:/models/seal-detector-v1:ro" `
  -v "${PWD}/models/seal-ocr-rec-v1:/models/seal-ocr-rec-v1:ro" `
  seal-ocr:uat
```

#### PROD Environment:
```powershell
docker stop seal-ocr-prod 2>$null
docker run -d --rm --name seal-ocr-prod `
  -p 127.0.0.1:8000:8000 `
  --env-file .env `
  -v "${PWD}/models/seal-detector-v1:/models/seal-detector-v1:ro" `
  -v "${PWD}/models/seal-ocr-rec-v1:/models/seal-ocr-rec-v1:ro" `
  seal-ocr:det-v1-rec-v1
```

---

### 2.3. Quy ước Tệp Cấu Hình Môi Trường Duy Nhất (.env)

Hệ thống sử dụng **duy nhất 1 file cấu hình `.env`** (được copy từ [`.env.example`](.env.example)), không phân tách thành nhiều file rời rạc:
- Để đổi môi trường hoạt động, chỉ cần điều chỉnh biến `APP_ENV`:
  - `APP_ENV=prod`: Chế độ sản xuất tiêu chuẩn (SLA 9s).
  - `APP_ENV=uat`: Chế độ kiểm thử nghiệm thu.
  - `APP_ENV=dev`: Chế độ lập trình/R&D (Timeout 15s).
- File `.env` được đưa vào `.gitignore` để tránh đẩy cấu hình nội bộ lên repository.

---

### 2.4. Quy ước Git & Release Workflow

1. **Phát triển (DEV)**:
   - Code và script thử nghiệm thực hiện trên nhánh `develop` hoặc `feature/*`.
   - Các thử nghiệm huấn luyện model đặt tên nhánh `experiment/{tên_mục_tiêu}`.
2. **Nghiệm thu (UAT)**:
   - Khi model đạt ngưỡng kiểm chuẩn sơ bộ, tạo nhánh `release/vX.Y.Z` và gắn tag `vX.Y.Z-rc1`.
   - Chạy bộ 300 frozen benchmark images đối soát với baseline.
3. **Phát hành (PROD)**:
   - Merge vào `master`. Gắn tag phiên bản chính thức: `v1.0.0`.
   - File model đưa vào thư mục `models/seal-ocr-rec-v1/`.
   - Baseline chuẩn được commit vào `baselines/seal-ocr-det-v1-rec-v1.json`.
