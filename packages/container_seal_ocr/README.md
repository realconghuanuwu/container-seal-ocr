# 🚢 Container Seal OCR (Flutter / Dart Library - 100% Offline On-Device)

Thư viện **Container Seal OCR** dạng **Blackbox SDK chạy 100% Offline trực tiếp trên thiết bị di động (Mobile On-Device)** mà **KHÔNG CẦN KẾT NỐI INTERNET** (Không tốn 4G/Wifi, không tốn tiền duy trì server).

---

## 🌟 Điểm nổi bật

- **100% Offline**: Chạy trực tiếp trên CPU/NPU của điện thoại qua engine **Microsoft ONNX Runtime Mobile**.
- **Siêu nhẹ**: Mô hình đã được lượng hóa động **INT8** chỉ còn **~18.5 MB** (tổng trọn gói ~28 MB).
- **Tốc độ cao**: Xử lý trong khoảng **~150ms – 250ms** trên chip di động.
- **Tự động nhận diện hãng tàu**: Phân giải tức thì các hãng tàu lớn (OOCL, SITC, Yang Ming, Wan Hai, Maersk, MSC, COSCO...).
- **Headless SDK**: Không chứa Widget UI thừa thãi, dev mobile toàn quyền custom giao diện theo ý muốn.

---

## 📦 1. Cài đặt vào ứng dụng Flutter

Thêm thư viện vào tệp `pubspec.yaml` của ứng dụng Flutter:

```yaml
dependencies:
  flutter:
    sdk: flutter

  # Thư viện Container Seal OCR Offline
  container_seal_ocr:
    path: ../packages/container_seal_ocr   # Hoặc git: https://github.com/...
```

---

## ⚡ 2. Cách Sử Dụng (Đúng 1 Dòng Code)

```dart
import 'package:container_seal_ocr/container_seal_ocr.dart';

void onImageCaptured(String imagePath) async {
  // 🌟 HÀM ĐỌC OFFLINE DUY NHẤT (KHÔNG CẦN CẤU HÌNH SERVER, KHÔNG CẦN MẠNG):
  final result = await ContainerSealOcr.recognizeFile(imagePath);

  if (result.isSuccess) {
    print("✅ Số seal: ${result.sealNumber}");       // VD: "OOLKCK28059"
    print("🚢 Hãng tàu: ${result.shippingLine}");     // VD: "OOCL"
    print("🎯 Độ tin cậy: ${result.confidencePercent}"); // VD: "98.5%"
    print("⏱️ Tốc độ xử lý: ${result.latencyMs} ms"); // VD: 185 ms
  } else if (result.isReview) {
    print("⚠️ Cần kiểm tra lại: ${result.sealNumber} (${result.reason})");
  } else {
    print("❌ Cần chụp lại: ${result.reason}");
  }
}
```

Nếu đọc từ mảng byte (`Uint8List` từ camera stream):
```dart
final result = await ContainerSealOcr.recognizeBytes(imageBytes);
```

---

## 📁 3. Các tệp mô hình đính kèm sẵn trong Assets

Thư viện đã được đóng gói sẵn toàn bộ các mô hình tối ưu cho di động:
- `assets/seal_detector.onnx`: YOLOv8 định vị và cắt vùng chứa seal (~9.5 MB).
- `assets/seal_ocr_int8.onnx`: PP-OCRv6 SVTR_LCNet nhận diện ký tự đã nén INT8 (~18.5 MB).
- `assets/char_dict.txt`: Từ điển ký tự giải mã CTC (18,708 ký tự chuẩn).

---

## 📋 4. Cấu trúc kết quả trả về (`SealResult`)

| Thuộc tính | Kiểu dữ liệu | Ý nghĩa | Ví dụ thực tế |
| :--- | :--- | :--- | :--- |
| **`sealNumber`** *(hoặc `seal`)* | `String?` | Số seal đã được sửa lỗi tự động | `"OOLKCK28059"` |
| **`shippingLine`** | `String?` | Tên hãng tàu tự động nhận diện | `"OOCL"`, `"SITC"`, `"Yang Ming"` |
| **`confidence`** | `double` | Độ tin cậy AI (từ `0.0` đến `1.0`) | `0.985` |
| **`confidencePercent`** | `String` | Độ tin cậy phần trăm | `"98.5%"` |
| **`status`** | `SealStatus` | `success`, `review`, `recapture`, `error` | `SealStatus.success` |
| **`latencyMs`** | `int` | Thời gian chạy trên chip điện thoại | `185` ms |
| **`isSuccess`** | `bool` | `true` nếu đọc chuẩn xác | `true` |
