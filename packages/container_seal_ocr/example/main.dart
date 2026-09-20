import 'package:flutter/material.dart';
import 'package:container_seal_ocr/container_seal_ocr.dart';

void main() {
  runApp(const MaterialApp(home: OfflineSealDemoScreen()));
}

class OfflineSealDemoScreen extends StatefulWidget {
  const OfflineSealDemoScreen({super.key});

  @override
  State<OfflineSealDemoScreen> createState() => _OfflineSealDemoScreenState();
}

class _OfflineSealDemoScreenState extends State<OfflineSealDemoScreen> {
  SealResult? _result;
  bool _isLoading = false;

  Future<void> _scanSealOffline(String imagePath) async {
    setState(() => _isLoading = true);

    // 🌟 CHẠY 100% OFFLINE TRÊN MÁY (KHÔNG CẦN WIFI/4G, KHÔNG TỐN TIỀN SERVER):
    final result = await ContainerSealOcr.recognizeFile(imagePath);

    setState(() {
      _result = result;
      _isLoading = false;
    });

    if (result.isSuccess) {
      print('✅ Đọc thành công: ${result.sealNumber}');
      print('🚢 Hãng tàu: ${result.shippingLine}');
      print('🎯 Độ tin cậy: ${result.confidencePercent}');
      print('⏱️ Tốc độ xử lý chip máy: ${result.latencyMs} ms');
    } else if (result.isReview) {
      print('⚠️ Cần người kiểm tra: ${result.sealNumber}');
    } else {
      print('❌ Cần chụp lại: ${result.reason}');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Offline Container Seal OCR')),
      body: Center(
        child: _isLoading
            ? const Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 12),
                  Text('AI đang quét và đọc seal trực tiếp trên máy…'),
                ],
              )
            : _result == null
                ? const Text('Bấm nút camera bên dưới để test offline')
                : Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          _result!.sealNumber ?? '(Không đọc được)',
                          style: const TextStyle(
                            fontSize: 32,
                            fontWeight: FontWeight.bold,
                            color: Colors.blueAccent,
                          ),
                        ),
                        const SizedBox(height: 12),
                        Card(
                          elevation: 2,
                          child: Padding(
                            padding: const EdgeInsets.all(16.0),
                            child: Column(
                              children: [
                                _row('Hãng tàu', _result!.shippingLine ?? 'Chưa rõ'),
                                _row('Độ tin cậy', _result!.confidencePercent),
                                _row('Trạng thái', _result!.status.name.toUpperCase()),
                                _row('Thời gian xử lý', '${_result!.latencyMs} ms'),
                                _row('Chế độ', '100% Offline Trên Máy'),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _scanSealOffline('/path/to/seal_image.jpg'),
        icon: const Icon(Icons.camera_alt),
        label: const Text('Quét Seal Offline'),
      ),
    );
  }

  Widget _row(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: Colors.grey)),
          Text(value, style: const TextStyle(fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}
