import 'package:flutter_test/flutter_test.dart';
import 'package:container_seal_ocr/container_seal_ocr.dart';

void main() {
  group('ShippingLineResolver Tests', () {
    test('resolves known carrier prefixes accurately', () {
      expect(ContainerSealOcr.resolveShippingLine('OOLKCK28059'), 'OOCL');
      expect(ContainerSealOcr.resolveShippingLine('OOLKCM1234'), 'OOCL');
      expect(ContainerSealOcr.resolveShippingLine('SJJA374088'), 'SITC');
      expect(ContainerSealOcr.resolveShippingLine('YMAT892120'), 'Yang Ming');
      expect(ContainerSealOcr.resolveShippingLine('WHAB123456'), 'Wan Hai Lines');
      expect(ContainerSealOcr.resolveShippingLine('FX37408884'), 'FedEx Express');
      expect(ContainerSealOcr.resolveShippingLine('MAEU992102'), 'Maersk Line');
      expect(ContainerSealOcr.resolveShippingLine('MSCU123456'), 'Mediterranean Shipping Co (MSC)');
      expect(ContainerSealOcr.resolveShippingLine('COSU882100'), 'COSCO Shipping');
    });

    test('resolves generic bolt seal series', () {
      expect(ContainerSealOcr.resolveShippingLine('A29296037'), 'Generic ISO 17712 Bolt Seal');
      expect(ContainerSealOcr.resolveShippingLine('R5935205'), 'Generic ISO 17712 Bolt Seal');
      expect(ContainerSealOcr.resolveShippingLine('28694002'), 'Generic ISO 17712 Numeric Seal');
    });
  });

  group('SealRules Postprocessing Tests', () {
    test('recovers prefix collapse and homoglyphs', () {
      // CTC multi-O collapse for OOCL
      expect(SealRules.cleanSealNumber('O0LKCK28059'), 'OOLKCK28059');
      expect(SealRules.cleanSealNumber('00LKCK28059'), 'OOLKCK28059');

      // Optical C3 -> CM for OOCL
      expect(SealRules.cleanSealNumber('OOLKC37120'), 'OOLKCM7120');

      // FedEx prefix recovery
      expect(SealRules.cleanSealNumber('X37408884'), 'FX37408884');

      // Homoglyph conversion in serial numbers
      expect(SealRules.cleanSealNumber('A29296O37'), 'A29296037'); // O -> 0
      expect(SealRules.cleanSealNumber('R59352OS'), 'R5935205');   // O -> 0, S -> 5
    });
  });

  group('SealResult Model Tests', () {
    test('parses JSON API response correctly', () {
      final json = {
        'status': 'SUCCESS',
        'sealNumber': 'OOLKCK28059',
        'rawText': 'oolkck28059',
        'confidence': 0.985,
        'source': 'OCR',
        'processingTimeMs': 142,
        'modelVersion': 'seal-ocr-det-v1-rec-v1',
      };

      final result = SealResult.fromJson(json);

      expect(result.sealNumber, 'OOLKCK28059');
      expect(result.seal, 'OOLKCK28059');
      expect(result.shippingLine, 'OOCL');
      expect(result.confidence, 0.985);
      expect(result.confidencePercent, '98.5%');
      expect(result.status, SealStatus.success);
      expect(result.isSuccess, true);
      expect(result.source, SealSource.ocr);
      expect(result.latencyMs, 142);
    });
  });
}
