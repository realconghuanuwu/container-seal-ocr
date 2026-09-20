/// Rule-based post-processing, homoglyph correction, and CTC recovery in pure Dart.
class SealRules {
  static const Map<String, String> homoglyphToDigit = {
    'O': '0', 'Q': '0', 'D': '0',
    'I': '1', 'L': '1',
    'B': '8',
    'S': '5',
    'Z': '2',
  };

  static const Map<String, String> homoglyphToAlpha = {
    '0': 'O',
    '1': 'I',
    '8': 'B',
    '5': 'S',
    '2': 'Z',
  };

  static const Set<String> allNoiseAndBrands = {
    'SEAL', 'SECURITY', 'DO NOT REMOVE', 'CONTAINER', 'MEGA FORTRIS',
    'ISO17712', 'H HIGH SECURITY', 'PASSED', 'CARGO', 'LOCK', 'BOLT',
    'SNAPPER', 'TYDENBROOKS', 'ONESEAL', 'LEGHORN', 'UNISTO',
  };

  static const List<String> sortedCarrierPrefixes = [
    'OOLKCK', 'OOLKCM', 'OOLKCR', 'OOLKCL', 'VNHPH',
    'OOLK', 'SJJA', 'YMAT', 'YMAS', 'YMAR', 'WHAB', 'WHAC', 'WHLU',
    'SITC', 'SITA', 'SITF', 'SITH', 'COSU', 'MAEU', 'MSCU',
    'HLCU', 'ONEY', 'EGLV', 'HDMU', 'YMLU', 'ZIMU', 'PILU',
    'KMTU', 'SMLU', 'TSLU', 'SKLU', 'HASU', 'RCOU', 'SNTO',
    'FX', 'SF', 'MSK', 'MSC', 'CMA', 'COS', 'ONE', 'WHL',
  ];

  /// Removes whitespace, converts to uppercase, and strips surrounding punctuation.
  static String normalize(String text) {
    var cleaned = text.toUpperCase().replaceAll(RegExp(r'\s+'), '');
    return cleaned.replaceAll(RegExp(r'^[-.:/_#"*~]+|[-.:/_#"*~]+$'), '');
  }

  /// Determines whether the string is non-seal noise (brand name or hardware mark).
  static bool isNoise(String text) {
    if (text.isEmpty) return true;
    if (allNoiseAndBrands.contains(text)) return true;

    final digitCount = text.codeUnits.where((c) => c >= 48 && c <= 57).length;
    if (digitCount < 3) return true;

    return false;
  }

  /// Recovers known optical misreads and CTC character collapse.
  static String recoverPrefixCollapse(String text) {
    if (text.isEmpty) return text;

    // 1. Multi-O collapse for OOCL (e.g. O0OLKCK -> OOLKCK, 00LK -> OOLK)
    if (RegExp(r'^[0O]{2,}LK').hasMatch(text)) {
      text = text.replaceFirst(RegExp(r'^[0O]{2,}LK'), 'OOLK');
    }

    // 2. SF Express optical prefix recovery
    final mSf = RegExp(r'^(SE|SEF|SFE|SER)([0-9]{6,8})$').firstMatch(text);
    if (mSf != null) {
      text = 'SF${mSf.group(2)}';
    }

    // 3. Jin Jiang Shipping optical variations
    final mSjj = RegExp(r'^(SJ[34]A?|SJA4A?)([0-9]{6,7})$').firstMatch(text);
    if (mSjj != null) {
      text = 'SJJA${mSjj.group(2)}';
    }

    // 4. OOCL optical letter misread (C3 -> CM)
    final mOocl = RegExp(r'^OOLKC3([0-9]{3,5})$').firstMatch(text);
    if (mOocl != null) {
      text = 'OOLKCM${mOocl.group(1)}';
    }

    // 5. FedEx prefix boundary cut (X37 -> FX37)
    final mFx = RegExp(r'^X(37[0-9]{6,8})$').firstMatch(text);
    if (mFx != null) {
      text = 'FX${mFx.group(1)}';
    }

    return text;
  }

  /// Context-aware homoglyph disambiguation (converts letters O/I/B/S/Z into digits 0/1/8/5/2 in serial tails).
  static String correctHomoglyphs(String text) {
    if (text.isEmpty) return text;

    // Case 1: Pure numeric seal (starts with digit, 5-12 chars)
    if (RegExp(r'^[0-9]').hasMatch(text) &&
        RegExp(r'^[0-9OIDQLBSZ]{5,12}$').hasMatch(text)) {
      return _replaceHomoglyphs(text);
    }

    // Case 2: Single-letter series prefix (e.g. A29296037, R5935205)
    if (RegExp(r'^[A-Z][0-9OIDQLBSZ]{5,11}$').hasMatch(text)) {
      final pref = text.substring(0, 1);
      final tail = _replaceHomoglyphs(text.substring(1));
      return '$pref$tail';
    }

    // Case 3: Registered global carrier prefix
    for (final pref in sortedCarrierPrefixes) {
      if (text.startsWith(pref)) {
        final rem = text.substring(pref.length);
        if (rem.length >= 3 && RegExp(r'^[0-9OIDQLBSZ]+$').hasMatch(rem)) {
          return '$pref${_replaceHomoglyphs(rem)}';
        }
      }
    }

    // Case 4: Generic alphanumeric seal (2-6 letters prefix + 4-10 serial digits)
    final m = RegExp(r'^([A-Z]{2,6})([0-9OIDQLBSZ]{4,10})$').firstMatch(text);
    if (m != null) {
      final pref = m.group(1)!;
      final tail = _replaceHomoglyphs(m.group(2)!);
      return '$pref$tail';
    }

    return text;
  }

  static String _replaceHomoglyphs(String s) {
    final sb = StringBuffer();
    for (var i = 0; i < s.length; i++) {
      final char = s[i];
      sb.write(homoglyphToDigit[char] ?? char);
    }
    return sb.toString();
  }

  /// Runs full cleaning, recovery, and homoglyph correction.
  static String cleanSealNumber(String text) {
    var cleaned = normalize(text);
    cleaned = recoverPrefixCollapse(cleaned);
    cleaned = correctHomoglyphs(cleaned);

    // Bevel stop-edge artifact for R-series
    if (RegExp(r'^R[0-9]{7}0$').hasMatch(cleaned)) {
      cleaned = cleaned.substring(0, cleaned.length - 1);
    }

    return cleaned;
  }
}
