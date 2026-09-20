/// Registry and resolver for global shipping lines, container carriers, and logistics operators.
class ShippingLineResolver {
  static const Map<String, String> _prefixMap = {
    // OOCL
    'OOLKCK': 'OOCL',
    'OOLKCM': 'OOCL',
    'OOLKCR': 'OOCL',
    'OOLKCL': 'OOCL',
    'OOLK': 'OOCL',

    // SITC
    'SJJA': 'SITC',
    'SITC': 'SITC',
    'SITA': 'SITC',
    'SITF': 'SITC',
    'SITH': 'SITC',

    // Yang Ming
    'YMAT': 'Yang Ming',
    'YMAS': 'Yang Ming',
    'YMAR': 'Yang Ming',
    'YMLU': 'Yang Ming',
    'YML': 'Yang Ming',

    // Wan Hai
    'WHAB': 'Wan Hai Lines',
    'WHAC': 'Wan Hai Lines',
    'WHLU': 'Wan Hai Lines',
    'WHL': 'Wan Hai Lines',

    // Maersk
    'MAEU': 'Maersk Line',
    'MSK': 'Maersk Line',

    // MSC
    'MSCU': 'Mediterranean Shipping Co (MSC)',
    'MSC': 'Mediterranean Shipping Co (MSC)',

    // COSCO
    'COSU': 'COSCO Shipping',
    'COS': 'COSCO Shipping',

    // CMA CGM / APL
    'CMCU': 'CMA CGM',
    'CMA': 'CMA CGM',
    'CGM': 'CMA CGM',
    'APL': 'APL / CMA CGM',
    'CNC': 'CNC Line',
    'ANL': 'ANL',

    // ONE (Ocean Network Express)
    'ONEY': 'Ocean Network Express (ONE)',
    'ONE': 'Ocean Network Express (ONE)',

    // Hapag-Lloyd
    'HLCU': 'Hapag-Lloyd',
    'HLC': 'Hapag-Lloyd',

    // Evergreen
    'EGLV': 'Evergreen Line',
    'EMC': 'Evergreen Line',

    // HMM (Hyundai Merchant Marine)
    'HDMU': 'HMM',
    'HMM': 'HMM',

    // ZIM
    'ZIMU': 'ZIM',
    'ZIM': 'ZIM',

    // PIL (Pacific International Lines)
    'PILU': 'Pacific International Lines (PIL)',
    'PIL': 'Pacific International Lines (PIL)',

    // KMTC
    'KMTU': 'KMTC',
    'KMT': 'KMTC',

    // TS Lines
    'TSLU': 'TS Lines',
    'TSL': 'TS Lines',

    // SM Line
    'SMLU': 'SM Line',
    'SML': 'SM Line',

    // Express Couriers / Logistics
    'FX': 'FedEx Express',
    'SF': 'SF Express',
    'VNHPH': 'VIMC Lines',
    'VNH': 'VIMC Lines',
  };

  /// Resolves the shipping line or logistics operator from a cleaned seal number.
  /// Returns null if the seal number does not match a known carrier prefix.
  static String? resolve(String? sealNumber) {
    if (sealNumber == null || sealNumber.isEmpty) return null;

    final upper = sealNumber.toUpperCase();

    // Match longer prefixes first
    for (final entry in _prefixMap.entries) {
      if (upper.startsWith(entry.key)) {
        return entry.value;
      }
    }

    // Check single-letter series bolt seals (ISO 17712 generic)
    if (RegExp(r'^[A-Z][0-9]{5,10}$').hasMatch(upper)) {
      return 'Generic ISO 17712 Bolt Seal';
    }

    // Check pure numeric bolt seals
    if (RegExp(r'^[0-9]{5,12}$').hasMatch(upper)) {
      return 'Generic ISO 17712 Numeric Seal';
    }

    return null;
  }
}
