"""Global Seal Knowledge Base & Carrier Registry (ISO 17712 & ISO 6346 BIC Standards).

Contains standardized dictionaries of global shipping lines, container leasing companies,
seal manufacturers, noise keywords, and automated CTC collapse recovery rules.
"""

from typing import Dict, Set

# 1. Top Global Shipping Lines, Regional Carriers, Logistics, and Leasing Prefixes
# Order matters: longer compound prefixes (e.g. OOLKCK) must match before shorter ones (OOLK).
GLOBAL_CARRIER_PREFIXES: list[str] = [
    # Compound prefixes (Carrier + Depot / Seal series)
    "OOLKCK", "OOLKCM", "OOLKCR", "OOLKCL", "VNHPH",
    # 4-character shipping line & BIC carrier prefixes
    "OOLK", "SJJA", "YMAT", "YMAS", "YMAR", "WHAB", "WHAC", "WHLU",
    "SITC", "SITR", "SITZ", "SITA", "SITF", "SITH", "COSU", "MAEU", "MSCU",
    "HLCU", "ONEY", "EGLV", "HDMU", "YMLU", "ZIMU", "PILU",
    "KMTU", "SMLU", "TSLU", "SKLU", "HASU", "RCOU", "SNTO",
    "SNTU", "ZGOU", "QASU", "MATU", "CMCU", "SBDU", "GRIU",
    "ARKU", "ESLU", "UNFU", "CULU", "MRAU", "SPIL", "SAMU",
    "HATS",
    # 3-character shipping line / courier codes
    "MSK", "MSC", "CMA", "CGM", "COS", "ONE", "HMM", "YML",
    "WHL", "WHA", "ZIM", "PIL", "KMT", "SML", "TSL", "RCL", "HLC",
    "TCS", "APL", "CNC", "ANL", "VNH", "EMC",
    # 2-character shipping line / express courier codes
    "FX", "SF", "ML", "EM", "CO", "MS", "HM", "HL", "VS",
    # Container Leasing Companies (BIC prefixes)
    "TRHU", "TRLU", "TCXU", "TEXU", "TGHU", "FCIU", "FBLU",
    "CAIU", "SEGU", "BOXU", "BCHU", "TXGU", "BSLU",
]

CARRIER_EXACT_LENGTH_RULES: dict[str, int] = {
    "FX33": 10,
    "FX37": 10,
    "FX38": 10,
    "FX39": 10,
    "FX40": 10,
    "FX": 10,
    "SITR": 10,
    "SITZ": 10,
    "SITF": 10,
    "SITC": 10,
    "WHA": 10,
    "WHLT": 10,
    "WHAB": 10,
    "SJJA": 10,
    "YMAR": 10,
    "YMAT": 10,
    "YMAS": 10,
    "OOLKC": 10,
    "SF": 9,
    "HLC": 10,
    "A29": 9,
    "A28": 9,
    "R58": 8,
    "R59": 8,
    "TCS": 10,
    "VNHPH": 12,
}

NUMERIC_PREFIX_LENGTH_RULES: dict[str, int] = {
    "2869": 8,
    "2870": 8,
    "2871": 8,
    "2886": 8,
    "2717": 7,
    "2718": 7,
}

_SORTED_EXACT_LENGTH_RULES = sorted(
    (CARRIER_EXACT_LENGTH_RULES | NUMERIC_PREFIX_LENGTH_RULES).items(),
    key=lambda item: len(item[0]),
    reverse=True,
)


def get_expected_seal_length(text: str) -> int | None:
    """Returns the required length for a registered seal-number prefix."""
    return next((length for prefix, length in _SORTED_EXACT_LENGTH_RULES
                 if text.startswith(prefix)), None)

# 2. Global ISO 17712 Seal Manufacturers & Product Brand Names
GLOBAL_SEAL_MANUFACTURERS: Set[str] = {
    "MEGAFORTRIS", "MEGA", "FORTRIS", "KLICKER",
    "TYDENBROOKS", "TYDEN", "BROOKS", "SNAPPER", "SNAPLOCK", "EZLOC",
    "ONESEAL", "1SEAL",
    "UNISTO",
    "LEGHORNGROUP", "LEGHORN",
    "ACME", "ACMESEALS",
    "BEDNORZ",
    "ENVASEAL",
    "HARCOR",
    "CAMBRIDGE",
    "UNIVERSEAL",
    "JWPRODUCTS",
}

# 3. Global Shipping Line Brand Names (When stamped separately from serial)
GLOBAL_CARRIER_BRANDS: Set[str] = {
    "MAERSK", "MSK", "APMOLLER",
    "MSC", "MEDLOG",
    "CMACGM", "CMA", "CGM", "APL", "ANL", "CNC",
    "COSCO", "COSCOSHIPPING", "CSCL",
    "HAPAG", "LLOYD", "HAPAGLLOYD",
    "ONE", "OCEANNETWORKEXPRESS",
    "EVERGREEN", "EMC",
    "HYUNDAI", "HMM",
    "YANGMING", "YML",
    "ZIM",
    "WANHAI", "WHL",
    "OOCL",
    "PIL", "PACIFICINTERNATIONALLINES",
    "SITC",
    "KMTC",
    "SMLINE", "SML",
    "TSLINES", "TSL",
    "SINOKOR", "SNKO",
    "HEUNGA",
    "RCL",
    "JJSHIPPING", "JINJIANG",
    "SINOTRANS",
    "ZHONGGU",
    "ANTONG",
    "MATSON",
    "CROWLEY",
    "SEABOARD",
    "GRIMALDI",
    "ARKAS",
    "EMIRATES",
    "UNIFEEDER",
    "SWIRE",
    "CULINES",
    "SEALEAD",
    "MERATUS",
    "SPIL",
    "SAMUDERA",
    "VINALINES", "VIMC", "GEMADEPT", "HAIPHONG",
    "FEDEX", "SFEXPRESS",
}

# 4. Universal Noise Lexicon: Container Hardware, Security, Origin & Warnings
# These words frequently appear on container doors, seal caps, and latches.
GLOBAL_NOISE_TERMS: Set[str] = {
    # Security terms
    "HIGH", "SECURITY", "HIGHSECURITY", "SEAL", "SEALS", "BOLT", "LOCK",
    "LOCKED", "CABLE", "CUSTOMS", "INSPECTION", "ISO", "ISO17712", "17712",
    "GRADE", "TYPE", "CLASS", "APPROVED", "VERIFIED", "ORIGINAL", "GENUINE",
    "SECURED", "PASSED", "CHECK", "TESTED", "BARRIER", "TAMPER", "EVIDENT",
    # Container hardware & logistics words
    "CONTAINER", "CONTAINERS", "DOOR", "DOORS", "CARGO", "FREIGHT",
    "LOGISTICS", "LINE", "LINES", "SHIPPING", "PORT", "TERMINAL", "OCEAN",
    "PACIFIC", "ATLANTIC", "GLOBAL", "WORLD", "EXPRESS", "INTERMODAL",
    "TRANSPORT", "TRANSPORTATION", "HANDLE", "LATCH", "BAR",
    # Operational notices & warnings
    "DO", "NOT", "REMOVE", "ENTER", "WARNING", "DANGER", "NOTICE", "CAUTION",
    "ATTENTION", "DONOTREMOVE",
    # Origin & Patent
    "MADE", "IN", "CHINA", "MALAYSIA", "USA", "VIETNAM", "TAIWAN", "GERMANY",
    "JAPAN", "KOREA", "DENMARK", "UK", "FRANCE", "CORP", "CO", "LTD", "INC",
    "PATENT", "PATENTED", "PAT",
    # Short isolated noise
    "H", "S", "I", "OK", "NO",
}

# Combined noise set for fast lookup
ALL_NOISE_AND_BRANDS: Set[str] = (
    GLOBAL_NOISE_TERMS
    | GLOBAL_CARRIER_BRANDS
    | GLOBAL_SEAL_MANUFACTURERS
)


def _build_ctc_collapse_map() -> Dict[str, str]:
    """Dynamically builds CTC collapse mapping for prefixes with duplicate adjacent letters,
    plus optical homoglyph substitutions (e.g. 00LK -> OOLK).
    """
    mapping: Dict[str, str] = {
        # Explicit optical substitutions
        "00LK": "OOLK",
        "O0LK": "OOLK",
        "0OLK": "OOLK",
        "S1TC": "SITC",
        "YMA": "YMAT",      # Yang Ming truncation
        "SJA": "SJJA",      # JJ Shipping CTC collapse
        "GITR": "SITR",     # SITC Logistics G/S optical confusion
        "FGITR": "SITR",    # SITC bounding box edge noise
        "SGITR": "SITR",    # SITC bounding box edge noise
        "G6ITR": "SITR",    # SITC bounding box edge noise
        "SIT7R": "SITR",    # SITC stroke split noise
        "WH4": "WHA",       # Wan Hai 4/A optical confusion
        "SIT27": "SITZ",    # SITC Z/27 stroke confusion
    }

    for prefix in GLOBAL_CARRIER_PREFIXES:
        # Check for adjacent duplicate characters (e.g. JJ in SJJA, OO in OOLK, etc.)
        collapsed = []
        for i, char in enumerate(prefix):
            if i > 0 and char == prefix[i - 1]:
                continue
            collapsed.append(char)

        collapsed_str = "".join(collapsed)
        if (
            collapsed_str != prefix
            and collapsed_str not in mapping
            and collapsed_str not in GLOBAL_CARRIER_PREFIXES
        ):
            mapping[collapsed_str] = prefix

    return mapping


AUTO_CTC_COLLAPSE_MAP: Dict[str, str] = _build_ctc_collapse_map()

# Sort prefixes by length descending so that compound prefixes match first
SORTED_CARRIER_PREFIXES: list[str] = sorted(GLOBAL_CARRIER_PREFIXES, key=len, reverse=True)
