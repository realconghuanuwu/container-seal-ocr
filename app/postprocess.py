import re

from .config import Settings
from .schemas import Candidate, OcrResponse
from .seal_registry import (
    ALL_NOISE_AND_BRANDS,
    AUTO_CTC_COLLAPSE_MAP,
    GLOBAL_CARRIER_BRANDS,
    GLOBAL_CARRIER_PREFIXES,
    GLOBAL_NOISE_TERMS,
    GLOBAL_SEAL_MANUFACTURERS,
    SORTED_CARRIER_PREFIXES,
    get_expected_seal_length,
)

# For backward compatibility
BRAND_NAMES = ALL_NOISE_AND_BRANDS
PREFIX_COLLAPSE_MAP = AUTO_CTC_COLLAPSE_MAP

HOMOGLYPH_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "B": "8",
    "S": "5",
    "Z": "2",
}

HOMOGLYPH_TO_ALPHA = {
    "0": "O",
    "1": "I",
    "8": "B",
    "5": "S",
    "2": "Z",
}


def normalize(text: str) -> str:
    """Removes whitespace, uppercases, and strips surrounding non-alphanumeric punctuation."""
    cleaned = "".join(text.upper().split())
    return cleaned.strip("-.:/_'\"#*~")


def is_noise(text: str) -> bool:
    """Determines whether text is noise (brand name, door hardware label, warning, or non-seal).

    In ISO 17712 container logistics, all mechanical seal serial numbers contain sequential
    numbers (at least 3-4 digits). Text with fewer than 3 digits represents carrier names,
    slogans, hardware markings, or isolated letters (e.g. H, S).
    """
    if not text:
        return True
    if text in ALL_NOISE_AND_BRANDS:
        return True

    # Check digit count: genuine container seals globally require at least 3-4 digits
    digit_count = sum(c.isdigit() for c in text)
    if digit_count < 3:
        return True

    # Fuzzy brand/noise check with digit-to-letter homoglyphs (e.g. YANGM1NG -> YANGMING)
    alpha_variant = "".join(HOMOGLYPH_TO_ALPHA.get(c, c) for c in text)
    if alpha_variant in ALL_NOISE_AND_BRANDS:
        return True

    return False


def recover_prefix_collapse(text: str) -> str:
    """Recovers known carrier prefix corruptions (CTC character collapse and optical errors)."""
    if not text:
        return text

    # 1. Multi-O collapse for OOCL (e.g. O0OLKCK -> OOLKCK, 00LK -> OOLK)
    if re.match(r"^[0O]{2,}LK", text):
        text = re.sub(r"^[0O]{2,}LK", "OOLK", text)

    # 2. SF Express optical prefix recovery (SE, SEF, SFE, SER -> SF when followed by 6-8 digits)
    m_sf = re.match(r"^(SE|SEF|SFE|SER)([0-9]{6,8})$", text)
    if m_sf:
        text = "SF" + m_sf.group(2)

    # 3. Jin Jiang Shipping optical variations (SJ4A, SJA4, SJ4, SJ3A -> SJJA when followed by 6-7 digits)
    m_sjj = re.match(r"^(SJ[34]A?|SJA4A?)([0-9]{6,7})$", text)
    if m_sjj:
        text = "SJJA" + m_sjj.group(2)

    # 4. OOCL optical letter misread (C3 -> CM when followed by 4 digits, e.g. OOLKC37120 -> OOLKCM7120)
    m_oocl = re.match(r"^OOLKC3([0-9]{3,5})$", text)
    if m_oocl:
        text = "OOLKCM" + m_oocl.group(1)

    # 5. FedEx prefix boundary cut (X37... -> FX37... when followed by 6-8 digits)
    m_fx = re.match(r"^X(37[0-9]{6,8})$", text)
    if m_fx:
        text = "FX" + m_fx.group(1)

    # 6. Global CTC collapse registry mapping
    for bad_pref, good_pref in AUTO_CTC_COLLAPSE_MAP.items():
        if text.startswith(bad_pref):
            rem = text[len(bad_pref):]
            if rem and (rem[0].isdigit() or rem[:2] in ("CK", "CM", "CR", "CL")):
                return good_pref + rem

    return text


def correct_homoglyphs(text: str) -> str:
    """Context-aware homoglyph disambiguation based on global ISO 17712 seal structures.

    Guarantees:
    - Never corrupts single-letter series prefixes (e.g. A29296037, R5935205).
    - Safely converts letter homoglyphs in numeric serial tails to digits.
    - Preserves registered carrier prefix codes.
    """
    if not text:
        return text

    # Case 1: Pure numeric seal (starts with a digit, 5-12 chars)
    if text[0].isdigit() and re.fullmatch(r"^[0-9OIDQLBSZ]{5,12}$", text):
        return "".join(HOMOGLYPH_TO_DIGIT.get(c, c) for c in text)

    # Case 2: Single-letter series prefix (e.g. A29296037, R5935205)
    # Prefix is 1 letter, serial is 5-11 digits. The 2nd character is strictly kept as digit.
    if re.fullmatch(r"^[A-Z][0-9OIDQLBSZ]{5,11}$", text):
        pref = text[0]
        tail = "".join(HOMOGLYPH_TO_DIGIT.get(c, c) for c in text[1:])
        return pref + tail

    # Case 3: Registered global carrier prefix (e.g. OOLKCK, SJJA, YMAT, WHAB, FX, SF, etc.)
    for pref in SORTED_CARRIER_PREFIXES:
        if text.startswith(pref):
            rem = text[len(pref):]
            if len(rem) >= 3 and re.fullmatch(r"^[0-9OIDQLBSZ]+$", rem):
                tail = "".join(HOMOGLYPH_TO_DIGIT.get(c, c) for c in rem)
                return pref + tail

    # Case 4: Generic alphanumeric seal (2-6 letters prefix + 4-10 serial digits)
    m = re.fullmatch(r"^([A-Z]{2,6})([0-9OIDQLBSZ]{4,10})$", text)
    if m:
        pref = m.group(1)
        tail = "".join(HOMOGLYPH_TO_DIGIT.get(c, c) for c in m.group(2))
        return pref + tail

    return text


def clean_seal_number(text: str) -> str:
    """Full normalization and rule-based correction pipeline."""
    cleaned = normalize(text)
    cleaned = recover_prefix_collapse(cleaned)
    cleaned = correct_homoglyphs(cleaned)

    # Bevel stop-edge artifact: R-series container bolt seals are strictly 8 chars (R + 7 digits).
    # The circular bevel stop ring on the pin tip is frequently misread as a trailing '0'.
    if re.match(r"^R[0-9]{7}0$", cleaned):
        cleaned = cleaned[:-1]

    return cleaned


def valid(text: str, settings: Settings) -> bool:
    """Validates whether cleaned text meets length, pattern, and non-noise criteria."""
    if is_noise(text):
        return False
    return (
        settings.seal_min_length <= len(text) <= settings.seal_max_length
        and re.fullmatch(settings.seal_allowed_pattern, text) is not None
    )


def _score_candidate(candidate: Candidate, cleaned: str) -> float:
    """Scores a candidate based on OCR confidence and structural conformance."""
    weight = 1.0

    # Higher weight for registered carrier prefixes with valid serials
    for pref in SORTED_CARRIER_PREFIXES:
        if cleaned.startswith(pref) and len(cleaned) > len(pref):
            weight = 1.25
            break
    else:
        # Standard pure numeric or single-letter series
        if cleaned and (cleaned[0].isdigit() or re.match(r"^[A-Z][0-9]{5,}$", cleaned)):
            weight = 1.15

    # Optimal length bonus (standard seal numbers are typically 6-12 chars)
    if 6 <= len(cleaned) <= 12:
        weight *= 1.05

    expected_length = get_expected_seal_length(cleaned)
    if expected_length:
        if len(cleaned) == expected_length:
            weight *= 1.35
        elif len(cleaned) == expected_length - 1:
            weight *= 0.85

    return candidate.confidence * weight


def classify(candidates: list[Candidate], settings: Settings) -> OcrResponse:
    """Ranks candidates by structural conformance and confidence to produce an OcrResponse."""
    processed_candidates = []
    for candidate in candidates:
        cleaned = clean_seal_number(candidate.text)
        if valid(cleaned, settings):
            score = _score_candidate(candidate, cleaned)
            processed_candidates.append((candidate, cleaned, score))

    ranked = sorted(
        processed_candidates,
        key=lambda item: item[2],
        reverse=True,
    )
    if not ranked:
        return OcrResponse(
            status="RECAPTURE", reason="NO_VALID_SEAL", processingTimeMs=0,
            modelVersion=settings.model_version,
        )
    candidate, text, _ = ranked[0]
    if candidate.confidence < settings.review_threshold:
        status, reason = "RECAPTURE", "LOW_CONFIDENCE"
    elif candidate.confidence < settings.success_threshold:
        status, reason = "REVIEW", None
    else:
        status, reason = "SUCCESS", None
    return OcrResponse(
        status=status, sealNumber=text, rawText=candidate.text,
        confidence=candidate.confidence, source="OCR", reason=reason,
        processingTimeMs=0, modelVersion=settings.model_version,
    )
