"""Unit tests for global rule-based postprocessing in app/postprocess.py (ISO 17712 standard)."""

import pytest
from app.config import Settings
from app.schemas import Candidate
from app.postprocess import (
    clean_seal_number,
    valid,
    classify,
    recover_prefix_collapse,
    correct_homoglyphs,
    is_noise,
    normalize,
)
from app.seal_registry import (
    CARRIER_EXACT_LENGTH_RULES,
    NUMERIC_PREFIX_LENGTH_RULES,
    get_expected_seal_length,
)


def test_expected_seal_lengths():
    for prefix, expected in (CARRIER_EXACT_LENGTH_RULES | NUMERIC_PREFIX_LENGTH_RULES).items():
        assert get_expected_seal_length(prefix + "0" * expected) == expected
    assert get_expected_seal_length("UNKNOWN123") is None


def test_prefix_collapse_recovery_global():
    # JJ Shipping collapse fix
    assert recover_prefix_collapse("SJA696362") == "SJJA696362"
    assert recover_prefix_collapse("SJA700615") == "SJJA700615"
    assert recover_prefix_collapse("SJA703451") == "SJJA703451"
    # JJ Shipping optical variants (hook on J misread as 3 or 4)
    assert recover_prefix_collapse("SJ4A696373") == "SJJA696373"
    assert recover_prefix_collapse("SJ4700417") == "SJJA700417"
    assert recover_prefix_collapse("SJA4703516") == "SJJA703516"

    # OOCL multi-O and collapse fixes
    assert recover_prefix_collapse("O0LKCK1351") == "OOLKCK1351"
    assert recover_prefix_collapse("00LKCK2859") == "OOLKCK2859"
    assert recover_prefix_collapse("O0OLKCK1476") == "OOLKCK1476"
    assert recover_prefix_collapse("O0LKCL7951") == "OOLKCL7951"
    assert recover_prefix_collapse("O0LKCL7932") == "OOLKCL7932"

    # Yang Ming
    assert recover_prefix_collapse("YMA304983") == "YMAT304983"

    # SF Express optical fixes (horizontal line on seal makes F look like E, EF, SFE, SER)
    assert recover_prefix_collapse("SE0764765") == "SF0764765"
    assert recover_prefix_collapse("SEF0786017") == "SF0786017"
    assert recover_prefix_collapse("SER0313092") == "SF0313092"
    assert recover_prefix_collapse("SFE0767676") == "SF0767676"


def test_single_letter_prefix_safety():
    """Ensure single-letter prefixes (A..., R..., H...) NEVER mutate subsequent digits into letters."""
    assert clean_seal_number("A29296037") == "A29296037"
    assert clean_seal_number("R5935205") == "R5935205"
    assert clean_seal_number("R5847075") == "R5847075"
    assert clean_seal_number("H1234567") == "H1234567"
    # Tail homoglyph in single-letter prefix
    assert clean_seal_number("A29296O37") == "A29296037"
    assert clean_seal_number("R59352OS") == "R5935205"


def test_homoglyph_numeric_seals():
    """Purely numeric seals with accidental letters converted to digits."""
    assert correct_homoglyphs("287O5257") == "28705257"
    assert correct_homoglyphs("235594B") == "2355948"
    assert correct_homoglyphs("27178I7") == "2717817"
    assert correct_homoglyphs("2717817") == "2717817"
    assert correct_homoglyphs("1526244") == "1526244"


def test_homoglyph_alphanumeric_seals():
    """Alphanumeric seals with letters in tail converted to digits."""
    assert clean_seal_number("SJJA69636Z") == "SJJA696362"
    assert clean_seal_number("SJA696362") == "SJJA696362"
    assert clean_seal_number("OOLKCM5826") == "OOLKCM5826"
    assert clean_seal_number("O0LKCM5826") == "OOLKCM5826"
    assert clean_seal_number("FX37408884") == "FX37408884"
    assert clean_seal_number("WHAB154425") == "WHAB154425"
    assert clean_seal_number("YMAS608028") == "YMAS608028"
    assert clean_seal_number("HLC2683306") == "HLC2683306"
    assert clean_seal_number("TCS0015751") == "TCS0015751"
    assert clean_seal_number("VNHPH2501745") == "VNHPH2501745"


def test_global_carrier_and_noise_filtering():
    """Tests that global noise terms, door hardware words, and pure alpha brands are rejected."""
    settings = Settings()
    # Major global carriers stamped as text
    assert valid("WANHAI", settings) is False
    assert valid("YANGMING", settings) is False
    assert valid("MAERSK", settings) is False
    assert valid("COSCO", settings) is False
    assert valid("EVERGREEN", settings) is False
    assert valid("HAPAGLLOYD", settings) is False

    # Corrupted / OCR-misread brands without digits (0 digits)
    assert valid("YANGMNG", settings) is False
    assert valid("YANGNG", settings) is False
    assert valid("YANGM1NG", settings) is False

    # Global seal manufacturers
    assert valid("MEGAFORTRIS", settings) is False
    assert valid("TYDENBROOKS", settings) is False
    assert valid("ONESEAL", settings) is False

    # Security words & container door hardware
    assert valid("HIGHSECURITY", settings) is False
    assert valid("CONTAINER", settings) is False
    assert valid("CUSTOMS", settings) is False
    assert valid("DONOTREMOVE", settings) is False
    assert valid("MADEINCHINA", settings) is False

    # Short isolated noise
    assert valid("H", settings) is False
    assert valid("S", settings) is False

    # Valid seals must pass
    assert valid("SJJA696362", settings) is True
    assert valid("28705257", settings) is True
    assert valid("A29296037", settings) is True
    assert valid("FX37408884", settings) is True
    assert valid("OOLKCK1476", settings) is True


def test_classify_prioritizes_seal_over_brand():
    settings = Settings()
    candidates = [
        Candidate(text="YANGMING", confidence=0.99),
        Candidate(text="YMAS608028", confidence=0.95),
    ]
    response = classify(candidates, settings)
    assert response.status == "SUCCESS"
    assert response.sealNumber == "YMAS608028"


def test_classify_discards_misread_brand_names():
    settings = Settings()
    candidates = [
        Candidate(text="YANGMNG", confidence=0.98),
        Candidate(text="YMAR886137", confidence=0.94),
    ]
    response = classify(candidates, settings)
    assert response.status == "SUCCESS"
    assert response.sealNumber == "YMAR886137"


def test_classify_fixes_user_image_case():
    settings = Settings()
    candidates = [
        Candidate(text="JJ SHIPPING", confidence=0.88),
        Candidate(text="SJA696362", confidence=0.9146),
    ]
    response = classify(candidates, settings)
    assert response.status == "SUCCESS"
    assert response.sealNumber == "SJJA696362"


def test_classify_ranks_registered_carrier_higher():
    settings = Settings()
    candidates = [
        Candidate(text="RANDOM12345", confidence=0.92),
        Candidate(text="OOLKCK1234", confidence=0.90),
    ]
    response = classify(candidates, settings)
    assert response.status == "SUCCESS"
    assert response.sealNumber == "OOLKCK1234"


def test_classify_prioritizes_exact_registered_length():
    response = classify([
        Candidate(text="FX3739524", confidence=0.99),
        Candidate(text="FX37395224", confidence=0.80),
    ], Settings())
    assert response.sealNumber == "FX37395224"
    assert response.confidence == 0.80


def test_unregistered_prefix_still_ranks_by_confidence():
    response = classify([
        Candidate(text="XY12345", confidence=0.91),
        Candidate(text="XY123456", confidence=0.90),
    ], Settings())
    assert response.sealNumber == "XY12345"


def test_bevel_edge_ghost_zero_strip():
    """Verify that trailing 0 from circular pin bevel ring on R-series bolt seals is stripped."""
    assert clean_seal_number("R59245600") == "R5924560"
    assert clean_seal_number("R59386940") == "R5938694"
    assert clean_seal_number("R59539460") == "R5953946"
    # Normal 8-char R seal should not be modified
    assert clean_seal_number("R5924560") == "R5924560"


def test_optical_prefix_recovery():
    """Verify optical misreading recoveries for OOCL and FedEx."""
    # OOCL C3 -> CM (M optically misread as 3)
    assert clean_seal_number("OOLKC37120") == "OOLKCM7120"
    assert clean_seal_number("00LKC37120") == "OOLKCM7120"
    assert clean_seal_number("OOLKC3187") == "OOLKCM187"

    # FedEx boundary truncation (X37... -> FX37...)
    assert clean_seal_number("X37374441") == "FX37374441"
    assert clean_seal_number("X37397435") == "FX37397435"
