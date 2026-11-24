"""Fast unit tests (no OCR engines needed)."""
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dualocr import evaluate as E, rules as R  # noqa: E402
from dualocr.align import Slot, align_tokens, split_scripts  # noqa: E402
from dualocr.structure import Para, PageLayout  # noqa: E402


# ---------------------------------------------------------------- alignment
def test_align_one_to_one():
    assert align_tokens(["שלום", "עולם"], ["שלום", "עולם"]) == [([0], [0]), ([1], [1])]


def test_align_glued_words():
    pairs = align_tokens(["זהו", "ביטוי", "למדינת", "ישראל"], ["זהוביטוילמדינת", "ישראל"])
    assert pairs[0] == ([0, 1, 2], [0])


def test_align_missing_word():
    pairs = align_tokens(["a", "b", "c"], ["a", "c"])
    assert ([1], []) in pairs


def test_align_cross_script_pairs_same_word():
    # Surya "מן" vs Tesseract "in" are one word read in two scripts
    pairs = align_tokens(["the", "מן,", "fell"], ["the", "in,", "fell"])
    assert ([1], [1]) in pairs


def test_split_scripts():
    assert split_scripts("Pesukim:ו-ט") == ["Pesukim", ":ו-ט"]
    assert split_scripts("מן-Manna") == ["מן-Manna"]
    assert split_scripts("פרק") == ["פרק"]


# ---------------------------------------------------------------- rules
def test_year_fix():
    assert R.year_fix('חשכ"ב') == 'תשכ"ב'
    assert R.year_fix('בחש"ח') == 'בתש"ח'
    assert R.year_fix("חשבון") is None
    assert R.year_fix('תש"י') is None


def test_fused_variants():
    assert "מציינים" in R.fused_variants("מציתים")       # ינ printed as ת
    assert "האזרחות" in R.fused_variants("האזרחת")       # swallowed ו


def test_bidi_punct_in_english_line():
    assert R.fix_bidi_punct(",מן", "en") == "מן,"
    assert R.fix_bidi_punct(':ל"ג', "en") == 'ל"ג:'
    assert R.fix_bidi_punct("-בקצה", "en") == "בקצה-"
    assert R.fix_bidi_punct(",word", "en") == ",word"      # English token untouched
    assert R.fix_bidi_punct(",מן", "he") == ",מן"          # only in left-to-right lines


def test_unmirror_parens():
    assert R.unmirror_parens(")1959(", "he") == "(1959)"
    assert R.unmirror_parens("(1959)", "he") == "(1959)"


def test_list_number():
    assert R.is_list_number("1.") and R.is_list_number("12.") and R.is_list_number("א.")
    assert not R.is_list_number("1.חוק")


# ---------------------------------------------------------------- scoring
def test_unbalanced_bracket_is_not_a_note():
    # a stray "[" read by an engine must not swallow the rest of the page
    assert E.normalize("בסיס 27[ לבן בין") == "בסיס 27[ לבן בין"
    assert E.normalize("טקסט [הערה] סוף") == "טקסט סוף"


def test_score_counts_all_error_types():
    s = E.score("a b c d", "a x c d e")
    assert s.n == 4 and s.word_errors == 2


# ---------------------------------------------------------------- docx
def _slot(text, status="agree"):
    s = Slot(surya=text, tess=text)
    s.text, s.status, s.conf = text, status, 0.99 if status == "agree" else 0.5
    return s


def test_docx_rtl_mixed_and_flags(tmp_path):
    from docx import Document
    from docx.oxml.ns import qn

    from dualocr.docx_writer import write_docx
    pl = PageLayout(0)
    pl.paras = [Para("heading", [_slot("כותרת"), _slot("ראשית")]),
                Para("text", [_slot("המונח"), _slot("liberty"), _slot("פירושו"), _slot("חופש", "uncertain")]),
                Para("text", [_slot("English"), _slot("line"), _slot("only")])]
    out = tmp_path / "t.docx"
    write_docx(out, [pl], [Image.new("RGB", (100, 100), "white")], review=True)
    d = Document(str(out))
    paras = [p for p in d.paragraphs if p.text.strip()]
    texts = [p.text.strip() for p in paras]
    assert "המונח liberty פירושו חופש" in texts
    heb = paras[texts.index("המונח liberty פירושו חופש")]
    assert heb._p.pPr.find(qn("w:bidi")) is not None                     # right-to-left paragraph
    eng = paras[texts.index("English line only")]
    assert eng._p.pPr.find(qn("w:bidi")) is None                         # left-to-right paragraph
    rtl_runs = [r.text for r in heb.runs if r._r.rPr is not None and r._r.rPr.find(qn("w:rtl")) is not None]
    assert any("המונח" in t for t in rtl_runs) and not any("liberty" in t for t in rtl_runs)
    flagged = [r for r in heb.runs if r.font.highlight_color is not None]
    assert flagged and "חופש" in flagged[0].text
    assert len(list(d.comments)) == 0          # no margin comments, highlight only


# ---------------------------------------------------------------- prep
def _text_image(rot=0):
    img = Image.new("RGB", (1700, 2200), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 46)
    y = 150
    for k in range(22):
        d.text((120, y), "The quick brown fox jumps over the lazy dog near the river bank.", font=f, fill="black")
        y += 80
    return img.rotate(-rot, expand=True) if rot else img


@pytest.mark.parametrize("rot", [0, 90, 180, 270])
def test_orientation(rot):
    from dualocr import prep
    o = prep.detect_orientation(_text_image(rot))
    assert (rot + o.rotate) % 360 == 0


def test_blank_page_kept_as_is():
    from dualocr import prep
    o = prep.detect_orientation(Image.new("RGB", (1200, 1600), "white"))
    assert o.rotate == 0


def test_skew_estimate():
    from dualocr import prep
    img = _text_image().rotate(1.5, expand=True, fillcolor="white")
    assert abs(prep.estimate_skew(img) + 1.5) < 0.4


# ---------------------------------------------------------------- AI tie-breaker (no network: faked)
def test_llm_tiebreaker_only_accepts_existing_candidates(monkeypatch):
    from dualocr import llm, pipeline
    from dualocr.align import AlignedBlock
    from dualocr.config import Settings
    from dualocr.model import Block

    def make(answer):
        sl = Slot(surya="המעון", tess="המנון", bbox=(100, 100, 200, 140))
        sl.text, sl.status, sl.conf = "המעון", "uncertain", 0.55
        sl.readings["_cands"] = {"המעון": 0.55, "המנון": 0.45}
        ab = AlignedBlock(Block("Text", (0, 0, 400, 400), "", "המעון", 0), [sl], 0)
        pw = pipeline.PageWork(0, "x", Image.new("RGB", (400, 400), "white"), aligned=[ab])
        monkeypatch.setattr(llm, "get_key", lambda p: "test-key")
        monkeypatch.setattr(llm, "ask", lambda *a, **k: answer)
        s = Settings(llm_enabled=True, llm_provider="claude", llm_max_words=5)
        n = pipeline._llm_pass([pw], s, None, None)
        return n, sl

    n, sl = make("המנון")                     # matches a candidate -> accepted
    assert n == 1 and sl.text == "המנון" and sl.status == "resolved"
    n, sl = make("משהו אחר")                  # invented text -> ignored, stays flagged
    assert n == 0 and sl.text == "המעון" and sl.status == "uncertain" and sl.readings["llm"] == "משהו אחר"


def test_llm_prompt_crop_marks_word():
    from dualocr import llm
    page = Image.new("RGB", (800, 300), "white")
    crop = llm.marked_crop(page, (100, 100, 700, 150), (300, 100, 400, 150))
    px = crop.convert("RGB").getdata()
    assert any(p[0] > 200 and p[1] < 60 and p[2] < 60 for p in px)    # red box drawn



def test_highlight_levels_and_old_settings(tmp_path, monkeypatch):
    import json
    from dualocr import config
    assert config.HIGHLIGHT_THRESHOLDS["rare"] < config.HIGHLIGHT_THRESHOLDS["some"] < config.HIGHLIGHT_THRESHOLDS["many"]
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"highlight_level": "normal", "quality": "fast"}))
    monkeypatch.setattr(config, "SETTINGS_FILE", f)
    s = config.Settings.load()
    assert s.highlight_level == "rare" and s.quality == "fast"


def test_chosen_output_name(tmp_path):
    from dualocr import pipeline
    # the name typed in the Save panel is used exactly (no " (2)"), and .docx is added if missing
    p = tmp_path / "My Notes"
    (tmp_path / "My Notes.docx").write_text("old")
    target = pipeline.Path(str(p))
    if target.suffix.lower() != ".docx":
        target = target.with_name(target.name + ".docx")
    assert target.name == "My Notes.docx"
