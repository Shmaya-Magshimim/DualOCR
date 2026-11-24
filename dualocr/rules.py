"""Text rules learned from checking real scans by eye.

1. Look-alike letters in common Hebrew fonts (bold David/Frank-Ruehl style):
   ת/ח, ד/ר, ה/ח, ב/כ, ם/ס, ו/י/ן, ע/צ.
2. Fused letter pairs: in tight bold fonts a י can melt into the following נ
   so "ינ" looks like "ת" ("מציינים" -> "מציתים", "כמדינה" -> "כמדיתה"), and
   a ו can sit on top of the letter before it (מו, או, חו, דו, טו) so it
   disappears ("האזרחות" -> "האזרחת").
3. Never "fix" a word that the engines and the zoom-in all read the same way,
   even if it is not in the dictionary: real documents contain typos, and the
   engines' habit of correcting spelling (לאומים -> לאומיים) is itself a
   major error source.
4. Latin letters inside a Hebrew line are almost always an engine failure on
   bold text ("ריבונות" -> "NINA"), and vice versa.
5. Bullets: Tesseract reads "•" as "+", "*", "e", "©"...
"""
from __future__ import annotations

import re

from .lexicon import script_of

QUOTE_MAP = str.maketrans({"״": '"', "”": '"', "“": '"', "„": '"', "׳": "'", "’": "'", "‘": "'",
                           "‏": "", "‎": "", " ": " "})

LOOKALIKE = {
    "ת": "חה", "ח": "תה", "ה": "חת", "ד": "ר", "ר": "ד", "ב": "כ", "כ": "ב",
    "ם": "ס", "ס": "ם", "ו": "יןז", "י": "ו", "ן": "ו", "ע": "צ", "צ": "ע", "ז": "ו",
    "ג": "נ", "נ": "ג",
}
SWALLOW_VAV_AFTER = set("מאחדטהי")
BULLET_JUNK = {"+", "*", "e", "©", "°", "¢", "«", "»", "»»", "ּ", "o", "-", "=", ".", "•", "●", "‏+"}
HEB_LETTER = re.compile(r"[א-ת]")


def norm(t: str) -> str:
    return t.translate(QUOTE_MAP).strip()


def fused_variants(word: str):
    """Spellings that look like `word` when letters fuse or look alike."""
    out = set()
    for i, ch in enumerate(word):
        if ch in ("ת", "ע"):      # fused ינ prints like ת (or ע)
            out.add(word[:i] + "ינ" + word[i + 1:])
        if ch in SWALLOW_VAV_AFTER and i + 1 < len(word) and HEB_LETTER.match(word[i + 1] if i + 1 < len(word) else ""):
            out.add(word[:i + 1] + "ו" + word[i + 1:])
        if ch in ("ק", "ך") and i == len(word) - 1:
            out.add(word[:i] + "ין")
    out.discard(word)
    return out


def lookalike_variants(word: str):
    out = set()
    for i, ch in enumerate(word):
        for alt in LOOKALIKE.get(ch, ""):
            out.add(word[:i] + alt + word[i + 1:])
    out.discard(word)
    return out


def is_junk_for_line(token: str, line_script: str) -> bool:
    """A token whose script doesn't fit its line (e.g. Latin in Hebrew)."""
    sc = script_of(token)
    if line_script == "he" and sc in ("en", "mixed"):
        return True
    if line_script == "en" and sc in ("he", "mixed"):
        return False  # Hebrew words inside English text are common (transliteration docs)
    return False


def is_bullet_junk(token: str) -> bool:
    return norm(token) in BULLET_JUNK


_LEAD = re.compile(r"^([,.:;!?-]+)(.+)$")


def fix_bidi_punct(token: str, line_script: str) -> str:
    """Engines reading a Hebrew word inside an English (left-to-right) line
    often put its trailing punctuation in front: ",מן" -> "מן,", ":ל"ג" -> "ל"ג:"."""
    if line_script != "en":
        return token
    m = _LEAD.match(token)
    if m and HEB_LETTER.search(m.group(2)) and not re.search(r"[,.:;!?-]$", m.group(2)) \
            and not re.search(r"[A-Za-z]", m.group(2)):
        return m.group(2) + m.group(1)
    return token


# Hebrew years are written in letters and always start with ת (400) for
# 5400+ (e.g. תש"ח, תשכ"ב, תשע"ח). In bold fonts ת is often read as ח.
YEAR = re.compile(r'^([בהומלכש]?)ח(ש[א-ת]{0,2}"[א-ת])(.*)$')


def year_fix(token: str):
    m = YEAR.match(token)
    if m:
        return m.group(1) + "ת" + m.group(2) + m.group(3)
    return None


_NUMDOT = re.compile(r"^\(?\d{1,3}[.)]$|^[א-י][.)]$")


def is_list_number(token: str) -> bool:
    return bool(_NUMDOT.match(token))


_MIRROR = re.compile(r"^\)([^()]*)\(([.,:;]?)$")


def unmirror_parens(token: str, line_script: str) -> str:
    """Right-to-left lines: engines sometimes return ')1959(' for '(1959)'."""
    m = _MIRROR.match(token)
    if m and line_script == "he":
        return "(" + m.group(1) + ")" + m.group(2)
    return token


def hard_fixes(text: str, line_script: str) -> str:
    """Deterministic corrections applied to the chosen text."""
    out = []
    for tok in text.split():
        y = year_fix(tok)
        if y:
            tok = y
        tok = unmirror_parens(tok, line_script)
        out.append(tok)
    return " ".join(out)


def one_vav_more(cand: str, other: str) -> bool:
    """cand == other with one extra ו (a ו the engine 'swallowed')."""
    if len(cand) != len(other) + 1:
        return False
    for i, ch in enumerate(cand):
        if ch == "ו" and cand[:i] + cand[i + 1:] == other:
            return True
    return False


def nun_yod_for_tav(cand: str, other: str) -> bool:
    """cand has 'ינ' where other has a single ת/ע (fused letters)."""
    if len(cand) != len(other) + 1:
        return False
    for i in range(len(other)):
        if other[i] in ("ת", "ע") and cand[:i] + "ינ" + cand[i + 2:] == other[:i] + "ינ" + other[i + 1:] \
                and cand[i:i + 2] == "ינ":
            return True
    return False
