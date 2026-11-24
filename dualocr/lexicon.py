"""Dictionaries: Hunspell (Hebrew with prefix handling, English) + word
frequency lists + the document's own vocabulary."""
from __future__ import annotations

import math
import re
import threading
from collections import Counter
from functools import lru_cache

from .config import LEXICONS

_HEB = re.compile(r"[א-ת]")
_LAT = re.compile(r"[A-Za-z]")
_STRIP = re.compile(r"^[^\w֐-׿]+|[^\w֐-׿\"']+$")
_lock = threading.Lock()
_he = _en = None
_freq_he: dict = {}
_freq_en: dict = {}


def _load():
    global _he, _en, _freq_he, _freq_en
    with _lock:
        if _he is not None:
            return
        from spylls.hunspell import Dictionary

        _he = Dictionary.from_files(str(LEXICONS / "he_IL"))
        _en = Dictionary.from_files(str(LEXICONS / "en_US"))
        for fname, target, limit in (("he_full.txt", _freq_he, 400000), ("en_50k.txt", _freq_en, 50000)):
            p = LEXICONS / fname
            if p.exists():
                with p.open(encoding="utf-8") as f:
                    for k, line in enumerate(f):
                        if k >= limit:
                            break
                        parts = line.split()
                        if len(parts) == 2:
                            target[parts[0]] = int(parts[1])


def script_of(word: str) -> str:
    h, l = len(_HEB.findall(word)), len(_LAT.findall(word))
    if h and not l:
        return "he"
    if l and not h:
        return "en"
    if h and l:
        return "mixed"
    return "other"   # digits, punctuation


def core(word: str) -> str:
    """Word without surrounding punctuation (keeps inner gershayim)."""
    w = _STRIP.sub("", word)
    return w.strip("\"'")


@lru_cache(maxsize=200000)
def known(word: str) -> bool:
    """In a dictionary (Hebrew via Hunspell incl. prefixes, English)."""
    _load()
    w = core(word)
    if not w:
        return True
    sc = script_of(w)
    try:
        if sc == "he":
            return bool(_he.lookup(w)) or w in _freq_he
        if sc == "en":
            # dictionary only: the subtitle frequency list also contains common misspellings
            return bool(_en.lookup(w)) or bool(_en.lookup(w.lower()))
    except Exception:
        return False
    return sc == "other"


@lru_cache(maxsize=200000)
def log_freq(word: str) -> float:
    """log10 of corpus frequency (0 if unseen)."""
    _load()
    w = core(word)
    sc = script_of(w)
    if sc == "he":
        c = _freq_he.get(w, 0)
    elif sc == "en":
        c = _freq_en.get(w.lower(), 0)
    else:
        return 3.0
    return math.log10(c + 1)


class DocVocab:
    """Words seen in this document (weighted by how sure we were)."""

    def __init__(self):
        self.c = Counter()

    def add(self, word: str, weight: float = 1.0):
        w = core(word)
        if w:
            self.c[w] += weight

    def remove(self, word: str, weight: float = 1.0):
        w = core(word)
        if w:
            self.c[w] -= weight

    def count(self, word: str) -> float:
        return self.c.get(core(word), 0.0)


_HEB_ABC = "אבגדהוזחטיכךלמםנןסעפףצץקרשת"
_EN_ABC = "abcdefghijklmnopqrstuvwxyz"


@lru_cache(maxsize=50000)
def near_known(word: str) -> bool:
    """Is there a dictionary word one edit away? (Real typos are near-misses
    of real words; names and transliterations usually are not.)"""
    w = core(word)
    sc = script_of(w)
    if sc not in ("he", "en") or len(w) < 3:
        return False
    abc = _HEB_ABC if sc == "he" else _EN_ABC
    base = w if sc == "he" else w.lower()
    splits = [(base[:i], base[i:]) for i in range(len(base) + 1)]
    cands = set()
    for a, b in splits:
        if b:
            cands.add(a + b[1:])                       # delete
        if len(b) > 1:
            cands.add(a + b[1] + b[0] + b[2:])         # swap
        for ch in abc:
            if b:
                cands.add(a + ch + b[1:])              # replace
            cands.add(a + ch + b)                      # insert
    cands.discard(base)
    return any(known(c) for c in cands if len(c) >= 2)
