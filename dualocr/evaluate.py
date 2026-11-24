"""Accuracy measurement against an answer key.

Text is normalized the same way on both sides (quote styles, dashes,
bullets, heading markers, editorial [notes] removed), then compared word by
word with a Levenshtein alignment, so missing, extra and wrong words all
count as errors:  accuracy = 1 - (substitutions + deletions + insertions) / N.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from rapidfuzz.distance import Levenshtein

_QUOTES = str.maketrans({"״": '"', "”": '"', "“": '"', "„": '"', "׳": "'", "’": "'", "‘": "'", "`": "'",
                         "–": "-", "—": "-", "־": "-", "●": "•", "▪": "•", "■": "•", "◦": "•",
                         " ": " ", "‏": "", "‎": ""})
_PUNCT = re.compile(r"[^\w֐-׿%]+", re.UNICODE)


def strip_editorial(t: str) -> str:
    """Remove [bracketed notes] (may nest)."""
    out, i, n = [], 0, len(t)
    while i < n:
        if t[i] == "[":
            depth, j = 0, i
            while j < n:
                if t[j] == "[":
                    depth += 1
                elif t[j] == "]":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j < n:          # balanced: skip the whole note
                i = j + 1
                continue
        out.append(t[i])
        i += 1
    return "".join(out)


def normalize(t: str, editorial: bool = True) -> str:
    t = t.translate(_QUOTES)
    if editorial:
        t = strip_editorial(t)
    lines = []
    for ln in t.split("\n"):
        ln = ln.strip()
        if ln.startswith("===== "):
            continue
        ln = re.sub(r"^#+\s*", "", ln)
        ln = re.sub(r"^•\s*", "", ln)
        lines.append(ln)
    t = " ".join(lines)
    t = t.replace("•", " ")
    return re.sub(r"\s+", " ", t).strip()


def tokens(t: str) -> List[str]:
    return normalize(t).split()


def letters(tok: str) -> str:
    return _PUNCT.sub("", tok)


@dataclass
class Score:
    n: int = 0
    word_errors: int = 0          # exact token (incl. punctuation)
    letter_errors: int = 0        # ignoring punctuation attached to words
    char_dist: int = 0
    char_n: int = 0
    errors: List[tuple] = field(default_factory=list)   # (op, gold, pred)

    @property
    def word_acc(self):
        return 1 - self.word_errors / self.n if self.n else 0.0

    @property
    def letter_acc(self):
        return 1 - self.letter_errors / self.n if self.n else 0.0

    @property
    def char_acc(self):
        return 1 - self.char_dist / self.char_n if self.char_n else 0.0

    def add(self, other: "Score"):
        self.n += other.n
        self.word_errors += other.word_errors
        self.letter_errors += other.letter_errors
        self.char_dist += other.char_dist
        self.char_n += other.char_n
        self.errors += other.errors

    def line(self, name=""):
        return (f"{name:<14} words {self.n:>5} | exact {self.word_acc:6.2%} | letters-only {self.letter_acc:6.2%}"
                f" | chars {self.char_acc:6.2%}")


def score(gold: str, pred: str, keep_errors: bool = True) -> Score:
    g, p = tokens(gold), tokens(pred)
    s = Score(n=len(g))
    ops = Levenshtein.editops(g, p)
    s.word_errors = len(ops)
    gl = [x for x in (letters(w) for w in g) if x]
    pl = [x for x in (letters(w) for w in p) if x]
    s.letter_errors = Levenshtein.distance(gl, pl)
    gs, ps = " ".join(g), " ".join(p)
    s.char_dist = Levenshtein.distance(gs, ps)
    s.char_n = len(gs)
    if keep_errors:
        for op in ops:
            if op.tag == "replace":
                s.errors.append(("sub", g[op.src_pos], p[op.dest_pos]))
            elif op.tag == "delete":
                s.errors.append(("miss", g[op.src_pos], ""))
            else:
                s.errors.append(("extra", "", p[op.dest_pos]))
    return s
