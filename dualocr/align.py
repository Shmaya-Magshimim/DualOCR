"""Line up Tesseract words with Surya's text, block by block.

Surya gives paragraphs (blocks) with good reading order; Tesseract gives
every word's box and confidence. Each Tesseract word is placed in the Surya
block that contains it, then the two word sequences are aligned with a
dynamic-programming alignment that also allows one engine to split a word the
other joined ("זהנובע" vs "זה נובע").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from rapidfuzz.fuzz import ratio

from .model import Block, Line, Word

_HEBC = re.compile(r"[\u05D0-\u05EA]")
_LATC = re.compile(r"[A-Za-z]")
_INVIS = re.compile(r"[‎‏‪-‮⁦-⁩]")
_NONLET = re.compile(r"[^\w֐-׿]+")


def clean_token(t: str) -> str:
    return _INVIS.sub("", t)


def _sc(ch: str) -> str:
    if "\u05D0" <= ch <= "\u05EA":
        return "he"
    if ch.isascii() and ch.isalpha():
        return "en"
    return ""


def split_scripts(t: str) -> List[str]:
    """'Pesukim:ו-ט' -> ['Pesukim', ':ו-ט'];  'u-i:The' -> ['u-i:', 'The'].
    Hyphen compounds ('מן-Manna') are kept whole."""
    if not (_HEBC.search(t) and _LATC.search(t)):
        return [t]
    letters_at = [(i, _sc(c)) for i, c in enumerate(t) if _sc(c)]
    cuts = []
    for (i, a), (j, b) in zip(letters_at, letters_at[1:]):
        if a == b:
            continue
        between = t[i + 1:j]
        if "-" in between:          # hyphen compound: keep together
            continue
        # punctuation in between goes with the Hebrew side (bidi fix moves it later)
        cuts.append(i + 1 if b == "he" else j)
    if not cuts:
        return [t]
    parts, prev = [], 0
    for c in cuts:
        parts.append(t[prev:c])
        prev = c
    parts.append(t[prev:])
    parts = [p for p in parts if p]
    return parts if len(parts) > 1 else [t]


def letters(t: str) -> str:
    return _NONLET.sub("", t)


@dataclass
class Slot:
    surya: Optional[str] = None
    tess: Optional[str] = None
    s_prob: Optional[float] = None
    s_alts: list = field(default_factory=list)
    t_conf: Optional[float] = None
    t_words: List[Word] = field(default_factory=list)
    bbox: Optional[tuple] = None
    line: int = 0                  # line index inside the Surya block
    t_line: Optional[tuple] = None
    # filled by consensus
    readings: dict = field(default_factory=dict)
    text: str = ""
    conf: float = 0.0
    status: str = ""
    kinds: dict = field(default_factory=dict)
    reason: str = ""
    bold: bool = False
    underline: bool = False


@dataclass
class AlignedBlock:
    block: Block
    slots: List[Slot]
    index: int


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _area(b):
    return max(1.0, (b[2] - b[0]) * (b[3] - b[1]))


def assign(lines: List[Line], blocks: List[Block]):
    """Map each Tesseract word to a block. Returns (per_block_lines, orphans)."""
    per_block = [dict() for _ in blocks]
    orphans: List[Word] = []
    for ln in lines:
        for w in ln.words:
            cx, cy = _center(w.bbox)
            h = w.bbox[3] - w.bbox[1]
            best, best_area = None, None
            for k, b in enumerate(blocks):
                m = max(6.0, 0.35 * h)
                if b.bbox[0] - m <= cx <= b.bbox[2] + m and b.bbox[1] - m <= cy <= b.bbox[3] + m:
                    a = _area(b.bbox)
                    if best is None or a < best_area:
                        best, best_area = k, a
            if best is None:
                orphans.append(w)
            else:
                per_block[best].setdefault(ln.key, []).append(w)
    return per_block, orphans


def _sim(a: str, b: str) -> float:
    la, lb = letters(a), letters(b)
    if not la and not lb:
        return 1.0 if a == b else 0.6
    if not la or not lb:
        return 0.0
    return ratio(la, lb) / 100.0


GAP = -0.45


def _script(t: str) -> str:
    h, l = bool(_HEBC.search(t)), bool(_LATC.search(t))
    return "he" if h and not l else "en" if l and not h else "other"


def align_tokens(S: List[str], T: List[str]):
    """Return list of (s_idx_list, t_idx_list) pairs covering both sequences."""
    n, m = len(S), len(T)
    NEG = -1e9
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    cache = {}

    def cross(a, b):
        return _script(a) != _script(b) and _script(a) in ("he", "en") and _script(b) in ("he", "en") \
            and abs(len(letters(a)) - len(letters(b))) <= 2

    def sim(a, b):
        k = (a, b)
        v = cache.get(k)
        if v is None:
            v = cache[k] = _sim(a, b)
        return v

    for i in range(n + 1):
        for j in range(m + 1):
            cur = dp[i][j]
            if cur == NEG:
                continue
            if i < n and j < m:
                v = cur + sim(S[i], T[j]) * 2 - 1
                # the same word read in two different scripts (Hebrew "מן" vs
                # Latin "in"): pairing them costs a bit less than two gaps
                if cross(S[i], T[j]):
                    v = max(v, cur - 0.8)
                if v > dp[i + 1][j + 1]:
                    dp[i + 1][j + 1], bt[i + 1][j + 1] = v, (i, j, 1, 1)
            if i < n:
                v = cur + GAP
                if v > dp[i + 1][j]:
                    dp[i + 1][j], bt[i + 1][j] = v, (i, j, 1, 0)
            if j < m:
                v = cur + GAP
                if v > dp[i][j + 1]:
                    dp[i][j + 1], bt[i][j + 1] = v, (i, j, 0, 1)
            # one engine glued 2-4 words that the other kept apart
            for k in (2, 3, 4):
                if i < n and j + k <= m:
                    sm = sim(S[i], "".join(T[j:j + k]))
                    # a glued word only counts when the glued spelling really matches
                    v = cur + k * (2 * sm - 1) - 0.15 if sm >= 0.85 else NEG
                    if v > dp[i + 1][j + k]:
                        dp[i + 1][j + k], bt[i + 1][j + k] = v, (i, j, 1, k)
                if i + k <= n and j < m:
                    sm = sim("".join(S[i:i + k]), T[j])
                    v = cur + k * (2 * sm - 1) - 0.15 if sm >= 0.85 else NEG
                    if v > dp[i + k][j + 1]:
                        dp[i + k][j + 1], bt[i + k][j + 1] = v, (i, j, k, 1)
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        pi, pj, di, dj = bt[i][j]
        pairs.append((list(range(pi, pi + di)), list(range(pj, pj + dj))))
        i, j = pi, pj
    pairs.reverse()
    return pairs


def _union(boxes):
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def align_block(block: Block, t_lines: dict) -> List[Slot]:
    # Surya tokens with their line index and confidence info
    S, S_line, S_info, S_style = [], [], [], []
    tp = block.token_probs or []
    k = 0
    styles = []
    try:
        from .engines.surya import html_token_styles
        st = html_token_styles(block.html or "")
        if [x[0] for x in st] == block.text.split():
            styles = [(b, u) for _, b, u in st]
    except Exception:
        styles = []
    for li, ln in enumerate(block.text.split("\n")):
        for tok in ln.split():
            info = tp[k] if k < len(tp) else None
            style = styles[k] if k < len(styles) else (False, False)
            k += 1
            parts = split_scripts(clean_token(tok))
            for part in parts:
                S.append(part)
                S_style.append(style)
                S_line.append(li)
                S_info.append(info if len(parts) == 1 else (part, info[1] if info else None, []))
    # Tesseract words, lines ordered top to bottom
    T_words: List[Word] = []
    for key, ws in sorted(t_lines.items(), key=lambda kv: min(_center(w.bbox)[1] for w in kv[1])):
        for w in ws:
            w.text = clean_token(w.text)
            if not w.text:
                continue
            parts = split_scripts(w.text)
            if len(parts) == 1:
                T_words.append(w)
            else:
                for part in parts:
                    T_words.append(Word(part, w.bbox, w.conf, w.line_key, w.engine))
    T_words = _reorder_to_surya(T_words, S)
    T = [w.text for w in T_words]
    slots: List[Slot] = []
    for si, ti in align_tokens(S, T):
        sl = Slot()
        if si:
            sl.surya = "".join(S[x] for x in si) if len(si) > 1 else S[si[0]]
            sl.line = S_line[si[0]]
            sl.bold = any(S_style[x][0] for x in si)
            sl.underline = any(S_style[x][1] for x in si)
            probs = [S_info[x][1] for x in si if S_info[x] and S_info[x][1] is not None]
            sl.s_prob = min(probs) if probs else None
            if len(si) == 1 and S_info[si[0]] and len(S_info[si[0]]) > 2:
                sl.s_alts = S_info[si[0]][2]
            if len(si) > 1:
                sl.readings["surya_split"] = " ".join(S[x] for x in si)
        if ti:
            ws = [T_words[x] for x in ti]
            sl.tess = "".join(w.text for w in ws) if len(ws) > 1 else ws[0].text
            sl.t_conf = min(w.conf for w in ws)
            sl.t_words = ws
            sl.bbox = _union([w.bbox for w in ws])
            sl.t_line = ws[0].line_key
            if len(ws) > 1:
                sl.readings["tess_split"] = " ".join(w.text for w in ws)
        slots.append(sl)
    # line index for Tesseract-only slots: inherit from neighbor
    for i, sl in enumerate(slots):
        if sl.surya is None:
            prev = next((slots[j].line for j in range(i - 1, -1, -1) if slots[j].surya is not None), 0)
            sl.line = prev
    _estimate_missing_boxes(slots)
    return slots


def _reorder_to_surya(T_words: List[Word], S: List[str]) -> List[Word]:
    """Mixed Hebrew/English lines: the engines may list an embedded run in
    opposite orders. Within each Tesseract line, sort the words by the
    position of their best match in Surya's text (when the match is clear)."""
    if not T_words or not S:
        return T_words
    out, i = [], 0
    while i < len(T_words):
        j = i
        while j < len(T_words) and T_words[j].line_key == T_words[i].line_key:
            j += 1
        line = T_words[i:j]
        scripts = {_script(w.text) for w in line} - {"other"}
        if len(scripts) > 1 and len(line) > 1:
            # expected position of this line inside S (proportional), then best match near it
            pos = []
            for k, w in enumerate(line):
                best, bj = 0.0, None
                for sj, st in enumerate(S):
                    v = _sim(w.text, st)
                    if v > best:
                        best, bj = v, sj
                pos.append(bj if best >= 0.75 else None)
            # unmatched words keep their place relative to neighbors
            filled, last = [], None
            for k, p in enumerate(pos):
                if p is None:
                    nxt = next((q for q in pos[k + 1:] if q is not None), None)
                    p = (last + 0.5) if last is not None else ((nxt - 0.5) if nxt is not None else k)
                filled.append(p)
                last = p
            order = sorted(range(len(line)), key=lambda k: (filled[k], k))
            line = [line[k] for k in order]
        out.extend(line)
        i = j
    return out


def _estimate_missing_boxes(slots: List[Slot]):
    """Surya-only words have no box; estimate one between their neighbors."""
    for i, sl in enumerate(slots):
        if sl.bbox is not None:
            continue
        prev = next((slots[j] for j in range(i - 1, -1, -1) if slots[j].bbox), None)
        nxt = next((slots[j] for j in range(i + 1, len(slots)) if slots[j].bbox), None)
        if prev and nxt and abs(_center(prev.bbox)[1] - _center(nxt.bbox)[1]) < (prev.bbox[3] - prev.bbox[1]):
            # same line: right-to-left (Hebrew) or left-to-right
            x0 = min(prev.bbox[0], nxt.bbox[0]); x1 = max(prev.bbox[2], nxt.bbox[2])
            y0 = min(prev.bbox[1], nxt.bbox[1]); y1 = max(prev.bbox[3], nxt.bbox[3])
            sl.bbox = (x0, y0, x1, y1)
            sl.reason = "box estimated from neighbors"
        elif prev or nxt:
            b = (prev or nxt).bbox
            sl.bbox = b
            sl.reason = "box estimated from neighbor"


def align_page(lines: List[Line], blocks: List[Block]):
    per_block, orphans = assign(lines, blocks)
    out = []
    for k, b in enumerate(blocks):
        out.append(AlignedBlock(b, align_block(b, per_block[k]), k))
    return out, orphans
