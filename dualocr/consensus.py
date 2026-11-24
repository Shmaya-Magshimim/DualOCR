"""Decide each word: candidates, evidence features, and a learned scorer.

For every aligned word slot we collect candidate spellings (Surya's reading,
Tesseract's, rescan readings, Surya's own letter alternatives, look-alike and
fused-letter variants), describe each with evidence features, and score them
with a small conditional-logit model whose weights were learned from the
answer-key pages (see eval/train.py). The winner's probability is the word's
confidence; low-confidence words get highlighted for review.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

import numpy as np

from . import lexicon as L
from . import rules as R
from .align import AlignedBlock, Slot, letters
from .config import MODELS

FEATURES = [
    "surya", "tess", "both", "surya_p", "tess_c", "surya_alt", "resc_t", "resc_s", "resc_any",
    "script_s", "script_t", "script_fit",
    "doc_stem", "doc_bigram", "width_best", "width_err", "single_letter", "year", "heb_ref", "vav_more", "inn_for_tav", "lookalike", "numdot_join", "gersh_s", "gap_ok", "gap_bad", "known_he", "known_en", "freq", "doc", "variant", "junk", "drop", "drop_tess_only", "drop_surya_only",
    "drop_conf", "drop_known", "drop_short",
    "punct_s", "punct_t", "split_s", "split_t", "unknown_both_he", "unknown_both_en", "tess_lowconf",
    "surya_lowp",
]
DEFAULT_W = {
    "surya": 1.6, "tess": 0.9, "both": 1.5, "surya_p": 1.5, "tess_c": 1.0, "surya_alt": 2.0,
    "resc_t": 1.5, "resc_s": 2.5, "resc_any": 0.5, "doc_stem": 0.6, "doc_bigram": 0.8, "width_best": 0.8, "width_err": -1.5,
    "single_letter": -0.3, "year": 3.0, "heb_ref": 1.0, "vav_more": 0.8, "inn_for_tav": 0.8, "lookalike": -0.5,
    "numdot_join": 0.5, "gersh_s": 1.5, "gap_ok": 1.0,
    "gap_bad": -1.0, "known_he": 0.8, "known_en": 0.8, "freq": 0.3,
    "doc": 0.8, "variant": -1.5, "junk": -4.0, "drop": -0.8, "drop_tess_only": 0.8, "drop_surya_only": -2.0,
    "drop_conf": -2.0, "drop_known": -1.0, "drop_short": 1.0,
    "punct_s": 0.4, "punct_t": 0.0, "split_s": 0.2, "split_t": 0.0, "unknown_both_he": 0.0,
    "unknown_both_en": 0.0,
    "tess_lowconf": -0.8, "surya_lowp": -0.8, "script_s": 0.5, "script_t": -0.3, "script_fit": 4.0,
}
WEIGHTS_FILE = MODELS / "consensus_weights.json"
import os as _os
IGNORE_SURYA_RESCANS = _os.environ.get("DUALOCR_NO_SURYA_RESCAN") == "1"   # for ablation tests
_HEBREF = __import__("re").compile(r'^[\u05D0-\u05EA]{1,2}("[\u05D0-\u05EA])?(-[\u05D0-\u05EA]{1,2}("[\u05D0-\u05EA])?)?[:,.]?$')
_NUMWORD = __import__("re").compile(r"^\(?\d{1,3}[.)][^\s\d.)]")
FLAG_THRESHOLD = 0.50     # set per run from Settings.highlight_level


def weights_file(quality: str):
    return MODELS / f"consensus_weights_{quality}.json"


def load_weights(quality: str = "balanced") -> np.ndarray:
    w = dict(DEFAULT_W)
    for f in (weights_file(quality), WEIGHTS_FILE):
        if f.exists():
            try:
                w.update(json.loads(f.read_text(encoding="utf-8"))["weights"])
                break
            except Exception:
                pass
    return np.array([w.get(f, 0.0) for f in FEATURES])


def tag_line_scripts(ab: AlignedBlock, line_heights: Optional[dict] = None) -> Dict[int, str]:
    ls = line_scripts(ab)
    prev = None
    for i, sl in enumerate(ab.slots):
        sl.readings["_line_script"] = ls.get(sl.line, "he")
        sl.readings["_prev"] = prev
        nxt = ab.slots[i + 1] if i + 1 < len(ab.slots) else None
        sl.readings["_next"] = L.core(surya_reading(nxt) or tess_reading(nxt) or "") if nxt else None
        prev = L.core(surya_reading(sl) or tess_reading(sl) or "") or None
        if line_heights and sl.t_line in line_heights:
            sl.readings["_line_h"] = line_heights[sl.t_line]
    return ls


def line_scripts(ab: AlignedBlock) -> Dict[int, str]:
    counts: Dict[int, Dict[str, int]] = {}
    for sl in ab.slots:
        t = sl.surya or sl.tess or ""
        sc = L.script_of(t)
        if sc in ("he", "en"):
            counts.setdefault(sl.line, {}).setdefault(sc, 0)
            counts[sl.line][sc] += 1
    out = {}
    for li, c in counts.items():
        out[li] = "he" if c.get("he", 0) >= c.get("en", 0) else "en"
    return out


def _fix(t: Optional[str], sl: Slot) -> Optional[str]:
    if t is None:
        return None
    t = R.norm(t)
    ls = sl.readings.get("_line_script", "he")
    return " ".join(R.fix_bidi_punct(x, ls) for x in t.split()) if ls == "en" else t


def surya_reading(sl: Slot) -> Optional[str]:
    return _fix(sl.readings.get("surya_split") or sl.surya, sl)


def tess_reading(sl: Slot) -> Optional[str]:
    return _fix(sl.readings.get("tess_split") or sl.tess, sl)


def build_vocab(pages: List[List[AlignedBlock]]) -> L.DocVocab:
    v = L.DocVocab()
    v.bigrams = {}
    for blocks in pages:
        for ab in blocks:
            prev = None
            for sl in ab.slots:
                s, t = surya_reading(sl), tess_reading(sl)
                ok = s and t and s == t and (sl.s_prob or 0) > 0.9 and (sl.t_conf or 0) > 80
                if ok:
                    for w in s.split():
                        v.add(w)
                    sl.readings["_vocab"] = s
                cur = L.core(s or t or "") if (s or t) else None
                if prev and cur:
                    v.bigrams[(prev, cur)] = v.bigrams.get((prev, cur), 0) + 1
                prev = cur
    return v


# ---------------------------------------------------------------- letter widths
_WCH = None


def fit_widths(blocks: List[AlignedBlock]) -> Optional[dict]:
    """Learn each character's width (in units of its line's height) on this
    page from words both engines read the same way with high confidence.
    Lets us check whether a spelling fits the printed word's measured width."""
    rows, ys, chars = [], [], {}
    for ab in blocks:
        for sl in ab.slots:
            s, t = surya_reading(sl), tess_reading(sl)
            if not (s and t and s == t and len(sl.t_words) == 1 and (sl.t_conf or 0) > 85 and " " not in s):
                continue
            lh = sl.readings.get("_line_h")
            if not lh:
                continue
            w = (sl.bbox[2] - sl.bbox[0]) / lh
            rows.append(s); ys.append(w)
            for ch in s:
                chars.setdefault(ch, len(chars))
    if len(rows) < 40:
        return None
    X = np.zeros((len(rows), len(chars)))
    for i, r in enumerate(rows):
        for ch in r:
            X[i, chars[ch]] += 1
    y = np.array(ys)
    lam = 0.5
    mean = float(y.sum() / max(1, X.sum()))
    A = X.T @ X + lam * np.eye(len(chars))
    b = X.T @ y + lam * mean
    a = np.linalg.solve(A, b)
    return {"w": {ch: float(max(0.05, a[k])) for ch, k in chars.items()}, "mean": mean}


def width_error(cand: str, sl: Slot, model: Optional[dict]) -> Optional[float]:
    if not model or not sl.bbox or len(sl.t_words) != 1 or " " in cand or not cand:
        return None
    lh = sl.readings.get("_line_h")
    if not lh:
        return None
    pred = sum(model["w"].get(ch, model["mean"]) for ch in cand)
    meas = (sl.bbox[2] - sl.bbox[0]) / lh
    return abs(meas - pred) / max(pred, 1e-6)


def _alt_support(cand: str, sread: Optional[str], alts: list) -> float:
    """Did Surya consider `cand`'s letter at the one position where it differs?"""
    if not sread or not alts or len(cand) != len(sread):
        return 0.0
    diff = [i for i in range(len(cand)) if cand[i] != sread[i]]
    if len(diff) != 1:
        return 0.0
    i = diff[0]
    for pos, opts in alts:
        if pos == i:
            for tok, p in opts:
                if tok == cand[i]:
                    return p
    return 0.0


def candidates(sl: Slot, line_script: str, vocab: L.DocVocab) -> Dict[str, set]:
    c: Dict[str, set] = {}
    s, t = surya_reading(sl), tess_reading(sl)
    if s is not None:
        c.setdefault(s, set()).add("surya")
    if t is not None:
        if R.is_bullet_junk(t) and not letters(t):
            t = "•" if (s is None or s == "•") else t
        c.setdefault(t, set()).add("tess")
    for src, r in sl.readings.get("rescan", []):
        if src.startswith("probe_") or (IGNORE_SURYA_RESCANS and src.startswith("surya")):
            continue
        r = R.norm(r)
        if r:
            c.setdefault(r, set()).add(src)
    # Surya's own letter alternatives (single-letter substitutions)
    if s and sl.s_alts and len(s.split()) == 1:
        for pos, opts in sl.s_alts:
            for tok, p in opts:
                if p >= 0.05 and len(tok) == 1 and pos < len(s) and tok != s[pos] and R.HEB_LETTER.match(tok) \
                        and R.HEB_LETTER.match(s[pos]):
                    v = s[:pos] + tok + s[pos + 1:]
                    c.setdefault(v, set()).add("alt")
    # fused-letter variants of what the engines read (kept only if plausible)
    for base in [x for x in (s, t) if x and len(x.split()) == 1 and L.script_of(x) == "he"]:
        core_b = L.core(base)
        for v in R.fused_variants(core_b):
            full = base.replace(core_b, v, 1)
            if L.known(v) and (vocab.count(v) > 0 or L.log_freq(v) > L.log_freq(core_b) + 1.0
                               or not L.known(core_b)):
                c.setdefault(full, set()).add("variant")
    # (Tried: single-letter variants of doubled י/ו with a letter-width check.
    #  On the answer key the width check was right 3 times and wrong 3 times:
    #  a narrow letter is ~10% of the word width, about the measurement noise.
    #  Not reliable enough to override two engines that agree, so not used.)
    # an engine's reading that is not a word: one look-alike letter away from a word?
    for base in [x for x in (s, t) if x and len(x.split()) == 1 and L.script_of(x) == "he"]:
        core_b = L.core(base)
        if core_b and not L.known(core_b):
            for v in R.lookalike_variants(core_b):
                if L.known(v) and L.log_freq(v) >= 2.0:
                    c.setdefault(base.replace(core_b, v, 1), set()).add("lookalike")
    for base in [x for x in (s, t) if x]:
        y = R.year_fix(base)
        if y:
            c.setdefault(y, set()).add("year")
    if s is None or t is None:
        c.setdefault("", set()).add("drop")
    return c


def featurize(cand: str, srcs: set, sl: Slot, line_script: str, vocab: L.DocVocab,
              letters_same: bool) -> np.ndarray:
    s, t = surya_reading(sl), tess_reading(sl)
    is_s = 1.0 if (s is not None and cand == s) else 0.0
    is_t = 1.0 if (t is not None and cand == t) else 0.0
    resc = sl.readings.get("rescan", [])
    probe = {src[6:]: r for src, r in resc if src.startswith("probe_")}
    resc = [(src, r) for src, r in resc if not src.startswith("probe_")
            and not (IGNORE_SURYA_RESCANS and src.startswith("surya"))]
    rt = [r for src, r in resc if src.startswith("tess")]
    rs = [r for src, r in resc if src.startswith("surya")]
    f = dict.fromkeys(FEATURES, 0.0)
    if cand == "":
        f["drop"] = 1.0
        f["drop_tess_only"] = 1.0 if (t is not None and s is None) else 0.0
        f["drop_surya_only"] = 1.0 if (s is not None and t is None) else 0.0
        single = s if s is not None else t
        if single is not None and (s is None) != (t is None):
            f["drop_conf"] = (sl.s_prob if sl.s_prob is not None else 0.7) if s is not None else (sl.t_conf or 0) / 100
            ws = single.split()
            is_known = bool(ws) and all(L.known(x) for x in ws) and bool(letters(single))
            f["drop_known"] = 0.0 if is_known else 1.0      # i.e. "the dropped reading is not a word"
            f["drop_short"] = 1.0 if (len(letters(single)) <= 1 and not is_known) else 0.0
        if rt or rs:
            f["resc_any"] = sum(1 for r in rt + rs if not letters(R.norm(r))) / len(rt + rs)
        return np.array([f[k] for k in FEATURES])
    f["surya"], f["tess"] = is_s, is_t
    f["both"] = is_s * is_t
    f["surya_p"] = (sl.s_prob if sl.s_prob is not None else 0.7) if is_s else 0.0
    f["tess_c"] = ((sl.t_conf or 0) / 100.0) if is_t else 0.0
    f["surya_alt"] = _alt_support(cand, s, sl.s_alts)
    if rt:
        f["resc_t"] = sum(1 for r in rt if R.norm(r) == cand) / len(rt)
    if rs:
        f["resc_s"] = sum(1 for r in rs if R.norm(r) == cand) / len(rs)
    if rt or rs:
        f["resc_any"] = sum(1 for r in rt + rs if R.norm(r) == cand) / len(rt + rs)
    words = cand.split()
    known = float(all(L.known(w) for w in words)) if words else 0.0
    sc = L.script_of(cand)
    f["known_he"] = known if sc == "he" else 0.0
    f["known_en"] = known if sc != "he" else 0.0
    f["freq"] = min(L.log_freq(w) for w in words) / 7.0 if words else 0.0
    dc = min(vocab.count(w) for w in words) if words else 0.0
    if sl.readings.get("_vocab") == cand:
        dc -= 1
    f["doc"] = math.log1p(max(0.0, dc)) / 3.0
    if len(words) == 1 and L.script_of(cand) == "he":
        core_c = L.core(cand)
        best = 0.0
        for k in (1, 2):   # strip up to two prefix letters (ו ה ב כ ל מ ש)
            if len(core_c) > k + 1 and all(ch in "והבכלמש" for ch in core_c[:k]):
                best = max(best, vocab.count(core_c[k:]))
        f["doc_stem"] = math.log1p(best) / 3.0
    f["variant"] = 1.0 if (srcs <= {"variant", "alt", "year", "lookalike", "single_letter"}) else 0.0
    f["single_letter"] = 1.0 if "single_letter" in srcs else 0.0
    bg = getattr(vocab, "bigrams", {})
    if bg and len(words) == 1:
        cc = L.core(cand)
        n = bg.get((sl.readings.get("_prev"), cc), 0) + bg.get((cc, sl.readings.get("_next")), 0)
        if sl.readings.get("_vocab") == cand:
            n = max(0, n - 2)
        f["doc_bigram"] = math.log1p(n) / 2.0
    werr = sl.readings.get("_werr", {}).get(cand)
    if werr is not None:
        f["width_err"] = min(1.0, werr * 4)
        best = sl.readings.get("_wbest")
        f["width_best"] = 1.0 if (best == cand) else 0.0
    f["lookalike"] = 1.0 if "lookalike" in srcs else 0.0
    lc = letters(cand)
    for other in (s, t):
        if other is not None and other != cand:
            lo = letters(other)
            if R.one_vav_more(lc, lo):
                f["vav_more"] = 1.0
            if R.nun_yod_for_tav(lc, lo):
                f["inn_for_tav"] = 1.0
    f["year"] = 1.0 if "year" in srcs else 0.0
    f["heb_ref"] = 1.0 if (line_script == "en" and _HEBREF.match(cand)) else 0.0
    if len(words) == 1:
        m = _NUMWORD.match(cand)
        if m:
            f["numdot_join"] = 1.0
    # Tesseract tends to drop gershayim/geresh (בס"ד -> בסד); Surya keeps them
    if is_s and t is not None and letters(cand) == letters(t) and cand.count('"') + cand.count("'") > \
            t.count('"') + t.count("'"):
        f["gersh_s"] = 1.0
    # pixel evidence about spaces (filled in by the pipeline when the page image is available)
    gaps = sl.readings.get("_gaps")
    if gaps is not None and letters(cand):
        k = len(words) - 1
        big = sorted(gaps, reverse=True)
        if k > 0:
            # each space needs a word-sized white gap inside the joined box
            f["gap_ok"] = 1.0 if len(big) >= k and min(big[:k]) >= 0.55 else 0.0
            f["gap_bad"] = 1.0 if len(big) < k or min(big[:k]) < 0.35 else 0.0
        else:
            f["gap_bad"] = 1.0 if big and big[0] >= 0.8 else 0.0
            f["gap_ok"] = 1.0 if not big or big[0] < 0.45 else 0.0
    f["junk"] = 1.0 if any(R.is_junk_for_line(w, line_script) for w in words) else 0.0
    f["punct_s"] = is_s * letters_same
    f["punct_t"] = is_t * letters_same
    f["split_s"] = is_s * (1.0 if len(words) > 1 else 0.0)
    f["split_t"] = is_t * (1.0 if len(words) > 1 else 0.0)
    f["unknown_both_he"] = (1.0 - known) * is_s * is_t * (1.0 if sc == "he" else 0.0)
    f["unknown_both_en"] = (1.0 - known) * is_s * is_t * (1.0 if sc == "en" else 0.0)
    if s is not None and t is not None:
        ss, ts = L.script_of(s), L.script_of(t)
        if ss != ts and "other" not in (ss, ts):
            f["script_s"] = is_s
            f["script_t"] = is_t
    if probe and cand:
        try:
            he = float(probe.get("heb", "0\t").split("\t")[0]); en = float(probe.get("eng", "0\t").split("\t")[0])
            sc = L.script_of(cand)
            if sc == "he":
                f["script_fit"] = max(-1.0, min(1.0, (he - en) / 40.0))
            elif sc == "en":
                f["script_fit"] = max(-1.0, min(1.0, (en - he) / 40.0))
        except ValueError:
            pass
    f["tess_lowconf"] = is_t * (1.0 if (sl.t_conf or 0) < 50 else 0.0)
    f["surya_lowp"] = is_s * (1.0 if (sl.s_prob is not None and sl.s_prob < 0.5) else 0.0)
    return np.array([f[k] for k in FEATURES])


def slot_matrix(sl: Slot, line_script: str, vocab: L.DocVocab):
    cands = candidates(sl, line_script, vocab)
    names = list(cands)
    model = sl.readings.get("_wmodel")
    if model:
        errs = {n: width_error(n, sl, model) for n in names if n}
        errs = {k: v for k, v in errs.items() if v is not None}
        sl.readings["_werr"] = errs
        if len(errs) >= 2:
            ranked = sorted(errs.items(), key=lambda kv: kv[1])
            sl.readings["_wbest"] = ranked[0][0] if ranked[1][1] - ranked[0][1] > 0.04 else None
    ls = {letters(x) for x in names if x}
    letters_same = 1.0 if len(ls) == 1 else 0.0
    X = np.stack([featurize(n, cands[n], sl, line_script, vocab, letters_same) for n in names]) if names else None
    return names, cands, X


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def decide_slot(sl: Slot, line_script: str, vocab: L.DocVocab, w: np.ndarray) -> None:
    names, cands, X = slot_matrix(sl, line_script, vocab)
    if not names:
        sl.text, sl.conf, sl.status = "", 1.0, "empty"
        return
    p = softmax(X @ w)
    k = int(p.argmax())
    sl.text = R.hard_fixes(names[k], line_script) if names[k] else names[k]
    sl.conf = float(p[k])
    s, t = surya_reading(sl), tess_reading(sl)
    sl.kinds = error_kinds(sl, sl.text, line_script)
    sl.readings["_cands"] = {names[i]: round(float(p[i]), 3) for i in np.argsort(-p)[:4]}
    if s is not None and t is not None and s == t == sl.text:
        sl.status = "agree" if sl.conf >= FLAG_THRESHOLD else "uncertain"
    else:
        sl.status = "resolved" if sl.conf >= FLAG_THRESHOLD else "uncertain"
    if sl.text and s == t == sl.text and letters(sl.text) and L.script_of(sl.text) in ("he", "en"):
        unknown = [x for x in sl.text.split() if not L.known(x)]
        # a real typo is a one-off near-miss of a dictionary word; repeated
        # unknown words are names / terms / transliterations
        if unknown and all(vocab.count(x) <= 1 and L.near_known(x) for x in unknown):
            sl.status = "source_typo" if sl.conf >= FLAG_THRESHOLD else "uncertain"
    sl.reason = ", ".join(sorted(cands.get(sl.text, set()))) if sl.text in cands else sl.reason


def error_kinds(sl: Slot, final: str, line_script: str) -> dict:
    """What kind of mistake did each engine make on this word?"""
    out = {}
    for eng, r in (("surya", surya_reading(sl)), ("tesseract", tess_reading(sl))):
        if r is None:
            if final:
                out[eng] = "missed word"
            continue
        if r == final:
            continue
        out[eng] = classify_error(r, final, line_script)
    return out


def classify_error(read: str, final: str, line_script: str) -> str:
    if final == "":
        return "extra text"
    if letters(read) == letters(final):
        if read.replace(" ", "") == final.replace(" ", ""):
            return "spacing"
        return "punctuation"
    if R.is_junk_for_line(read, line_script) or (L.script_of(read) != L.script_of(final)
                                                  and L.script_of(final) in ("he", "en")):
        return "wrong script / garbage"
    if read.replace(" ", "") == final.replace(" ", ""):
        return "word split/join"
    lr, lf = letters(read), letters(final)
    if len(lr) == len(lf):
        diffs = [(a, b) for a, b in zip(lr, lf) if a != b]
        if len(diffs) == 1:
            a, b = diffs[0]
            if b in R.LOOKALIKE.get(a, "") or a in R.LOOKALIKE.get(b, ""):
                return f"look-alike letters ({a}→{b})"
            return f"one letter ({a}→{b})"
    if lf.replace("ינ", "ת") == lr or (len(lf) == len(lr) + 1 and "ו" in lf):
        return "fused letters"
    if L.known(read) and not L.known(final):
        return "spelling 'corrected' by engine"
    if L.known(read) and L.known(final) and len(lr) != len(lf):
        return "spelling 'corrected' by engine"
    return "misread word"


def needs_rescan(sl: Slot, line_script: str, vocab: L.DocVocab) -> bool:
    s, t = surya_reading(sl), tess_reading(sl)
    if s is None and t is None:
        return False
    if s is None or t is None:
        r = s if s is not None else t
        return bool(letters(r)) and not R.is_bullet_junk(r)
    if s != t:
        return True
    if (sl.s_prob is not None and sl.s_prob < 0.85) or (sl.t_conf or 0) < 60:
        return True
    if L.script_of(s) == "he" and len(s.split()) == 1:
        core = L.core(s)
        for v in R.fused_variants(core):
            if L.known(v) and (vocab.count(v) > 0 or L.log_freq(v) > L.log_freq(core) + 1.5):
                return True
    return False
