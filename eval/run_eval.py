"""Score systems against the answer keys.

usage: python eval/run_eval.py [system ...]      systems: tess4 tess3 surya combined
Tuning split for heb23: pages 1-10 = train, 11-20 = test (held out).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dualocr import evaluate as E, pagecache  # noqa: E402

CACHE = ROOT / "eval" / "cache"
GOLD = ROOT / "eval" / "gold"
_PAGENUM = re.compile(r"^\s*[-–]?\s*\d{1,3}\s*[-–]?\s*$")


def tess_text(doc, n, psm):
    lines = pagecache.load_lines(CACHE / doc / f"p{n:02d}_tess{psm}.json")
    rows = [" ".join(w.text for w in l.words) for l in lines]
    while rows and _PAGENUM.match(rows[-1]):
        rows.pop()
    return "\n".join(rows)


def surya_text(doc, n):
    blocks = pagecache.load_blocks(CACHE / doc / f"p{n:02d}_surya.json")
    out = []
    for b in sorted(blocks, key=lambda b: b.order):
        if b.label in ("PageFooter",) and _PAGENUM.match(b.text or ""):
            continue
        if b.text:
            out.append(b.text)
    return "\n".join(out)


def system_text(system, doc, n):
    if system == "tess4":
        return tess_text(doc, n, 4)
    if system == "tess3":
        return tess_text(doc, n, 3)
    if system == "surya":
        return surya_text(doc, n)
    if system in ("combined", "fast", "balanced", "max"):
        from dualocr.pipeline import page_text_from_cache
        q = "fast" if system == "combined" else system
        return page_text_from_cache(CACHE / doc, n, quality=q)
    raise ValueError(system)


def eval_heb23(system, show_errors=0):
    tr, te = E.Score(), E.Score()
    for n in range(1, 21):
        g = (GOLD / "heb23" / f"page_{n:02d}.txt").read_text()
        s = E.score(g, system_text(system, "heb23", n))
        (tr if n <= 10 else te).add(s)
    allsc = E.Score(); allsc.add(tr); allsc.add(te)
    print(tr.line(f"{system} p1-10"))
    print(te.line(f"{system} p11-20"))
    if show_errors:
        for e in te.errors[:show_errors]:
            print("   ", e)
    return tr, te


def _gold_order(doc, gold_text):
    """Scanned pages may be out of order: sort them by where each page's
    opening words best match the answer key."""
    from rapidfuzz.fuzz import partial_ratio_alignment
    n = len(list(CACHE.joinpath(doc).glob("p*_tess3.json")))
    g = E.normalize(gold_text)
    pos = {}
    for i in range(1, n + 1):
        t = " ".join(E.tokens(surya_text(doc, i))[:25])
        if not t:
            pos[i] = 1e9
            continue
        al = partial_ratio_alignment(t, g)
        pos[i] = al.dest_start
    return sorted(pos, key=pos.get)


def eval_eng14(system, show_errors=0, doc="eng14"):
    # the 22.41 scan lacks one page that the answer key (made from both scans) has
    g = (GOLD / "eng14" / ("scan2241.txt" if doc == "eng14" else "full.txt")).read_text()
    order = _gold_order(doc, g)
    pred = "\n".join(system_text(system, doc, i) for i in order)
    s = E.score(g, pred)
    print(s.line(f"{system} {doc}"))
    if show_errors:
        for e in s.errors[:show_errors]:
            print("   ", e)
    return s


if __name__ == "__main__":
    systems = [a for a in sys.argv[1:] if not a.startswith("-")] or ["tess4", "tess3", "surya"]
    show = 25 if "-v" in sys.argv else 0
    for sname in systems:
        eval_heb23(sname, show)
        for d in ("eng14", "eng14_blurry"):
            if CACHE.joinpath(d).exists() and list(CACHE.joinpath(d).glob("p*_surya.json")):
                eval_eng14(sname, show, d)
