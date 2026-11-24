"""The whole process: import -> prep -> engines -> align -> vote -> rescan ->
structure -> outputs, with progress reporting and cancellation."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from PIL import Image

from . import consensus as C
from . import lexicon as L
from . import prep, rescan
from .align import AlignedBlock, Slot, align_page, letters
from .config import Settings
from .model import Block, Line
from .structure import PageLayout, Para, layout_page, numbering_report

log = logging.getLogger("dualocr")


class Cancelled(Exception):
    pass


@dataclass
class PageWork:
    index: int
    source: str
    image: Image.Image                 # upright RGB
    gray: Optional[Image.Image] = None
    lines: List[Line] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    aligned: List[AlignedBlock] = field(default_factory=list)
    orphans: list = field(default_factory=list)
    layout: Optional[PageLayout] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class Progress:
    stage: str
    page: int = 0          # 1-based page currently worked on
    pages: int = 0
    fraction: float = 0.0  # 0..1 overall
    detail: str = ""


STAGE_WEIGHTS = [("prepare", 0.10), ("read", 0.45), ("check", 0.35), ("write", 0.10)]


def _overall(stage: str, frac: float) -> float:
    done = 0.0
    for name, w in STAGE_WEIGHTS:
        if name == stage:
            return done + w * max(0.0, min(1.0, frac))
        done += w
    return done


# ---------------------------------------------------------------- deciding

def _orphan_paras(pw: PageWork) -> List[Para]:
    """Tesseract lines that fall outside every Surya block (Surya skipped them)."""
    from .structure import PICTURE_LABELS
    lines = {}
    for w in pw.orphans:
        lines.setdefault(w.line_key, []).append(w)
    out = []
    # with no Surya text on the page at all, Tesseract is all we have
    min_words, min_conf = (1, 40) if not any(b.text for b in pw.blocks) else (2, 75)
    for key, ws in sorted(lines.items(), key=lambda kv: min(w.bbox[1] for w in kv[1])):
        txt = [w for w in ws if letters(w.text)]
        if len(txt) < min_words or np.mean([w.conf for w in txt]) < min_conf:
            continue
        slots = []
        for w in ws:
            s = Slot(tess=w.text, t_conf=w.conf, bbox=w.bbox)
            s.text, s.conf, s.status, s.reason = w.text, min(0.79, w.conf / 100 * 0.8), "only_one", "only Tesseract"
            slots.append(s)
        box = (min(w.bbox[0] for w in ws), min(w.bbox[1] for w in ws), max(w.bbox[2] for w in ws),
               max(w.bbox[3] for w in ws))
        p = Para("text", slots, box)
        p._orphan = True
        out.append(p)
    return out


def _rescan_items(pw: PageWork, vocab, scripts, limit: int):
    items = []
    for ab in pw.aligned:
        if ab.block.label in ("Picture", "Figure", "Image", "Diagram"):
            continue
        ls = scripts[ab.index]
        for i, sl in enumerate(ab.slots):
            if not sl.bbox or not C.needs_rescan(sl, ls.get(sl.line, "he"), vocab):
                continue
            # context: neighbors on the same Tesseract line
            lo, hi = i, i + 1
            if i > 0 and ab.slots[i - 1].bbox and ab.slots[i - 1].t_line == sl.t_line:
                lo = i - 1
            if i + 1 < len(ab.slots) and ab.slots[i + 1].bbox and ab.slots[i + 1].t_line == sl.t_line:
                hi = i + 2
            boxes = [ab.slots[k].bbox for k in range(lo, hi) if ab.slots[k].bbox]
            box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
                   max(b[3] for b in boxes))
            expect, target = [], None
            for k in range(lo, hi):
                s2 = ab.slots[k]
                t = C.surya_reading(s2) or C.tess_reading(s2) or ""
                toks = t.split() or ["?"]
                if k == i:
                    target = (len(expect), len(expect) + len(toks))
                expect.extend(toks)
            items.append(rescan.RescanItem((ab.index, i), box, expect, target))
    # most doubtful first if over budget
    if len(items) > limit:
        def doubt(it):
            sl = pw.aligned[it.key[0]].slots[it.key[1]]
            return (sl.s_prob or 0.5) + (sl.t_conf or 0) / 100
        items.sort(key=doubt)
        items = items[:limit]
    return items


def measure_gaps(pw: PageWork) -> None:
    """For words where the engines disagree about spaces, measure the white
    gaps inside the word's box, relative to the line's normal word gap."""
    import cv2
    need = []
    for ab in pw.aligned:
        for sl in ab.slots:
            if not sl.bbox or not sl.t_words:
                continue
            s, t = C.surya_reading(sl), C.tess_reading(sl)
            if s and t and s != t and s.replace(" ", "") == t.replace(" ", ""):
                need.append(sl)
            elif sl.readings.get("tess_split") or sl.readings.get("surya_split"):
                need.append(sl)
    if not need:
        return
    if pw.gray is None:
        if pw.image is None:
            return
        pw.gray = prep.ocr_gray(pw.image)
    g = np.asarray(pw.gray)
    for sl in need:
        med = _median_gap(pw.lines, sl.t_line)
        if not med:
            continue
        x0, y0, x1, y1 = (int(v) for v in sl.bbox)
        crop = g[max(0, y0):y1, max(0, x0):x1]
        if crop.size == 0:
            continue
        _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink = (bw > 0).sum(axis=0) > 0
        runs, cur = [], 0
        seen_ink = False
        for v in ink:
            if v:
                if seen_ink and cur:
                    runs.append(cur)
                cur, seen_ink = 0, True
            else:
                cur += 1
        sl.readings["_gaps"] = [r / med for r in runs]


def _median_gap(lines, line_key):
    for ln in lines:
        if ln.key == line_key:
            xs = sorted((w.bbox for w in ln.words), key=lambda b: b[0])
            gaps = [b2[0] - b1[2] for b1, b2 in zip(xs, xs[1:]) if b2[0] - b1[2] > 0]
            if len(gaps) >= 2:
                return float(np.median(gaps))
    return None


def join_tight_numbers(ab: AlignedBlock, lines) -> None:
    """'2.' followed by a word with almost no gap is printed as one token
    ('2.חוק'); engines tend to insert a space. Decide from the pixels: the gap
    between the two boxes compared with the line's normal word gap."""
    from . import rules as R
    for a, b in zip(ab.slots, ab.slots[1:]):
        if not (a.text and b.text and a.bbox and b.bbox and a.t_line and a.t_line == b.t_line):
            continue
        if not R.is_list_number(a.text) or not letters(b.text):
            continue
        med = _median_gap(lines, a.t_line)
        if not med:
            continue
        # right-to-left: the number sits to the right of the word
        gap = a.bbox[0] - b.bbox[2] if L.script_of(b.text) == "he" else b.bbox[0] - a.bbox[2]
        h = max(1.0, b.bbox[3] - b.bbox[1])
        if gap < 0.35 * med and gap < 0.3 * h:
            b.text = a.text + b.text
            b.reason = (b.reason + "; " if b.reason else "") + "joined to number (tight gap)"
            a.text = ""


def _script_in_doubt(sl: Slot) -> bool:
    """The engines disagree on the alphabet, or one engine alone saw a word in a
    different alphabet than its line."""
    s, t = C.surya_reading(sl), C.tess_reading(sl)
    ls = sl.readings.get("_line_script", "he")
    if s is not None and t is not None:
        a, b = L.script_of(s), L.script_of(t)
        return a != b and {a, b} <= {"he", "en", "mixed"}
    r = s if s is not None else t
    return bool(r) and L.script_of(r) in ("he", "en") and L.script_of(r) != ls


def _letters_in_doubt(sl: Slot) -> bool:
    """Surya's (slower) zoomed re-read is used only when the letters themselves are in doubt."""
    s, t = C.surya_reading(sl), C.tess_reading(sl)
    if s is None or t is None:
        r = s if s is not None else t
        return bool(r and letters(r))
    return letters(s) != letters(t) or (sl.s_prob is not None and sl.s_prob < 0.85)


def decide_document(works: List[PageWork], settings: Settings, report: Callable = None,
                    cancelled: Callable = None, weights=None, do_rescans: bool = True):
    preset = settings.preset()
    C.IGNORE_SURYA_RESCANS = preset.get("surya_variants", 0) == 0
    from .config import HIGHLIGHT_THRESHOLDS
    C.FLAG_THRESHOLD = HIGHLIGHT_THRESHOLDS.get(getattr(settings, "highlight_level", "rare"), 0.50)
    w = weights if weights is not None else C.load_weights(settings.quality)
    for pw in works:
        lh = {ln.key: max(1.0, ln.bbox[3] - ln.bbox[1]) for ln in pw.lines}
        for ab in pw.aligned:
            C.tag_line_scripts(ab, lh)
        model = C.fit_widths(pw.aligned)
        for ab in pw.aligned:
            for sl in ab.slots:
                sl.readings["_wmodel"] = model
    vocab = C.build_vocab([pw.aligned for pw in works])
    preset = settings.preset()
    for n, pw in enumerate(works):
        if cancelled and cancelled():
            raise Cancelled()
        lh = {ln.key: max(1.0, ln.bbox[3] - ln.bbox[1]) for ln in pw.lines}
        scripts = {ab.index: C.tag_line_scripts(ab, lh) for ab in pw.aligned}
        rounds = preset["rescan_rounds"] if do_rescans else 0
        for rnd in range(rounds):
            if rnd == 0:
                items = _rescan_items(pw, vocab, scripts, limit=400)
            else:
                # later rounds: only words still unsure after the first look
                measure_gaps(pw)
                for ab in pw.aligned:
                    for sl in ab.slots:
                        C.decide_slot(sl, scripts[ab.index].get(sl.line, "he"), vocab, w)
                items = [it for it in _rescan_items(pw, vocab, scripts, limit=400)
                         if pw.aligned[it.key[0]].slots[it.key[1]].conf < C.FLAG_THRESHOLD]
            if not items:
                break
            if report:
                report(Progress("check", n + 1, len(works), _overall("check", (n + 0.3) / len(works)),
                                f"Page {n + 1} of {len(works)}: zooming in on {len(items)} doubtful words"
                                + (f" (round {rnd + 1})" if rounds > 1 else "")))
            sur = [it.key for it in items if _letters_in_doubt(pw.aligned[it.key[0]].slots[it.key[1]])]
            probes = [(it.key, pw.aligned[it.key[0]].slots[it.key[1]].bbox) for it in items
                      if rnd == 0 and _script_in_doubt(pw.aligned[it.key[0]].slots[it.key[1]])]
            res = rescan.run(pw.image, items, preset["variants"], use_surya=preset.get("surya_variants", 0) > 0,
                             surya_items=sur,
                             surya_variants=preset.get("surya_variants", 1), variant_offset=rnd, probes=probes)
            for (bi, si), reads in res.items():
                pw.aligned[bi].slots[si].readings.setdefault("rescan", []).extend(reads)
        measure_gaps(pw)
        for ab in pw.aligned:
            ls = scripts[ab.index]
            for sl in ab.slots:
                C.decide_slot(sl, ls.get(sl.line, "he"), vocab, w)
            join_tight_numbers(ab, pw.lines)
        pw.layout = layout_page(pw.index, pw.aligned, _orphan_paras(pw))
        if report:
            report(Progress("check", n + 1, len(works), _overall("check", (n + 1) / len(works)),
                            f"Page {n + 1} of {len(works)} checked"))
    return vocab


def page_text(pl: PageLayout) -> str:
    out = []
    for p in pl.paras:
        if p.kind == "picture":
            continue
        t = p.text
        if not t:
            continue
        if p.kind == "list_item" and not t.startswith("•") and not t[:3].strip(" .)").isdigit():
            pass
        out.append(("# " if p.kind == "heading" else "") + t)
    return "\n".join(out)


# ---------------------------------------------------------------- eval helper

_DOC_MEMO = {}


def _rkey(block_index: int, sl: Slot) -> str:
    b = sl.bbox or (0, 0, 0, 0)
    return f"{block_index}|{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])}|{sl.surya or ''}"


def decide_cached(cache_dir: Path, settings: Settings = None, weights=None, rescan_cache=True):
    """Run alignment + decisions on cached engine output (for evaluation).
    Rescan readings are cached per document so tuning runs are fast."""
    import json
    from . import pagecache

    settings = settings or Settings(quality="fast")
    works = []
    n = len(list(cache_dir.glob("p*_surya.json")))
    for i in range(1, n + 1):
        lines = pagecache.load_lines(cache_dir / f"p{i:02d}_tess3.json")
        blocks = pagecache.load_blocks(cache_dir / f"p{i:02d}_surya.json")
        aligned, orphans = align_page(lines, blocks)
        gray = Image.open(cache_dir / f"p{i:02d}.png").convert("L")
        works.append(PageWork(i - 1, cache_dir.name, None, gray, lines, blocks, aligned, orphans))
    rc = cache_dir / f"rescan_{settings.quality}.json"
    if settings.preset()["rescan_rounds"] and rescan_cache and rc.exists():
        stored = json.loads(rc.read_text(encoding="utf-8"))
        for pw in works:
            got = stored.get(str(pw.index), {})
            for ab in pw.aligned:
                for sl in ab.slots:
                    reads = got.get(_rkey(ab.index, sl))
                    if reads:
                        sl.readings["rescan"] = [tuple(r) for r in reads]
        works_vocab = decide_document(works, settings, weights=weights, do_rescans=False)
        works[0].notes.append(works_vocab)
        return works
    if settings.preset()["rescan_rounds"]:
        for pw in works:
            pw.image = Image.open(cache_dir / f"p{pw.index + 1:02d}.png").convert("RGB")
        works_vocab = decide_document(works, settings, weights=weights)
        works[0].notes.append(works_vocab)
        stored = {}
        for pw in works:
            d = {}
            for ab in pw.aligned:
                for sl in ab.slots:
                    if sl.readings.get("rescan"):
                        d[_rkey(ab.index, sl)] = sl.readings["rescan"]
            stored[str(pw.index)] = d
            pw.image = None
        rc.write_text(json.dumps(stored, ensure_ascii=False), encoding="utf-8")
        return works
    works_vocab = decide_document(works, settings, weights=weights)
    works[0].notes.append(works_vocab)
    return works


def page_text_from_cache(cache_dir: Path, n: int, quality: str = "fast") -> str:
    key = (str(cache_dir), quality)
    if key not in _DOC_MEMO:
        _DOC_MEMO[key] = decide_cached(cache_dir, Settings(quality=quality))
    return page_text(_DOC_MEMO[key][n - 1].layout)


# ---------------------------------------------------------------- full run

def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    k = 2
    while True:
        p = path.with_name(f"{path.stem} ({k}){path.suffix}")
        if not p.exists():
            return p
        k += 1


def _llm_pass(works: List[PageWork], settings: Settings, report: Callable, cancelled: Callable) -> int:
    from . import llm, rules as R
    key = llm.get_key(settings.llm_provider)
    if not key:
        return 0
    todo = []
    for pw in works:
        for ab in pw.aligned:
            for sl in ab.slots:
                if sl.status == "uncertain" and sl.bbox and sl.text is not None:
                    todo.append((sl.conf, pw, ab, sl))
    todo.sort(key=lambda x: x[0])
    todo = todo[: max(0, settings.llm_max_words)]
    done = 0
    for k, (_, pw, ab, sl) in enumerate(todo):
        if cancelled and cancelled():
            raise Cancelled()
        if report:
            report(Progress("check", pw.index + 1, len(works), _overall("check", 0.95),
                            f"Asking the AI tie-breaker about word {k + 1} of {len(todo)}"))
        same = [s.bbox for s in ab.slots if s.bbox and s.t_line == sl.t_line] or [sl.bbox]
        line_box = (min(b[0] for b in same), min(b[1] for b in same), max(b[2] for b in same), max(b[3] for b in same))
        cands = [c for c in (sl.readings.get("_cands") or {}) if c]
        try:
            ans = llm.ask(settings.llm_provider, key, settings.llm_model,
                          llm.marked_crop(pw.image, line_box, sl.bbox), cands)
        except Exception as e:
            log.warning("tie-breaker failed: %s", e)
            continue
        if not ans:
            continue
        ans = R.norm(ans)
        sl.readings["llm"] = ans
        if ans in cands or ans == sl.text:
            sl.text, sl.conf, sl.status = ans, max(sl.conf, 0.9), "resolved"
            sl.reason = (sl.reason + "; " if sl.reason else "") + "confirmed by AI tie-breaker"
            done += 1
    return done


def _verify_page_numbers(works: List[PageWork]) -> None:
    """Before reporting a page-number mismatch, re-read the footer digits-only."""
    from .engines import tesseract as T
    from collections import Counter
    have = [(pw.index + 1, pw.layout.printed_number) for pw in works if pw.layout.printed_number is not None]
    if not have:
        return
    offset = Counter(n - i for i, n in have).most_common(1)[0][0]
    for pw in works:
        pl = pw.layout
        expected = pw.index + 1 + offset
        if pl.printed_number is None or pl.printed_number == expected or not pl.footer_bbox:
            continue
        x0, y0, x1, y1 = pl.footer_bbox
        h = y1 - y0
        crop = pw.gray.crop((max(0, int(x0 - 2 * h)), max(0, int(y0 - h * .5)), min(pw.gray.width, int(x1 + 2 * h)),
                             min(pw.gray.height, int(y1 + h * .5))))
        crop = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)
        reads = set()
        for psm in (7, 8):
            tsv = T._run(crop, "eng", psm, extra=("-c", "tessedit_char_whitelist=0123456789"))
            for ln in T.parse_tsv(tsv):
                for w in ln.words:
                    if w.text.isdigit():
                        reads.add(int(w.text))
        if expected in reads:
            pw.notes.append(f"printed page number re-read as {expected}")
            pl.printed_number = expected


def process(paths, settings: Optional[Settings] = None, out_dir: Optional[Path] = None,
            report: Callable = None, cancelled: Callable = None, out_path: Optional[str] = None) -> dict:
    """Run everything and write the outputs. Returns paths and statistics."""
    from concurrent.futures import ThreadPoolExecutor

    from . import docx_writer, ingest
    from . import report as rep_mod
    from .engines import surya as SU, tesseract as T

    settings = settings or Settings.load()
    t0 = time.time()
    paths = [Path(p) for p in paths]
    total = ingest.count_pages(paths)

    def say(stage, page, frac, detail):
        if report:
            report(Progress(stage, page, total, _overall(stage, frac), detail))

    def check():
        if cancelled and cancelled():
            raise Cancelled()

    # 1. prepare pages
    works: List[PageWork] = []
    for p in ingest.load_pages(paths):
        check()
        say("prepare", p.index + 1, p.index / max(1, total), f"Page {p.index + 1} of {total}: turning upright")
        up, notes, _rot = prep.prepare(p.image, settings.languages)
        works.append(PageWork(p.index, p.source, up, prep.ocr_gray(up), notes=list(p.notes) + notes))
    total = len(works)
    if not works:
        raise ValueError("No pages found in the selected files.")

    # 2. read with both engines (Tesseract on the CPU while Surya uses the GPU)
    say("read", 1, 0.0, f"Reading {total} pages with Tesseract and Surya")
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(T.ocr_pages, [pw.gray for pw in works], settings.languages, 3)
        say("read", 1, 0.02, f"Starting Surya (page 1 of {total})")

        def page_done(k):
            check()
            say("read", k, k / total * 0.95, f"Surya has read {k} of {total} pages")
        caps = [SU.MAX_TOKENS_LOW_TEXT if any("little printed text" in n for n in pw.notes)
                else SU.MAX_TOKENS_PRINTED for pw in works]
        su_notes = [[] for _ in works]
        for pw, blocks, extra in zip(works, SU.ocr_pages([pw.image for pw in works], on_page=page_done, caps=caps,
                                                         notes=su_notes), su_notes):
            pw.blocks = blocks
            pw.notes.extend(extra)
        say("read", total, 0.97, "Waiting for Tesseract to finish")
        all_lines = fut.result()
    for pw, lines in zip(works, all_lines):
        pw.lines = lines
        pw.aligned, pw.orphans = align_page(pw.lines, pw.blocks)
    check()

    # 3. vote, zoom in on doubtful words, decide
    decide_document(works, settings, report=report, cancelled=cancelled)
    llm_fixed = 0
    if settings.llm_enabled:
        llm_fixed = _llm_pass(works, settings, report, cancelled)
    layouts = [pw.layout for pw in works]
    _verify_page_numbers(works)
    numbering = numbering_report(layouts)

    # 4. write outputs
    check()
    say("write", total, 0.1, "Writing the Word document")
    first = paths[0]
    if out_path:
        # the name the user typed in the Save panel (it already asked about replacing)
        target = Path(out_path).expanduser()
        if target.suffix.lower() != ".docx":
            target = target.with_name(target.name + ".docx")
        out_dir, stem = target.parent, target.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        named = lambda p: p          # noqa: E731  (overwrite: the user chose this name)
    else:
        out_dir = Path(out_dir or settings.output_dir or first.parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = first.stem + (f" +{len(paths) - 1}" if len(paths) > 1 else "")
        named = _unique
    out = {"docx": str(named(out_dir / f"{stem}.docx"))}
    images = [pw.image for pw in works]
    docx_writer.write_docx(out["docx"], layouts, images, title=stem, review=settings.review_highlights,
                           include_pictures=settings.include_pictures)
    if settings.write_txt:
        out["txt"] = str(named(out_dir / f"{stem}.txt"))
        docx_writer.write_txt(out["txt"], layouts)
    seconds = time.time() - t0
    if settings.write_report:
        say("write", total, 0.6, "Writing the review report")
        out["report"] = str(named(out_dir / f"{stem} - OCR report.html"))
        stats = rep_mod.write_report(out["report"], layouts, images, [p.name for p in paths], seconds, numbering,
                                     {pw.index: pw.notes for pw in works},
                                     f"quality: {settings.quality}" + (", AI tie-breaker on" if settings.llm_enabled else ""))
    else:
        stats = {}
    stats["llm_fixed"] = llm_fixed
    # keep everything needed to review and fix flagged words later
    try:
        from . import review as RV
        from .config import DATA
        rdir = DATA / "reviews" / (time.strftime("%Y%m%d-%H%M%S-") + re.sub(r"[^\w.-]+", "_", stem)[:60])
        out["review"] = str(RV.save(rdir, layouts, images, {
            "inputs": [str(p) for p in paths], "outputs": dict(out), "title": stem,
            "include_pictures": settings.include_pictures, "stats": stats, "numbering": numbering}))
    except Exception as e:  # reviewing is a convenience; never fail the job over it
        log.warning("could not save review data: %s", e)
    say("write", total, 1.0, "Done")
    return {"outputs": out, "stats": stats, "numbering": numbering, "pages": total,
            "seconds": round(time.time() - t0, 1)}
