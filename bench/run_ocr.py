"""Run every system on a prepared dataset, one engine pass per page.

usage: python bench/run_ocr.py DATASET [--limit N] [--batch 8] [--systems a,b,...]

Per page:
  1. DualOCR page prep (orientation, deskew, highlighter removal)
  2. Tesseract psm 3 on the prepped page (exactly DualOCR's Tesseract pass)
     + Tesseract psm 3 on the raw image (what a plain `tesseract` user gets)
  3. Surya full-page OCR on the prepped page (exactly DualOCR's Surya pass)
  4. DualOCR alignment + voting (+ zoom-in rescans for "balanced") on 2 and 3
Each page is treated as its own one-page document. Engine output is cached in
work/cache/DATASET/<id>/, so re-running only redoes the voting step.

Engine-alone text follows eval/run_eval.py: Tesseract lines joined in order;
Surya blocks in Surya's reading order. Page-number-only footers are dropped
for all systems and picture blocks are dropped for Surya (as DualOCR does).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # the dualocr package (repo root)
sys.path.insert(0, str(HERE))

from common import DATA, RESULTS, WORK, PAGENUM, SYSTEMS, manifest, pred_path  # noqa: E402

from PIL import Image  # noqa: E402

from dualocr import pagecache, prep  # noqa: E402
from dualocr.align import align_page  # noqa: E402
from dualocr.config import Settings  # noqa: E402
from dualocr.engines import surya as SU, tesseract as T  # noqa: E402
from dualocr.pipeline import PageWork, decide_document, page_text  # noqa: E402
from dualocr.structure import PICTURE_LABELS  # noqa: E402

LANGS = "heb+eng"


def tess_text(lines) -> str:
    rows = [" ".join(w.text for w in ln.words) for ln in lines]
    while rows and PAGENUM.match(rows[-1]):
        rows.pop()
    return "\n".join(rows)


def surya_text(blocks) -> str:
    out = []
    for b in sorted(blocks, key=lambda b: b.order):
        if b.label in PICTURE_LABELS:
            continue
        if b.label == "PageFooter" and PAGENUM.match(b.text or ""):
            continue
        if b.text:
            out.append(b.text)
    return "\n".join(out)


def _timed(fn, *a, **kw):
    t = time.time()
    r = fn(*a, **kw)
    return r, time.time() - t


def engine_pass(ids, pages_dir: Path, cache_root: Path, timing: dict) -> None:
    """Prep + Tesseract (prepped and raw) + Surya for pages not cached yet."""
    todo = [i for i in ids if not (cache_root / i / "surya.json").exists()]
    if not todo:
        return
    ups, grays, raws = {}, {}, {}

    def do_prep(pid):
        img = Image.open(pages_dir / f"{pid}.png").convert("RGB")
        (up, notes, rot), dt = _timed(prep.prepare, img, LANGS)
        d = cache_root / pid
        d.mkdir(parents=True, exist_ok=True)
        up.save(d / "page.png")
        (d / "prep.json").write_text(json.dumps({"notes": notes, "rotation": rot, "seconds": dt}), encoding="utf-8")
        return pid, img, up, notes, dt

    with ThreadPoolExecutor(4) as ex:
        for pid, img, up, notes, dt in ex.map(do_prep, todo):
            ups[pid], raws[pid] = (up, notes), img
            grays[pid] = prep.ocr_gray(up)
            timing.setdefault(pid, {})["prep"] = dt

    def do_tess(pid):
        lines, dt = _timed(T.ocr_page, grays[pid], LANGS, 3)
        raw_lines, dt_raw = _timed(T.ocr_page, raws[pid].convert("L"), LANGS, 3)
        pagecache.save_lines(cache_root / pid / "tess3.json", lines)
        pagecache.save_lines(cache_root / pid / "tessraw.json", raw_lines)
        return pid, dt, dt_raw

    with ThreadPoolExecutor(2) as tex:
        tfut = tex.map(do_tess, todo)
        caps = [SU.MAX_TOKENS_LOW_TEXT if any("little printed text" in n for n in ups[p][1]) else SU.MAX_TOKENS_PRINTED
                for p in todo]
        notes = [[] for _ in todo]
        t0 = time.time()
        blocks_all = SU.ocr_pages([ups[p][0] for p in todo], on_page=lambda k: None, caps=caps, notes=notes)
        per_page = (time.time() - t0) / len(todo)
        for pid, blocks, nt in zip(todo, blocks_all, notes):
            pagecache.save_blocks(cache_root / pid / "surya.json", blocks)
            timing[pid]["surya"] = per_page
            timing[pid]["surya_notes"] = nt
        for pid, dt, dt_raw in tfut:
            timing[pid]["tesseract"] = dt
            timing[pid]["tesseract_raw"] = dt_raw


def decide(pid: str, cache_root: Path, quality: str):
    d = cache_root / pid
    lines = pagecache.load_lines(d / "tess3.json")
    blocks = pagecache.load_blocks(d / "surya.json")
    up = Image.open(d / "page.png").convert("RGB")
    aligned, orphans = align_page(copy.deepcopy(lines), copy.deepcopy(blocks))
    pw = PageWork(0, pid, up, prep.ocr_gray(up), lines, blocks, aligned, orphans)
    decide_document([pw], Settings(quality=quality))
    return page_text(pw.layout)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    a = ap.parse_args(argv)
    systems = a.systems.split(",")
    m = manifest(a.dataset)
    ids = m["ids"][: a.limit or None]
    pages_dir = DATA / a.dataset / "pages"
    cache_root = WORK / "cache" / a.dataset
    out_dir = RESULTS / a.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    tpath = out_dir / "timing.jsonl"
    timing = {}
    if tpath.exists():
        for ln in tpath.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            timing[r["id"]] = r
    t_start = time.time()
    try:
        for k in range(0, len(ids), a.batch):
            chunk = ids[k:k + a.batch]
            try:
                engine_pass(chunk, pages_dir, cache_root, timing)
            except Exception as e:   # keep going; the page is reported as failed
                print(f"[{a.dataset}] engine pass failed for {chunk}: {type(e).__name__}: {e}", flush=True)
                for pid in chunk:
                    timing.setdefault(pid, {})["error"] = f"engine: {type(e).__name__}: {e}"
                continue
            for pid in chunk:
                d = cache_root / pid
                if not (d / "surya.json").exists():
                    continue
                t = timing.setdefault(pid, {})
                outs = {}
                if "tesseract_raw" in systems:
                    outs["tesseract_raw"] = tess_text(pagecache.load_lines(d / "tessraw.json"))
                if "tesseract" in systems:
                    outs["tesseract"] = tess_text(pagecache.load_lines(d / "tess3.json"))
                if "surya" in systems:
                    outs["surya"] = surya_text(pagecache.load_blocks(d / "surya.json"))
                for q in ("fast", "balanced"):
                    sname = f"dualocr_{q}"
                    if sname not in systems:
                        continue
                    if pred_path(a.dataset, sname, pid).exists() and f"decide_{q}" in t:
                        continue
                    try:
                        txt, dt = _timed(decide, pid, cache_root, q)
                        outs[sname] = txt
                        t[f"decide_{q}"] = dt
                    except Exception as e:
                        import traceback
                        t[f"error_{q}"] = f"{type(e).__name__}: {e}"
                        print(f"[{a.dataset}] {pid} {q} failed:\n{traceback.format_exc()}", flush=True)
                        outs[sname] = ""
                for sname, txt in outs.items():
                    p = pred_path(a.dataset, sname, pid)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(txt, encoding="utf-8")
                t["id"] = pid
                t["loadavg_1m"] = round(__import__("os").getloadavg()[0], 1)   # timings are only comparable on a quiet machine
            with open(tpath, "w", encoding="utf-8") as f:
                for pid in ids:
                    if pid in timing:
                        f.write(json.dumps(timing[pid], ensure_ascii=False) + "\n")
            done = min(k + a.batch, len(ids))
            el = time.time() - t_start
            print(f"[{a.dataset}] {done}/{len(ids)} pages, {el / 60:.1f} min", flush=True)
    finally:
        SU.shutdown()


if __name__ == "__main__":
    main()
