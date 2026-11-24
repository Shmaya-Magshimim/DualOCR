"""Run import + prep + both engines on the test documents once and cache it.

usage: python eval/build_cache.py heb23 eng14 ...
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dualocr import ingest, prep, pagecache  # noqa: E402
from dualocr.engines import surya as SU, tesseract as T  # noqa: E402

SRC = ROOT / "eval" / "sources"
CACHE = ROOT / "eval" / "cache"


def build(doc: str):
    out = CACHE / doc
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pages = list(ingest.load_pages([SRC / f"{doc}.pdf"]))
    ups = []
    for p in pages:
        up, notes, rot = prep.prepare(p.image)
        up.save(out / f"p{p.index + 1:02d}.png")
        (out / f"p{p.index + 1:02d}_prep.json").write_text(json.dumps({"notes": notes, "rotation": rot}))
        ups.append(up)
    print(f"[{doc}] prep {len(pages)} pages {time.time() - t0:.0f}s", flush=True)
    grays = [prep.ocr_gray(u) for u in ups]
    for psm in (4, 3):
        t = time.time()
        res = T.ocr_pages(grays, psm=psm)
        for i, lines in enumerate(res):
            pagecache.save_lines(out / f"p{i + 1:02d}_tess{psm}.json", lines)
        print(f"[{doc}] tesseract psm{psm} {time.time() - t:.0f}s", flush=True)
    t = time.time()
    res = SU.ocr_pages(ups)
    for i, blocks in enumerate(res):
        pagecache.save_blocks(out / f"p{i + 1:02d}_surya.json", blocks)
    print(f"[{doc}] surya {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        for d in sys.argv[1:]:
            build(d)
    finally:
        SU.shutdown()
