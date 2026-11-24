"""olmOCR-bench (Ai2) subsets as page images + the benchmark's own unit tests.

usage: python bench/prepare_olmocr.py [SUBSET=old_scans] [--limit N] [--seed 0]

Downloads bench_data/<subset>.jsonl and the PDFs it references from
https://huggingface.co/datasets/allenai/olmOCR-bench (ODC-BY 1.0), then turns
each PDF page into the image DualOCR itself would read (dualocr.ingest: scanned
pages at the scan's native resolution). Every system gets that same image.
Needs the DualOCR environment (pymupdf).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import DATA, download, write_manifest  # noqa: E402

BASE = "https://huggingface.co/datasets/allenai/olmOCR-bench/resolve/main/bench_data/"


def _cap(path: Path, max_side: int) -> None:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None     # some archival scans are >250 megapixels
    im = Image.open(path)
    if max(im.size) > max_side:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        im.save(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("subset", nargs="?", default="old_scans")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=2400,
                    help="downscale larger scans (old_scans are ~7000x9000 px archival TIFFs; "
                         "2400 px is about 220 dpi for a letter page, Surya's own runner uses 192 dpi)")
    a = ap.parse_args(argv)
    name = f"olmocr_{a.subset}"
    root = DATA / name
    tests_file = download(BASE + f"{a.subset}.jsonl", root / f"{a.subset}.jsonl")
    tests = [json.loads(l) for l in tests_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    pdfs = sorted({(t["pdf"], t.get("page", 1)) for t in tests})
    if a.limit and a.limit < len(pdfs):
        pdfs = sorted(random.Random(a.seed).sample(pdfs, a.limit))
    from dualocr import ingest
    ids, items = [], []
    for pdf, page in pdfs:
        local = download(BASE + "pdfs/" + pdf, root / "pdfs" / pdf)
        pid = Path(pdf).stem + (f"_p{page}" if page != 1 else "")
        out = root / "pages" / f"{pid}.png"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            pages = list(ingest.load_pages([local]))
            pages[page - 1].image.convert("RGB").save(out)
        _cap(out, a.max_side)
        ids.append(pid)
        items.append({"id": pid, "pdf": pdf, "page": page})
        print(pid, flush=True)
    keep = {(i["pdf"], i["page"]) for i in items}
    sel = [t for t in tests if (t["pdf"], t.get("page", 1)) in keep]
    (root / "tests.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in sel) + "\n",
                                      encoding="utf-8")
    write_manifest(name, {
        "name": name, "ids": ids, "items": items, "seed": a.seed, "n_tests": len(sel),
        "source": f"https://huggingface.co/datasets/allenai/olmOCR-bench ({a.subset})",
        "license": "ODC-BY 1.0 (dataset); test harness: allenai/olmocr, Apache-2.0",
        "gt_format": "olmOCR-bench unit tests (present / absent / order) in tests.jsonl",
        "max_side": a.max_side,
    })
    print(f"wrote {len(ids)} pages, {len(sel)} tests to {root}")


if __name__ == "__main__":
    main()
