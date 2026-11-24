"""Tesseract engine (word boxes + confidences), run as parallel subprocesses."""
from __future__ import annotations

import io
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List

from PIL import Image

from .. import config  # noqa: F401
from ..model import Line, Word

MAX_WORKERS = max(2, (os.cpu_count() or 4) - 2)


def _run(img: Image.Image, langs: str, psm: int, dpi: int = 300, extra: tuple = ()) -> str:
    b = io.BytesIO()
    img.save(b, format="PNG")
    r = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", langs, "--psm", str(psm), "--dpi", str(dpi),
         "-c", "preserve_interword_spaces=0", *extra, "tsv"],
        input=b.getvalue(), capture_output=True, timeout=600)
    return r.stdout.decode("utf-8", "replace")


def parse_tsv(tsv: str, offset=(0, 0), scale: float = 1.0) -> List[Line]:
    lines: dict = {}
    line_boxes: dict = {}
    ox, oy = offset
    for row in tsv.splitlines()[1:]:
        p = row.split("\t")
        if len(p) != 12:
            continue
        level = int(p[0])
        key = (int(p[2]), int(p[3]), int(p[4]))
        x, y, w, h = (int(v) for v in p[6:10])
        box = (ox + x / scale, oy + y / scale, ox + (x + w) / scale, oy + (y + h) / scale)
        if level == 4:
            line_boxes[key] = box
        elif level == 5:
            t = p[11].strip()
            if not t:
                continue
            try:
                c = float(p[10])
            except ValueError:
                c = 0.0
            lines.setdefault(key, []).append(Word(t, box, max(c, 0.0), key, "tesseract"))
    out = []
    for key in sorted(lines):
        ws = lines[key]
        out.append(Line(ws, line_boxes.get(key, _union([w.bbox for w in ws])), key))
    return out


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def ocr_page(img: Image.Image, langs: str = "heb+eng", psm: int = 4) -> List[Line]:
    return parse_tsv(_run(img, langs, psm))


def ocr_pages(images: List[Image.Image], langs: str = "heb+eng", psm: int = 4) -> List[List[Line]]:
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        return list(ex.map(lambda im: ocr_page(im, langs, psm), images))


def ocr_crops(crops: List[Image.Image], langs: str = "heb+eng", psm: int = 7) -> List[List[Line]]:
    """OCR many small crops (single line / word) in parallel."""
    def one(im):
        return parse_tsv(_run(im, langs, psm, dpi=300))
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        return list(ex.map(one, crops))


def text_of(lines: List[Line]) -> str:
    return "\n".join(" ".join(w.text for w in ln.words) for ln in lines)
