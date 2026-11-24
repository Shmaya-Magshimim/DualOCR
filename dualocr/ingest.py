"""Turn input files (PDF, JPG, PNG, HEIC, TIFF...) into page images.

PDF pages that are just one scanned photo (CamScanner, phone scanners) are
extracted at the photo's native resolution instead of being re-rendered, which
keeps the most detail. Other PDF pages are rendered at 300 dpi. A real text
layer, if a PDF has one, is kept as an extra opinion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from PIL import Image, ImageOps

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif", ".gif"}
PDF_EXT = {".pdf"}
SUPPORTED_EXT = IMAGE_EXT | PDF_EXT

try:  # HEIC support (iPhone photos)
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass


@dataclass
class TextLayerWord:
    text: str
    bbox: tuple  # x0, y0, x1, y1 in image pixels


@dataclass
class PageImage:
    index: int                 # 0-based position in the output document
    source: str                # file name
    source_page: int           # 1-based page number inside the source file
    image: Image.Image         # RGB, upright after prep
    text_layer: List[TextLayerWord] = field(default_factory=list)
    rotation_applied: int = 0  # degrees clockwise applied during prep
    notes: List[str] = field(default_factory=list)


def count_pages(paths: List[Path]) -> int:
    n = 0
    for p in paths:
        if p.suffix.lower() in PDF_EXT:
            import pymupdf

            with pymupdf.open(p) as d:
                n += d.page_count
        else:
            try:
                with Image.open(p) as im:
                    n += getattr(im, "n_frames", 1)
            except Exception:
                n += 1
    return n


def _pdf_pages(path: Path) -> Iterator[tuple]:
    import pymupdf

    doc = pymupdf.open(path)
    for pno, page in enumerate(doc):
        rect = page.rect
        page_area = rect.width * rect.height
        infos = [i for i in page.get_image_info(xrefs=True) if i.get("xref")]
        area = lambda b: max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
        clip = None
        scale = 300 / 72
        if infos:
            info = max(infos, key=lambda i: area(i["bbox"]))
            if area(info["bbox"]) > 0.5 * page_area:
                # A scanned photo: render just its area (this skips scanner-app
                # logos/watermarks beside it) at the photo's native resolution.
                # Rendering, rather than extracting raw bytes, applies any
                # rotation the PDF puts on the image.
                bw = info["bbox"][2] - info["bbox"][0]
                bh = info["bbox"][3] - info["bbox"][1]
                sx = max(info["width"], info["height"]) / max(bw, bh)
                scale = min(max(sx, 300 / 72), 600 / 72)
                clip = pymupdf.Rect(info["bbox"]) & rect
        mat = pymupdf.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False, annots=False, clip=clip)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        ox, oy = (clip.x0, clip.y0) if clip is not None else (0.0, 0.0)
        words = []
        try:
            for w in page.get_text("words"):
                x0, y0, x1, y1, t = w[:5]
                if clip is not None and not pymupdf.Rect(x0, y0, x1, y1).intersects(clip):
                    continue
                words.append(TextLayerWord(t, ((x0 - ox) * scale, (y0 - oy) * scale,
                                               (x1 - ox) * scale, (y1 - oy) * scale)))
        except Exception:
            pass
        # CamScanner and similar add a short invisible watermark text; ignore tiny layers
        if len(words) < 15:
            words = []
        yield pno + 1, img, words
    doc.close()


def _image_pages(path: Path) -> Iterator[tuple]:
    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1)
        for f in range(n):
            if n > 1:
                im.seek(f)
            frame = ImageOps.exif_transpose(im.copy()) if f == 0 else im.copy()
            yield f + 1, frame.convert("RGB"), []


def load_pages(paths: List[Path]) -> Iterator[PageImage]:
    """Yield pages from all inputs in order (files in the order given)."""
    idx = 0
    for p in paths:
        p = Path(p)
        ext = p.suffix.lower()
        if ext in PDF_EXT:
            gen = _pdf_pages(p)
        elif ext in IMAGE_EXT:
            gen = _image_pages(p)
        else:
            raise ValueError(f"Unsupported file type: {p.name}")
        for spno, img, words in gen:
            yield PageImage(index=idx, source=p.name, source_page=spno, image=img, text_layer=words)
            idx += 1
