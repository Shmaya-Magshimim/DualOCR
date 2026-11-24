"""Saved review data: every decided word with its evidence and box, plus the
page images, so the review screen can reopen a finished job, let you fix
words, and rewrite the Word file."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from PIL import Image

from .align import Slot
from .structure import PageLayout, Para

KEEP_READINGS = ("surya_split", "tess_split", "_cands", "rescan", "llm")


def _slot_to_dict(sl: Slot) -> dict:
    return {"text": sl.text, "status": sl.status, "conf": round(float(sl.conf or 0), 4),
            "bbox": [round(v, 1) for v in sl.bbox] if sl.bbox else None,
            "surya": sl.surya, "tess": sl.tess, "kinds": sl.kinds, "reason": sl.reason,
            "bold": getattr(sl, "bold", False), "underline": getattr(sl, "underline", False),
            "readings": {k: sl.readings[k] for k in KEEP_READINGS if k in sl.readings}}


def _slot_from_dict(d: dict) -> Slot:
    sl = Slot(surya=d.get("surya"), tess=d.get("tess"), bbox=tuple(d["bbox"]) if d.get("bbox") else None)
    sl.text, sl.status, sl.conf = d.get("text", ""), d.get("status", ""), d.get("conf", 0.0)
    sl.kinds, sl.reason = d.get("kinds") or {}, d.get("reason", "")
    sl.readings = dict(d.get("readings") or {})
    sl.reviewed = d.get("reviewed", False)
    sl.bold, sl.underline = d.get("bold", False), d.get("underline", False)
    return sl


def save(review_dir: Path, layouts: List[PageLayout], images: List[Image.Image], meta: dict) -> Path:
    review_dir = Path(review_dir)
    (review_dir / "pages").mkdir(parents=True, exist_ok=True)
    pages = []
    for pl, img in zip(layouts, images):
        name = f"pages/p{pl.index + 1:03d}.jpg"
        if img is not None:
            img.convert("RGB").save(review_dir / name, quality=88)
        pages.append({"index": pl.index, "image": name, "printed_number": pl.printed_number,
                      "paras": [{"kind": p.kind, "bold": p.bold, "underline": p.underline,
                                 "bbox": list(p.bbox) if p.bbox else None,
                                 "words": [_slot_to_dict(s) for s in p.words]} for p in pl.paras]})
    data = dict(meta, pages=pages)
    path = review_dir / "review.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def load(path: Path):
    """-> (meta dict, layouts, image paths)"""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    layouts, images = [], []
    for pg in data["pages"]:
        pl = PageLayout(pg["index"], printed_number=pg.get("printed_number"))
        for p in pg["paras"]:
            pl.paras.append(Para(p["kind"], [_slot_from_dict(w) for w in p["words"]],
                                 tuple(p["bbox"]) if p.get("bbox") else None, p.get("bold", False),
                                 p.get("underline", False)))
        layouts.append(pl)
        images.append(path.parent / pg["image"])
    return data, layouts, images


def write_back(path: Path, layouts: List[PageLayout]) -> None:
    """Store edits (text/status/reviewed) back into review.json."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    for pg, pl in zip(data["pages"], layouts):
        for p, para in zip(pg["paras"], pl.paras):
            for w, sl in zip(p["words"], para.words):
                w["text"], w["status"] = sl.text, sl.status
                w["reviewed"] = getattr(sl, "reviewed", False)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def rewrite_outputs(path: Path, review_highlights: bool = True) -> dict:
    """Regenerate the DOCX/TXT from the (edited) review data."""
    from . import docx_writer
    data, layouts, img_paths = load(path)
    images = [Image.open(p).convert("RGB") if Path(p).exists() else None for p in img_paths]
    out = data.get("outputs", {})
    if out.get("docx"):
        docx_writer.write_docx(out["docx"], layouts, images, title=data.get("title", ""),
                               review=review_highlights, include_pictures=data.get("include_pictures", True))
    if out.get("txt"):
        docx_writer.write_txt(out["txt"], layouts)
    return out
