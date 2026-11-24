"""Zoom-in rescans for doubtful words.

A doubtful word is cropped together with its neighbors (context helps both
engines), enlarged, and re-read from several cleaned-up versions of the crop.
The reading of the target word is pulled out of each re-read by aligning it
against the expected  [left neighbor, target, right neighbor]  sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .align import align_tokens, clean_token

VARIANT_ORDER = ["up3", "sharp", "otsu", "adaptive", "thin", "thick", "up2"]


@dataclass
class RescanItem:
    key: tuple                       # (block_index, slot_index)
    bbox: tuple                      # crop region (already includes neighbors)
    expect: List[str]                # expected tokens for the crop region
    target: Tuple[int, int]          # [start, end) indices of the target inside expect


def _to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"))


def make_variants(crop: Image.Image, n: int) -> List[Tuple[str, Image.Image]]:
    g = crop.convert("L")
    h = max(1, g.height)
    # aim for ~90 px tall text lines after scaling (Tesseract likes 30-40 px x-height)
    f3 = min(4.0, max(1.5, 110 / h * 1.0)) if h < 110 else 1.0
    up = g.resize((max(1, int(g.width * f3)), max(1, int(g.height * f3))), Image.LANCZOS) if f3 != 1 else g
    out = [("up3", up)]
    if n <= 1:
        return out
    arr = _to_np(up)
    sharp = ImageOps.autocontrast(up, cutoff=1).filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=2))
    out.append(("sharp", sharp))
    _, otsu = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(("otsu", Image.fromarray(otsu)))
    blk = max(15, (int(up.height * 0.6) // 2) * 2 + 1)
    ad = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, 12)
    out.append(("adaptive", Image.fromarray(ad)))
    k = np.ones((2, 2), np.uint8)
    # "thin": grow the white background -> thinner strokes, helps separate fused letters
    out.append(("thin", Image.fromarray(cv2.dilate(otsu, k, iterations=1))))
    # "thick": grow the black strokes -> helps faint / broken letters
    out.append(("thick", Image.fromarray(cv2.erode(otsu, k, iterations=1))))
    f2 = f3 * 0.66
    out.append(("up2", g.resize((max(1, int(g.width * f2)), max(1, int(g.height * f2))), Image.LANCZOS)))
    return out[:n]


def pad_box(bbox, img_size, px=0.25, py=0.45):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    W, H = img_size
    return (max(0, int(x0 - px * h - 6)), max(0, int(y0 - py * h)),
            min(W, int(x1 + px * h + 6)), min(H, int(y1 + py * h)))


def extract_target(tokens: List[str], expect: List[str], target: Tuple[int, int]) -> Optional[str]:
    """Which of the crop's tokens line up with the target part of `expect`?"""
    toks = [clean_token(t) for t in tokens if clean_token(t)]
    if not toks:
        return None
    pairs = align_tokens(expect, toks)
    got = []
    hit = False
    for ei, ti in pairs:
        if any(target[0] <= e < target[1] for e in ei):
            hit = True
            got.extend(toks[t] for t in ti)
    if not hit:
        return None
    return " ".join(got) if got else ""


def script_probe(page: Image.Image, probes: List[Tuple[tuple, tuple]]) -> Dict[tuple, dict]:
    """Which alphabet do the pixels of this word fit? Read the word crop with
    Hebrew-only and English-only Tesseract models and compare confidences."""
    from .engines import tesseract as T
    from concurrent.futures import ThreadPoolExecutor

    jobs = []
    for key, box in probes:
        c = page.crop(pad_box(box, page.size, px=0.15, py=0.35))
        up = make_variants(c, 1)[0][1]
        jobs.append((key, up))

    def one(job):
        key, img = job
        out = {}
        for lang in ("heb", "eng"):
            lines = T.parse_tsv(T._run(img, lang, 8))
            ws = [w for ln in lines for w in ln.words]
            out[lang] = (" ".join(w.text for w in ws), float(np.mean([w.conf for w in ws])) if ws else 0.0)
        return key, out

    with ThreadPoolExecutor(T.MAX_WORKERS) as ex:
        return dict(ex.map(one, jobs))


def run(page: Image.Image, items: List[RescanItem], n_variants: int, use_surya: bool = True,
        surya_items=None, surya_variants: int = 1, variant_offset: int = 0, probes=None,
        progress=None) -> Dict[tuple, List[Tuple[str, str]]]:
    """Returns key -> [(source_name, reading)].
    surya_items: keys of the items whose letters are in doubt (Surya re-reads only those)."""
    results: Dict[tuple, List[Tuple[str, str]]] = {it.key: [] for it in items}
    if not items or n_variants <= 0:
        return results
    crops, owners = [], []
    variants_by_item = {}
    for it in items:
        c = page.crop(pad_box(it.bbox, page.size))
        allv = make_variants(c, len(VARIANT_ORDER))
        variants_by_item[it.key] = dict(allv)
        # round 2+ uses the variants round 1 did not try
        start = (variant_offset * n_variants) % len(allv)
        vs = (allv[start:] + allv[:start])[:n_variants]
        for name, v in vs:
            crops.append(v)
            owners.append((it, name))
    from .engines import tesseract as T
    t_res = T.ocr_crops(crops, psm=7)
    for (it, name), lines in zip(owners, t_res):
        toks = [w.text for ln in lines for w in ln.words]
        r = extract_target(toks, it.expect, it.target)
        if r is not None:
            results[it.key].append((f"tess:{name}", r))
    if probes:
        for key, res in script_probe(page, probes).items():
            for lang, (txt, conf) in res.items():
                results.setdefault(key, []).append((f"probe_{lang}", f"{conf:.0f}\t{txt}"))
    if progress:
        progress(0.6)
    if use_surya:
        from .engines import surya as SU

        wanted = set(surya_items) if surya_items is not None else {it.key for it in items}
        names = (["up3", "otsu"] if variant_offset == 0 else ["sharp", "thin"])[:max(1, surya_variants)]
        s_crops, s_owners = [], []
        for it in items:
            if it.key not in wanted:
                continue
            for name in names:
                v = variants_by_item[it.key].get(name)
                if v is None:
                    v = dict(make_variants(page.crop(pad_box(it.bbox, page.size)), 3)).get(name)
                if v is not None:
                    s_crops.append(v.convert("RGB"))
                    s_owners.append((it, name))
        for (it, name), (text, _wp) in zip(s_owners, SU.ocr_crops(s_crops)):
            r = extract_target(text.split(), it.expect, it.target)
            if r is not None:
                results[it.key].append((f"surya:{name}", r))
    return results
