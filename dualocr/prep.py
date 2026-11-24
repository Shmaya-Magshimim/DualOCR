"""Page preparation: orientation, deskew, OCR-friendly grayscale."""
from __future__ import annotations

import io
import re
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from . import config  # noqa: F401  (sets TESSDATA_PREFIX)


@dataclass
class OrientationResult:
    rotate: int          # degrees clockwise to apply (0/90/180/270)
    confidence: float
    method: str          # "osd" | "probe" | "none"
    script: str = ""


def _png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def _tess(img: Image.Image, args: list, timeout: int = 120) -> str:
    r = subprocess.run(["tesseract", "stdin", "stdout", *args], input=_png_bytes(img),
                       capture_output=True, timeout=timeout)
    return r.stdout.decode("utf-8", "replace")


def rotate_cw(img: Image.Image, deg: int) -> Image.Image:
    deg %= 360
    if deg == 0:
        return img
    return img.rotate(-deg, expand=True)


# printed pages score 20,000+ when upright; handwriting stays below ~2,000
MIN_PROBE_TEXT = 5000


def _probe_score(img: Image.Image, langs: str) -> float:
    """Quick OCR of the whole page; higher = more readable horizontal text.
    Lines that come out taller than wide (text read sideways) don't count."""
    small = img.convert("L")
    small.thumbnail((1700, 1700))
    tsv = _tess(small, ["-l", langs, "--psm", "3", "tsv"])
    lines = {}
    for line in tsv.splitlines()[1:]:
        p = line.split("\t")
        if len(p) != 12 or not p[11].strip():
            continue
        try:
            c = float(p[10])
            x, y, w, h = (int(v) for v in p[6:10])
        except ValueError:
            continue
        lines.setdefault((p[2], p[3], p[4]), []).append((c, p[11].strip(), x, y, w, h))
    score = 0.0
    for ws in lines.values():
        x0 = min(w[2] for w in ws); x1 = max(w[2] + w[4] for w in ws)
        y0 = min(w[3] for w in ws); y1 = max(w[3] + w[5] for w in ws)
        if len(ws) >= 2 and (y1 - y0) > (x1 - x0):
            continue  # vertical line: text is sideways in this orientation
        for c, t, *_ in ws:
            if c > 60 and len(t) >= 2:
                score += c * min(len(t), 8)
    return score


def detect_orientation(img: Image.Image, langs: str = "heb+eng") -> OrientationResult:
    g = img.convert("L")
    g.thumbnail((2600, 2600))
    out = _tess(g, ["--psm", "0", "-l", "osd"])
    m_rot = re.search(r"Rotate:\s*(\d+)", out)
    m_conf = re.search(r"Orientation confidence:\s*([\d.]+)", out)
    m_scr = re.search(r"Script:\s*(\w+)", out)
    osd_rot = int(m_rot.group(1)) % 360 if m_rot else None
    osd_conf = float(m_conf.group(1)) if m_conf else 0.0
    if osd_rot is not None and osd_conf >= 8.0:
        return OrientationResult(osd_rot, osd_conf, "osd", m_scr.group(1) if m_scr else "")
    # OSD unsure (few characters, photos): read the page all four ways and keep
    # the one with the most confidently-read horizontal text.
    scores = {d: _probe_score(rotate_cw(img, d), langs) for d in (0, 90, 180, 270)}
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    if ranked[0] < MIN_PROBE_TEXT:
        # Too little confidently-read text in any direction (handwriting,
        # photos, near-empty pages): don't guess; keep the page as scanned.
        return OrientationResult(0, 0.0, "none")
    conf = (ranked[0] - ranked[1]) / (ranked[0] + 1e-6) * 10
    # a weak probe result does not override a moderately sure OSD answer
    if osd_rot is not None and osd_conf >= 3.0 and best != osd_rot and conf < 2.0:
        return OrientationResult(osd_rot, osd_conf, "osd")
    return OrientationResult(best, conf, "probe")


def estimate_skew(img: Image.Image, max_deg: float = 5.0) -> float:
    """Small-angle skew (degrees, positive = rotate counter-clockwise to fix)."""
    g = np.array(img.convert("L"))
    scale = 1200 / max(g.shape)
    if scale < 1:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if bw.mean() < 1:  # blank page
        return 0.0
    h, w = bw.shape
    center = (w / 2, h / 2)

    def score(a):
        M = cv2.getRotationMatrix2D(center, a, 1.0)
        r = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST)
        prof = r.sum(axis=1, dtype=np.float64)
        return float(np.var(prof))

    coarse = np.arange(-max_deg, max_deg + 1e-9, 0.5)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.5, best + 0.5 + 1e-9, 0.05)
    best = max(fine, key=score)
    return float(round(best, 2))


def deskew(img: Image.Image, angle: float) -> Image.Image:
    if abs(angle) < 0.2:
        return img
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def prepare(img: Image.Image, langs: str = "heb+eng") -> tuple:
    """Return (upright image, notes list, rotation applied)."""
    notes = []
    o = detect_orientation(img, langs)
    up = rotate_cw(img, o.rotate)
    if o.method == "none":
        notes.append("little printed text on this page (handwriting or a photo?): orientation kept as scanned")
    if o.rotate:
        notes.append(f"rotated {o.rotate}° ({o.method}, confidence {o.confidence:.1f})")
    ang = estimate_skew(up)
    if abs(ang) >= 0.2:
        up = deskew(up, ang)
        notes.append(f"deskewed {ang:+.2f}°")
    return up, notes, o.rotate


def ocr_gray(img: Image.Image) -> Image.Image:
    """Grayscale for OCR. Uses the brightest channel when the page has colored
    highlighting, so a yellow marker background turns white and black text
    stays black."""
    arr = np.asarray(img.convert("RGB"))
    lum = np.asarray(img.convert("L")).copy()
    mx = arr.max(axis=2)
    sat = mx.astype(np.int16) - arr.min(axis=2).astype(np.int16)
    # Only light, colorful pixels (highlighter backgrounds) are brightened;
    # dark colored text (blue/red ink) keeps its luminance and stays dark.
    mask = (sat > 60) & (lum > 140)
    if mask.any():
        lum[mask] = mx[mask]
    return Image.fromarray(lum, "L")
