"""Generate test pages with known text: Hebrew, English and mixed lines,
rendered in different fonts, then damaged (rotation, skew, blur, JPEG).

usage: python eval/synth.py OUTDIR [N_PAGES]
Writes OUTDIR/sNN.jpg + OUTDIR/sNN.txt (ground truth) + OUTDIR/meta.json
"""
import json
import random
import re
import sys
from pathlib import Path

from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",   # (Arial Hebrew has no Latin letters)
]
MIXED = [
    "בשנת 1948 נחתמה מגילת העצמאות (Declaration of Independence) בתל אביב.",
    "המונח liberty פירושו חופש, ובעברית: חירות.",
    "ראו סעיף 7(ב) לחוק, וכן את הספר Basic Laws of Israel עמ' 25.",
    "The word שבת means rest; the Torah calls it a sign (אות) for Israel.",
    "Rashi explains פסוק ג: the people complained about the manna (מן).",
    "Chapter 3, pages 10-14: the Knesset (הכנסת) has 120 members.",
    "הוועדה התכנסה ב-12.5.2024 בשעה 10:30 ודנה בהצעת החוק.",
    "Email: info@example.org | טלפון: 03-1234567",
]


def _lines(path, script):
    out = []
    for ln in Path(path).read_text(encoding="utf-8").split("\n"):
        ln = re.sub(r"\[[^\]]*\]", "", ln).lstrip("#•- ").strip()
        if len(ln) < 25:
            continue
        he = len(re.findall(r"[א-ת]", ln)); en = len(re.findall(r"[A-Za-z]", ln))
        if (script == "he" and he > en) or (script == "en" and en > he):
            out.append(ln)
    return out


def wrap(draw, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(get_display(t), font=font) <= width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_page(paras, font_path, size, rtl_doc):
    W, H = 2480, 3508
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size)
    y, margin = 220, 200
    truth = []
    for p in paras:
        he = len(re.findall(r"[א-ת]", p)); en = len(re.findall(r"[A-Za-z]", p))
        rtl = he >= en
        for ln in wrap(d, p, font, W - 2 * margin):
            vis = get_display(ln, base_dir="R" if rtl else "L")
            tw = d.textlength(vis, font=font)
            x = W - margin - tw if rtl else margin
            d.text((x, y), vis, font=font, fill="black")
            y += int(size * 1.55)
            if y > H - 250:
                break
        truth.append(p)
        y += int(size * 0.9)
        if y > H - 250:
            break
    return img, "\n".join(truth)


def damage(img, rng):
    rot = rng.choice([0, 0, 0, 90, 180, 270])
    skew = rng.uniform(-2.5, 2.5)
    img = img.rotate(skew, resample=Image.BICUBIC, expand=True, fillcolor="white")
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.4)))
    if rot:
        img = img.rotate(-rot, expand=True)
    # slight gray paper tone
    img = Image.blend(img, Image.new("RGB", img.size, (238, 234, 226)), rng.uniform(0.0, 0.35))
    return img, rot, skew


def main(out, n=8, seed=7):
    rng = random.Random(seed)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    he = []
    for f in sorted((ROOT / "eval/gold/heb23").glob("page_*.txt")):
        he += _lines(f, "he")
    en = _lines(ROOT / "eval/gold/eng14/full.txt", "en")
    meta = []
    for k in range(n):
        kind = ["he", "en", "mixed", "mixed"][k % 4]
        if kind == "he":
            paras = rng.sample(he, 7)
        elif kind == "en":
            paras = rng.sample(en, 5)
        else:
            paras = rng.sample(MIXED, 4) + rng.sample(he, 2) + rng.sample(en, 1)
            rng.shuffle(paras)
        font = FONTS[k % len(FONTS)]
        img, truth = render_page(paras, font, rng.choice([44, 50, 56]), kind != "en")
        img, rot, skew = damage(img, rng)
        q = rng.choice([45, 60, 80])
        img.save(out / f"s{k + 1:02d}.jpg", quality=q)
        (out / f"s{k + 1:02d}.txt").write_text(truth, encoding="utf-8")
        meta.append({"page": k + 1, "kind": kind, "font": Path(font).name, "rotation": rot, "skew": round(skew, 2),
                     "jpeg": q})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=0))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
