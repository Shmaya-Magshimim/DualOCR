"""Write the result as a Word document (and plain text).

- Hebrew paragraphs are right-to-left; English paragraphs left-to-right.
  Inside a paragraph, Hebrew runs are marked RTL and Latin/number runs are
  not, so mixed lines display in the right order.
- Headings, list items, captions; pictures (maps, diagrams, photos) are cut
  from the page and placed where they appear.
- Review mode: uncertain words are highlighted yellow with a Word comment
  listing what each engine read; probable typos in the original are
  highlighted light blue ("kept as printed").
"""
from __future__ import annotations

import io
import re
from typing import List, Optional

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from . import lexicon as L

FONT = "Arial"

# Element order required by the Word schema (CT_RPr / CT_PPr / CT_SectPr).
_RPR_ORDER = ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike", "outline",
              "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish", "webHidden", "color", "spacing",
              "w", "kern", "position", "sz", "szCs", "highlight", "u", "effect", "bdr", "shd", "fitText",
              "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath"]
_PPR_ORDER = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
              "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
              "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
              "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc",
              "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr",
              "sectPr", "pPrChange"]
_SECT_ORDER = ["headerReference", "footerReference", "footnotePr", "endnotePr", "type", "pgSz", "pgMar",
               "paperSrc", "pgBorders", "lnNumType", "pgNumType", "cols", "formProt", "vAlign", "noEndnote",
               "titlePg", "textDirection", "bidi", "rtlGutter", "docGrid", "printerSettings", "sectPrChange"]


def _sort_children(el, order):
    if el is None:
        return
    rank = {qn("w:" + n): i for i, n in enumerate(order)}
    kids = list(el)
    kids.sort(key=lambda c: rank.get(c.tag, len(order)))
    for c in kids:
        el.remove(c)
    for c in kids:
        el.append(c)


def _fix_order(doc):
    body = doc.element.body
    for pPr in body.iter(qn("w:pPr")):
        _sort_children(pPr, _PPR_ORDER)
    for rPr in body.iter(qn("w:rPr")):
        _sort_children(rPr, _RPR_ORDER)
    for sp in body.iter(qn("w:sectPr")):
        _sort_children(sp, _SECT_ORDER)
_RUN_SPLIT = re.compile(r"([֐-׿][֐-׿\"'\-]*|[A-Za-z][A-Za-z'\-]*)")


def _set_rtl_par(p, rtl: bool):
    pPr = p._p.get_or_add_pPr()
    for el in pPr.findall(qn("w:bidi")):
        pPr.remove(el)
    if rtl:
        b = OxmlElement("w:bidi")
        b.set(qn("w:val"), "1")
        pPr.append(b)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if rtl else WD_ALIGN_PARAGRAPH.LEFT


def _style_run(r, size: float, bold=False, underline=False, rtl=False, italic=False, color=None):
    r.font.name = FONT
    r.font.size = Pt(size)
    r.bold = bold or None
    r.underline = underline or None
    r.italic = italic or None
    if color:
        r.font.color.rgb = RGBColor(*color)
    rPr = r._r.get_or_add_rPr()
    rf = rPr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rPr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), FONT)
    szcs = OxmlElement("w:szCs")
    szcs.set(qn("w:val"), str(int(size * 2)))
    rPr.append(szcs)
    if bold:
        rPr.append(OxmlElement("w:bCs"))
    if rtl:
        e = OxmlElement("w:rtl")
        e.set(qn("w:val"), "1")
        rPr.append(e)


def _para_is_rtl(text: str) -> bool:
    he = len(re.findall(r"[א-ת]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    return he >= en


def _add_text_runs(p, text: str, size: float, bold: bool, underline: bool, highlight=None):
    """Add text as runs split by script so mixed Hebrew/English displays right."""
    runs = []
    pos = 0
    for m in _RUN_SPLIT.finditer(text):
        if m.start() > pos:
            runs.append((text[pos:m.start()], None))
        seg = m.group(0)
        runs.append((seg, "he" if "֐" <= seg[0] <= "׿" else "en"))
        pos = m.end()
    if pos < len(text):
        runs.append((text[pos:], None))
    # merge neutral pieces (spaces, punctuation, digits) into the previous run
    merged = []
    for seg, sc in runs:
        if sc is None and merged:
            merged[-1] = (merged[-1][0] + seg, merged[-1][1])
        elif merged and merged[-1][1] == sc:
            merged[-1] = (merged[-1][0] + seg, sc)
        else:
            merged.append((seg, sc))
    out = []
    for seg, sc in merged:
        r = p.add_run(seg)
        _style_run(r, size, bold, underline, rtl=(sc == "he"))
        if highlight is not None:
            r.font.highlight_color = highlight
        out.append(r)
    return out


def _comment_text(sl) -> str:
    bits = []
    s = sl.readings.get("surya_split") or sl.surya
    t = sl.readings.get("tess_split") or sl.tess
    if s is not None:
        bits.append(f"Surya read: {s}")
    if t is not None:
        bits.append(f"Tesseract read: {t}")
    resc = sl.readings.get("rescan") or []
    if resc:
        from collections import Counter
        c = Counter(r for _, r in resc)
        bits.append("Zoomed-in re-reads: " + ", ".join(f"{k or '(nothing)'} ×{v}" for k, v in c.most_common(3)))
    cands = sl.readings.get("_cands") or {}
    if len(cands) > 1:
        bits.append("Candidates: " + ", ".join(f"{k or '(drop)'} {v:.0%}" for k, v in cands.items()))
    if sl.kinds:
        bits.append("Errors: " + "; ".join(f"{k}: {v}" for k, v in sl.kinds.items()))
    if sl.readings.get("llm"):
        bits.append(f"AI tie-breaker read: {sl.readings['llm']}")
    return "\n".join(bits) or "Low confidence"


def _add_para_words(doc, p, words, size, bold, underline, review: bool, comments: list, force_bold=False):
    """Words of one paragraph; flagged words become separate highlighted runs."""
    buf, buf_style = [], None
    word_level = any(getattr(sl, "bold", False) or getattr(sl, "underline", False) for sl in words)

    def style_of(sl):
        # headings are always bold; otherwise follow the word-level formatting
        # Surya saw, falling back to the paragraph's formatting
        if word_level:
            return bool(force_bold or getattr(sl, "bold", False)), bool(getattr(sl, "underline", False))
        return bool(force_bold or bold), bool(underline)

    def flush():
        if buf:
            b, u = buf_style
            _add_text_runs(p, " ".join(buf) + " ", size, b, u)
            buf.clear()

    for sl in words:
        if not sl.text:
            continue
        st = style_of(sl)
        # only words the app is really unsure of are marked; no comments (they
        # clutter the margin) - the readings are in the review screen and report
        flag = review and sl.status == "uncertain"
        if not flag:
            if buf and st != buf_style:
                flush()
            buf_style = st
            buf.append(sl.text)
            continue
        flush()
        _add_text_runs(p, sl.text, size, st[0], st[1], highlight=WD_COLOR_INDEX.YELLOW)
        tail = p.add_run(" ")
        _style_run(tail, size)
    flush()


def write_docx(path, pages, images, title: str = "", review: bool = True, include_pictures: bool = True,
               page_markers: bool = True) -> None:
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(12)
    st.paragraph_format.space_after = Pt(4)
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    usable_w = sec.page_width - sec.left_margin - sec.right_margin
    all_text = " ".join(p.text for pl in pages for p in pl.paras)
    doc_rtl = _para_is_rtl(all_text)
    if doc_rtl:
        sp = sec._sectPr
        b = OxmlElement("w:bidi")
        sp.append(b)
    comments = []
    for k, pl in enumerate(pages):
        first_of_page = len(doc.paragraphs)
        if page_markers:
            p = doc.add_paragraph()
            _set_rtl_par(p, doc_rtl)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(f"עמוד {pl.index + 1}" if doc_rtl else f"Page {pl.index + 1}")
            _style_run(r, 9, color=(0x88, 0x88, 0x88), rtl=doc_rtl)
        img = images[k] if images else None
        for para in pl.paras:
            if para.kind == "picture":
                if include_pictures and img is not None and para.bbox:
                    _add_picture(doc, img, para.bbox, usable_w)
                continue
            text = para.text
            if not text:
                continue
            p = doc.add_paragraph()
            rtl = _para_is_rtl(text) if text.strip() else doc_rtl
            _set_rtl_par(p, rtl)
            size, bold, und = 12, para.bold, para.underline
            if para.kind == "heading":
                size, bold = 13, True
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.keep_with_next = True
            elif para.kind == "header":
                size = 10
            elif para.kind == "caption":
                size = 10
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif para.kind == "list_item":
                p.paragraph_format.left_indent = Cm(0.6) if not rtl else None
                p.paragraph_format.right_indent = Cm(0.6) if rtl else None
                p.paragraph_format.space_after = Pt(2)
            _add_para_words(doc, p, para.words, size, para.bold, und, review, comments,
                            force_bold=(para.kind == "heading"))
        if k > 0 and len(doc.paragraphs) > first_of_page:
            doc.paragraphs[first_of_page].paragraph_format.page_break_before = True
    for runs, text in comments:
        try:
            doc.add_comment(runs, text=text, author="DualOCR", initials="OCR")
        except Exception:
            pass
    _fix_order(doc)
    doc.core_properties.title = title or "OCR result"
    doc.core_properties.author = "DualOCR"
    doc.save(str(path))


def _add_picture(doc, page_img, bbox, usable_w):
    x0, y0, x1, y1 = (int(v) for v in bbox)
    pad = 6
    crop = page_img.crop((max(0, x0 - pad), max(0, y0 - pad), min(page_img.width, x1 + pad),
                          min(page_img.height, y1 + pad)))
    if crop.width < 20 or crop.height < 20:
        return
    b = io.BytesIO()
    crop.convert("RGB").save(b, format="PNG")
    b.seek(0)
    # keep the picture's size relative to the page width
    frac = min(1.0, max(0.15, crop.width / page_img.width))
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(b, width=int(usable_w * frac))


def write_txt(path, pages) -> None:
    out = []
    for pl in pages:
        out.append(f"===== עמוד {pl.index + 1} / Page {pl.index + 1} =====")
        for para in pl.paras:
            if para.kind == "picture":
                out.append("[תמונה / picture]")
                continue
            if para.text:
                out.append(para.text)
        out.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out).rstrip() + "\n")
