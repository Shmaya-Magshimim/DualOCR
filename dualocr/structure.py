"""Turn decided blocks into document structure (paragraphs, headings, lists,
pictures) and check numbering."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .align import AlignedBlock

PICTURE_LABELS = {"Picture", "Figure", "Image", "Diagram", "Chart", "ChemicalBlock", "Handwriting"}
HEADER_LABELS = {"PageHeader"}
FOOTER_LABELS = {"PageFooter"}
HEADING_LABELS = {"SectionHeader", "Title"}
LIST_LABELS = {"ListGroup", "ListItem"}
CAPTION_LABELS = {"Caption", "Footnote"}
BULLETS = {"•", "●", "▪", "■", "◦", "-", "–", "*"}
PAGENUM = re.compile(r"^\s*[-–(]?\s*(\d{1,4})\s*[-–)]?\s*$")


@dataclass
class Para:
    kind: str                     # heading | text | list_item | header | footer | caption | picture
    words: list                   # list of Slot (decided)
    bbox: Optional[tuple] = None
    bold: bool = False
    underline: bool = False
    level: int = 0

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.words if s.text).strip()


@dataclass
class PageLayout:
    index: int
    paras: List[Para] = field(default_factory=list)
    printed_number: Optional[int] = None
    footer_bbox: Optional[tuple] = None
    notes: List[str] = field(default_factory=list)


def _style(html: str):
    h = html or ""
    return ("<b>" in h or "<strong>" in h), ("<u>" in h)


def layout_page(index: int, ablocks: List[AlignedBlock], orphan_paras: List[Para] = ()) -> PageLayout:
    pl = PageLayout(index)
    items = []
    for ab in sorted(ablocks, key=lambda a: a.block.order):
        b = ab.block
        bold, und = _style(b.html)
        if b.label in PICTURE_LABELS:
            items.append(Para("picture", [], b.bbox))
            continue
        text = " ".join(s.text for s in ab.slots if s.text).strip()
        if b.label in FOOTER_LABELS:
            m = PAGENUM.match(text)
            if m:
                pl.printed_number = int(m.group(1))
                pl.footer_bbox = b.bbox
                continue
        if not text:
            continue
        if b.label in LIST_LABELS:
            # one paragraph per Surya line (list items); keep the bullet Surya saw
            lines = {}
            for s in ab.slots:
                lines.setdefault(s.line, []).append(s)
            src_lines = (b.text or "").split("\n")
            for li in sorted(lines):
                ws = lines[li]
                if not any(w.text for w in ws):
                    continue
                src = src_lines[li].strip() if li < len(src_lines) else ""
                first = next((w for w in ws if w.text), None)
                if src[:1] in BULLETS and first is not None and first.text[:1] not in BULLETS:
                    from .align import Slot
                    bullet = Slot(surya=src[:1])
                    bullet.text, bullet.status, bullet.conf = src[:1], "agree", 1.0
                    ws = [bullet] + ws
                items.append(Para("list_item", ws, b.bbox, bold, und))
            continue
        kind = ("heading" if b.label in HEADING_LABELS else "header" if b.label in HEADER_LABELS
                else "footer" if b.label in FOOTER_LABELS else "caption" if b.label in CAPTION_LABELS
                else "text")
        # paragraphs inside a Text block: Surya separates <p> with new lines
        lines = {}
        for s in ab.slots:
            lines.setdefault(s.line, []).append(s)
        if kind == "text" and len(lines) > 1 and "<p>" in (b.html or "") and b.html.count("<p>") > 1:
            for li in sorted(lines):
                items.append(Para(kind, lines[li], b.bbox, bold, und))
        else:
            items.append(Para(kind, list(ab.slots), b.bbox, bold, und))
    items.extend(orphan_paras)
    # keep reading order; orphan paragraphs are placed by their vertical position
    pl.paras = _merge_by_position(items)
    return pl


def _merge_by_position(items: List[Para]) -> List[Para]:
    main = [p for p in items if getattr(p, "_orphan", False) is False]
    orph = [p for p in items if getattr(p, "_orphan", False)]
    for o in orph:
        oy = o.bbox[1] if o.bbox else 0
        pos = len(main)
        for k, p in enumerate(main):
            if p.bbox and p.bbox[1] > oy:
                pos = k
                break
        main.insert(pos, o)
    return main


_NUM = re.compile(r"^\(?(\d{1,3})[.)]")
_HEB_NUM = re.compile(r"^([א-י])[.)]")
HEB_ORDER = "אבגדהוזחטי"


def numbering_report(pages: List[PageLayout]) -> List[str]:
    """Printed page numbers vs file order, and gaps/repeats in numbered lists."""
    out = []
    printed = [(p.index + 1, p.printed_number) for p in pages]
    have = [(i, n) for i, n in printed if n is not None]
    offset = None
    if have:   # documents may start at another number; use the most common offset
        from collections import Counter
        offset = Counter(n - i for i, n in have).most_common(1)[0][0]
    for i, n in have:
        if n - i != offset:
            out.append(f"Page {i} of the file is printed as page {n} (expected {i + offset}).")
    missing = [i for i, n in printed if n is None]
    if have and missing and len(missing) <= len(pages) // 2:
        out.append("No printed page number found on page(s) " + ", ".join(map(str, missing)) + ".")
    # numbered paragraphs: a list may restart at 1 at any time; report skips and repeats
    prev = None
    for p in pages:
        for para in p.paras:
            t = para.text.lstrip("•●▪- ").strip()
            m = _NUM.match(t)
            if not m:
                continue
            v = int(m.group(1))
            if prev is not None and v != 1:
                pv, ppage = prev
                where = f"page {ppage}" if ppage == p.index + 1 else f"pages {ppage}-{p.index + 1}"
                if v == pv:
                    out.append(f"Item {v}. appears twice in a row ({where}).")
                elif pv < v <= pv + 4 and v != pv + 1:
                    missing_n = ", ".join(f"{k}." for k in range(pv + 1, v))
                    out.append(f"List goes from {pv}. to {v}. ({where}): {missing_n} missing, or misread.")
            prev = (v, p.index + 1)
    return out
