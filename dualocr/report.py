"""A self-contained HTML review report: summary, what each engine got wrong,
numbering problems, and every flagged word with a zoomed crop."""
from __future__ import annotations

import base64
import html
import io
import time
from collections import Counter
from typing import List

from PIL import Image

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e4e4e2;--warn:#fff3bf;--typo:#d3f3f1;--accent:#3b5bdb}
@media (prefers-color-scheme: dark){:root{--bg:#161618;--card:#202023;--ink:#ececee;--muted:#9a9aa0;--line:#333338;--warn:#5c4b00;--typo:#0d4d49;--accent:#8ea2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}
main{max-width:1000px;margin:0 auto;padding:28px 16px 60px}h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 10px}
.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:16px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.stat b{display:block;font-size:22px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
td,th{padding:8px 10px;border-bottom:1px solid var(--line);text-align:start;vertical-align:top}th{font-size:13px;color:var(--muted);font-weight:600}
.word{font-size:18px}.he{direction:rtl;unicode-bidi:plaintext}.crop{max-width:420px;max-height:90px;border:1px solid var(--line);border-radius:6px;background:#fff}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px}.u{background:var(--warn)}.t{background:var(--typo)}
ul{padding-inline-start:20px}
"""


def _img_b64(img: Image.Image) -> str:
    b = io.BytesIO()
    img.convert("RGB").save(b, format="JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def write_report(path, pages, images, inputs: List[str], seconds: float, numbering: List[str],
                 page_notes: dict, settings_desc: str) -> dict:
    slots = [sl for pl in pages for para in pl.paras for sl in para.words if sl.text]
    n = len(slots)
    status = Counter(sl.status for sl in slots)
    kinds = {"surya": Counter(), "tesseract": Counter()}
    for sl in slots:
        for eng, k in (sl.kinds or {}).items():
            kinds.setdefault(eng, Counter())[k.split(" (")[0]] += 1
    flagged = []
    for pl in pages:
        for para in pl.paras:
            for i, sl in enumerate(para.words):
                if sl.text and sl.status in ("uncertain", "source_typo"):
                    ctx_l = " ".join(w.text for w in para.words[max(0, i - 4):i] if w.text)
                    ctx_r = " ".join(w.text for w in para.words[i + 1:i + 5] if w.text)
                    flagged.append((pl.index, sl, ctx_l, ctx_r))
    stats = {"words": n, "agree": status.get("agree", 0), "resolved": status.get("resolved", 0),
             "uncertain": status.get("uncertain", 0), "source_typo": status.get("source_typo", 0),
             "only_one": status.get("only_one", 0)}
    e = html.escape
    out = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>OCR review report</title><style>{CSS}</style></head><body><main>"]
    out.append("<h1>OCR review report</h1>")
    out.append(f"<div class='muted'>{e(', '.join(inputs))} · {len(pages)} pages · {seconds / 60:.1f} min · "
               f"{e(settings_desc)} · {time.strftime('%Y-%m-%d %H:%M')}</div>")
    pct = lambda x: f"{(x / n * 100 if n else 0):.1f}%"
    out.append("<div class='grid'>")
    for label, val, sub in (("Words", n, ""), ("Both engines agreed", stats["agree"], pct(stats["agree"])),
                            ("Disagreements resolved", stats["resolved"], pct(stats["resolved"])),
                            ("Highlighted (really unsure)", stats["uncertain"], pct(stats["uncertain"])),
                            ("Possible typos in original", stats["source_typo"], "kept as printed")):
        out.append(f"<div class='stat'><span class='muted'>{e(label)}</span><b>{val}</b><span class='muted'>{e(sub)}</span></div>")
    out.append("</div>")
    out.append("<h2>What each engine got wrong (before voting)</h2><table><tr><th>Kind of error</th><th>Surya</th><th>Tesseract</th></tr>")
    allk = sorted(set(kinds["surya"]) | set(kinds["tesseract"]), key=lambda k: -(kinds["surya"][k] + kinds["tesseract"][k]))
    for k in allk:
        out.append(f"<tr><td>{e(k)}</td><td>{kinds['surya'][k]}</td><td>{kinds['tesseract'][k]}</td></tr>")
    if not allk:
        out.append("<tr><td colspan=3 class='muted'>No disagreements.</td></tr>")
    out.append("</table>")
    out.append("<h2>Page and list numbering</h2>")
    out.append("<ul>" + "".join(f"<li>{e(x)}</li>" for x in numbering) + "</ul>" if numbering
               else "<p class='muted'>No numbering problems found.</p>")
    notes = [(k, v) for k, v in sorted(page_notes.items()) if v]
    if notes:
        out.append("<h2>Page preparation</h2><ul>" + "".join(
            f"<li>Page {k + 1}: {e('; '.join(v))}</li>" for k, v in notes) + "</ul>")
    out.append(f"<h2>Words to check ({len(flagged)})</h2>")
    if flagged:
        out.append("<table><tr><th>Page</th><th>Word in context</th><th>Scan</th><th>Readings</th></tr>")
        for pidx, sl, cl, cr in flagged[:400]:
            crop = ""
            if images and images[pidx] is not None and sl.bbox:
                x0, y0, x1, y1 = sl.bbox
                h = y1 - y0
                im = images[pidx].crop((max(0, int(x0 - 3 * h)), max(0, int(y0 - h * .5)),
                                        min(images[pidx].width, int(x1 + 3 * h)), min(images[pidx].height, int(y1 + h * .5))))
                im.thumbnail((420, 90))
                crop = f"<img class='crop' src='{_img_b64(im)}'>"
            tag = "<span class='tag u'>uncertain</span>" if sl.status == "uncertain" else \
                "<span class='tag t'>possible typo in original</span>"
            reads = []
            for eng, key in (("Surya", "surya"), ("Tesseract", "tess")):
                v = sl.readings.get(key + "_split") or getattr(sl, key)
                if v is not None:
                    reads.append(f"{eng}: <span class='he'>{e(v)}</span>")
            if sl.readings.get("llm"):
                reads.append(f"AI: <span class='he'>{e(sl.readings['llm'])}</span>")
            out.append(f"<tr><td>{pidx + 1}</td><td><div class='he'>{e(cl)} <b class='word'>{e(sl.text)}</b> {e(cr)}</div>{tag}</td>"
                       f"<td>{crop}</td><td>{'<br>'.join(reads)}</td></tr>")
        out.append("</table>")
    else:
        out.append("<p class='muted'>Nothing flagged.</p>")
    out.append("</main></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(out))
    return stats
