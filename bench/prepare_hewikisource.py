"""Hebrew printed pages with human-proofread text, from Hebrew Wikisource.

Wikisource volunteers transcribe scanned books page by page (ProofreadPage).
We sample pages marked "proofread" (quality 3) or "validated" (quality 4),
download the scan of that page from Wikimedia Commons (1920 px wide) and the
page's rendered text as the answer key.

usage: python bench/prepare_hewikisource.py [N_PAGES=60] [--per-book 3] [--seed 0]

Sources: https://he.wikisource.org (text CC BY-SA 4.0), scans on Wikimedia
Commons (mostly public domain; each page's file URL is in the manifest).
Answer-key caveats: the running header, page number and catchword printed on
the page are usually not in the transcription, so every system gets a few
"extra" words per page, and scholarly footnotes or variant-reading apparatus
printed under the text often aren't either (score.py's span word accuracy
ignores output before and after the transcribed text for that reason).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, download, user_agent, write_manifest  # noqa: E402

API = "https://he.wikisource.org/w/api.php"
NAME = "he_wikisource"
WIDTH = 1920          # a standard Wikimedia thumbnail width
SKIP_CLASSES = {"prp-page-qualityheader", "TooltipSpan", "mw-editsection", "reference", "noprint",
                "mw-cite-backlink", "mw-empty-elt", "ws-noexport"}
BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt", "table", "center", "poem"}


def api(**p):
    p.setdefault("format", "json")
    p.setdefault("formatversion", "2")
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(p), headers=user_agent())
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 + 10 * attempt)


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], []

    def handle_starttag(self, tag, attrs):
        cls = set((dict(attrs).get("class") or "").split())
        if self.skip:
            if tag not in ("br", "img", "hr", "wbr"):
                self.skip.append(tag)
            return
        if tag in ("style", "script") or cls & SKIP_CLASSES:
            if tag not in ("br", "img", "hr", "wbr"):
                self.skip.append(tag)
            return
        if tag in BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if self.skip:
            self.skip.pop()
            return
        if tag in BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_text(h: str) -> str:
    p = _Text()
    p.feed(h)
    t = "".join(p.out).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def list_pages():
    m = api(action="query", meta="allmessages",
            ammessages="proofreadpage_quality4_category|proofreadpage_quality3_category")
    cats = {x["name"][-10]: x["content"] for x in m["query"]["allmessages"]}   # '4' / '3'
    out = []
    for q, cat in sorted(cats.items()):
        cont = {}
        while True:
            r = api(action="query", list="categorymembers", cmtitle="Category:" + cat, cmnamespace=104,
                    cmlimit=500, **cont)
            out += [(x["title"], int(q)) for x in r["query"]["categorymembers"]]
            if "continue" not in r:
                break
            cont = r["continue"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=60)
    ap.add_argument("--per-book", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    root = DATA / NAME
    (root / "pages").mkdir(parents=True, exist_ok=True)
    (root / "gt").mkdir(parents=True, exist_ok=True)
    pages = list_pages()
    print(f"{len(pages)} proofread/validated pages on he.wikisource")
    books = {}
    for title, q in pages:
        if "/" not in title:
            continue
        books.setdefault(title.split(":", 1)[1].rsplit("/", 1)[0], []).append((title, q))
    rng = random.Random(a.seed)
    names = sorted(books)
    rng.shuffle(names)
    for b in names:
        rng.shuffle(books[b])
    # round-robin over books so no single book dominates
    order = []
    for k in range(a.per_book):
        order += [books[b][k] for b in names if len(books[b]) > k]
    ids, items = [], []
    mf = root / "manifest.json"
    if mf.exists():                       # resume an interrupted run
        old = json.loads(mf.read_text(encoding="utf-8"))
        ids, items = old["ids"], old["items"]
    done_titles = {it["page"] for it in items}

    def save():
        write_manifest(NAME, {
            "name": NAME, "ids": ids, "items": items, "seed": a.seed, "per_book": a.per_book,
            "source": "https://he.wikisource.org (ProofreadPage, quality 3 'proofread' and 4 'validated')",
            "license": "text: CC BY-SA 4.0 (Wikisource); scans: Wikimedia Commons, per-file license "
                       "(mostly public domain)",
            "gt_format": "plain text, one paragraph per line",
        })

    for title, q in order:
        if len(ids) >= a.n:
            break
        if title in done_titles:
            continue
        fname, num = title.split(":", 1)[1].rsplit("/", 1)
        if not num.isdigit() or not fname.lower().endswith((".pdf", ".djvu")):
            continue
        try:
            h = api(action="parse", page=title, prop="text")["parse"]["text"]
        except Exception as e:
            print("skip (parse):", title, e)
            continue
        if any(x in h for x in ("<math", "<table", "mwe-math")):
            continue
        text = html_text(h)
        heb = len(re.findall(r"[א-ת]", text))
        if len(text.split()) < 40 or heb < 0.6 * len(re.findall(r"\w", text)):
            continue
        try:
            ii = api(action="query", prop="imageinfo", titles="File:" + fname, iiprop="url",
                     iiurlparam=f"page{num}-{WIDTH}px")["query"]["pages"][0]
        except Exception as e:
            print("skip (imageinfo):", title, e)
            continue
        info = (ii.get("imageinfo") or [{}])[0]
        if not info.get("thumburl"):
            continue
        # the API may return a smaller bucketed URL; ask for the standard 1920 px rendering
        url = re.sub(r"/page\d+-\d+px-", f"/page{num}-{WIDTH}px-", info["thumburl"].split("?")[0])
        pid = f"ws{len(ids) + 1:03d}"
        try:
            download(url, root / "raw" / f"{pid}.jpg")
        except Exception as e:
            print("skip (image):", title, e)
            continue
        from PIL import Image
        Image.open(root / "raw" / f"{pid}.jpg").convert("RGB").save(root / "pages" / f"{pid}.png")
        (root / "gt" / f"{pid}.txt").write_text(text, encoding="utf-8")
        ids.append(pid)
        items.append({"id": pid, "page": title, "quality": q, "book": fname,
                      "text_url": "https://he.wikisource.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                      "image_url": url, "file_page": info.get("descriptionurl")})
        print(pid, q, title, len(text.split()), "words", flush=True)
        save()
        time.sleep(1.0)
    save()
    print(f"wrote {len(ids)} pages to {root}")


if __name__ == "__main__":
    main()
