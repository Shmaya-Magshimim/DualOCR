"""Surya engine: layout blocks + text, with per-word confidence from token
log-probabilities. Runs Surya's model in a local llama.cpp server on the GPU.
"""
from __future__ import annotations

import html as htmlmod
import logging
import math
import re
import threading
from typing import List, Optional

from PIL import Image

from ..model import Block

log = logging.getLogger("dualocr.surya")

_manager = None
_lock = threading.Lock()


def _windows_fixes():
    """Surya checks and stops its llama.cpp server with POSIX calls
    (os.WNOHANG, os.kill(pid, 0)) that don't work on Windows."""
    import os

    if os.name != "nt":
        return
    from surya.inference.backends import spawn

    from .. import osutil

    if getattr(spawn, "_dualocr_patched", False):
        return
    orig_alive = spawn._server_is_alive

    def server_is_alive(handle):
        if handle.cleanup_kind == "process" and handle.pid:
            return osutil.pid_alive(handle.pid)
        return orig_alive(handle)

    spawn._server_is_alive = server_is_alive
    spawn._stop_process = lambda pid, name: osutil.kill_pid(pid)
    spawn._dualocr_patched = True


def manager():
    """One Surya inference server per process, started lazily."""
    global _manager
    with _lock:
        if _manager is None:
            from surya.inference import SuryaInferenceManager

            _windows_fixes()
            _manager = SuryaInferenceManager(method="llamacpp", lazy=True)
        return _manager


def shutdown():
    global _manager
    with _lock:
        if _manager is not None:
            try:
                _manager.stop()
            except Exception:
                pass
            _manager = None


_TAG = re.compile(r"<[^>]+>")


def _canon(label: str) -> str:
    try:
        from surya.layout.label import LAYOUT_PRED_RELABEL
        return LAYOUT_PRED_RELABEL.get(label, label)
    except Exception:
        return label


_MARKER = re.compile(r"<\|[^|<>]{0,40}\|>")


def html_to_text(h: str) -> str:
    h = re.sub(r"<br\s*/?>", "\n", h, flags=re.I)
    h = re.sub(r"</(p|li|h\d|div|tr|ul|ol)>", "\n", h, flags=re.I)
    t = htmlmod.unescape(_TAG.sub("", h))
    t = _MARKER.sub("", t)          # model control markers (sometimes arrive escaped)
    t = "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n"))
    return re.sub(r"\n{2,}", "\n", t).strip()


def html_token_styles(h: str):
    """[(bold, underline)] for each whitespace token of html_to_text(h)."""
    out_chars = []   # (char, bold, underline)
    bold = und = 0
    i, n = 0, len(h)
    while i < n:
        if h[i] == "<":
            j = h.find(">", i)
            if j < 0:
                break
            tag = h[i + 1:j].strip().lower()
            name = tag.lstrip("/").split()[0].rstrip("/") if tag.lstrip("/") else ""
            closing = tag.startswith("/")
            if name in ("b", "strong", "h1", "h2", "h3", "h4", "h5", "h6"):
                bold += -1 if closing else 1
            elif name == "u":
                und += -1 if closing else 1
            elif name in ("br", "p", "li", "div", "tr", "ul", "ol") and (closing or name == "br"):
                out_chars.append(("\n", 0, 0))
            i = j + 1
            continue
        if h[i] == "&":
            j = h.find(";", i)
            if 0 < j - i < 10:
                for ch in htmlmod.unescape(h[i:j + 1]):
                    out_chars.append((ch, bold > 0, und > 0))
                i = j + 1
                continue
        out_chars.append((h[i], bold > 0, und > 0))
        i += 1
    toks, cur = [], None
    for ch, b, u in out_chars:
        if ch.isspace():
            if cur is not None:
                toks.append(cur); cur = None
            continue
        if cur is None:
            cur = [ch, b, u]
        else:
            cur[0] += ch
            cur[2] = cur[2] or u
    if cur is not None:
        toks.append(cur)
    return [(t[0], bool(t[1]), bool(t[2])) for t in toks]


def visible_stream(logprobs: Optional[list]) -> tuple:
    """Rebuild the model's output as visible text only (tags and <|markers|>
    removed, entities decoded). Returns (text, per-char info) where info is
    (prob, [(alternative_token, prob), ...])."""
    if not logprobs:
        return "", []
    chars = []   # (char, prob, alts)
    for tok in logprobs:
        if not isinstance(tok, dict):
            continue
        t = tok.get("token", "")
        lp = tok.get("logprob")
        p = math.exp(lp) if lp is not None else 1.0
        alts = [(a.get("token", ""), math.exp(a["logprob"])) for a in (tok.get("top_logprobs") or [])[:6]
                if a.get("logprob") is not None]
        for ch in t:
            chars.append((ch, p, alts))
    out_text, out_info = [], []
    i, n = 0, len(chars)
    while i < n:
        ch = chars[i][0]
        if ch == "<":                      # skip <tag> and <|marker|>
            j = i
            while j < n and chars[j][0] != ">":
                j += 1
            i = j + 1
            continue
        if ch == "&":                      # entity
            j = i
            while j < n and j - i < 10 and chars[j][0] != ";":
                j += 1
            ent = "".join(c[0] for c in chars[i:j + 1])
            dec = htmlmod.unescape(ent)
            if len(dec) == 1 and j < n:
                out_text.append(dec)
                out_info.append((min(c[1] for c in chars[i:j + 1]), []))
                i = j + 1
                continue
        out_text.append(ch)
        out_info.append((chars[i][1], chars[i][2]))
        i += 1
    return "".join(out_text), out_info


def word_probs(text: str, vis: str, info: list, start: int = 0) -> tuple:
    """Align the words of `text` to the visible stream in order.
    Returns ([(word, min_prob or None, letter_alts)], new_cursor). letter_alts
    lists, for letters the model was unsure of, (index_in_word, [(alt, p)])."""
    out = []
    cur = start
    for w in text.split():
        i = vis.find(w, cur)
        if i < 0 or i - cur > 3000:
            out.append((w, None, []))
            continue
        seg = info[i:i + len(w)]
        probs = [s[0] for s in seg]
        alts = [(k, s[1]) for k, s in enumerate(seg) if s[0] < 0.9 and s[1]]
        out.append((w, min(probs) if probs else None, alts))
        cur = i + len(w)
    return out, cur


MAX_TOKENS_PRINTED = 8000     # a dense printed page here needs ~2,500
MAX_TOKENS_LOW_TEXT = 3000    # handwriting / photos: stop early instead of looping


def _full_page_items(images, request_logprobs=True, caps=None):
    from surya.inference.schema import BatchInputItem
    from surya.inference.prompts import PROMPT_TYPE_HIGH_ACCURACY_BBOX

    return [BatchInputItem(image=im, prompt_type=PROMPT_TYPE_HIGH_ACCURACY_BBOX,
                           max_tokens=(caps[i] if caps else MAX_TOKENS_PRINTED),
                           request_logprobs=request_logprobs, metadata={"i": i})
            for i, im in enumerate(images)]


def ocr_pages(images: List[Image.Image], on_page=None, parallel: int = 4, caps=None,
              notes=None) -> List[List[Block]]:
    """Full-page OCR for each image -> blocks in reading order.
    on_page(done_count) is called as each page finishes. caps: max tokens per
    page. notes: optional list of lists that receives per-page remarks."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from surya.inference.parsers import parse_full_page_html
    from surya.settings import settings

    mgr = manager()
    items = _full_page_items(images, caps=caps)
    outs = {}
    if on_page is None:
        outs = {o.metadata["i"]: o for o in mgr.generate(items)}
    else:
        mgr.start() if getattr(mgr.backend, "handle", None) is None else None
        with ThreadPoolExecutor(parallel) as ex:
            futs = [ex.submit(mgr.generate, [it]) for it in items]
            for k, f in enumerate(as_completed(futs)):
                for o in f.result():
                    outs[o.metadata["i"]] = o
                on_page(k + 1)
    results: List[Optional[List[Block]]] = [None] * len(images)
    fallback = []
    for i, im in enumerate(images):
        o = outs.get(i)
        w, h = im.size
        hit_cap = o is not None and getattr(o, "finish_reason", None) == "length"
        ok = o is not None and not o.error and o.raw
        parsed = []
        if ok:
            try:
                parsed = parse_full_page_html(o.raw)
            except Exception:
                parsed = []
        if hit_cap:
            # stopped at the token cap (often a loop on handwriting): keep the
            # complete blocks, drop the cut-off last one, don't retry slowly
            if parsed and parsed[-1].truncated:
                parsed = parsed[:-1]
            if notes is not None:
                notes[i].append("Surya stopped early on this page (very dense or handwritten); "
                                "text from Tesseract fills the rest")
        elif not parsed or (parsed and parsed[-1].truncated):
            low_text = caps is not None and caps[i] <= MAX_TOKENS_LOW_TEXT
            if low_text:
                # handwriting/photos: the slow block-by-block retry rarely helps
                parsed = [b for b in parsed if not b.truncated]
                if notes is not None:
                    notes[i].append("Surya could not lay out this page cleanly; text from Tesseract fills in")
            else:
                if notes is not None:
                    notes[i].append("Surya re-read this page block by block")
                fallback.append(i)
                continue
        if not parsed:
            results[i] = []
            continue
        vis, info = visible_stream(o.logprobs)
        cursor = 0
        blocks = []
        for k, b in enumerate(parsed):
            box = (b.bbox[0] / settings.BBOX_SCALE * w, b.bbox[1] / settings.BBOX_SCALE * h,
                   b.bbox[2] / settings.BBOX_SCALE * w, b.bbox[3] / settings.BBOX_SCALE * h)
            text = html_to_text(b.html)
            tp = None
            if vis:
                tp, cursor = word_probs(text, vis, info, cursor)
            blocks.append(Block(label=_canon(b.label), bbox=box, html=b.html, text=text, order=k,
                                conf=o.mean_token_prob or 1.0, token_probs=tp))
        results[i] = blocks
    if fallback:
        log.info("surya: block-mode fallback for %d page(s)", len(fallback))
        from surya.recognition import RecognitionPredictor

        rec = RecognitionPredictor(mgr)
        fb = rec([images[i] for i in fallback], full_page=True)
        for j, i in enumerate(fallback):
            blocks = []
            for b in fb[j].blocks:
                text = html_to_text(b.html or "")
                blocks.append(Block(label=_canon(b.label), bbox=tuple(b.bbox), html=b.html or "", text=text,
                                    order=b.reading_order, conf=b.confidence or 1.0))
            results[i] = blocks
    return [r or [] for r in results]


def ocr_crops(crops: List[Image.Image]) -> List[tuple]:
    """Block-mode OCR of small crops. Returns [(text, [(word, prob)])]."""
    from surya.inference.parsers import clean_block_html
    from surya.inference.prompts import PROMPT_TYPE_BLOCK
    from surya.inference.schema import BatchInputItem

    if not crops:
        return []
    items = [BatchInputItem(image=c, prompt_type=PROMPT_TYPE_BLOCK, max_tokens=512,
                            request_logprobs=True, metadata={"i": i}) for i, c in enumerate(crops)]
    outs = {o.metadata["i"]: o for o in manager().generate(items)}
    res = []
    for i in range(len(crops)):
        o = outs.get(i)
        if o is None or o.error or not o.raw:
            res.append(("", []))
            continue
        try:
            text = html_to_text(clean_block_html(o.raw))
        except Exception:
            text = html_to_text(o.raw)
        vis, info = visible_stream(o.logprobs)
        wp, _ = word_probs(text, vis, info) if vis else ([(w, None, []) for w in text.split()], 0)
        res.append((text, wp))
    return res
