"""Score every system's output and write results/scores.json + results/summary.md.

usage: python bench/score.py [DATASET ...]      (default: every dataset with results)

Metrics
  Text sets (he_wikisource, recbench_*):
    word acc     DualOCR's own metric (dualocr/evaluate.py): 1 - word-level
                 Levenshtein distance / reference words, after normalizing
                 quotes/dashes/bullets and dropping [editorial notes]. Order
                 matters, so missing, extra and out-of-order words all count.
    letters acc  same, ignoring punctuation attached to words
    char acc     1 - character edit distance / reference characters
    char acc (no spaces)  the same with all whitespace removed first, so a word
                 split or joined differently (rec_bench breaks words mid-word
                 at line ends; engines that rejoin them are not penalized)
    BoW F1       order-free: F1 of the multiset of punctuation-free, lowercased
                 words (robust to reading-order differences)
    span word acc  in-order, punctuation-free word accuracy against the best
                 matching stretch of the output: text the output has before or
                 after the reference (footnotes, apparatus, running heads the
                 reference doesn't transcribe) is not counted as errors
    Hebrew sets are scored with niqqud/cantillation removed on both sides.
    Unlike dualocr/evaluate.py's default, [bracketed text] is kept on both
    sides: here it is printed text, and stripping it let one stray "[" in an
    output delete everything up to a much later "]".
  recbench_* also gets "Surya line score": Surya's benchmark metric
    (rapidfuzz ratio per reference line, weighted by sqrt(line length)), adapted
    to page-level output: each reference line is matched against its best
    partial match in the whole output (fuzz.partial_ratio) instead of a 1:1
    predicted line, because only Surya's script supplies the line boxes.
  recbench_he drops pages whose answer key is stored backwards (visual order)
    or has no Hebrew at all (an English page labeled Hebrew).
  funsd: order-free word precision / recall / F1 (forms have no single
    reading order).
  olmocr_*: the olmOCR-bench unit tests (present / absent / order), run with
    the benchmark's own test code (vendored in olmocr_tests/); score = share of
    tests passed, as in the official harness.
scores.json also has seconds/page (median, from results/<dataset>/timing.jsonl);
they are only comparable when nothing else was running (see loadavg_1m there).
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common import DATA, RESULTS, SYSTEMS, SYSTEM_NAMES, WORK, manifest, pred_path, strip_marks  # noqa: E402
from dualocr import evaluate as E  # noqa: E402

E.strip_editorial = lambda t: t

HEBREW_SETS = ("he_wikisource", "recbench_he")
_FINAL_START = re.compile(r"(?:^|\s)[ךםןףץ][א-ת]")


def bow(text: str) -> Counter:
    return Counter(t for t in (E.letters(w).lower() for w in E.tokens(text)) if t)


def span_errors(gold: list, pred: list) -> int:
    """Word edit distance between gold and the closest substring of pred."""
    import numpy as np
    if not gold:
        return 0
    ids = {}
    g = np.array([ids.setdefault(w, len(ids)) for w in gold])
    p = np.array([ids.setdefault(w, len(ids)) for w in pred], dtype=np.int64)
    m = len(p)
    col = np.arange(m + 1)
    prev = np.zeros(m + 1, dtype=np.int64)
    for i, gw in enumerate(g, 1):
        a = np.empty(m + 1, dtype=np.int64)
        a[0] = i
        a[1:] = np.minimum(prev[:-1] + (p != gw), prev[1:] + 1)
        prev = col + np.minimum.accumulate(a - col)
    return int(prev.min())


def plain_words(text: str) -> list:
    return [t for t in (E.letters(w).lower() for w in E.tokens(text)) if t]


def prf(gold: Counter, pred: Counter):
    inter = sum((gold & pred).values())
    p = inter / max(1, sum(pred.values()))
    r = inter / max(1, sum(gold.values()))
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), inter, sum(gold.values()), sum(pred.values())


# --- Surya's benchmark normalization (surya v0.17 benchmark/recognition.py) ---
_BULLETS = re.compile(r"•\s+|·\s+|○\s+|◦\s+|▪\s+|▫\s+|➢\s+|➤\s+|★\s+|✓\s+|✗\s+|✦\s+|\\bullet\s+")


def surya_norm(t: str) -> str:
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    t = _BULLETS.sub("*", t)
    t = unicodedata.normalize("NFKC", t)
    return t.strip().lower().replace(",", ".")


def surya_line_score(ref_lines, pred_text: str) -> tuple:
    from rapidfuzz import fuzz
    pred = surya_norm(pred_text.replace("\n", " "))
    num = den = 0.0
    for ln in ref_lines:
        r = surya_norm(ln)
        if not r:
            continue
        w = math.sqrt(len(r))
        s = fuzz.partial_ratio(r, pred, score_cutoff=20) / 100 if pred else 0.0
        num += s * w
        den += w
    return num, den


def reversed_hebrew(lines) -> bool:
    """rec_bench has some Hebrew pages whose answer key is in visual (reversed)
    order, e.g. 'ןכותםיניינע' for 'תוכן עניינים'. Final letters at word starts
    give it away."""
    t = " ".join(lines)
    heb_words = len(re.findall(r"[א-ת]{2,}", t))
    return heb_words > 10 and len(_FINAL_START.findall(t)) > 0.1 * heb_words


def load_gt(ds: str, pid: str):
    g = DATA / ds / "gt"
    if (g / f"{pid}.txt").exists():
        return {"text": (g / f"{pid}.txt").read_text(encoding="utf-8")}
    d = json.loads((g / f"{pid}.json").read_text(encoding="utf-8"))
    if "lines" in d:
        d["text"] = "\n".join(d["lines"])
    return d


def timing(ds: str) -> dict:
    p = RESULTS / ds / "timing.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
    out = {}
    for s in SYSTEMS:
        v = []
        for r in rows:
            try:
                if s == "tesseract_raw":
                    v.append(r["tesseract_raw"])
                elif s == "tesseract":
                    v.append(r["prep"] + r["tesseract"])
                elif s == "surya":
                    v.append(r["surya"])
                else:
                    q = s.split("_")[1]
                    v.append(r["prep"] + max(r["surya"], r["tesseract"]) + r[f"decide_{q}"])
            except KeyError:
                pass
        out[s] = statistics.median(v) if v else None   # median: robust to a batch slowed by other load
    return out


def _labels(ds: str) -> dict:
    p = DATA / ds / "labels.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def score_text(ds: str, ids, systems):
    from rapidfuzz.distance import Levenshtein
    heb = ds in HEBREW_SETS
    lab = _labels(ds)
    excluded = list(lab.get("exclude", {}).get("ids", []))
    if ds.startswith("recbench") and heb:
        excluded += [pid for pid in ids if reversed_hebrew(load_gt(ds, pid)["lines"])
                     or not re.search(r"[א-ת]", load_gt(ds, pid)["text"])]
    ids = [i for i in ids if i not in set(excluded)]
    groups = {"all": ids, **{g: [i for i in v if i in set(ids)] for g, v in lab.get("groups", {}).items()}}
    res = {}
    for s in systems:
        per = {}       # pid -> raw counts
        for pid in ids:
            p = pred_path(ds, s, pid)
            if not p.exists():
                continue
            gt = load_gt(ds, pid)
            gold, pred = gt["text"], p.read_text(encoding="utf-8")
            if heb:
                gold, pred = strip_marks(gold), strip_marks(pred)
            sc = E.score(gold, pred, keep_errors=False)
            g0, p0 = re.sub(r"\s+", "", E.normalize(gold)), re.sub(r"\s+", "", E.normalize(pred))
            _, _, _, inter, g, pr = prf(bow(gold), bow(pred))
            gw = plain_words(gold)
            c = {"sc": sc, "ns": (Levenshtein.distance(g0, p0), len(g0)), "bw": (inter, g, pr),
                 "sp": (span_errors(gw, plain_words(pred)), len(gw))}
            if ds.startswith("recbench"):
                c["ls"] = surya_line_score(gt["lines"], pred)
            per[pid] = c
        out = {}
        for gname, gids in groups.items():
            tot, bw, ns, sp, ls, n = E.Score(), [0, 0, 0], [0, 0], [0, 0], [0.0, 0.0], 0
            for pid in gids:
                if pid not in per:
                    continue
                c = per[pid]
                tot.add(c["sc"])
                ns[0] += c["ns"][0]; ns[1] += c["ns"][1]
                sp[0] += c["sp"][0]; sp[1] += c["sp"][1]
                for k in range(3):
                    bw[k] += c["bw"][k]
                if "ls" in c:
                    ls[0] += c["ls"][0]; ls[1] += c["ls"][1]
                n += 1
            p_, r_ = bw[0] / max(1, bw[2]), bw[0] / max(1, bw[1])
            r = {"pages": n, "words": tot.n, "word_acc": tot.word_acc, "letters_acc": tot.letter_acc,
                 "char_acc": tot.char_acc, "char_acc_nospace": 1 - ns[0] / max(1, ns[1]),
                 "span_word_acc": 1 - sp[0] / max(1, sp[1]),
                 "bow_f1": 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0, "bow_precision": p_, "bow_recall": r_}
            if ds.startswith("recbench"):
                r["surya_line_score"] = ls[0] / ls[1] if ls[1] else 0.0
            out[gname] = r
        res[s] = dict(out["all"])
        res[s]["per_page_word_acc"] = {pid: round(c["sc"].word_acc, 4) for pid, c in per.items()}
        res[s]["per_page_span_word_acc"] = {pid: round(1 - c["sp"][0] / max(1, c["sp"][1]), 4) for pid, c in per.items()}
        if len(groups) > 1:
            res[s]["groups"] = {g: v for g, v in out.items() if g != "all"}
    return res, excluded


def score_funsd(ds: str, ids, systems):
    res = {}
    for s in systems:
        tp = g = pr = n = 0
        for pid in ids:
            p = pred_path(ds, s, pid)
            if not p.exists():
                continue
            gold = Counter(t for t in (E.letters(w).lower() for w in load_gt(ds, pid)["words"]) if t)
            _, _, _, inter, gg, pp = prf(gold, bow(p.read_text(encoding="utf-8")))
            tp += inter; g += gg; pr += pp; n += 1
        prec, rec = tp / max(1, pr), tp / max(1, g)
        res[s] = {"pages": n, "words": g, "bow_precision": prec, "bow_recall": rec,
                  "bow_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    return res


def score_olmocr(ds: str, ids, systems):
    from olmocr_tests.tests import load_tests
    tests = load_tests(str(DATA / ds / "tests.jsonl"))
    items = {(i["pdf"], i["page"]): i["id"] for i in manifest(ds)["items"]}
    # DualOCR's own page classifier: pages where prep found little printed text
    hand = set()
    for pid in ids:
        pj = WORK / "cache" / ds / pid / "prep.json"
        if pj.exists() and any("little printed text" in n for n in json.loads(pj.read_text())["notes"]):
            hand.add(pid)
    res = {}
    for s in systems:
        by_type, fails = {}, []
        groups = {"all": [0, 0], "printed": [0, 0], "little_printed_text": [0, 0]}
        for t in tests:
            pid = items.get((t.pdf, t.page))
            p = pred_path(ds, s, pid) if pid else None
            if p is None or not p.exists():
                continue
            ok, why = t.run(p.read_text(encoding="utf-8"))
            bt = by_type.setdefault(t.type, [0, 0])
            bt[0] += int(ok); bt[1] += 1
            for gname in ("all", "little_printed_text" if pid in hand else "printed"):
                groups[gname][0] += int(ok); groups[gname][1] += 1
            if not ok:
                fails.append((t.id, why[:120]))
        res[s] = {"pages": len({items[k] for k in items}), "tests": groups["all"][1],
                  "pass_rate": groups["all"][0] / max(1, groups["all"][1]),
                  "pass_rate_printed": groups["printed"][0] / max(1, groups["printed"][1]),
                  "tests_printed": groups["printed"][1],
                  "pass_rate_little_text": groups["little_printed_text"][0] / max(1, groups["little_printed_text"][1]),
                  "tests_little_text": groups["little_printed_text"][1],
                  "by_type": {k: v[0] / v[1] for k, v in by_type.items()},
                  "n_by_type": {k: v[1] for k, v in by_type.items()}}
    return res, sorted(hand)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    dss = argv or sorted(p.name for p in RESULTS.iterdir() if (p / "pred").is_dir())
    allres = {}
    for ds in dss:
        m = manifest(ds)
        ids = m["ids"]
        systems = [s for s in SYSTEMS if (RESULTS / ds / "pred" / s).is_dir()]
        extra = {}
        if ds.startswith("olmocr"):
            r, hand = score_olmocr(ds, ids, systems)
            extra["little_printed_text_pages"] = hand
        elif ds == "funsd":
            r = score_funsd(ds, ids, systems)
        else:
            r, excl = score_text(ds, ids, systems)
            extra["excluded"] = excl
        tm = timing(ds)
        for s in r:
            r[s]["sec_per_page"] = tm.get(s)
        allres[ds] = {"systems": r, "source": m.get("source"), "license": m.get("license"), **extra}
    (RESULTS / "scores.json").write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")
    md = summary(allres)
    (RESULTS / "summary.md").write_text(md, encoding="utf-8")
    print(md)


def _pct(x):
    return "" if x is None else f"{100 * x:.1f}"


def summary(allres) -> str:
    out = ["| Benchmark | System | Main metric | Word acc (exact) | Letters acc | Char acc | BoW F1 | Pages |",
           "|---|---|---|---|---|---|---|---|"]
    for ds, d in allres.items():
        for s, r in d["systems"].items():
            if ds.startswith("olmocr"):
                main_m = f"tests passed {_pct(r['pass_rate'])}% (printed {_pct(r['pass_rate_printed'])}%, " \
                         f"handwritten/little text {_pct(r['pass_rate_little_text'])}%)"
            elif ds == "funsd":
                main_m = f"word F1 {_pct(r['bow_f1'])} (P {_pct(r['bow_precision'])} / R {_pct(r['bow_recall'])})"
            elif ds.startswith("recbench"):
                main_m = (f"Surya line score {_pct(r['surya_line_score'])}; "
                          f"char acc (no spaces) {_pct(r['char_acc_nospace'])}")
            else:
                main_m = (f"span word acc {_pct(r['span_word_acc'])}; ref-word recall {_pct(r['bow_recall'])}; "
                          f"char acc (no spaces) {_pct(r['char_acc_nospace'])}")
            out.append(f"| {ds} | {SYSTEM_NAMES[s]} | {main_m} | {_pct(r.get('word_acc'))} | {_pct(r.get('letters_acc'))} | "
                       f"{_pct(r.get('char_acc'))} | {_pct(r.get('bow_f1'))} | {r['pages']} |")
    # sub-groups (e.g. he_wikisource by script), word acc / BoW recall
    for ds, d in allres.items():
        any_s = next(iter(d["systems"].values()), {})
        if "groups" not in any_s:
            continue
        out.append("")
        out.append(f"{ds} by group (span word acc / order-free recall of the reference words / char acc no-spaces):")
        out.append("")
        gnames = list(any_s["groups"])
        out.append("| System | " + " | ".join(f"{g} ({any_s['groups'][g]['pages']} p.)" for g in gnames) + " |")
        out.append("|---|" + "---|" * len(gnames))
        for s, r in d["systems"].items():
            cells = [f"{_pct(r['groups'][g]['span_word_acc'])} / {_pct(r['groups'][g]['bow_recall'])} / "
                     f"{_pct(r['groups'][g]['char_acc_nospace'])}" for g in gnames]
            out.append(f"| {SYSTEM_NAMES[s]} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
