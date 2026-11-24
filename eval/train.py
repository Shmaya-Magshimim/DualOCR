"""Learn the consensus weights from the answer-key pages.

Training data: Hebrew pages 1-10, English scan pages 1, 2, 4.
Held out (never used for training): Hebrew pages 11-20, English 5, 6, 7, 3.

usage: python eval/train.py [fast|balanced|max]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from dualocr import consensus as C, evaluate as E, rules as R  # noqa: E402
from dualocr.align import align_tokens  # noqa: E402
from dualocr.config import Settings  # noqa: E402
from dualocr.pipeline import decide_cached  # noqa: E402
import run_eval  # noqa: E402

CACHE = ROOT / "eval" / "cache"
GOLD = ROOT / "eval" / "gold"
TRAIN = {"heb23": set(range(1, 11)), "eng14": {1, 2, 4}}


def slot_sequence(pw):
    out = []
    for para in pw.layout.paras:
        if para.kind == "picture":
            continue
        out.extend(para.words)
    return out


def label_slots(slots, gold_tokens):
    """Pair each slot with the answer-key text it corresponds to."""
    anchors = []
    for sl in slots:
        a = C.surya_reading(sl) or C.tess_reading(sl) or ""
        anchors.append(a.replace(" ", "") if a else "∅")
    pairs = align_tokens(anchors, gold_tokens)
    labels = [None] * len(slots)
    for si, gi in pairs:
        if not si:
            continue
        g = R.norm(" ".join(gold_tokens[x] for x in gi)) if gi else ""
        if len(si) == 1:
            labels[si[0]] = g
        else:  # two slots matched one gold word: first gets it, second should vanish
            labels[si[0]] = g
            for x in si[1:]:
                labels[x] = ""
    return labels


def examples(doc, quality, weights=None, pages=None):
    works = decide_cached(CACHE / doc, Settings(quality=quality), weights=weights)
    vocab = works[0].notes[-1]
    ex = []
    if doc == "heb23":
        for pw in works:
            n = pw.index + 1
            if pages is not None and n not in pages:
                continue
            gold = E.tokens((GOLD / "heb23" / f"page_{n:02d}.txt").read_text())
            sl = slot_sequence(pw)
            for s, lab in zip(sl, label_slots(sl, gold)):
                ex.append((n, s, lab))
    else:
        raw = (GOLD / "eng14" / "scan2241.txt").read_text()
        order = run_eval._gold_order(doc, raw)
        seq, page_of = [], []
        for n in order:
            s = slot_sequence(works[n - 1])
            seq.extend(s)
            page_of.extend([n] * len(s))
        for n, s, lab in zip(page_of, seq, label_slots(seq, E.tokens(raw))):
            if pages is None or n in pages:
                ex.append((n, s, lab))
    data = []
    reach = total = 0
    for n, sl, lab in ex:
        if lab is None:
            continue
        ls = sl.readings.get("_line_script", "he")
        names, cands, X = C.slot_matrix(sl, ls, vocab)
        if X is None or len(names) < 2:
            continue
        normed = [R.norm(x) for x in names]
        if all(not E.letters(x) for x in normed if x) and lab and not E.letters(lab):
            continue  # bullets / punctuation-only slots that eval ignores
        total += 1
        if lab in normed:
            reach += 1
            data.append((X, normed.index(lab)))
    return data, reach, total


PRIOR = np.array([C.DEFAULT_W.get(f, 0) for f in C.FEATURES])
# Rules with too few training examples to learn safely stay at their prior:
# Surya identifies the script (Hebrew vs Latin) of a word far more reliably
# than Tesseract; Hebrew years start with ת.
FROZEN = {"script_fit", "year", "heb_ref"}
FROZEN_IDX = [C.FEATURES.index(f) for f in FROZEN if f in C.FEATURES]


def train(data, l2=0.02, iters=600, lr=0.05, verbose=True):
    """Conditional logit, regularized toward the hand-set prior weights."""
    w = PRIOR.copy()
    m = np.zeros_like(w); v = np.zeros_like(w)
    for t in range(1, iters + 1):
        g = l2 * (w - PRIOR)
        loss = 0.5 * l2 * ((w - PRIOR) @ (w - PRIOR))
        for X, y in data:
            z = X @ w
            z = z - z.max()
            p = np.exp(z); p /= p.sum()
            loss -= np.log(p[y] + 1e-12)
            g += X.T @ p - X[y]
        g /= len(data); loss /= len(data)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        w[FROZEN_IDX] = PRIOR[FROZEN_IDX]
        if verbose and t % 200 == 0:
            acc = np.mean([int(np.argmax(X @ w) == y) for X, y in data])
            print(f"  iter {t}: loss {loss:.4f}  train choice acc {acc:.4f}")
    return w


if __name__ == "__main__":
    quality = sys.argv[1] if len(sys.argv) > 1 else "fast"
    data = []
    for doc, pages in TRAIN.items():
        d, reach, total = examples(doc, quality, pages=pages)
        print(f"{doc}: {len(d)} decision examples; correct answer among candidates in {reach}/{total}")
        data += d
    # pick the regularization strength by 2-fold cross-validation inside the training pages
    folds = [d for d in (data[0::2], data[1::2])]
    best = None
    for l2 in (0.01, 0.03, 0.1, 0.3, 1.0):
        acc = []
        for k in range(2):
            wk = train(folds[1 - k], l2=l2, iters=400, verbose=False)
            acc.append(np.mean([int(np.argmax(X @ wk) == y) for X, y in folds[k]]))
        print(f"  l2={l2}: held-out choice accuracy {np.mean(acc):.4f}")
        if best is None or np.mean(acc) > best[1]:
            best = (l2, np.mean(acc))
    print("  using l2 =", best[0])
    w = train(data, l2=best[0])
    C.weights_file(quality).write_text(json.dumps({"quality": quality,
                                                   "weights": dict(zip(C.FEATURES, map(float, w)))}, indent=1))
    print("weights:", {f: round(float(x), 2) for f, x in zip(C.FEATURES, w)})
