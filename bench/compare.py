"""Per-page head-to-head from results/scores.json (text datasets only).

usage: python bench/compare.py [SYSTEM_A=dualocr_balanced] [SYSTEM_B=surya]

For every page: is A's word accuracy (span word acc for he_wikisource) better, worse or within 0.5 point of B's?
Also prints a paired bootstrap 95% interval for the mean per-page difference.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RESULTS  # noqa: E402


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else "dualocr_balanced"
    b = sys.argv[2] if len(sys.argv) > 2 else "surya"
    scores = json.loads((RESULTS / "scores.json").read_text(encoding="utf-8"))
    rng = random.Random(0)
    for ds, d in scores.items():
        s = d["systems"]
        key = "per_page_span_word_acc" if ds == "he_wikisource" else "per_page_word_acc"
        if a not in s or b not in s or key not in s[a]:
            continue
        pa, pb = s[a][key], s[b][key]
        diffs = [pa[k] - pb[k] for k in pa if k in pb]
        if not diffs:
            continue
        win = sum(x > 0.005 for x in diffs)
        loss = sum(x < -0.005 for x in diffs)
        boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(2000))
        print(f"{ds:<18} {a} vs {b}: better on {win}, worse on {loss}, tie {len(diffs) - win - loss} "
              f"of {len(diffs)} pages; mean per-page diff {100 * sum(diffs) / len(diffs):+.2f} pts "
              f"[{100 * boots[50]:+.2f}, {100 * boots[1949]:+.2f}]")


if __name__ == "__main__":
    main()
