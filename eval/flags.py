"""How useful are the highlights? Of the words that are still wrong, how many
are highlighted for review (recall), and how many highlights are on correct
words (wasted review time)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
from dualocr import evaluate as E, rules as R, consensus as C
from dualocr.config import Settings
from dualocr.pipeline import decide_cached
from train import slot_sequence, label_slots

quality = sys.argv[1] if len(sys.argv) > 1 else "balanced"
works = decide_cached(ROOT / "eval/cache/heb23", Settings(quality=quality))
rows = []
for pw in works:
    n = pw.index + 1
    gold = E.tokens((ROOT / "eval/gold/heb23" / f"page_{n:02d}.txt").read_text())
    sl = slot_sequence(pw)
    for s, lab in zip(sl, label_slots(sl, gold)):
        if lab is None or not E.letters(s.text or lab or ""):
            continue
        rows.append((n, s, lab, R.norm(s.text or "") == lab))
for name, pages in (("train p1-10", range(1, 11)), ("held-out p11-20", range(11, 21))):
    rs = [r for r in rows if r[0] in pages]
    wrong = [r for r in rs if not r[3]]
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        flagged = [r for r in rs if r[1].conf < thr]
        caught = [r for r in flagged if not r[3]]
        prec = len(caught) / max(1, len(flagged))
        print(f"{name}: below {thr:.1f}: highlights {len(flagged):3} words ({len(flagged)/len(rs):.1%}), "
              f"{prec:.0%} of them really wrong, catches {len(caught)}/{len(wrong)} errors")
    typos = [r for r in rs if r[1].status == "source_typo"]
    print(f"{name}: 'possible typo' marks: {len(typos)}, of which really typos kept as printed: "
          f"{sum(1 for r in typos if r[3])}")
