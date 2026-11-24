import sys; sys.path.insert(0,'.'); sys.path.insert(0,'eval')
import run_eval as R
from dualocr import evaluate as E
from collections import Counter
pages = range(int(sys.argv[1]), int(sys.argv[2]) + 1)
errs = []
for n in pages:
    g = (R.GOLD/'heb23'/f'page_{n:02d}.txt').read_text()
    s = E.score(g, R.system_text(sys.argv[4] if len(sys.argv) > 4 else 'combined','heb23',n))
    errs += [(n,)+e for e in s.errors]
kinds = Counter()
for n, op, g, p in errs:
    kinds[op if op != 'sub' else ('punct/space' if E.letters(g) == E.letters(p) else 'letters')] += 1
print(kinds)
for e in errs[:int(sys.argv[3]) if len(sys.argv) > 3 else 60]: print(e)
