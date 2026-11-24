import sys; sys.path.insert(0,'.'); sys.path.insert(0,'eval')
import run_eval as R
from dualocr import evaluate as E
from rapidfuzz.distance import Levenshtein
doc, gname, system = sys.argv[1], sys.argv[2], sys.argv[3]
raw = open(f'eval/gold/eng14/{gname}').read()
g = E.tokens(raw)
order = R._gold_order(doc, raw)
p = E.tokens('\n'.join(R.system_text(system, doc, i) for i in order))
print('order', order, 'gold', len(g), 'pred', len(p))
ops = Levenshtein.editops(g, p)
for tag, posk, seq in (('delete', 'src_pos', g), ('insert', 'dest_pos', p)):
    runs, cur = [], []
    for op in ops:
        if op.tag != tag: continue
        v = getattr(op, posk)
        if cur and v == cur[-1] + 1: cur.append(v)
        else:
            if len(cur) > 4: runs.append(cur)
            cur = [v]
    if len(cur) > 4: runs.append(cur)
    for r in runs: print(len(r), 'MISSING' if tag == 'delete' else 'EXTRA', ' '.join(seq[r[0]:r[0] + 12]))
