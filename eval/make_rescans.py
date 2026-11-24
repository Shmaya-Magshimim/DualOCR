"""Run the zoom-in rescans on cached documents and store their readings."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dualocr.config import Settings
from dualocr.pipeline import decide_cached
from dualocr.engines import surya as SU
q = sys.argv[1]
try:
    for doc in sys.argv[2:]:
        t = time.time()
        rc = ROOT / "eval" / "cache" / doc / f"rescan_{q}.json"
        if rc.exists(): rc.unlink()
        works = decide_cached(ROOT / "eval" / "cache" / doc, Settings(quality=q))
        n = sum(1 for pw in works for ab in pw.aligned for sl in ab.slots if sl.readings.get("rescan"))
        print(f"[{doc}] {q}: rescanned {n} words in {time.time() - t:.0f}s", flush=True)
finally:
    SU.shutdown()
