"""Surya's own recognition benchmark (vikp/rec_bench) as page images.

usage: python bench/prepare_recbench.py [--en 40] [--seed 1]

https://huggingface.co/datasets/vikp/rec_bench is the set used by Surya's
benchmark/recognition.py (Surya <= 0.17; no license is stated on the dataset
card). Each row is a rendered document page with the bounding box and text of
every line, and a language code. Surya's script crops the given line boxes and
scores line recognition; DualOCR is page-level, so here every system reads the
whole page and is scored against all of its lines (see score.py).
We take every Hebrew row and a seeded sample of English rows (Surya's script
also samples with shuffle(seed=1)).
Needs pyarrow and pillow.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, download, write_manifest  # noqa: E402

URL = "https://huggingface.co/datasets/vikp/rec_bench/resolve/main/data/train-00000-of-00001.parquet"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", type=int, default=40)
    ap.add_argument("--he", type=int, default=0, help="0 = all Hebrew rows")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    import pyarrow.parquet as pq
    from PIL import Image

    src = download(URL, DATA / "rec_bench" / "train.parquet")
    t = pq.read_table(src)
    langs = t.column("language").to_pylist()
    rng = random.Random(a.seed)
    for lang, n in (("he", a.he), ("en", a.en)):
        rows = [i for i, l in enumerate(langs) if l == lang]
        if n and n < len(rows):
            rows = sorted(rng.sample(rows, n))
        name = f"recbench_{lang}"
        root = DATA / name
        (root / "pages").mkdir(parents=True, exist_ok=True)
        (root / "gt").mkdir(parents=True, exist_ok=True)
        ids = []
        for i in rows:
            r = t.slice(i, 1).to_pylist()[0]
            img = r["image"]
            im = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            pid = f"rb{i:05d}"
            im.save(root / "pages" / f"{pid}.png")
            (root / "gt" / f"{pid}.json").write_text(json.dumps(
                {"lines": r["text"], "bboxes": r["bboxes"], "language": r["language"], "size": im.size},
                ensure_ascii=False), encoding="utf-8")
            ids.append(pid)
        write_manifest(name, {
            "name": name, "ids": ids, "seed": a.seed, "rows": rows,
            "source": "https://huggingface.co/datasets/vikp/rec_bench (Surya recognition benchmark)",
            "license": "not stated on the dataset card",
            "gt_format": "json: lines (text per line), bboxes, language",
        })
        print(f"{name}: {len(ids)} pages")


if __name__ == "__main__":
    main()
