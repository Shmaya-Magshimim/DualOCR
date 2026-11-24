"""FUNSD test set: 50 real scanned business forms (noisy, English).

usage: python bench/prepare_funsd.py

https://guillaumejaume.github.io/FUNSD/ (Jaume et al., ICDAR-OST 2019).
License: free for non-commercial, research and educational use.
Ground truth is word-level; forms have no single reading order, so score.py
uses order-free (bag-of-words) metrics for this set.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, download, write_manifest  # noqa: E402

URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
NAME = "funsd"


def main():
    root = DATA / NAME
    z = download(URL, root / "dataset.zip")
    (root / "pages").mkdir(parents=True, exist_ok=True)
    (root / "gt").mkdir(parents=True, exist_ok=True)
    ids = []
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        anns = sorted(n for n in names if "testing_data/annotations/" in n and n.endswith(".json") and "__MACOSX" not in n)
        for a in anns:
            stem = Path(a).stem
            img = next(n for n in names if f"testing_data/images/{stem}." in n and "__MACOSX" not in n)
            doc = json.loads(zf.read(a))
            words = []
            for ent in sorted(doc["form"], key=lambda e: (e["box"][1], e["box"][0])):
                ws = sorted(ent.get("words", []), key=lambda w: (w["box"][1] // 10, w["box"][0]))
                words += [w["text"] for w in ws if w["text"].strip()]
            pid = "f_" + stem
            (root / "pages" / f"{pid}.png").write_bytes(zf.read(img))
            (root / "gt" / f"{pid}.json").write_text(json.dumps({"words": words}, ensure_ascii=False), encoding="utf-8")
            ids.append(pid)
    write_manifest(NAME, {
        "name": NAME, "ids": ids, "source": URL,
        "license": "FUNSD: non-commercial, research and educational use",
        "gt_format": "json: words (all annotated words; reading order not defined)",
    })
    print(f"{NAME}: {len(ids)} pages")


if __name__ == "__main__":
    main()
