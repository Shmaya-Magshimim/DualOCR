"""Shared paths and helpers for the DualOCR benchmark harness.

Layout (all relative to BENCH_ROOT):

  data/<dataset>/pages/<id>.png     page images fed to every system
  data/<dataset>/gt/<id>.json       ground truth (format depends on the dataset)
  data/<dataset>/manifest.json      ids, source URL, license, sampling seed
  work/cache/<dataset>/<id>/        one engine pass per page (reused by every system)
  results/<dataset>/pred/<system>/<id>.txt
  results/<dataset>/timing.jsonl
  results/summary.md, results/scores.json

Set BENCH_ROOT to put data/work/results elsewhere (default: this folder).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_ROOT = Path(os.environ.get("BENCH_ROOT", HERE)).resolve()
DATA = BENCH_ROOT / "data"
WORK = BENCH_ROOT / "work"
RESULTS = BENCH_ROOT / "results"

SYSTEMS = ["tesseract_raw", "tesseract", "surya", "dualocr_fast", "dualocr_balanced"]
SYSTEM_NAMES = {
    "tesseract_raw": "Tesseract (raw image, psm 3)",
    "tesseract": "Tesseract (DualOCR-prepped page, psm 3)",
    "surya": "Surya 2.5 (llama.cpp)",
    "dualocr_fast": "DualOCR fast",
    "dualocr_balanced": "DualOCR balanced",
}

PAGENUM = re.compile(r"^\s*[-–]?\s*\d{1,3}\s*[-–]?\s*$")
HEB_MARKS = re.compile(r"[֑-ׇ]")          # niqqud + cantillation


def manifest(dataset: str) -> dict:
    return json.loads((DATA / dataset / "manifest.json").read_text(encoding="utf-8"))


def write_manifest(dataset: str, m: dict) -> None:
    d = DATA / dataset
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


def pred_path(dataset: str, system: str, pid: str) -> Path:
    return RESULTS / dataset / "pred" / system / f"{pid}.txt"


def strip_marks(t: str) -> str:
    """NFKC (typographic forms like the wide/alternative ayin U+FB20 and
    presentation-form letters become plain letters), then drop niqqud and
    cantillation marks."""
    import unicodedata
    return HEB_MARKS.sub("", unicodedata.normalize("NFKC", t))


def user_agent() -> dict:
    return {"User-Agent": "DualOCR-bench/0.1 (OCR benchmark script; python-urllib)"}


def download(url: str, dest: Path, timeout: int = 300) -> Path:
    """Download url to dest (skips if present). Uses a temp file so an
    interrupted download is not mistaken for a finished one."""
    import shutil
    import urllib.request

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=user_agent())
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, dest)
    return dest
