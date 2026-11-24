"""End-to-end tests with the real engines (slow: run with  pytest -m slow).

Uses the generated test pages in eval/synth (python eval/synth.py eval/synth 8):
mixed Hebrew/English text, random rotation, skew, blur and JPEG damage.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SYNTH = ROOT / "eval" / "synth"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    if not (SYNTH / "meta.json").exists():
        pytest.skip("run: python eval/synth.py eval/synth 8")
    from dualocr import pipeline
    from dualocr.config import Settings
    from dualocr.engines import surya
    out = tmp_path_factory.mktemp("out")
    files = sorted(SYNTH.glob("s*.jpg"))
    progress = []
    try:
        res = pipeline.process(files, Settings(quality="balanced"), out_dir=out, report=progress.append)
    finally:
        surya.shutdown()
    return res, files, progress


def test_outputs_exist(result):
    res, files, _ = result
    for k in ("docx", "txt", "report", "review"):
        assert Path(res["outputs"][k]).exists(), k
    assert res["pages"] == len(files)


def test_progress_reports(result):
    _, files, progress = result
    fr = [p.fraction for p in progress]
    assert fr == sorted(fr), "progress must never go backwards"
    assert fr[-1] == pytest.approx(1.0)
    assert {p.stage for p in progress} >= {"prepare", "read", "check", "write"}
    assert any(p.page == len(files) for p in progress)


def test_accuracy_on_generated_pages(result):
    from docx import Document
    from dualocr import evaluate as E
    res, files, _ = result
    meta = json.loads((SYNTH / "meta.json").read_text())
    txt = Path(res["outputs"]["txt"]).read_text(encoding="utf-8")
    pages = txt.split("===== ")[1:]
    scores = []
    for f, m, page in zip(files, meta, pages):
        gold = f.with_suffix(".txt").read_text(encoding="utf-8")
        body = page.split("=====", 1)[1] if "=====" in page else page
        s = E.score(gold, body)
        scores.append((m["kind"], m["rotation"], s.word_acc))
    for kind, rot, acc in scores:
        print(f"  {kind:6} rotated {rot:3}°: {acc:.1%}")
    assert min(a for _, _, a in scores) > 0.90, scores
    assert sum(a for *_, a in scores) / len(scores) > 0.96, scores
    Document(res["outputs"]["docx"])  # opens
