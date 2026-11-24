"""Command line:  python -m dualocr FILE [FILE ...] [-o OUTDIR] [--quality fast|balanced|max]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def check() -> int:
    """Print what DualOCR found; exit code 1 if something it needs is missing."""
    import os
    import shutil
    import subprocess

    from . import config

    ok = True

    def line(good, what, detail):
        nonlocal ok
        ok = ok and good
        print(f"  [{'ok' if good else 'MISSING'}] {what}: {detail}")

    print(f"DualOCR {__version__}")
    tess = shutil.which("tesseract")
    ver = ""
    if tess:
        r = subprocess.run([tess, "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=30)
        ver = (r.stdout or r.stderr).splitlines()[0] if (r.stdout or r.stderr) else ""
    line(bool(tess), "Tesseract", f"{tess} ({ver})" if tess else config.install_hint())
    for name in ("heb", "eng", "osd"):
        f = config.TESSDATA / f"{name}.traineddata"
        line(f.exists(), f"Tesseract model {name}", f)
    for name in ("he_IL.dic", "en_US.dic", "he_full.txt", "en_50k.txt"):
        f = config.LEXICONS / name
        line(f.exists(), f"dictionary {name}", f)
    llama = os.environ.get("LLAMA_CPP_BINARY", "")
    line(bool(llama) and Path(llama).exists(), "llama.cpp server", llama or config.install_hint())
    try:
        from huggingface_hub import hf_hub_download
        from surya.settings import settings as ss
        paths = [hf_hub_download(ss.SURYA_GGUF_REPO, f, local_files_only=True)
                 for f in (ss.SURYA_GGUF_MODEL_FILE, ss.SURYA_GGUF_MMPROJ_FILE)]
        line(True, "Surya model", Path(paths[0]).parent)
    except Exception:
        print("  [later] Surya model: not downloaded yet (about 1.4 GB, downloaded on first use)")
    print(f"  data folder: {config.DATA}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dualocr", description="Hebrew + English OCR to DOCX (Tesseract + Surya).")
    ap.add_argument("files", nargs="*", help="PDF or image files (combined into one document, in order)")
    ap.add_argument("-o", "--out", default="", help="output folder (default: next to the first file)")
    ap.add_argument("-n", "--name", default="", help="name (or full path) of the Word file to write")
    ap.add_argument("-q", "--quality", choices=["fast", "balanced", "max"], default=None)
    ap.add_argument("--no-review", action="store_true", help="don't highlight uncertain words")
    ap.add_argument("--no-pictures", action="store_true", help="don't copy pictures into the DOCX")
    ap.add_argument("--ai", choices=["claude", "gemini"], help="use the AI tie-breaker (key from Settings/Keychain)")
    ap.add_argument("--background", action="store_true", help="queue the job for the background worker and exit")
    ap.add_argument("--check", action="store_true", help="check that Tesseract, llama.cpp and the models are installed")
    ap.add_argument("--version", action="version", version=f"DualOCR {__version__}")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):      # Hebrew file names on a Windows console
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if a.check:
        return check()
    if not a.files:
        ap.error("give at least one PDF or image file")

    from .config import Settings
    s = Settings.load()
    if a.quality:
        s.quality = a.quality
    if a.no_review:
        s.review_highlights = False
    if a.no_pictures:
        s.include_pictures = False
    if a.ai:
        s.llm_enabled, s.llm_provider = True, a.ai
    for f in a.files:
        if not Path(f).exists():
            print(f"Not found: {f}", file=sys.stderr)
            return 2
    if a.background:
        from . import worker
        jid = worker.submit(a.files, s, a.out, str(Path(a.out or ".") / a.name) if a.name else "")
        print(f"Queued job {jid}. The app (or a notification) will tell you when it's done.")
        return 0

    from . import pipeline
    from .engines import surya

    def report(p):
        bar = "#" * int(p.fraction * 30)
        sys.stdout.write(f"\r[{bar:<30}] {p.fraction:5.1%}  {p.detail[:70]:<70}")
        sys.stdout.flush()

    try:
        res = pipeline.process(a.files, s, out_dir=a.out or None, report=report,
                               out_path=str(Path(a.out or Path(a.files[0]).parent) / a.name) if a.name else None)
    finally:
        surya.shutdown()
    print()
    for k, v in res["outputs"].items():
        print(f"{k:>7}: {v}")
    st = res["stats"]
    if st:
        print(f"{res['pages']} pages, {st.get('words', 0)} words: {st.get('uncertain', 0)} flagged for review, "
              f"{st.get('source_typo', 0)} possible typos in the original, {res['seconds']:.0f}s")
    for line in res["numbering"]:
        print("numbering:", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
