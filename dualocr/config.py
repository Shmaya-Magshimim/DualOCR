"""Paths and user settings for DualOCR.

Read-only resources (Tesseract models, dictionaries, the llama.cpp server)
live in the project folder, or inside the built app. Jobs, settings and review
data go to the usual per-user folder of each system (see osutil.user_data_dir),
or to $DUALOCR_DATA if that is set. The optional API key goes to the system's
password store (macOS Keychain, Windows Credential Manager, Secret Service on
Linux), never to a file.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import osutil

APP_NAME = "DualOCR"

FROZEN = getattr(sys, "frozen", False)          # running from the built app
if FROZEN:
    # bundled read-only resources live inside the app
    ROOT = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
else:
    ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
TESSDATA = MODELS / "tessdata"
LEXICONS = MODELS / "lexicons"
LLAMA_DIR = ROOT / "vendor" / "llama.cpp"       # downloaded by scripts/setup.py
DATA = Path(os.environ.get("DUALOCR_DATA") or osutil.user_data_dir(APP_NAME))
JOBS = DATA / "jobs"
SETTINGS_FILE = DATA / "settings.json"
KEYCHAIN_SERVICE = "DualOCR"

for _p in (DATA, JOBS):
    _p.mkdir(parents=True, exist_ok=True)

# Apps opened from the desktop get a minimal PATH: add the usual install folders.
_extra = [str(p) for p in osutil.tool_dirs()]
os.environ["PATH"] = os.pathsep.join(_extra + [p for p in os.environ.get("PATH", "").split(os.pathsep)
                                               if p and p not in _extra])
# Surya runs its model in llama.cpp's server: prefer the copy setup.py
# downloaded (a tested version), then whatever is installed.
_bundled_llama = LLAMA_DIR / ("llama-server" + osutil.EXE)
if _bundled_llama.exists():
    os.environ.setdefault("LLAMA_CPP_BINARY", str(_bundled_llama))
elif shutil.which("llama-server"):
    os.environ.setdefault("LLAMA_CPP_BINARY", shutil.which("llama-server"))

# Windows: no console window for every Tesseract / llama.cpp run
osutil.hide_console_windows()


def missing_tools() -> list:
    """External programs DualOCR needs that aren't installed."""
    out = []
    if not shutil.which("tesseract"):
        out.append("Tesseract")
    if not (os.environ.get("LLAMA_CPP_BINARY") and Path(os.environ["LLAMA_CPP_BINARY"]).exists()):
        out.append("llama.cpp")
    return out


def install_hint() -> str:
    """One line telling the user how to get the missing tools."""
    if osutil.WINDOWS:
        return "Run setup.cmd in the DualOCR folder again (it installs Tesseract with winget)."
    if osutil.MACOS:
        return "Run ./setup.sh in the DualOCR folder again (or: brew install tesseract)."
    return "Run ./setup.sh in the DualOCR folder again (or install the 'tesseract-ocr' package)."


# Tesseract must use our "best" models (a system-wide TESSDATA_PREFIX may point
# at a folder without Hebrew), and must not spawn its own thread pool per
# process (we run many processes in parallel).
if (TESSDATA / "heb.traineddata").exists():
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA)
else:
    os.environ.setdefault("TESSDATA_PREFIX", str(TESSDATA))
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
# Surya: never wait forever on one page (handwriting can make the model loop)
os.environ.setdefault("SURYA_INFERENCE_TIMEOUT_SECONDS", "300")

QUALITY_PRESETS = {
    # rescan_rounds: how many zoom-in rounds for doubtful words
    "fast": {"rescan_rounds": 0, "variants": 0, "surya_variants": 0},
    # Surya's zoomed re-reads cost ~1 s per word and measured ~0.0-0.2% on the
    # held-out pages, so only "max" uses them.
    "balanced": {"rescan_rounds": 1, "variants": 4, "surya_variants": 0},
    "max": {"rescan_rounds": 2, "variants": 7, "surya_variants": 1},
}


@dataclass
class Settings:
    quality: str = "balanced"          # fast | balanced | max
    languages: str = "heb+eng"         # tesseract language string
    include_pictures: bool = True
    review_highlights: bool = True     # highlight uncertain words in the DOCX
    highlight_level: str = "rare"      # rare (~0.5% of words) | some (~4%) | many (~9%)
    write_txt: bool = True
    write_report: bool = True
    output_dir: str = ""               # "" = next to the input file
    open_when_done: bool = False
    # Optional cloud tie-breaker. Off by default; the key itself is in the system password store.
    llm_enabled: bool = False
    llm_provider: str = "claude"       # claude | gemini
    llm_model: str = ""                # "" = provider default
    llm_max_words: int = 60            # cap per document (cost control)
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Settings":
        if SETTINGS_FILE.exists():
            try:
                raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
                # older names: "normal" highlighted ~6% of words; the default is now rare
                known["highlight_level"] = {"normal": "rare", "thorough": "many"}.get(
                    known.get("highlight_level", "rare"), known.get("highlight_level", "rare"))
                return cls(**known)
            except Exception:
                pass
        return cls()

    def save(self) -> None:
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    def preset(self) -> dict:
        return QUALITY_PRESETS.get(self.quality, QUALITY_PRESETS["balanced"])


# confidence below which a word is highlighted (measured on the held-out answer-key pages)
HIGHLIGHT_THRESHOLDS = {"rare": 0.50, "some": 0.70, "many": 0.90}
