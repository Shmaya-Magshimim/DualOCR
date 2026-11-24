# PyInstaller recipe for DualOCR (macOS .app, Windows and Linux folder).
# Built by scripts/setup.py, which first downloads models/ and vendor/llama.cpp.
#
# Two programs share one folder: DualOCR (the window, also the background
# worker) and dualocr-cli (the command line, with a console on Windows).
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
MAC = sys.platform == "darwin"
WIN = sys.platform == "win32"
VERSION = "0.1.0"

datas = [
    (str(ROOT / "models" / "tessdata"), "models/tessdata"),
    (str(ROOT / "models" / "lexicons"), "models/lexicons"),
]
datas += [(str(p), "models") for p in (ROOT / "models").glob("consensus_weights*.json")]
datas += [(str(p), "dualocr/ui") for p in (ROOT / "dualocr" / "ui").glob("*.png")]
llama = ROOT / "vendor" / "llama.cpp"
if llama.is_dir():
    datas.append((str(llama), "vendor/llama.cpp"))
datas += collect_data_files("surya")
hidden = (collect_submodules("surya") + collect_submodules("dualocr") + collect_submodules("spylls")
          + collect_submodules("keyring.backends") + ["pillow_heif", "bidi.algorithm"])

a = Analysis([str(ROOT / "dualocr" / "app_main.py")], pathex=[str(ROOT)], datas=datas, hiddenimports=hidden,
             excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore",
                       "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtPdf", "matplotlib",
                       "IPython", "pytest", "torch"],
             noarchive=False)
pyz = PYZ(a.pure)
icon = str(ROOT / "packaging" / ("DualOCR.icns" if MAC else "icon.ico")) if (MAC or WIN) else None
gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DualOCR", console=False, argv_emulation=False,
          icon=icon)
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name="dualocr-cli", console=True,
          icon=icon if WIN else None)
coll = COLLECT(gui, cli, a.binaries, a.datas, name="DualOCR")

if MAC:
    app = BUNDLE(coll, name="DualOCR.app", icon=icon, bundle_identifier="io.github.shmaya-magshimim.dualocr", version=VERSION,
                 info_plist={
                     "CFBundleName": "DualOCR", "CFBundleDisplayName": "DualOCR",
                     "CFBundleShortVersionString": VERSION, "LSMinimumSystemVersion": "12.0",
                     "NSHighResolutionCapable": True,
                     "LSApplicationCategoryType": "public.app-category.productivity",
                     "CFBundleDocumentTypes": [{
                         "CFBundleTypeName": "Scanned document", "CFBundleTypeRole": "Viewer",
                         "LSHandlerRank": "Alternate",
                         "LSItemContentTypes": ["com.adobe.pdf", "public.jpeg", "public.png", "public.heic",
                                                "public.tiff", "public.image"]}],
                 })
