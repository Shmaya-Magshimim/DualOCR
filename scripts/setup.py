"""Set up DualOCR on macOS, Windows or Linux, and build the app.

Run through setup.sh (macOS/Linux) or setup.cmd (Windows), which first install
uv and the Python packages. Every step is skipped when it's already done, so
running it again is quick (it just rebuilds the app).

Steps:
  1. Tesseract (Homebrew on macOS, winget on Windows, the distro's package
     manager on Linux), plus the libraries the window needs on Linux
  2. Tesseract's "best" Hebrew/English models and the spelling dictionaries
  3. llama.cpp's server (a tested release, Metal / Vulkan / CPU build)
  4. Surya's model (about 1.4 GB, into the Hugging Face cache)
  5. Build the app with PyInstaller and install it with a shortcut
"""
from __future__ import annotations

import argparse
import ctypes.util
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
VENDOR = ROOT / "vendor"
LLAMA_DIR = VENDOR / "llama.cpp"
BUILD = ROOT / "build.noindex"       # ".noindex": Spotlight won't list the build as a second app

WINDOWS = os.name == "nt"
MACOS = sys.platform == "darwin"
LINUX = not WINDOWS and not MACOS
EXE = ".exe" if WINDOWS else ""
ARCH = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"

# ---------------------------------------------------------------- pinned downloads
# Each file is checked against its SHA-256, so a changed or broken download is caught.
_TESS = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/e12c65a915945e4c28e237a9b52bc4a8f39a0cec"
_DICT = "https://raw.githubusercontent.com/LibreOffice/dictionaries/c0865a85eb67dc377f2cf8e8fe2c4fa9e53f0dcb"
_FREQ = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/525f9b560de45753a5ea01069454e72e9aa541c6"
DATA_FILES = [
    # (path under models/, url, sha256)
    ("tessdata/heb.traineddata", f"{_TESS}/heb.traineddata",
     "dbaa827aea6bc21215638447f17783a1004987c2d0bf5573d111fee397abdae5"),
    ("tessdata/eng.traineddata", f"{_TESS}/eng.traineddata",
     "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba"),
    ("tessdata/osd.traineddata", f"{_TESS}/osd.traineddata",
     "9cf5d576fcc47564f11265841e5ca839001e7e6f38ff7f7aacf46d15a96b00ff"),
    ("lexicons/he_IL.dic", f"{_DICT}/he_IL/he_IL.dic",
     "5f5331f90ed775bd527f6fb7ad1ead9a1b7d8ce46ad640c2387d9d1dc91d3058"),
    ("lexicons/he_IL.aff", f"{_DICT}/he_IL/he_IL.aff",
     "6caf86b3a545be5614f135d33a48baa244a59aca43f051dda6173d5d9cbc7700"),
    ("lexicons/en_US.dic", f"{_DICT}/en/en_US.dic",
     "f0b1a234bd178bdd01875b2a392a9647f888b8fe879f79c52aae62c2759b3647"),
    ("lexicons/en_US.aff", f"{_DICT}/en/en_US.aff",
     "e746c882dd6f303c2c46e7452804b9201115a6942cfeb15f18f8edf774d2e24e"),
    ("lexicons/he_full.txt", f"{_FREQ}/content/2018/he/he_full.txt",
     "a69a3f390eb53183bf191c4eac18282592992aef9ff184c5dcf8919f5ea0f773"),
    ("lexicons/en_50k.txt", f"{_FREQ}/content/2018/en/en_50k.txt",
     "5351ff405b1126ef555791dd4d9798a48e3e9a501a9fc481a9da957752cfb458"),
]

LLAMA_BUILD = "b12359"      # the llama.cpp release DualOCR is tested with
_LLAMA_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/"
LLAMA_ASSETS = {
    # (system, arch, flavor): (file, sha256)
    ("mac", "arm64", "metal"): ("llama-{b}-bin-macos-arm64.tar.gz",
                                "82b00abc740160e1b9980e6cc79c117345fa34af058c80e6c0a69aa8ddccde75"),
    ("mac", "x64", "metal"): ("llama-{b}-bin-macos-x64.tar.gz",
                              "25320d529077ef5a3d16e7b5da26438f221d88c8c0930a32b0c79457f5eede56"),
    ("linux", "x64", "vulkan"): ("llama-{b}-bin-ubuntu-vulkan-x64.tar.gz",
                                 "19b60f00395a3f4954e46a5037695bc1d843404fc9d3124f82a612ecd8fbe9f0"),
    ("linux", "x64", "cpu"): ("llama-{b}-bin-ubuntu-x64.tar.gz",
                              "0bc1f091e8916102b6d8f2cf944233a620e90a0f856c72d0ce41150c65c2e72e"),
    ("linux", "arm64", "vulkan"): ("llama-{b}-bin-ubuntu-vulkan-arm64.tar.gz",
                                   "3f8ae812a58ac92e5a596ca9fbc988f52854d147c5a7fe1264dddc90c1d085b4"),
    ("linux", "arm64", "cpu"): ("llama-{b}-bin-ubuntu-arm64.tar.gz",
                                "dc7aa17a67884619b7223870aca9fd6addbc01717713b04a714add1c31137359"),
    ("windows", "x64", "vulkan"): ("llama-{b}-bin-win-vulkan-x64.zip",
                                   "e8d18c6e4210cca87e7307c2ea78363ac8d158e4e1397358ec96ab70b53a0d5e"),
    ("windows", "x64", "cpu"): ("llama-{b}-bin-win-cpu-x64.zip",
                                "837c1655e87df3b66989f4be33a792a1399c785bf259038dc75c900096683e69"),
    ("windows", "arm64", "vulkan"): ("llama-{b}-bin-win-vulkan-arm64.zip",
                                     "62cd50089cffdd825524a91c5a8819041eb75d8e2beacd6eca68194dba50574f"),
    ("windows", "arm64", "cpu"): ("llama-{b}-bin-win-cpu-arm64.zip",
                                  "d27564c754478d2832789dfcebfc5ace018e36a224c41d810b7cbc1ba8243ec0"),
}

# ---------------------------------------------------------------- output helpers

_tty = sys.stdout.isatty()


def step(msg: str) -> None:
    print(("\n\033[1m==> " + msg + "\033[0m") if _tty else f"\n==> {msg}", flush=True)


def info(msg: str) -> None:
    print(f"    {msg}", flush=True)


def fail(msg: str) -> None:
    print(("\n\033[31mSetup stopped:\033[0m " if _tty else "\nSetup stopped: ") + msg, file=sys.stderr)
    sys.exit(1)


def ask(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        return False
    return input(f"    {question} [Y/n] ").strip().lower() in ("", "y", "yes")


def run(cmd, check=True, **kw):
    info("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], **kw)
    if check and r.returncode != 0:
        fail(f"this command failed (exit code {r.returncode}): {' '.join(str(c) for c in cmd)}")
    return r


# ---------------------------------------------------------------- downloads

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, digest: str) -> None:
    """Download url to dest unless it's already there with the right hash."""
    if dest.exists() and sha256(dest) == digest:
        info(f"have {dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DualOCR-setup"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done, last = 0, 0.0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if _tty and total and time.time() - last > 0.2:
                        last = time.time()
                        print(f"\r    {dest.name}: {done * 100 // total}% of {total / 1e6:.0f} MB", end="", flush=True)
            if _tty and total:
                print("\r" + " " * 70 + "\r", end="")
            got = sha256(tmp)
            if got != digest:
                raise OSError(f"checksum mismatch for {url}: got {got}")
            os.replace(tmp, dest)
            info(f"got  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            return
        except Exception as e:  # network hiccup: try again
            tmp.unlink(missing_ok=True)
            if attempt == 3:
                fail(f"couldn't download {url}\n    {e}")
            info(f"retrying ({e})")
            time.sleep(3 * attempt)


# ---------------------------------------------------------------- 1. system packages

def _sudo() -> list:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return []
    if shutil.which("sudo"):
        return ["sudo"]
    fail("installing system packages needs root, and 'sudo' isn't available. Run this as root.")
    return []


def _linux_lib(name: str) -> bool:
    return ctypes.util.find_library(name) is not None


def find_tesseract() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found
    if WINDOWS:
        for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                     os.environ.get("LOCALAPPDATA")):
            for sub in ("Tesseract-OCR", "Programs/Tesseract-OCR"):
                p = Path(base or "") / sub / "tesseract.exe"
                if base and p.exists():
                    return str(p)
    for p in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/opt/local/bin/tesseract"):
        if Path(p).exists():
            return p
    return None


def system_packages(a) -> None:
    step("1/5  Tesseract and system libraries")
    if MACOS:
        if find_tesseract():
            info(f"Tesseract: {find_tesseract()}")
            return
        brew = shutil.which("brew") or next((p for p in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew")
                                             if Path(p).exists()), None)
        if not brew and shutil.which("port"):
            run(["sudo", "port", "install", "tesseract"])
            return
        if not brew:
            info("Tesseract comes from Homebrew (https://brew.sh), which isn't installed.")
            if not ask("Install Homebrew now? (it asks for your password)", a.yes):
                fail("install Homebrew from https://brew.sh, then run ./setup.sh again.")
            run(["/bin/bash", "-c",
                 "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"])
            brew = next((p for p in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew") if Path(p).exists()), None)
            if not brew:
                fail("Homebrew didn't install. Install it from https://brew.sh and run ./setup.sh again.")
        run([brew, "install", "tesseract"])
        return

    if WINDOWS:
        if find_tesseract():
            info(f"Tesseract: {find_tesseract()}")
            return
        if not shutil.which("winget"):
            fail("Tesseract isn't installed and winget isn't available. Install Tesseract from\n"
                 "    https://github.com/UB-Mannheim/tesseract/wiki  then run setup.cmd again.")
        info("Installing Tesseract with winget (Windows may ask for permission)")
        run(["winget", "install", "--id", "UB-Mannheim.TesseractOCR", "-e", "--silent",
             "--accept-source-agreements", "--accept-package-agreements"], check=False)
        if not find_tesseract():
            fail("Tesseract didn't install. Install it from https://github.com/UB-Mannheim/tesseract/wiki\n"
                 "    then run setup.cmd again.")
        info(f"Tesseract: {find_tesseract()}")
        return

    # Linux: Tesseract, the libraries Qt's window needs, binutils for PyInstaller
    need_tess = not find_tesseract()
    need_qt = not a.no_app and not (_linux_lib("xcb-cursor") and _linux_lib("xkbcommon-x11") and _linux_lib("EGL"))
    need_bin = not a.no_app and not shutil.which("objdump")
    if not (need_tess or need_qt or need_bin):
        info(f"Tesseract: {find_tesseract()}; window libraries present")
        return
    managers = {
        "apt-get": (["apt-get", "install", "-y"], ["tesseract-ocr", "libxcb-cursor0", "libxkbcommon-x11-0", "libegl1",
                                                   "libgl1", "libfontconfig1", "libdbus-1-3", "binutils"]),
        "dnf": (["dnf", "install", "-y"], ["tesseract", "xcb-util-cursor", "libxkbcommon-x11", "mesa-libEGL",
                                           "mesa-libGL", "fontconfig", "dbus-libs", "binutils"]),
        "pacman": (["pacman", "-S", "--needed", "--noconfirm"], ["tesseract", "tesseract-data-eng", "xcb-util-cursor",
                                                                 "libxkbcommon-x11", "libglvnd", "fontconfig", "dbus",
                                                                 "binutils"]),
        "zypper": (["zypper", "--non-interactive", "install"], ["tesseract-ocr", "libxcb-cursor0",
                                                                "libxkbcommon-x11-0", "Mesa-libEGL1", "Mesa-libGL1",
                                                                "fontconfig", "libdbus-1-3", "binutils"]),
    }
    pm = next((m for m in managers if shutil.which(m)), None)
    if not pm:
        fail("couldn't find apt-get, dnf, pacman or zypper. Install Tesseract (version 4 or newer) and the\n"
             "    Qt runtime libraries (xcb-util-cursor, libxkbcommon-x11) yourself, then run ./setup.sh again.")
    cmd, pkgs = managers[pm]
    info(f"Installing with {pm}: {' '.join(pkgs)}")
    if not ask("OK to install these system packages? (needs your password)", a.yes):
        fail("these packages are needed. Install them and run ./setup.sh again.")
    if pm == "apt-get":
        run(_sudo() + ["apt-get", "update"])
    run(_sudo() + cmd + pkgs)
    if not find_tesseract():
        fail("Tesseract still isn't installed.")


# ---------------------------------------------------------------- 2. models

def data_files() -> None:
    step("2/5  Tesseract models and dictionaries")
    for rel, url, digest in DATA_FILES:
        download(url, MODELS / rel, digest)


# ---------------------------------------------------------------- 3. llama.cpp

def _has_vulkan() -> bool:
    if WINDOWS:
        return (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vulkan-1.dll").exists()
    return _linux_lib("vulkan")


def llama_cpp(a) -> None:
    step(f"3/5  llama.cpp server (runs Surya's model; release {LLAMA_BUILD})")
    if a.llama == "system":
        found = shutil.which("llama-server")
        if not found:
            fail("--llama system: no llama-server on the PATH. Install llama.cpp first.")
        shutil.rmtree(LLAMA_DIR, ignore_errors=True)
        info(f"using {found}")
        return
    system = "mac" if MACOS else "windows" if WINDOWS else "linux"
    flavor = "metal" if MACOS else a.llama if a.llama in ("vulkan", "cpu") else ("vulkan" if _has_vulkan() else "cpu")
    key = (system, ARCH, flavor)
    if key not in LLAMA_ASSETS:
        fail(f"no llama.cpp download for {system} {ARCH}. Install llama.cpp yourself and run setup with --llama system.")
    name, digest = LLAMA_ASSETS[key]
    name = name.format(b=LLAMA_BUILD)
    stamp = LLAMA_DIR / ".release"
    server = LLAMA_DIR / f"llama-server{EXE}"
    if server.exists() and stamp.exists() and stamp.read_text().strip() == name:
        info(f"have {name}")
    else:
        archive = VENDOR / "downloads" / name
        download(_LLAMA_URL + name, archive, digest)
        with tempfile.TemporaryDirectory(dir=VENDOR) as tmp:
            if name.endswith(".zip"):
                with zipfile.ZipFile(archive) as z:
                    z.extractall(tmp)
            else:
                with tarfile.open(archive) as t:
                    t.extractall(tmp, filter="data")
            found = next(Path(tmp).rglob(f"llama-server{EXE}"), None)
            if not found:
                fail(f"llama-server isn't in {name}")
            shutil.rmtree(LLAMA_DIR, ignore_errors=True)
            LLAMA_DIR.mkdir(parents=True)
            # keep the server, its libraries and the license; drop the other tools
            for f in found.parent.iterdir():
                n = f.name.lower()
                if f.is_dir() or not (n.startswith(("llama-server", "license")) or n.endswith((".dll", ".dylib"))
                                      or ".so" in n):
                    continue
                dest = LLAMA_DIR / f.name
                if f.is_symlink():
                    os.symlink(os.readlink(f), dest)
                else:
                    shutil.copy2(f, dest)
        stamp.write_text(name)
        archive.unlink(missing_ok=True)
        info(f"installed {name} ({flavor})")
    r = subprocess.run([str(server), "--version"], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=120)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        fail(f"the downloaded llama-server doesn't run on this system:\n{out[-800:]}\n"
             "    Install llama.cpp yourself (https://github.com/ggml-org/llama.cpp) and run setup with --llama system.")
    info(next((ln for ln in out.splitlines() if ln.startswith("version")), out.splitlines()[0] if out else ""))


# ---------------------------------------------------------------- 4. Surya model

def surya_model(a) -> None:
    step("4/5  Surya's model (about 1.4 GB, downloaded once)")
    if a.skip_surya_model:
        info("skipped (--skip-surya-model): it downloads the first time you convert something")
        return
    from huggingface_hub import hf_hub_download
    from surya.settings import settings

    for f in (settings.SURYA_GGUF_MODEL_FILE, settings.SURYA_GGUF_MMPROJ_FILE):
        try:
            p = hf_hub_download(settings.SURYA_GGUF_REPO, f, local_files_only=True)
            info(f"have {f}")
        except Exception:
            p = hf_hub_download(settings.SURYA_GGUF_REPO, f)
            info(f"got  {f}")
    info(f"in {Path(p).parent}")


# ---------------------------------------------------------------- 5. build + install

def build_app() -> Path:
    import PyInstaller.__main__

    step("5/5  Building the app (a few minutes)")
    PyInstaller.__main__.run(["--noconfirm", "--clean", "--log-level", "WARN",
                              "--distpath", str(BUILD / "dist"), "--workpath", str(BUILD / "work"),
                              str(ROOT / "packaging" / "DualOCR.spec")])
    dist = BUILD / "dist" / ("DualOCR.app" if MACOS else "DualOCR")
    if not dist.exists():
        fail("PyInstaller didn't produce the app.")
    return dist


def cli_in(app: Path) -> Path:
    return app / "Contents" / "MacOS" / "dualocr-cli" if MACOS else app / f"dualocr-cli{EXE}"


def check_build(app: Path) -> None:
    info("Checking the built app")
    r = subprocess.run([str(cli_in(app)), "--check"], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300)
    print("\n".join("    " + ln for ln in (r.stdout + r.stderr).strip().splitlines()))
    if r.returncode != 0:
        fail("the built app is missing something (see above).")


def install_app(dist: Path) -> str:
    if MACOS:
        target = Path.home() / "Applications" / "DualOCR.app"
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            old = BUILD / "DualOCR.app.previous"
            shutil.rmtree(old, ignore_errors=True)
            shutil.move(str(target), old)
        run(["ditto", dist, target])
        lsreg = ("/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/"
                 "Support/lsregister")
        if Path(lsreg).exists():
            subprocess.run([lsreg, "-f", str(target)], capture_output=True)
        return f"{target}  (open DualOCR from Launchpad or Spotlight)"

    if WINDOWS:
        target = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "DualOCR"
        if target.exists():
            try:
                shutil.rmtree(target)
            except OSError:
                fail(f"couldn't replace {target}. Close DualOCR (and wait for running conversions), then run "
                     "setup.cmd again.")
        shutil.copytree(dist, target)
        lnk = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "DualOCR.lnk"
        ps = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK);"
              "$s.TargetPath = $env:TARGET; $s.WorkingDirectory = $env:WORKDIR; $s.Save()")
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                       env={**{k: v for k, v in os.environ.items() if k.upper() != "PSMODULEPATH"},
                            "LNK": str(lnk), "TARGET": str(target / "DualOCR.exe"), "WORKDIR": str(target)})
        if not lnk.exists():
            info("couldn't add the Start menu shortcut (the app itself is installed)")
        return f"{target}\\DualOCR.exe  (in the Start menu as DualOCR)"

    target = Path.home() / ".local" / "opt" / "DualOCR"
    shutil.rmtree(target, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dist, target, symlinks=True)
    share = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    icon_dir = share / "icons" / "hicolor" / "256x256" / "apps"
    icon_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    with Image.open(ROOT / "dualocr" / "ui" / "icon.png") as im:
        im.resize((256, 256), Image.LANCZOS).save(icon_dir / "dualocr.png")
    apps = share / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    (apps / "dualocr.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=DualOCR\n"
        "Comment=Turn scanned Hebrew and English documents into Word files\n"
        f'Exec="{target / "DualOCR"}" %F\nIcon=dualocr\nTerminal=false\nCategories=Office;Graphics;OCR;\n'
        "MimeType=application/pdf;image/png;image/jpeg;image/tiff;image/heic;image/webp;\n", encoding="utf-8")
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(apps)], capture_output=True)
    return f"{target / 'DualOCR'}  (in your applications menu as DualOCR)"


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Install DualOCR's dependencies and build the app.")
    ap.add_argument("-y", "--yes", action="store_true", help="don't ask before installing system packages")
    ap.add_argument("--llama", choices=["auto", "vulkan", "cpu", "system"], default="auto",
                    help="llama.cpp build: auto (Metal on Macs, Vulkan if the GPU supports it, else CPU), "
                         "or 'system' to use an installed llama-server")
    ap.add_argument("--skip-surya-model", action="store_true", help="don't download Surya's model now")
    ap.add_argument("--no-app", action="store_true", help="only set up the command line (uv run dualocr)")
    ap.add_argument("--no-install", action="store_true", help="build the app but don't install it")
    a = ap.parse_args()

    if sys.version_info < (3, 12):
        fail("DualOCR needs Python 3.12 or newer (run setup.sh / setup.cmd, which use uv to get it).")
    print(f"DualOCR setup: {platform.system()} {platform.machine()}, Python {platform.python_version()}")
    system_packages(a)
    data_files()
    llama_cpp(a)
    surya_model(a)
    where = None
    if not a.no_app:
        dist = build_app()
        check_build(dist)
        where = str(dist) if a.no_install else install_app(dist)

    step("Done")
    if where:
        info(f"App: {where}")
    again = ".\\setup.cmd" if WINDOWS else "./setup.sh"
    info("Command line (from this folder):  uv run dualocr scan.pdf")
    info(f"After pulling new code, run {again} again to rebuild.")


if __name__ == "__main__":
    main()
