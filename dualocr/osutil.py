"""Small helpers that differ between macOS, Windows and Linux."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

WINDOWS = os.name == "nt"
MACOS = sys.platform == "darwin"
LINUX = not WINDOWS and not MACOS

EXE = ".exe" if WINDOWS else ""


def user_data_dir(app: str = "DualOCR") -> Path:
    """Where jobs, settings and review data go (the usual place on each system)."""
    if WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif MACOS:
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / app


def tool_dirs() -> list:
    """Folders where package managers put Tesseract (and llama.cpp) but that an
    app started from the desktop may not have on its PATH."""
    if WINDOWS:
        pf = [os.environ.get(k) for k in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
        cands = [Path(p) / "Tesseract-OCR" for p in pf if p]
        cands += [Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Tesseract-OCR"] if os.environ.get("LOCALAPPDATA") else []
    elif MACOS:
        cands = [Path("/opt/homebrew/bin"), Path("/usr/local/bin"), Path("/opt/local/bin")]
    else:
        cands = [Path("/usr/local/bin"), Path("/home/linuxbrew/.linuxbrew/bin"), Path.home() / ".linuxbrew" / "bin"]
    return [p for p in cands if p.is_dir()]


def hide_console_windows() -> None:
    """On Windows, every program a windowed app starts (Tesseract runs many
    times per page, plus the llama.cpp server) would flash a console window.
    Make all child processes windowless."""
    if not WINDOWS or getattr(subprocess.Popen, "_dualocr_no_window", False):
        return
    base = subprocess.Popen
    flag = subprocess.CREATE_NO_WINDOW

    class Popen(base):
        _dualocr_no_window = True

        def __init__(self, *args, **kw):
            kw["creationflags"] = kw.get("creationflags", 0) | flag
            super().__init__(*args, **kw)

    subprocess.Popen = Popen


def detached_popen_kwargs() -> dict:
    """Start a background process that outlives the window that started it."""
    if WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def pid_alive(pid: int) -> bool:
    """Is a process with this id running? (os.kill(pid, 0) does something else
    on Windows: it sends Ctrl+C.)"""
    if not pid or pid <= 0:
        return False
    if WINDOWS:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        h = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ctypes.get_last_error() == 5           # access denied: it exists
        try:
            code = wintypes.DWORD()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def kill_pid(pid: int) -> None:
    try:
        if WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=30)
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


def open_path(path) -> None:
    """Open a file with its default app."""
    p = str(path)
    if WINDOWS:
        os.startfile(p)  # noqa: S606 (Windows only)
    elif MACOS:
        subprocess.Popen(["open", p])
    else:
        subprocess.Popen(["xdg-open", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def reveal(path) -> None:
    """Show a file in Finder / Explorer / the file manager."""
    p = Path(path)
    if WINDOWS:
        subprocess.Popen(f'explorer /select,"{p}"')
    elif MACOS:
        subprocess.Popen(["open", "-R", str(p)])
    else:
        # the freedesktop file-manager interface can select the file; fall back to opening the folder
        try:
            r = subprocess.run(["dbus-send", "--session", "--print-reply", "--dest=org.freedesktop.FileManager1",
                                "/org/freedesktop/FileManager1", "org.freedesktop.FileManager1.ShowItems",
                                f"array:string:{p.resolve().as_uri()}", "string:"],
                               capture_output=True, timeout=5)
            if r.returncode == 0:
                return
        except Exception:
            pass
        subprocess.Popen(["xdg-open", str(p.parent)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


file_manager_name = "Explorer" if WINDOWS else "Finder" if MACOS else "File Manager"

# Windows toast through PowerShell (no extra packages). The text arrives in
# environment variables, so nothing in it is ever run as code.
_WIN_TOAST = (
    "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] "
    "| Out-Null;"
    "$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
    "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
    "$x = $t.GetElementsByTagName('text');"
    "$x.Item(0).AppendChild($t.CreateTextNode($env:DUALOCR_TITLE)) | Out-Null;"
    "$x.Item(1).AppendChild($t.CreateTextNode($env:DUALOCR_TEXT)) | Out-Null;"
    "$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe';"
    "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show("
    "[Windows.UI.Notifications.ToastNotification]::new($t))"
)


def clean_powershell_env(**extra) -> dict:
    """Environment for starting Windows PowerShell 5.1: a module path inherited
    from PowerShell 7 stops it from loading its own modules."""
    env = {k: v for k, v in os.environ.items() if k.upper() != "PSMODULEPATH"}
    env.update(extra)
    return env


def notify(title: str, text: str) -> None:
    """Desktop notification. Never raises."""
    try:
        if MACOS:
            subprocess.run(["osascript", "-e", "on run argv", "-e",
                            "display notification (item 2 of argv) with title (item 1 of argv)",
                            "-e", "end run", title, text], capture_output=True, timeout=10)
        elif WINDOWS:
            env = clean_powershell_env(DUALOCR_TITLE=title, DUALOCR_TEXT=text)
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _WIN_TOAST],
                           env=env, capture_output=True, timeout=20)
        elif shutil.which("notify-send"):
            subprocess.run(["notify-send", "-a", "DualOCR", title, text], capture_output=True, timeout=10)
    except Exception:
        pass
