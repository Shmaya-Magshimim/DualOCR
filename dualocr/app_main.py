"""Entry point of the built app.

  DualOCR              -> the window
  DualOCR --worker     -> the background job runner (started by the window)
  DualOCR --cli ARGS   -> command line (same as `dualocr ARGS`)
  dualocr-cli ARGS     -> command line (the console program next to DualOCR)
"""
import sys
from pathlib import Path


def main():
    args = sys.argv[1:]
    if "--worker" in args:
        from dualocr import worker
        worker.main()
        return
    if Path(sys.executable).stem.lower() == "dualocr-cli" or args[:1] == ["--cli"]:
        from dualocr.__main__ import main as cli
        sys.exit(cli(args[1:] if args[:1] == ["--cli"] else args))
    # Finder adds a -psn_... argument on some systems; files arrive as open events
    sys.argv = [sys.argv[0]] + [a for a in args if not a.startswith("-psn")]
    from dualocr.ui.app import main as ui
    ui()


if __name__ == "__main__":
    main()
