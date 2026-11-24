# DualOCR

Turns scanned PDFs and photos (Hebrew, English, or both mixed) into Word
documents, word for word. It reads every page with two OCR engines,
[Tesseract](https://github.com/tesseract-ocr/tesseract) and
[Surya](https://github.com/datalab-to/surya), lines their readings up word by
word, and votes on each word. Everything runs on your own computer.

Works on **macOS, Windows and Linux**. There's a desktop app (drop files on the
window) and a command line.

## Install

You need `git` and an internet connection. Setup downloads about 2 GB the
first time (mostly Surya's model) and takes 5 to 15 minutes.

**macOS / Linux** (Terminal):

```sh
git clone https://github.com/Shmaya-Magshimim/DualOCR.git && cd DualOCR && ./setup.sh
```

**Windows** (PowerShell):

```powershell
git clone https://github.com/Shmaya-Magshimim/DualOCR.git; cd DualOCR; .\setup.cmd
```

No `git` on Windows? Run `winget install Git.Git` first, then open a new
PowerShell window.

That's it. Setup installs and downloads everything DualOCR needs, builds the
app, and adds it to Launchpad (macOS), the Start menu (Windows) or your
applications menu (Linux):

| Step | What | From |
|---|---|---|
| 1 | [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and the Python packages | astral.sh, PyPI |
| 2 | Tesseract | Homebrew (macOS), winget (Windows), apt / dnf / pacman / zypper (Linux) |
| 3 | Tesseract's best Hebrew and English models, Hebrew and English spelling dictionaries | GitHub, checked against fixed SHA-256 hashes |
| 4 | llama.cpp's server, which runs Surya's model (Metal on Macs, Vulkan on Windows/Linux graphics cards, else CPU) | llama.cpp release `b12359`, checked against its SHA-256 |
| 5 | Surya's model (about 1.4 GB) | Hugging Face, into `~/.cache/huggingface` |
| 6 | Builds the app with PyInstaller and installs it | |

On macOS and Linux it may ask for your password to install system packages.
Run setup again any time: finished steps are skipped, so it just rebuilds the
app (do this after `git pull`).

Setup options (`./setup.sh --help`):

| Option | |
|---|---|
| `--yes` | don't ask before installing system packages |
| `--llama cpu` | use the CPU build of llama.cpp (if Surya fails on your graphics card) |
| `--llama system` | use a `llama-server` you installed yourself |
| `--no-app` | skip building the app; use only the command line |
| `--skip-surya-model` | don't download Surya's model now (it downloads on first use) |

**Hardware:** 8 GB of RAM or more. A Mac with Apple Silicon or a PC with a
graphics card is much faster: on an M4 Pro a page takes about 15-20 seconds.
On CPU only it works, but expect minutes per page.

## Use

**App:** open DualOCR, then drop PDFs or images on the window (or use File →
Open). On macOS you can also drop files on its Dock icon, or right-click a file
in Finder and choose Open With → DualOCR. A Save panel asks where to put the
Word document. Conversions keep running in the background if you close the
window, and you get a notification when each is done.

**Command line** (from the DualOCR folder):

```sh
uv run dualocr scan.pdf [more files] [-o OUTDIR] [-q fast|balanced|max]
uv run dualocr --check        # what's installed and where
```

Several files given together become one document, in order. Supported: PDF,
JPEG, PNG, TIFF, HEIC, WebP, BMP and GIF.

**Where things go:**

| | macOS | Windows | Linux |
|---|---|---|---|
| App | `~/Applications/DualOCR.app` | `%LOCALAPPDATA%\Programs\DualOCR` | `~/.local/opt/DualOCR` |
| Jobs, settings, review data | `~/Library/Application Support/DualOCR` | `%LOCALAPPDATA%\DualOCR` | `~/.local/share/DualOCR` |

Set `DUALOCR_DATA` to use another data folder. Surya's model is in
`~/.cache/huggingface`, and the llama.cpp server's log in
`~/.cache/datalab/surya`.

## How it works

1. **Import:** PDF pages that are a scanned photo are taken at the photo's
   native resolution. Scanner-app logos and watermarks are skipped.
2. **Page prep:**
   - Orientation is checked with Tesseract's detector. If that isn't sure,
     each of the four directions is test-read and the clearest one wins.
   - Pages with too little printed text, like handwriting, are left as scanned.
   - Tilted pages are straightened, and highlighter backgrounds are removed
     for reading.
3. **Two engines:** Tesseract (best Hebrew and English models) gives every
   word's box and confidence. Surya (a vision model running locally through
   llama.cpp) gives paragraphs, headings, lists, pictures, reading order, and
   per-letter alternatives.
4. **Alignment:** both readings are lined up word by word. This handles words
   one engine glued together and the other split, and the same word read in
   two scripts ("מן" vs "in").
5. **Zoom-in rescans:** doubtful words are cropped with their neighbors,
   enlarged, cleaned up several ways and re-read by both engines.
6. **Voting:** each candidate spelling gets evidence features. A small model
   trained on answer-key pages scores them. The evidence comes from:
   - both engines, the rescans, and Surya's own letter alternatives
   - Hunspell dictionaries (Hebrew prefixes are handled) and word frequency
   - the document's own vocabulary
   - fixed rules learned from real scans (see `dualocr/rules.py`):
     - ינ printed as ת, a ו swallowed by the letter before it, ח/ת and ד/ר
       look-alikes
     - Hebrew years start with ת
     - punctuation on the wrong side of Hebrew words inside English lines
     - mirrored parentheses
     - pixel gaps decide "1.חוק" vs "1. חוק"
     - a word that the engines and the zoom-in all read the same way is never
       "corrected", even if it isn't in the dictionary. It is kept and marked
       as a possible typo in the original.
7. **Outputs:**
   - **DOCX:** right-to-left Hebrew paragraphs, English kept in the right
     order inside them, headings, lists, and pictures cropped from the page.
     The few words it's really unsure of are highlighted yellow.
   - **Also:** a `.txt`, and an HTML review report with every flagged word,
     what each engine got wrong, and page and list numbering problems.
8. **Review screen:** step through the flagged words next to the scan, pick or
   type the right reading, and save the Word file again.

**Optional AI tie-breaker** (Settings): with a Claude or Gemini API key, only
small crops of the few words still in doubt are sent. The key is stored in the
system's password store (macOS Keychain, Windows Credential Manager, or the
Secret Service keyring on Linux). It's off unless you turn it on.

## Accuracy

Measured on the author's own answer-key documents (not included in this
repository): a 20-page Hebrew civics handout and a 7-page English handout with
Hebrew words. The voting model was trained on Hebrew pages 1-10 and English
scan pages 1, 2 and 4. Hebrew pages 11-20 were never used for training.
"Exact" counts every word including its punctuation.

| | Tesseract alone | Surya alone | DualOCR |
|---|---|---|---|
| Hebrew, held-out pages 11-20 | 88.2% | 93.7% | **96.2%** exact · 97.4% ignoring punctuation · 99.2% of characters |
| English (+ Hebrew words) | 97.4% | 98.1% | **98.3%** |
| Generated mixed Hebrew/English pages (rotated, blurred, JPEG) | | | 97-100% per page |
| Blurry phone scan of the English handout | | | 63-64% (rescans help a little) |

- **Highlights:** by default only words the app is really unsure of are
  highlighted. On the 20-page Hebrew handout that's 25 words, about 1 per page.
  Settings → Review → "Sometimes" or "Often" highlights more words and catches
  more errors (about 4% / 9% of words).
- **Most remaining errors:** words both engines agree on but "correct" the
  spelling of (e.g. printed לאומים read as לאומיים), real-word confusions
  (המנון/המעון), and punctuation details.
- **Quality presets:** zoomed-in re-reads gave no measurable gain on clean
  scans and a small gain on the blurry one. Fast is as accurate on clean scans;
  Balanced is the default for safety.

## Troubleshooting

- **`uv run dualocr --check`** shows what DualOCR found and what's missing.
- **Surya fails or crashes on a Windows/Linux graphics card:** run setup again
  with `--llama cpu`.
- **Linux: "Could not load the Qt platform plugin xcb":** install
  `libxcb-cursor0` (Debian/Ubuntu) or `xcb-util-cursor` (Fedora/Arch).
- **Linux: the downloaded llama.cpp doesn't run** (older distributions): build
  or install llama.cpp yourself and run setup with `--llama system`.
- **Windows: "couldn't replace ... DualOCR":** close DualOCR, wait a couple of
  minutes for background conversions to finish, and run setup again.

## Development

```sh
uv sync --extra dev
uv run pytest tests/test_core.py        # fast unit tests
uv run pytest -m slow                   # real engines (needs eval/synth, see tests/test_e2e.py)
```

`eval/` has the scripts that measured the table above (`run_eval.py`,
`train.py`, `flags.py`, `errs.py`). They expect your own scans in
`eval/sources/` and answer keys in `eval/gold/`, which are not in the
repository.

## Licenses

DualOCR's code is under the [MIT license](LICENSE). Setup downloads other
projects' files, each under its own license:

- [Surya](https://github.com/datalab-to/surya): code Apache-2.0. The model
  weights have a modified OpenRAIL-M license: free for personal use and
  research, and commercial use is limited for organizations above $5M in
  revenue or funding (see the
  [model's license](https://huggingface.co/datalab-to/surya-ocr-2.5-gguf)).
- [Tesseract](https://github.com/tesseract-ocr/tesseract) and
  [tessdata_best](https://github.com/tesseract-ocr/tessdata_best): Apache-2.0.
- [llama.cpp](https://github.com/ggml-org/llama.cpp): MIT.
- Hebrew dictionary from [Hspell](http://hspell.ivrix.org.il/) via
  [LibreOffice dictionaries](https://github.com/LibreOffice/dictionaries):
  AGPL-3.0. English dictionary: SCOWL, via the same repository.
- Word frequency lists from
  [FrequencyWords](https://github.com/hermitdave/FrequencyWords): CC-BY-SA-4.0.
