"""DualOCR window (PySide6), kept close to Apple's Human Interface Guidelines
on macOS (native controls, the system font at the standard text sizes) and
using the native look on Windows and Linux: one window with a drop area and a
list of jobs, options in Settings.

The window only manages jobs; the work runs in the background worker process
(dualocr.worker), so closing the window does not stop a conversion.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import (QEasingCurve, QEvent, QPropertyAnimation, QRectF, QSize, Qt, QTimer,
                            QVariantAnimation, Signal)
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QKeySequence, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QMenu, QSizePolicy,
                               QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSystemTrayIcon,
                               QTabWidget, QToolButton, QVBoxLayout, QWidget)

from .. import osutil, worker
from ..config import ROOT, Settings
from ..ingest import SUPPORTED_EXT

ICON_PATH = ROOT / "dualocr" / "ui" / "icon.png"
KEY_STORE = ("the macOS Keychain" if osutil.MACOS else "Windows Credential Manager" if osutil.WINDOWS
             else "your system's keyring")
TRAY_PATH = ROOT / "dualocr" / "ui" / "tray.png"

# text styles (points): Apple HIG > Typography > macOS. Windows and Linux
# render points larger, so their sizes are a little smaller.
if osutil.MACOS:
    TITLE3, HEADLINE, BODY, CALLOUT, SUBHEADLINE, FOOTNOTE = 15, 13, 13, 12, 11, 10
else:
    TITLE3, HEADLINE, BODY, CALLOUT, SUBHEADLINE, FOOTNOTE = 12, 10, 10, 9, 9, 8


def font(size: int, weight: QFont.Weight = QFont.Normal) -> QFont:
    if osutil.MACOS:
        f = QFont(".AppleSystemUIFont", size)        # SF Pro
    else:
        f = QFont(QApplication.font() if QApplication.instance() else QFont())
        f.setPointSize(size)
    f.setWeight(weight)
    return f


def secondary(widget: QWidget) -> QWidget:
    """macOS 'secondary label' color: the normal text color at reduced opacity."""
    pal = widget.palette()
    c = QColor(pal.color(QPalette.WindowText))
    c.setAlphaF(0.55)
    pal.setColor(QPalette.WindowText, c)
    widget.setPalette(pal)
    return widget


def label(text: str = "", size: int = BODY, weight=QFont.Normal, dim: bool = False, wrap=False) -> QLabel:
    lab = QLabel(text)
    lab.setFont(font(size, weight))
    lab.setWordWrap(wrap)
    if dim:
        secondary(lab)
    return lab


def fmt_time(s) -> str:
    if not s:
        return ""
    s = int(s)
    if s < 60:
        return "less than a minute left"
    return f"about {round(s / 60)} min left"


# ------------------------------------------------------------------ drop area

class DropZone(QFrame):
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(170)
        self.hover = False
        self.t = 0.0      # 0 = idle, 1 = a file is being dragged over
        self.anim = QVariantAnimation(self, duration=160, easingCurve=QEasingCurve.OutCubic)
        self.anim.valueChanged.connect(self._set_t)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(6)
        title = label("Drop scans here", TITLE3, QFont.DemiBold)
        sub = label("PDF or images · Hebrew and English", SUBHEADLINE, dim=True)
        btn = QPushButton("Choose Files…")
        btn.clicked.connect(self.choose)
        for w in (title, sub):
            w.setAlignment(Qt.AlignCenter)
            lay.addWidget(w)
        lay.addSpacing(6)
        row = QHBoxLayout(); row.addStretch(); row.addWidget(btn); row.addStretch()
        lay.addLayout(row)

    def choose(self):
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXT))
        files, _ = QFileDialog.getOpenFileNames(self.window(), "Choose Scans", str(Path.home() / "Downloads"),
                                                f"Scans ({exts})")
        if files:
            self.files_dropped.emit(files)

    def _set_t(self, v):
        self.t = float(v)
        self.update()

    def _hover(self, on: bool):
        self.hover = on
        self.anim.stop()
        self.anim.setStartValue(self.t)
        self.anim.setEndValue(1.0 if on else 0.0)
        self.anim.start()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pal = self.palette()
        a, b = QColor(pal.color(QPalette.WindowText)), QColor(pal.color(QPalette.Highlight))
        t = self.t
        line = QColor(int(a.red() + (b.red() - a.red()) * t), int(a.green() + (b.green() - a.green()) * t),
                      int(a.blue() + (b.blue() - a.blue()) * t))
        line.setAlphaF(0.25 + 0.75 * t)
        fill = QColor(b)
        fill.setAlphaF(0.08 * t)
        pen = QPen(line, 1.5, Qt.CustomDashLine)
        pen.setDashPattern([5, 4])
        p.setPen(pen)
        p.setBrush(fill)
        p.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 10, 10)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction(); self._hover(True)

    def dragLeaveEvent(self, e):
        self._hover(False)

    def dropEvent(self, e):
        self._hover(False)
        files = [u.toLocalFile() for u in e.mimeData().urls()]
        files = [f for f in files if Path(f).suffix.lower() in SUPPORTED_EXT]
        if files:
            self.files_dropped.emit(files)


# ------------------------------------------------------------------ job row

class JobRow(QFrame):
    removed = Signal(str)

    def __init__(self, job: dict, parent=None):
        super().__init__(parent)
        self.job_id = job["id"]
        self.job = job
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 10, 0, 10)
        lay.setSpacing(4)
        top = QHBoxLayout()
        names = [Path(p).name for p in job["inputs"]]
        title = Path(job["out_path"]).name if job.get("out_path") else (
            names[0] + (f" + {len(names) - 1} more" if len(names) > 1 else ""))
        self.name = label(title, BODY, QFont.Medium)
        self.name.setToolTip("From: " + ", ".join(names))
        top.addWidget(self.name, 1)
        self.primary = QPushButton()
        self.primary.clicked.connect(self.primary_action)
        top.addWidget(self.primary)
        self.more = QToolButton()
        self.more.setText("⋯")
        self.more.setToolTip("More")
        self.more.setPopupMode(QToolButton.InstantPopup)
        self.more.setAutoRaise(True)
        self.more.setFont(font(TITLE3))
        self.more.setStyleSheet("QToolButton { border: none; padding: 0 4px; }"
                                "QToolButton::menu-indicator { image: none; width: 0; }")
        self.menu = QMenu(self)
        self.more.setMenu(self.menu)
        top.addWidget(self.more)
        lay.addLayout(top)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        sp = self.bar.sizePolicy(); sp.setRetainSizeWhenHidden(True); self.bar.setSizePolicy(sp)
        self.bar_anim = QPropertyAnimation(self.bar, b"value", self)
        self.bar_anim.setDuration(450)
        self.bar_anim.setEasingCurve(QEasingCurve.OutCubic)
        lay.addWidget(self.bar)
        self.status = label("", SUBHEADLINE, dim=True, wrap=True)
        lay.addWidget(self.status)
        self._state = None
        self._sig = None
        self.update_job(job)

    def outputs(self):
        return self.job.get("result", {}).get("outputs", {})

    def update_job(self, job: dict):
        sig = (job["status"], tuple(sorted((job.get("progress") or {}).items())))
        if sig == self._sig:
            return            # nothing changed: no repaint
        self._sig = sig
        self.job = job
        st = job["status"]
        pr = job.get("progress", {})
        frac = pr.get("fraction", 0.0)
        self.bar.setVisible(st in ("queued", "running"))
        target = int(frac * 1000)
        if target != self.bar_anim.endValue():
            self.bar_anim.stop()
            self.bar_anim.setStartValue(self.bar.value())
            self.bar_anim.setEndValue(target)
            self.bar_anim.start()
        if st == "running":
            bits = []
            if pr.get("pages"):
                bits.append(f"Page {pr.get('page')} of {pr.get('pages')}")
            bits.append(f"{frac:.0%}")
            t = fmt_time(pr.get("eta"))
            if t:
                bits.append(t)
            self.status.setText(" · ".join(bits))
            self.status.setToolTip(pr.get("detail", ""))
            self.primary.setText("Cancel")
        elif st == "queued":
            self.status.setText("Waiting for the previous document")
            self.primary.setText("Cancel")
        elif st == "done":
            res = job.get("result", {})
            s = res.get("stats", {})
            check = s.get("uncertain", 0)
            n = res.get("pages", 0)
            text = f"Done · {n} page{'' if n == 1 else 's'}"
            text += (f" · {check} word{'' if check == 1 else 's'} to check") if check else " · nothing to check"
            if res.get("numbering"):
                text += " · see the report for numbering notes"
            self.status.setText(text)
            self.primary.setText("Open")
        elif st == "failed":
            self.status.setText("Couldn't convert: " + (job.get("error") or "unknown error"))
            self.primary.setText("Try Again")
        else:
            self.status.setText("Cancelled")
            self.primary.setText("Try Again")
        if st != self._state:       # rebuild the menu only when the state changes
            self._state = st
            self.primary.setDefault(st == "done")
            self._build_menu(st)

    def _build_menu(self, st):
        self.menu.clear()
        outs = self.outputs()
        if st == "done":
            s = self.job.get("result", {}).get("stats", {})
            if "review" in outs and (s.get("uncertain", 0) + s.get("source_typo", 0)):
                self.menu.addAction("Review Words to Check…", self.open_review)
            if "report" in outs:
                self.menu.addAction("Open Report", lambda: self._open(outs["report"]))
            self.menu.addAction(f"Show in {osutil.file_manager_name}", lambda: osutil.reveal(outs.get("docx", "")))
            self.menu.addSeparator()
        if st not in ("queued", "running"):
            self.menu.addAction("Remove from List", self.do_remove)
        self.more.setVisible(not self.menu.isEmpty())

    def primary_action(self):
        st = self.job["status"]
        if st in ("queued", "running"):
            worker.cancel(self.job_id)
        elif st == "done":
            self._open(self.outputs().get("docx"))
        else:   # failed / cancelled: run it again
            s = Settings(**{k: v for k, v in self.job["settings"].items() if k in Settings.__dataclass_fields__})
            worker.submit(self.job["inputs"], s, self.job.get("out_dir", ""), self.job.get("out_path", ""))
            self.do_remove()

    def _open(self, p):
        if p and Path(p).exists():
            osutil.open_path(p)
        else:
            QMessageBox.warning(self.window(), "File Not Found", f"The file was moved or deleted.\n\n{p}")

    def open_review(self):
        from .review import ReviewWindow
        p = self.outputs().get("review")
        if p and Path(p).exists():
            ReviewWindow(p, self.window()).exec()

    def do_remove(self):
        worker.remove(self.job_id)
        self.removed.emit(self.job_id)


# ------------------------------------------------------------------ settings

class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        from .. import llm
        self.llm = llm
        self.s = settings
        self.setWindowTitle("Settings")
        self.setMinimumWidth(540)
        tabs = QTabWidget()
        tabs.addTab(self._general(), "General")
        tabs.addTab(self._ai(), "AI Tie-Breaker")
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.save)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        lay.addWidget(bb)

    @staticmethod
    def _form(w):
        form = QFormLayout(w)
        form.setLabelAlignment(Qt.AlignRight)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setVerticalSpacing(12)
        form.setContentsMargins(16, 16, 16, 16)
        return form

    def _general(self):
        w = QWidget()
        form = self._form(w)
        self.quality = QComboBox()
        self.quality.addItems(["Fast", "Balanced (recommended)", "Max"])
        self.quality.setCurrentIndex({"fast": 0, "balanced": 1, "max": 2}.get(self.s.quality, 1))
        qhint = label("Balanced takes a second look at doubtful words, which helps on blurry scans.",
                      SUBHEADLINE, dim=True, wrap=True)
        qbox = QVBoxLayout(); qbox.setSpacing(4); qbox.addWidget(self.quality); qbox.addWidget(qhint)
        form.addRow("Quality:", qbox)
        self.hl = QCheckBox("Highlight words to check")
        self.hl.setChecked(self.s.review_highlights)
        self.level = QComboBox()
        self.level.addItems(["Rarely: only words it's really unsure of",
                             "Sometimes: about 1 word in 25",
                             "Often: about 1 word in 10, catches the most"])
        self.levels = ["rare", "some", "many"]
        self.level.setCurrentIndex(self.levels.index(self.s.highlight_level)
                                   if self.s.highlight_level in self.levels else 0)
        self.hl.toggled.connect(self.level.setEnabled)
        self.level.setEnabled(self.hl.isChecked())
        hbox = QVBoxLayout(); hbox.setSpacing(6); hbox.addWidget(self.hl); hbox.addWidget(self.level)
        form.addRow("Review:", hbox)
        self.pics = QCheckBox("Copy pictures into the document")
        self.pics.setChecked(self.s.include_pictures)
        self.txt = QCheckBox("Also save a text file")
        self.txt.setChecked(self.s.write_txt)
        self.rep = QCheckBox("Also save a report")
        self.rep.setChecked(self.s.write_report)
        obox = QVBoxLayout(); obox.setSpacing(6)
        for c in (self.pics, self.txt, self.rep):
            obox.addWidget(c)
        form.addRow("Output:", obox)
        self.open_done = QCheckBox("Open the document when it's ready")
        self.open_done.setChecked(self.s.open_when_done)
        form.addRow("When done:", self.open_done)
        return w

    def _ai(self):
        w = QWidget()
        form = self._form(w)
        form.addRow(label("Off by default, so everything stays on this computer. When on, only small images of "
                          "the few words the engines still disagree on are sent to the service you choose. "
                          f"Your key is kept in {KEY_STORE}.", SUBHEADLINE, dim=True, wrap=True))
        self.enabled = QCheckBox("Ask an AI model about the hardest words")
        self.enabled.setChecked(self.s.llm_enabled)
        form.addRow("", self.enabled)
        self.provider = QComboBox()
        self.provider.addItems(["Claude (Anthropic)", "Gemini (Google)"])
        self.provider.setCurrentIndex(1 if self.s.llm_provider == "gemini" else 0)
        self.provider.currentIndexChanged.connect(self._load_key)
        form.addRow("Service:", self.provider)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("Paste your API key")
        form.addRow("API key:", self.key)
        self.model = QLineEdit(self.s.llm_model)
        self.model.setPlaceholderText("Automatic")
        form.addRow("Model:", self.model)
        self.maxw = QSpinBox()
        self.maxw.setRange(1, 1000)
        self.maxw.setValue(self.s.llm_max_words)
        self.maxw.setSuffix(" words per document")
        form.addRow("Limit:", self.maxw)
        trow = QHBoxLayout()
        tb = QPushButton("Test Key")
        tb.clicked.connect(self.test)
        self.test_msg = label("", SUBHEADLINE, dim=True, wrap=True)
        trow.addWidget(tb); trow.addWidget(self.test_msg, 1)
        form.addRow("", trow)
        self._load_key()
        return w

    def _prov(self):
        return "gemini" if self.provider.currentIndex() == 1 else "claude"

    def _load_key(self):
        self.key.setText(self.llm.get_key(self._prov()) or "")

    def test(self):
        self.test_msg.setText("Testing…")
        QApplication.processEvents()
        ok, msg = self.llm.test_key(self._prov(), self.key.text().strip(), self.model.text().strip())
        self.test_msg.setText(("✓ " if ok else "✗ ") + msg)

    def save(self):
        try:
            self.llm.set_key(self._prov(), self.key.text().strip())
        except Exception as e:
            QMessageBox.warning(self, "API Key", f"Couldn't save the key in {KEY_STORE}.\n\n{e}")
        s = self.s
        s.quality = ["fast", "balanced", "max"][self.quality.currentIndex()]
        s.review_highlights = self.hl.isChecked()
        s.highlight_level = self.levels[self.level.currentIndex()]
        s.include_pictures, s.write_txt, s.write_report = (self.pics.isChecked(), self.txt.isChecked(),
                                                           self.rep.isChecked())
        s.open_when_done = self.open_done.isChecked()
        s.llm_enabled = self.enabled.isChecked() and bool(self.key.text().strip())
        s.llm_provider = self._prov()
        s.llm_model = self.model.text().strip()
        s.llm_max_words = self.maxw.value()
        s.save()
        self.accept()


# ------------------------------------------------------------------ main window

class Divider(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(1)

    def paintEvent(self, e):
        p = QPainter(self)
        c = QColor(self.palette().color(QPalette.WindowText))
        c.setAlphaF(0.1)
        p.fillRect(self.rect(), c)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings.load()
        self.setWindowTitle("DualOCR")
        self.setMinimumSize(QSize(520, 460))
        self.resize(600, 600)
        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(20, 20, 20, 16)
        lay.setSpacing(14)
        self.drop = DropZone()
        self.drop.files_dropped.connect(self.add_files)
        lay.addWidget(self.drop)
        self.list_box = QWidget()
        self.list_lay = QVBoxLayout(self.list_box)
        self.list_lay.setContentsMargins(0, 0, 0, 0)
        self.list_lay.setSpacing(0)
        self.list_lay.addStretch()
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setWidget(self.list_box)
        lay.addWidget(sc, 1)
        self.rows = {}
        self._menus()
        self.tray = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(500)
        self.poll()

    def _menus(self):
        mb = self.menuBar()
        app_menu = mb.addMenu("DualOCR")
        pref = QAction("Settings…", self)
        pref.setMenuRole(QAction.PreferencesRole)
        pref.setShortcut(QKeySequence.Preferences if osutil.MACOS else QKeySequence("Ctrl+,"))
        pref.triggered.connect(self.open_settings)
        app_menu.addAction(pref)
        f = mb.addMenu("File")
        op = QAction("Open…", self)
        op.setShortcut(QKeySequence.Open)
        op.triggered.connect(self.drop.choose)
        f.addAction(op)
        f.addSeparator()
        cl = QAction("Clear Finished", self)
        cl.triggered.connect(self.clear_finished)
        f.addAction(cl)

    def open_settings(self):
        if SettingsDialog(self.settings, self).exec():
            self.settings = Settings.load()
            self.poll()

    def add_files(self, files):
        self.settings = Settings.load()
        groups = [[f] for f in files]
        if len(files) > 1:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setText(f"Make one document from these {len(files)} files?")
            box.setInformativeText("Combine them in this order, or make a separate document for each file.")
            combine = box.addButton("Combine", QMessageBox.AcceptRole)
            box.addButton("Separate Documents", QMessageBox.RejectRole)
            box.setDefaultButton(combine)
            box.exec()
            if box.clickedButton() == combine:
                groups = [files]
        for g in groups:
            out = self.ask_name(g)
            if not out:
                continue      # Save panel cancelled: don't start this one
            worker.submit(g, self.settings, str(Path(out).parent), out)
        self.poll()

    def ask_name(self, files) -> str:
        """Standard Save panel: where to put the Word document and what to call it."""
        first = Path(files[0])
        suggested = first.parent / f"{first.stem}.docx"
        path, _ = QFileDialog.getSaveFileName(self, "Save Word Document", str(suggested), "Word Document (*.docx)")
        if path and not path.lower().endswith(".docx"):
            path += ".docx"
        return path

    def clear_finished(self):
        for j in worker.list_jobs():
            if j["status"] in ("done", "failed", "cancelled"):
                worker.remove(j["id"])
        self.poll()

    def _drop_row(self, jid):
        row = self.rows.pop(jid, None)
        if row:
            self._fade(row, 1.0, 0.0, lambda: (row.setParent(None), row.deleteLater(), self._relayout_dividers()))

    def _fade(self, w, a, b, done=None):
        eff = QGraphicsOpacityEffect(w)
        w.setGraphicsEffect(eff)
        an = QPropertyAnimation(eff, b"opacity", w)
        an.setDuration(220)
        an.setStartValue(a)
        an.setEndValue(b)
        an.setEasingCurve(QEasingCurve.OutCubic)

        def finished():
            if b >= 1.0:
                w.setGraphicsEffect(None)   # keep text crisp once visible
            if done:
                done()
        an.finished.connect(finished)
        an.start()
        w._fade_anim = an

    def _relayout_dividers(self):
        # rows separated by hairlines; rebuild the order newest-first
        for i in reversed(range(self.list_lay.count())):
            it = self.list_lay.itemAt(i)
            w = it.widget()
            if isinstance(w, Divider):
                self.list_lay.takeAt(i)
                w.setParent(None)
                w.deleteLater()
        rows = [self.list_lay.itemAt(i).widget() for i in range(self.list_lay.count())
                if isinstance(self.list_lay.itemAt(i).widget(), JobRow)]
        for r in rows[1:]:
            self.list_lay.insertWidget(self.list_lay.indexOf(r), Divider())

    def poll(self):
        jobs = worker.list_jobs()
        ids = {j["id"] for j in jobs}
        for jid in list(self.rows):
            if jid not in ids:
                self._drop_row(jid)
        added = False
        for j in jobs:   # each new row goes on top, so the newest ends up first
            if j["id"] in self.rows:
                self.rows[j["id"]].update_job(j)
            else:
                row = JobRow(j)
                row.removed.connect(self._drop_row)
                self.rows[j["id"]] = row
                self.list_lay.insertWidget(0, row)
                self._fade(row, 0.0, 1.0)
                added = True
        if added:
            self._relayout_dividers()
        active = [j for j in jobs if j["status"] in ("queued", "running")]
        if active and not worker.worker_alive():
            worker.ensure_worker()
        self._update_tray(active)

    # menu bar extra: only while work continues with the window closed
    def _update_tray(self, active):
        want = bool(active) and not self.isVisible()
        if want and self.tray is None and QSystemTrayIcon.isSystemTrayAvailable():
            if osutil.MACOS:     # black template glyph, tinted by the menu bar
                icon = QIcon(str(TRAY_PATH))
                icon.setIsMask(True)
            else:                # a black glyph would vanish on a dark taskbar
                icon = QIcon(str(ICON_PATH))
            self.tray = QSystemTrayIcon(icon, self)
            m = QMenu()
            self.tray_status = QAction("", self)
            self.tray_status.setEnabled(False)
            m.addAction(self.tray_status)
            m.addSeparator()
            m.addAction("Show DualOCR", self.show_normal)
            self.tray.setContextMenu(m)
            self.tray.show()
        if self.tray is not None:
            if not want:
                self.tray.hide()
                self.tray.deleteLater()
                self.tray = None
            else:
                p = active[0].get("progress", {})
                self.tray_status.setText(f"{Path(active[0]['inputs'][0]).name}: {p.get('fraction', 0):.0%}")

    def show_normal(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, e):
        # closing the window never stops work: the worker keeps going
        e.accept()
        if not [j for j in worker.list_jobs() if j["status"] in ("queued", "running")]:
            QApplication.instance().quit()


def make_icons():
    """Draw the app icon and the menu-bar template icon (no image assets needed)."""
    if not ICON_PATH.exists():
        pm = QPixmap(1024, 1024); pm.fill(Qt.transparent)
        p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor("#2f54eb")); p.setPen(Qt.NoPen); p.drawRoundedRect(64, 64, 896, 896, 200, 200)
        p.setBrush(QColor("#ffffff")); p.drawRoundedRect(250, 180, 524, 664, 40, 40)
        p.setPen(QColor("#2f54eb"))
        f = QFont("Arial", 230); f.setBold(True); p.setFont(f)
        p.drawText(250, 230, 262, 380, Qt.AlignCenter, "א")
        p.drawText(512, 230, 262, 380, Qt.AlignCenter, "A")
        p.setPen(Qt.NoPen); p.setBrush(QColor("#9fb2ff"))
        for k in range(3):
            p.drawRoundedRect(320, 620 + k * 60, 384 - k * 70, 26, 13, 13)
        p.end(); pm.save(str(ICON_PATH))
    if not TRAY_PATH.exists():
        pm = QPixmap(44, 44); pm.fill(Qt.transparent)
        p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen); p.setBrush(QColor("black")); p.drawRoundedRect(6, 2, 32, 40, 5, 5)
        p.setCompositionMode(QPainter.CompositionMode_Clear)
        for k in range(4):
            p.drawRect(12, 10 + k * 8, 20 - (k % 2) * 6, 3)
        p.end(); pm.save(str(TRAY_PATH))


class DualOCRApp(QApplication):
    """Receives files dropped on the Dock icon or opened with "Open With"."""

    def __init__(self, argv):
        super().__init__(argv)
        self.window = None
        self.pending = []

    def event(self, e):
        if e.type() == QEvent.FileOpen:
            path = e.file()
            if path and Path(path).suffix.lower() in SUPPORTED_EXT:
                if self.window:
                    self.window.add_files([path]); self.window.show_normal()
                else:
                    self.pending.append(path)
            return True
        return super().event(e)


def main():
    app = DualOCRApp(sys.argv)
    app.setApplicationName("DualOCR")
    app.setApplicationDisplayName("DualOCR")
    app.setFont(font(BODY))
    app.setQuitOnLastWindowClosed(False)
    try:
        make_icons()
    except OSError:
        pass   # inside the app bundle the icons are already there
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    w = MainWindow()
    app.window = w
    w.show()
    from ..config import install_hint, missing_tools
    miss = missing_tools()
    if miss:
        QMessageBox.warning(w, "DualOCR Is Missing a Tool",
                            "Not found: " + " and ".join(miss) + ".\n\n" + install_hint()
                            + "\n\nThen open DualOCR again.")
    files = [a for a in sys.argv[1:] if Path(a).suffix.lower() in SUPPORTED_EXT and Path(a).exists()]
    if files or app.pending:
        w.add_files(files + app.pending)
    # clicking the Dock icon while the window is closed brings it back
    app.applicationStateChanged.connect(
        lambda st: w.show_normal() if st == Qt.ApplicationActive and not w.isVisible() else None)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
