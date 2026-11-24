"""Review screen: go through the flagged words of a finished job, see the
scan next to each one, pick or type the right reading, and save the Word file."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (QDialog, QFrame, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
                               QGraphicsView, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import review as RV
from .. import osutil
from .app import BODY, HEADLINE, SUBHEADLINE, TITLE3, label


def _qpix(img) -> QPixmap:
    img = img.convert("RGB")
    data = img.tobytes("raw", "RGB")
    q = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(q.copy())


class PageView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.page_item = None
        self.marks = []

    def show_page(self, pix: QPixmap, boxes, current=None):
        sc = self.scene(); sc.clear(); self.marks = []
        self.page_item = QGraphicsPixmapItem(pix); sc.addItem(self.page_item)
        for b, is_cur in boxes:
            x0, y0, x1, y1 = b
            r = QGraphicsRectItem(QRectF(x0 - 4, y0 - 4, x1 - x0 + 8, y1 - y0 + 8))
            col = QColor(47, 84, 235) if is_cur else QColor(240, 180, 0)
            r.setPen(QPen(col, 5 if is_cur else 3))
            fill = QColor(col); fill.setAlpha(40); r.setBrush(QBrush(fill))
            sc.addItem(r); self.marks.append(r)
        self.setSceneRect(QRectF(pix.rect()))
        if current:
            x0, y0, x1, y1 = current
            self.resetTransform()
            self.fitInView(QRectF(max(0, x0 - 700), max(0, y0 - 350), 1400 + (x1 - x0), 700 + (y1 - y0)),
                           Qt.KeepAspectRatio)
        else:
            self.fitInView(self.page_item, Qt.KeepAspectRatio)

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)


class ReviewWindow(QDialog):
    def __init__(self, review_path: str, parent=None):
        super().__init__(parent)
        self.path = Path(review_path)
        self.meta, self.layouts, self.img_paths = RV.load(self.path)
        self._imgs = {}
        self.setWindowTitle("Review Words to Check")
        self.resize(1250, 820)
        self.items = []   # (page index, para, word index, slot)
        for k, pl in enumerate(self.layouts):
            for para in pl.paras:
                for i, sl in enumerate(para.words):
                    if sl.text and sl.status in ("uncertain", "source_typo"):
                        self.items.append((k, para, i, sl))
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.title = label("", HEADLINE)
        top.addWidget(self.title, 1)
        self.save_btn = QPushButton("Save Document")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self.save)
        top.addWidget(self.save_btn)
        lay.addLayout(top)
        split = QSplitter()
        lay.addWidget(split, 1)
        self.list = QTreeWidget(); self.list.setMinimumWidth(240)
        self.list.setColumnCount(3)
        self.list.setHeaderLabels(["", "Word", "Page"])
        self.list.setRootIsDecorated(False)
        self.list.setUniformRowHeights(True)
        hdr = self.list.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.list.currentItemChanged.connect(lambda cur, _prev: self.select(self.list.indexOfTopLevelItem(cur)))
        split.addWidget(self.list)
        mid = QWidget(); ml = QVBoxLayout(mid); ml.setContentsMargins(0, 0, 0, 0)
        self.view = PageView(); ml.addWidget(self.view, 1)
        split.addWidget(mid)
        side = QWidget(); side.setMinimumWidth(320)
        sl = QVBoxLayout(side); sl.setContentsMargins(12, 0, 0, 0); sl.setSpacing(6)
        self.ctx = label("", TITLE3, wrap=True); self.ctx.setLayoutDirection(Qt.RightToLeft)
        self.crop = QLabel(); self.crop.setAlignment(Qt.AlignCenter); self.crop.setMinimumHeight(110)
        self.why = label("", SUBHEADLINE, dim=True, wrap=True)
        sl.addWidget(label("In context", SUBHEADLINE, dim=True)); sl.addWidget(self.ctx)
        sl.addSpacing(8)
        sl.addWidget(label("On the scan", SUBHEADLINE, dim=True)); sl.addWidget(self.crop)
        sl.addSpacing(8)
        sl.addWidget(label("Choose the right reading", SUBHEADLINE, dim=True))
        self.choices = QVBoxLayout(); sl.addLayout(self.choices)
        row = QHBoxLayout()
        self.edit = QLineEdit(); self.edit.setPlaceholderText("Or type it exactly as printed")
        self.edit.setLayoutDirection(Qt.RightToLeft)
        self.edit.returnPressed.connect(lambda: self.apply(self.edit.text()))
        ok = QPushButton("Use"); ok.clicked.connect(lambda: self.apply(self.edit.text()))
        row.addWidget(self.edit, 1); row.addWidget(ok)
        sl.addLayout(row)
        sl.addWidget(self.why)
        nav = QHBoxLayout()
        prev = QPushButton("Previous"); prev.clicked.connect(lambda: self.move(-1))
        nxt = QPushButton("Next"); nxt.clicked.connect(lambda: self.move(1))
        nav.addWidget(prev); nav.addStretch(); nav.addWidget(nxt)
        sl.addStretch(); sl.addLayout(nav)
        split.addWidget(side)
        split.setSizes([280, 640, 330])
        for k, para, i, s in self.items:
            it = QTreeWidgetItem(["", s.text, str(k + 1)])
            it.setTextAlignment(2, Qt.AlignRight | Qt.AlignVCenter)
            self.list.addTopLevelItem(it)
        self._update_title()
        if self.items:
            self.list.setCurrentItem(self.list.topLevelItem(0))
        else:
            self.ctx.setText("Nothing to check in this document.")


    def _update_title(self):
        done = sum(1 for *_, s in self.items if getattr(s, "reviewed", False))
        self.title.setText(f"{done} of {len(self.items)} checked")

    def image(self, k):
        if k not in self._imgs:
            from PIL import Image
            self._imgs[k] = Image.open(self.img_paths[k]).convert("RGB")
        return self._imgs[k]

    def select(self, row):
        if row < 0 or row >= len(self.items):
            return
        k, para, i, s = self.items[row]
        img = self.image(k)
        boxes = [(it[3].bbox, it[3] is s) for it in self.items if it[0] == k and it[3].bbox]
        self.view.show_page(_qpix(img), boxes, s.bbox)
        left = " ".join(w.text for w in para.words[max(0, i - 6):i] if w.text)
        right = " ".join(w.text for w in para.words[i + 1:i + 7] if w.text)
        self.ctx.setText(f"<div dir='rtl'>{left} <b><u>{s.text}</u></b> {right}</div>")
        if s.bbox:
            x0, y0, x1, y1 = s.bbox
            h = y1 - y0
            c = img.crop((max(0, int(x0 - 2.5 * h)), max(0, int(y0 - .6 * h)), min(img.width, int(x1 + 2.5 * h)),
                          min(img.height, int(y1 + .6 * h))))
            c.thumbnail((320, 120))
            self.crop.setPixmap(_qpix(c))
        while self.choices.count():
            w = self.choices.takeAt(0).widget()
            if w:
                w.hide(); w.setParent(None); w.deleteLater()
        opts = []
        for v in [s.text] + list((s.readings.get("_cands") or {}).keys()) + [s.surya, s.tess, s.readings.get("llm")]:
            if v and v not in opts:
                opts.append(v)
        for v in opts[:6]:
            b = QPushButton(v); b.setLayoutDirection(Qt.RightToLeft)
            b.clicked.connect(lambda _=False, v=v: self.apply(v))
            self.choices.addWidget(b)
        self.edit.setText(s.text)
        why = []
        if s.status == "source_typo":
            why.append("Both engines read this the same way, but it isn't in the dictionary: maybe a typo in the original.")
        if s.surya is not None:
            why.append(f"Surya: {s.surya}")
        if s.tess is not None:
            why.append(f"Tesseract: {s.tess}")
        if s.kinds:
            why.append("; ".join(f"{e}: {k}" for e, k in s.kinds.items()))
        self.why.setText("\n".join(why))

    def apply(self, text):
        row = self.list.indexOfTopLevelItem(self.list.currentItem())
        if row < 0:
            return
        k, para, i, s = self.items[row]
        s.text = text.strip()
        s.reviewed = True
        s.status = "reviewed"
        it = self.list.topLevelItem(row)
        it.setText(0, "✓"); it.setText(1, s.text)
        self._update_title()
        self.move(1)

    def move(self, d):
        r = self.list.indexOfTopLevelItem(self.list.currentItem()) + d
        if 0 <= r < len(self.items):
            self.list.setCurrentItem(self.list.topLevelItem(r))

    def save(self):
        RV.write_back(self.path, self.layouts)
        try:
            out = RV.rewrite_outputs(self.path, review_highlights=True)
        except Exception as e:
            QMessageBox.warning(self, "Could not save", str(e))
            return
        n = sum(1 for *_, s in self.items if getattr(s, "reviewed", False))
        QMessageBox.information(self, "Document Saved", f"Updated with {n} checked words.\n\n{out.get('docx')}")
        if out.get("docx"):
            osutil.reveal(out["docx"])
