"""Save/load per-page engine results (used for job resume and evaluation)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from .model import Block, Line, Word


def save_lines(path: Path, lines: List[Line]) -> None:
    path.write_text(json.dumps([asdict(l) for l in lines], ensure_ascii=False), encoding="utf-8")


def load_lines(path: Path) -> List[Line]:
    out = []
    for l in json.loads(path.read_text(encoding="utf-8")):
        ws = [Word(w["text"], tuple(w["bbox"]), w["conf"], tuple(w["line_key"]), w["engine"]) for w in l["words"]]
        out.append(Line(ws, tuple(l["bbox"]), tuple(l["key"])))
    return out


def save_blocks(path: Path, blocks: List[Block]) -> None:
    path.write_text(json.dumps([asdict(b) for b in blocks], ensure_ascii=False), encoding="utf-8")


def load_blocks(path: Path) -> List[Block]:
    out = []
    for b in json.loads(path.read_text(encoding="utf-8")):
        tp = b.get("token_probs")
        if tp is not None:
            tp = [tuple(x) for x in tp]
        out.append(Block(b["label"], tuple(b["bbox"]), b["html"], b["text"], b["order"], b.get("conf", 1.0), tp))
    return out
