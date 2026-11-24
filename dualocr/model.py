"""Shared data types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Word:
    text: str
    bbox: tuple                 # x0, y0, x1, y1 (page pixels)
    conf: float                 # 0..100
    line_key: tuple = ()        # engine-specific line id
    engine: str = ""


@dataclass
class Line:
    words: List[Word]
    bbox: tuple
    key: tuple = ()


@dataclass
class Block:
    """A layout block (from Surya): paragraph, heading, list, picture..."""
    label: str                  # Text, SectionHeader, ListGroup, Picture, PageHeader...
    bbox: tuple
    html: str
    text: str                   # plain text, lines separated by \n
    order: int
    conf: float = 1.0
    token_probs: Optional[list] = None   # [(token_text, prob)] if logprobs were requested


@dataclass
class Choice:
    """One decided word slot in the final text."""
    text: str
    bbox: Optional[tuple]
    conf: float                           # 0..1 final confidence
    status: str                           # agree | resolved | uncertain | source_typo | only_one
    readings: dict = field(default_factory=dict)   # engine/variant -> text
    error_kinds: dict = field(default_factory=dict)  # engine -> kind of error
    reason: str = ""
