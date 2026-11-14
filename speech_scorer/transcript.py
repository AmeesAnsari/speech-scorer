"""Load word-level transcripts and turn text into comparable tokens."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

from arabic_toolkit import is_arabic, normalize_loose, tokenize


@dataclass(frozen=True)
class Word:
    text: str  # normalised token used for comparison
    raw: str  # as transcribed
    start: float
    end: float
    probability: Optional[float] = None


def make_normalizer(reference: str) -> Callable[[str], str]:
    """Arabic: fold diacritics, alef/ya/ta marbuta/hamza variants. Other scripts: lower-case."""
    if is_arabic(reference):
        return normalize_loose
    return lambda text: re.sub(r"\s+", " ", text.lower()).strip()


def tokens(text: str, normalizer: Callable[[str], str]) -> list[str]:
    return tokenize(normalizer(text))


def load_words(source: Union[str, Path, dict[str, Any]], normalizer: Callable[[str], str]) -> list[Word]:
    """Read a Whisper / faster-whisper style transcript with word timestamps.

    Accepts ``{"segments": [{"words": [...]}, ...]}`` or ``{"words": [...]}``, where each word
    has ``word`` (or ``text``), ``start``, ``end`` and optionally ``probability``.
    A transcribed "word" can hold punctuation or several tokens; each token keeps its word's timing.
    """
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    if "words" in data:
        items = data["words"]
    else:
        items = [w for seg in data.get("segments", []) for w in seg.get("words") or []]
    if not items:
        raise ValueError("Transcript has no word-level timestamps (enable word_timestamps when transcribing).")

    words: list[Word] = []
    for item in items:
        raw = str(item.get("word", item.get("text", ""))).strip()
        for token in tokens(raw, normalizer):
            words.append(Word(token, raw, float(item["start"]), float(item["end"]), item.get("probability")))
    words.sort(key=lambda w: w.start)
    return words
