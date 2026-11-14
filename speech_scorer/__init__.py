"""Score a timed speech transcript against the script it should follow."""

from .scoring import Report, Settings, score
from .transcript import Word, load_words, make_normalizer

__version__ = "0.1.0"
__all__ = ["score", "Settings", "Report", "load_words", "make_normalizer", "Word"]
