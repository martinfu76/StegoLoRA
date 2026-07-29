"""Synthetic carrier helpers used when no real watermark corpus is supplied.

The production experiment path uses ``hash_watermark.py`` and ``mcp_server.py``.
This deterministic word-position scheme keeps dataset and evaluator smoke tests
independent of model downloads.
"""
import random
from typing import List


_FILLER_WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "and", "runs", "through", "forest", "with", "great", "speed",
    "while", "birds", "sing", "in", "trees", "above", "softly",
    "today", "weather", "is", "quite", "pleasant", "for", "walk",
    "people", "often", "find", "peaceful", "moments", "in", "nature",
    "shadows", "drift", "across", "open", "fields", "at", "dusk",
]


def extract_message(stego_text: str, message_length: int = 8, word_step: int = 5) -> str:
    """Read the first letter of every ``word_step``-th word."""
    words = stego_text.split()
    if not words:
        return ""
    chars: List[str] = []
    for i in range(message_length):
        idx = i * word_step
        if idx < len(words) and words[idx]:
            chars.append(words[idx][0])
        else:
            chars.append("?")
    return "".join(chars)


def generate_stego_text(message: str, word_step: int = 5, rng: random.Random = None) -> str:
    """Create synthetic text that round-trips through ``extract_message``."""
    if rng is None:
        rng = random.Random()
    words: List[str] = []
    for i, ch in enumerate(message):
        while len(words) < i * word_step:
            words.append(rng.choice(_FILLER_WORDS))
        filler = rng.choice(_FILLER_WORDS)
        words.append(ch.upper() + filler[1:] if filler else ch.upper())
    while len(words) < (len(message) + 1) * word_step:
        words.append(rng.choice(_FILLER_WORDS))
    return " ".join(words)
