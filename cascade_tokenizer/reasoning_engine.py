"""
6-1-6 Mapping Engine — tokenize, window, count.

No storage. No ranking. No queries. Just the mapping step.
Storage is in binary_cell.py. Predictions are in predict.py.
"""

import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# ── Suppression policy ────────────────────────────────────────

SUPPRESSED_ANCHOR_CLASSES: frozenset = frozenset({
    "function", "structural", "punctuation", "emoji",
})

# ── Content-only tokenizer ────────────────────────────────────

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0000FE00-\U0000FE0F"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF\U0000200D\U00002B50\U000023F0-\U000023FA"
    "]+", re.UNICODE
)
_WORD_PAT = re.compile(r"\b\w+\b|[^\w\s]")
_PUNCT_CHARS = frozenset(
    ".,;:!?\"'`()[]{}/<>@#$%^&*+-=~|\\\u2018\u2019\u201c\u201d\u2014\u2013\u2026"
)
_NUMBER_RE = re.compile(r"^\d+$")
_MIN_WORD_LEN = 2


def content_tokenize(text: str) -> List[str]:
    """Content words only. Punctuation, emoji, numbers, single chars skipped."""
    text = _EMOJI_RE.sub(" ", text)
    tokens = _WORD_PAT.findall(text.lower().strip())
    out = []
    for t in tokens:
        if not t.strip():
            continue
        if len(t) < _MIN_WORD_LEN:
            continue
        if all(c in _PUNCT_CHARS for c in t):
            continue
        if _NUMBER_RE.match(t):
            continue
        if any(0x2300 <= ord(c) <= 0x2BFF for c in t):
            continue
        out.append(t)
    return out


# ── Window counting ───────────────────────────────────────────

def compute_window_counts(
    tokens: List[str], window: int = 6,
) -> Tuple[Dict[str, Dict[int, Counter]], Dict[str, Dict[int, Counter]]]:
    """Compute 6-1-6 co-occurrence counts from a token sequence.

    Returns (before_map, after_map) where each is:
        {anchor_word: {distance: Counter({neighbor: count})}}

    Paragraph splitting is the caller's responsibility.
    """
    before = defaultdict(lambda: defaultdict(Counter))
    after = defaultdict(lambda: defaultdict(Counter))
    length = len(tokens)

    for i, focus in enumerate(tokens):
        for offset in range(1, window + 1):
            bp = i - offset
            if bp < 0:
                break
            before[focus][offset][tokens[bp]] += 1
        for offset in range(1, window + 1):
            ap = i + offset
            if ap >= length:
                break
            after[focus][offset][tokens[ap]] += 1

    return before, after


# ── Map Report ────────────────────────────────────────────────

@dataclass
class MapReport:
    """Result of mapping a text through the 6-1-6 window."""
    source: str
    window: int
    total_tokens: int
    content_tokens: int
    duration_ms: int
    before_map: Dict[str, Dict[int, Counter]] = field(default_factory=dict)
    after_map: Dict[str, Dict[int, Counter]] = field(default_factory=dict)


def map_text(text: str, source: str = "inline", window: int = 6) -> MapReport:
    """Map text into 6-1-6 co-occurrence structure.

    Paragraph boundary = double newline (wall).
    Content-only tokenization (punct/emoji/numbers transparent).
    """
    paragraphs = re.split(r"\n\s*\n", text)
    merged_before: Dict[str, Dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    merged_after: Dict[str, Dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    total_tokens = 0
    content_tokens = 0
    t0 = time.time()

    for para in paragraphs:
        tokens = content_tokenize(para)
        if not tokens:
            continue
        total_tokens += len(para.split())
        content_tokens += len(tokens)

        before, after = compute_window_counts(tokens, window)

        for anchor, dists in before.items():
            for d, counter in dists.items():
                merged_before[anchor][d] += counter
        for anchor, dists in after.items():
            for d, counter in dists.items():
                merged_after[anchor][d] += counter

    return MapReport(
        source=source,
        window=window,
        total_tokens=total_tokens,
        content_tokens=content_tokens,
        duration_ms=int((time.time() - t0) * 1000),
        before_map=dict(merged_before),
        after_map=dict(merged_after),
    )
