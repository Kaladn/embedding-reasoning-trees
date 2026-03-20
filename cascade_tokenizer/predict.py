"""
6-1-6 Prediction Engine — anchor → neighbors → chains → predictions.

No abstractions. No ranking math. No prose generation.
Reads binary cells, returns Top-K positional neighbors and chains.

Usage:
    from cascade_tokenizer.predict import Predictor
    p = Predictor("reasoning_cache/evidence.bin", "Lexical Data")
    p.predict_next("ai", k=5)
    p.forward_chain("ai", steps=6)
    p.backward_chain("ai", steps=6)
"""

from typing import Dict, List, Optional, Tuple
from cascade_tokenizer.binary_cell import CellStore, BinaryCellV2

# ── Hardcoded stoplist — the real gate, not lexicon tags ──────
# These are never anchors, never chain links, never predictions.
# They exist in the cell store as neighbors (raw truth preserved)
# but the prediction engine skips them.

STOP_WORDS: frozenset = frozenset({
    # determiners / articles
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    # prepositions
    "of", "in", "to", "for", "with", "on", "at", "from", "by",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "about", "against", "along", "around",
    "behind", "beneath", "beside", "beyond", "down", "except", "inside",
    "near", "off", "onto", "outside", "past", "since", "toward",
    "towards", "until", "upon", "within", "without", "up",
    # conjunctions
    "and", "but", "or", "nor", "so", "yet", "both", "either",
    "neither", "whether",
    # pronouns
    "it", "he", "him", "she", "her", "we", "us", "you", "they", "them",
    "who", "whom", "whose", "which", "what", "whoever", "whatever",
    "me", "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "themselves",
    # aux / modal / be / have / do
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "shall", "should", "can", "could", "may",
    "might", "must",
    # other function
    "not", "no", "if", "then", "else", "when", "where", "how",
    "why", "while", "as", "than", "too", "also", "very", "just",
    "only", "even", "still", "already", "there", "here", "now",
    # code/structural noise
    "self", "re", "like", "such", "each", "every", "any", "all",
    "some", "most", "other", "more", "many", "much", "own",
})


class Predictor:
    """Reads binary cell store. Returns predictions and chains."""

    def __init__(self, store_path: str, lexicon_dir: str = None):
        self.store = CellStore(store_path)
        self.store.load_index()

        # Word → symbol lookup (for querying by word instead of hex)
        # Built from the cell store index + display text
        self._word_to_hex: Dict[str, str] = {}
        self._hex_to_word: Dict[str, str] = {}

    def _ensure_word_index(self):
        """Lazy-build word→hex index by scanning all cells."""
        if self._word_to_hex:
            return
        for sym in self.store.index._index:
            cell = self.store.read_cell(sym)
            if cell:
                word = cell.display_text.lower()
                self._word_to_hex[word] = sym
                self._hex_to_word[sym] = word

    def _resolve(self, anchor: str) -> Optional[BinaryCellV2]:
        """Resolve anchor (word or hex) to a cell."""
        # Try direct hex lookup
        cell = self.store.read_cell(anchor)
        if cell:
            return cell
        # Try word lookup
        self._ensure_word_index()
        sym = self._word_to_hex.get(anchor.lower())
        if sym:
            return self.store.read_cell(sym)
        return None

    def top_neighbors(self, anchor: str, position: int, k: int = 10,
                      skip_stops: bool = True) -> List[Tuple[str, int]]:
        """Top-K neighbors at a signed position (-6..-1, +1..+6).

        Returns [(word, count), ...] sorted by count desc.
        Stopwords filtered by default.
        """
        cell = self._resolve(anchor)
        if not cell:
            return []

        entries = cell.get_bucket(position)
        results = []
        for e in entries:
            word = e.neighbor_word.lower()
            if skip_stops and word in STOP_WORDS:
                continue
            results.append((word, e.count))
            if len(results) >= k:
                break
        return results

    def predict_next(self, anchor: str, k: int = 5) -> List[Tuple[str, int]]:
        """What comes after this anchor? Top-K at position +1."""
        return self.top_neighbors(anchor, +1, k)

    def predict_previous(self, anchor: str, k: int = 5) -> List[Tuple[str, int]]:
        """What comes before this anchor? Top-K at position -1."""
        return self.top_neighbors(anchor, -1, k)

    def full_context(self, anchor: str, k: int = 5,
                     skip_stops: bool = True) -> Dict[str, List[Tuple[str, int]]]:
        """All 12 positions, Top-K each. The complete 6-1-6 view."""
        result = {}
        for pos in range(-6, 7):
            if pos == 0:
                continue
            label = f"{'before' if pos < 0 else 'after'}_{abs(pos)}"
            neighbors = self.top_neighbors(anchor, pos, k, skip_stops)
            if neighbors:
                result[label] = neighbors
        return result

    def forward_chain(self, anchor: str, steps: int = 6,
                      strategy: str = "top1") -> List[Tuple[str, int]]:
        """Follow the chain forward: anchor → top at +1 → top at +1 → ...

        strategy="top1": always follow highest count neighbor
        Returns [(word, count), ...] for each step.
        """
        chain = []
        current = anchor.lower()
        seen = {current}

        for _ in range(steps):
            nexts = self.predict_next(current, k=5)
            if not nexts:
                break

            # Pick best unseen
            picked = None
            for word, count in nexts:
                if word not in seen:
                    picked = (word, count)
                    break

            if not picked:
                break

            chain.append(picked)
            seen.add(picked[0])
            current = picked[0]

        return chain

    def backward_chain(self, anchor: str, steps: int = 6,
                       strategy: str = "top1") -> List[Tuple[str, int]]:
        """Follow the chain backward: ... → top at -1 → anchor."""
        chain = []
        current = anchor.lower()
        seen = {current}

        for _ in range(steps):
            prevs = self.predict_previous(current, k=5)
            if not prevs:
                break

            picked = None
            for word, count in prevs:
                if word not in seen:
                    picked = (word, count)
                    break

            if not picked:
                break

            chain.append(picked)
            seen.add(picked[0])
            current = picked[0]

        return chain

    def stats(self) -> Dict[str, int]:
        return {"cells": len(self.store)}

    def close(self):
        self.store.close()
