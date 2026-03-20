"""
Cascade 6-1-6 Engine — Binary cell prediction engine.

Self-contained. No imports from other plugins.
Reads its own evidence.bin store and returns predictions/chains.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Stopword gate — the real filter ─────────────────────────
STOP_WORDS: frozenset = frozenset({
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "of", "in", "to", "for", "with", "on", "at", "from", "by",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "about", "against", "along", "around",
    "behind", "beneath", "beside", "beyond", "down", "except", "inside",
    "near", "off", "onto", "outside", "past", "since", "toward",
    "towards", "until", "upon", "within", "without", "up",
    "and", "but", "or", "nor", "so", "yet", "both", "either",
    "neither", "whether",
    "it", "he", "him", "she", "her", "we", "us", "you", "they", "them",
    "who", "whom", "whose", "which", "what", "whoever", "whatever",
    "me", "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "themselves",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "shall", "should", "can", "could", "may",
    "might", "must",
    "not", "no", "if", "then", "else", "when", "where", "how",
    "why", "while", "as", "than", "too", "also", "very", "just",
    "only", "even", "still", "already", "there", "here", "now",
    "self", "re", "like", "such", "each", "every", "any", "all",
    "some", "most", "other", "more", "many", "much", "own",
})


class CascadeEngine:
    """Self-contained 6-1-6 prediction engine. No external plugin deps."""

    def __init__(self, store_path: str = "", lexicon_path: str = ""):
        self._store = None
        self._store_path = store_path
        self._lexicon_path = lexicon_path
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        if not self._store_path:
            raise RuntimeError("store_path not configured")

        store_file = Path(self._store_path)
        if not store_file.exists():
            raise FileNotFoundError(f"Evidence store not found: {store_file}")

        # Import from the cascade_tokenizer package
        # Add the project root to sys.path if needed
        project_root = str(store_file.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from cascade_tokenizer.binary_cell import CellStore
        self._store = CellStore(str(store_file))
        self._store.load_index()
        self._loaded = True
        logger.info("Cascade engine loaded: %d cells from %s",
                     len(self._store), store_file)

    def _resolve(self, anchor: str):
        """Resolve word to binary cell."""
        self._ensure_loaded()
        cell = self._store.read_cell(anchor)
        if cell:
            return cell
        sym = self._store.resolve_word(anchor.lower())
        if sym:
            return self._store.read_cell(sym)
        return None

    def _word_for(self, sym: str) -> str:
        """Resolve hex symbol back to word."""
        return self._store.resolve_hex(sym) or sym

    def top_neighbors(self, anchor: str, position: int, k: int = 10,
                      skip_stops: bool = True) -> List[Dict]:
        """Top-K neighbors at signed position (-6..-1, +1..+6)."""
        cell = self._resolve(anchor)
        if not cell:
            return []
        entries = cell.get_bucket(position)
        results = []
        for e in entries:
            word = e.neighbor_word.lower()
            if skip_stops and word in STOP_WORDS:
                continue
            results.append({
                "token": word,
                "count": e.count,
                "symbol": e.neighbor_symbol,
                "position": position,
            })
            if len(results) >= k:
                break
        return results

    def predict_next(self, anchor: str, k: int = 5) -> List[Dict]:
        return self.top_neighbors(anchor, +1, k)

    def predict_previous(self, anchor: str, k: int = 5) -> List[Dict]:
        return self.top_neighbors(anchor, -1, k)

    def full_context(self, anchor: str, k: int = 5) -> Dict:
        """All 12 positions, Top-K each. The complete 6-1-6 view."""
        cell = self._resolve(anchor)
        if not cell:
            return {"error": "anchor not found", "anchor": anchor}

        context = {}
        for pos in range(-6, 7):
            if pos == 0:
                continue
            label = f"{'before' if pos < 0 else 'after'}_{abs(pos)}"
            neighbors = self.top_neighbors(anchor, pos, k)
            if neighbors:
                context[label] = neighbors
        return {
            "anchor": anchor,
            "symbol": cell.symbol_hex,
            "display": cell.display_text,
            "category": cell.category,
            "total_count": cell.total_count,
            "positions": context,
        }

    def forward_chain(self, anchor: str, steps: int = 6) -> List[Dict]:
        """Follow anchor → top at +1 → top at +1 → ..."""
        chain = []
        current = anchor.lower()
        seen = {current}

        for _ in range(steps):
            nexts = self.top_neighbors(current, +1, k=5)
            if not nexts:
                break
            picked = None
            for n in nexts:
                if n["token"] not in seen:
                    picked = n
                    break
            if not picked:
                break
            chain.append(picked)
            seen.add(picked["token"])
            current = picked["token"]
        return chain

    def backward_chain(self, anchor: str, steps: int = 6) -> List[Dict]:
        """Follow ... → top at -1 → anchor."""
        chain = []
        current = anchor.lower()
        seen = {current}

        for _ in range(steps):
            prevs = self.top_neighbors(current, -1, k=5)
            if not prevs:
                break
            picked = None
            for n in prevs:
                if n["token"] not in seen:
                    picked = n
                    break
            if not picked:
                break
            chain.append(picked)
            seen.add(picked["token"])
            current = picked["token"]
        return chain

    def stats(self) -> Dict:
        self._ensure_loaded()
        return {
            "cells": len(self._store),
            "store_path": self._store_path,
            "loaded": self._loaded,
        }
