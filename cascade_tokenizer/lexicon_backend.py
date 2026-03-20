"""
LexiconBackend: Symbol-authority layer backed by the persistent lexicon.

Replaces the old build_vocabulary() approach with deterministic symbol lookup:
  1. Assigned symbol exists → use it (Canonical / Medical / Structural)
  2. No assigned symbol → pull next available temp slot from Temp_Pool
  3. Cascade / reasoning attaches to the symbol ID, not the raw word
  4. Raw word is metadata, not authority

Pools have distinct roles:
  - Canonical / Medical / Structural: permanent, governed inventory
  - Temp_Pool: runtime / session assignment lane (ephemeral until promoted)
  - Spare_Slots: reserved future permanent inventory -- never touched at runtime
"""

import json
import glob
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Symbol status enum
# ---------------------------------------------------------------------------

class SymbolStatus(Enum):
    ASSIGNED = "ASSIGNED"           # Permanent mapping in Canonical/Medical
    STRUCTURAL = "STRUCTURAL"       # Reserved structural symbol (digits, punct)
    TEMP_ASSIGNED = "TEMP_ASSIGNED" # Runtime temp mapping from Temp_Pool
    AVAILABLE = "AVAILABLE"         # Unassigned slot


# ---------------------------------------------------------------------------
# SymbolRecord: what the cascade engine consumes
# ---------------------------------------------------------------------------

@dataclass
class SymbolRecord:
    """Full symbol record returned by the lexicon for every resolved token."""
    symbol_id: str          # hex address -- the real identity
    hex: str                # same as symbol_id (canonical field name)
    binary: str             # 40/46-bit binary string
    status: SymbolStatus
    source_pool: str        # "canonical", "medical", "structural", "temp_pool"
    surface_word: str       # the word as it appeared in input text
    canonical_word: str     # the word stored in the lexicon (may differ in case)
    tone_signature: str     # e.g. "TONE_310"
    font_symbol: str        # e.g. "CHAR_0101011000"
    pack: str = ""          # original pack field from the JSON
    mapped_at: str = ""     # ISO timestamp of original or temp mapping
    display: str = ""       # display form if present
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_token_id(self) -> int:
        """Convert hex symbol to a deterministic integer token ID."""
        return int(self.hex, 16)


# ---------------------------------------------------------------------------
# LexiconBackend
# ---------------------------------------------------------------------------

class LexiconBackend:
    """
    Loads the full lexicon from disk and exposes match → temp-assign → emit
    as the sole vocabulary authority for the cascade system.
    """

    def __init__(self, lexicon_dir: str, load_medical: bool = True):
        self.lexicon_dir = Path(lexicon_dir)

        # Permanent lookup indexes  (word_lower → raw dict entry)
        self._canonical: Dict[str, dict] = {}
        self._medical: Dict[str, dict] = {}
        self._structural: Dict[str, dict] = {}

        # Reverse index: hex → raw dict entry  (for all assigned symbols)
        self._hex_index: Dict[str, dict] = {}

        # Temp pool: list of available raw entries (FIFO queue)
        self._temp_available: List[dict] = []
        self._temp_cursor: int = 0  # next index to hand out

        # Session temp assignments:  word_lower → SymbolRecord
        self._temp_assigned: Dict[str, SymbolRecord] = {}

        # Stats
        self._stats = {
            "canonical_lookups": 0,
            "medical_lookups": 0,
            "structural_lookups": 0,
            "temp_assignments": 0,
            "cache_hits": 0,
        }

        # Load
        self._load_lexicon(load_medical=load_medical)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_lexicon(self, load_medical: bool = True) -> None:
        """Load all pools from the lexicon directory."""

        # --- Canonical ---
        canonical_dir = self.lexicon_dir / "Canonical"
        if canonical_dir.exists():
            for fp in sorted(canonical_dir.glob("*.json")):
                for entry in json.loads(fp.read_text(encoding="utf-8")):
                    word = entry.get("word", "")
                    if word:
                        key = word.lower()
                        self._canonical[key] = entry
                        self._hex_index[entry.get("hex", entry.get("symbol", ""))] = entry
            logger.info(f"Canonical: {len(self._canonical):,} words loaded")

        # --- Medical ---
        if load_medical:
            medical_dir = self.lexicon_dir / "Medical"
            if medical_dir.exists():
                for fp in sorted(medical_dir.glob("*.json")):
                    for entry in json.loads(fp.read_text(encoding="utf-8")):
                        word = entry.get("word", "")
                        if word:
                            key = word.lower()
                            # Canonical takes precedence on collisions
                            if key not in self._canonical:
                                self._medical[key] = entry
                            self._hex_index[entry.get("hex", entry.get("symbol", ""))] = entry
                logger.info(f"Medical: {len(self._medical):,} words loaded")

        # --- Structural ---
        structural_dir = self.lexicon_dir / "Structural"
        if structural_dir.exists():
            for fp in sorted(structural_dir.glob("*.json")):
                for entry in json.loads(fp.read_text(encoding="utf-8")):
                    word = entry.get("word", "")
                    self._structural[word] = entry
                    self._hex_index[entry.get("hex", entry.get("symbol", ""))] = entry
            logger.info(f"Structural: {len(self._structural):,} entries loaded")

        # --- Temp_Pool (runtime lane only -- Spare_Slots intentionally excluded) ---
        temp_dir = self.lexicon_dir / "Temp_Pool"
        if temp_dir.exists():
            for fp in sorted(temp_dir.glob("*.json")):
                data = json.loads(fp.read_text(encoding="utf-8"))
                for entry in data:
                    if entry.get("status") == "AVAILABLE":
                        self._temp_available.append(entry)
            logger.info(f"Temp_Pool: {len(self._temp_available):,} slots loaded")

        total = len(self._canonical) + len(self._medical) + len(self._structural)
        logger.info(
            f"Lexicon ready: {total:,} assigned symbols, "
            f"{len(self._temp_available):,} temp slots available"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup_symbol(self, word: str) -> Optional[SymbolRecord]:
        """
        Look up a word in the permanent pools.
        Returns SymbolRecord if found, None otherwise.
        Does NOT create temp assignments.
        """
        key = word.lower()

        # 1. Structural (exact match -- single chars, digits)
        if word in self._structural:
            self._stats["structural_lookups"] += 1
            return self._make_record(self._structural[word], word, "structural")

        # 2. Canonical
        if key in self._canonical:
            self._stats["canonical_lookups"] += 1
            return self._make_record(self._canonical[key], word, "canonical")

        # 3. Medical
        if key in self._medical:
            self._stats["medical_lookups"] += 1
            return self._make_record(self._medical[key], word, "medical")

        return None

    def assign_temp_symbol(self, word: str, context: str = None) -> SymbolRecord:
        """
        Pull the next available Temp_Pool slot and assign it to *word*.
        Raises RuntimeError if the temp pool is exhausted.
        """
        key = word.lower()

        # Already temp-assigned this session?
        if key in self._temp_assigned:
            self._stats["cache_hits"] += 1
            return self._temp_assigned[key]

        if self._temp_cursor >= len(self._temp_available):
            raise RuntimeError(
                f"Temp_Pool exhausted ({self._temp_cursor:,} slots used). "
                "Cannot assign more temporary symbols this session."
            )

        slot = self._temp_available[self._temp_cursor]
        self._temp_cursor += 1

        record = SymbolRecord(
            symbol_id=slot.get("hex", slot.get("symbol", "")),
            hex=slot.get("hex", slot.get("symbol", "")),
            binary=slot.get("binary", ""),
            status=SymbolStatus.TEMP_ASSIGNED,
            source_pool="temp_pool",
            surface_word=word,
            canonical_word=key,
            tone_signature=slot.get("tone_signature", ""),
            font_symbol=slot.get("font_symbol", ""),
            pack="temp_pool",
            mapped_at=datetime.now(timezone.utc).isoformat(),
            display=word,
            metadata={"context": context} if context else {},
        )

        self._temp_assigned[key] = record
        self._stats["temp_assignments"] += 1
        return record

    def resolve_token(self, word: str, context: str = None) -> SymbolRecord:
        """
        The main entry point. Returns a SymbolRecord for any word:
          - permanent match → return it
          - session temp hit → return cached temp record
          - miss → assign next temp slot
        """
        # Fast path: already temp-assigned this session
        key = word.lower()
        if key in self._temp_assigned:
            self._stats["cache_hits"] += 1
            return self._temp_assigned[key]

        # Permanent lookup
        record = self.lookup_symbol(word)
        if record is not None:
            return record

        # Temp assign
        return self.assign_temp_symbol(word, context=context)

    def resolve_sequence(self, words: List[str], context: str = None) -> List[SymbolRecord]:
        """Resolve a list of words to symbol records."""
        return [self.resolve_token(w, context=context) for w in words]

    def promote_temp_symbol(
        self, word: str, target_pool: str = "canonical"
    ) -> Optional[SymbolRecord]:
        """
        Promote a temp-assigned symbol to a governed queue for permanent assignment.

        This does NOT write to the lexicon JSON files directly -- it returns the
        record marked for promotion so the caller (or a governance pipeline) can
        persist it. Keeping writes out of the hot path.
        """
        key = word.lower()
        if key not in self._temp_assigned:
            return None

        record = self._temp_assigned[key]
        record.metadata["promoted_to"] = target_pool
        record.metadata["promoted_at"] = datetime.now(timezone.utc).isoformat()
        return record

    def serialize_temp_registry(self) -> List[dict]:
        """
        Serialize all session temp assignments for persistence / audit.
        Returns a list of dicts ready for json.dump().
        """
        out = []
        for record in self._temp_assigned.values():
            d = asdict(record)
            d["status"] = d["status"].value  # Enum → string
            out.append(d)
        return out

    def save_temp_registry(self, filepath: str) -> None:
        """Write the temp registry to a JSON file."""
        data = self.serialize_temp_registry()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Temp registry saved: {len(data)} entries → {filepath}")

    def stats(self) -> Dict[str, Any]:
        """Return lookup / assignment statistics."""
        return {
            **self._stats,
            "permanent_symbols": len(self._canonical) + len(self._medical) + len(self._structural),
            "canonical_count": len(self._canonical),
            "medical_count": len(self._medical),
            "structural_count": len(self._structural),
            "temp_pool_total": len(self._temp_available),
            "temp_pool_used": self._temp_cursor,
            "temp_pool_remaining": len(self._temp_available) - self._temp_cursor,
            "session_temp_assigned": len(self._temp_assigned),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_record(self, entry: dict, surface_word: str, source_pool: str) -> SymbolRecord:
        """Build a SymbolRecord from a raw lexicon JSON entry."""
        status_str = entry.get("status", "ASSIGNED")
        try:
            status = SymbolStatus(status_str)
        except ValueError:
            status = SymbolStatus.ASSIGNED

        return SymbolRecord(
            symbol_id=entry.get("hex", entry.get("symbol", "")),
            hex=entry.get("hex", entry.get("symbol", "")),
            binary=entry.get("binary", ""),
            status=status,
            source_pool=source_pool,
            surface_word=surface_word,
            canonical_word=entry.get("word", surface_word),
            tone_signature=entry.get("tone_signature", ""),
            font_symbol=entry.get("font_symbol", ""),
            pack=entry.get("pack", source_pool),
            mapped_at=entry.get("mapped_at", ""),
            display=entry.get("display", surface_word),
        )
