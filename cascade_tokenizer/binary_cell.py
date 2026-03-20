"""
BinaryCell V2 — Raw evidence storage for 6-1-6 anchor neighborhoods.

Design principles:
  - 12 fixed positional buckets (before_6..before_1, after_1..after_6)
  - Symbol-first identity (hex address is the primary key, text is metadata)
  - Full neighbor symbol hex stored in each entry (no hash degradation)
  - Metadata header: status, category, source_pool, tone_signature
  - NO derived stats in the cell (IDF/rank live in CorpusStats side index)
  - Bloom-gated MasterIndex for O(1) anchor existence checks
  - Batch write-once, memory-mapped reads
  - CRC32 verified on every read

Binary neighbor entry format:
  sym_len:  1 byte
  sym_hex:  variable (full hex address, up to 20 bytes)
  count:    4 bytes  (uint32)
  tone_sig: 2 bytes  (uint16)
"""

import hashlib
import json
import math
import mmap
import struct
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

MAGIC_BYTES = 0xB1C3
MAX_BUCKET_ENTRIES = 65535

BUCKET_ORDER = [
    "before_6", "before_5", "before_4", "before_3", "before_2", "before_1",
    "after_1", "after_2", "after_3", "after_4", "after_5", "after_6",
]

# Status encoding
STATUS_MAP = {"ASSIGNED": 0, "TEMP_ASSIGNED": 1, "STRUCTURAL": 2, "AVAILABLE": 3}
STATUS_REVERSE = {0: "ASSIGNED", 1: "TEMP_ASSIGNED", 2: "STRUCTURAL", 3: "AVAILABLE"}

# Category encoding — explicit both directions, no sloppy overrides
CATEGORY_MAP = {"content": 0, "function": 1, "structural": 2, "punctuation": 3, "emoji": 4}
CATEGORY_REVERSE = {0: "content", 1: "function", 2: "structural", 3: "punctuation", 4: "emoji"}

# Source pool encoding
POOL_MAP = {"canonical": 0, "medical": 1, "structural": 2, "temp_pool": 3}
POOL_REVERSE = {0: "canonical", 1: "medical", 2: "structural", 3: "temp_pool"}


# ── Neighbor entry ────────────────────────────────────────────

@dataclass
class NeighborEntry:
    """A single neighbor in a positional bucket.

    Full symbol hex is stored — no hash degradation.
    """
    neighbor_symbol: str       # full hex address, stored in binary
    neighbor_word: str = ""    # display text, stored alongside for convenience
    count: int = 0
    tone_sig: int = 0

    def to_bytes(self) -> bytes:
        sym_bytes = self.neighbor_symbol.encode("utf-8")
        word_bytes = self.neighbor_word.encode("utf-8")[:64]
        return (
            struct.pack(">B", len(sym_bytes)) + sym_bytes
            + struct.pack(">B", len(word_bytes)) + word_bytes
            + struct.pack(">IH", min(self.count, 0xFFFFFFFF), min(self.tone_sig, 0xFFFF))
        )

    @classmethod
    def from_bytes(cls, data: bytes, pos: int) -> Tuple["NeighborEntry", int]:
        sym_len = data[pos]; pos += 1
        neighbor_symbol = data[pos:pos + sym_len].decode("utf-8"); pos += sym_len
        word_len = data[pos]; pos += 1
        neighbor_word = data[pos:pos + word_len].decode("utf-8"); pos += word_len
        count, tone = struct.unpack(">IH", data[pos:pos + 6]); pos += 6
        return cls(neighbor_symbol=neighbor_symbol, neighbor_word=neighbor_word,
                   count=count, tone_sig=tone), pos


# ── Binary Cell (one per anchor) ──────────────────────────────

class BinaryCellV2:
    """Raw evidence cell for one anchor in the 6-1-6 structure.

    Accumulation uses dict-per-bucket (O(1) neighbor lookup).
    Serialization freezes dicts into sorted lists.
    """

    __slots__ = (
        "symbol_hex", "display_text", "status", "category",
        "source_pool", "tone_signature", "total_count", "_bucket_maps",
    )

    def __init__(
        self,
        symbol_hex: str,
        display_text: str,
        status: str = "ASSIGNED",
        category: str = "content",
        source_pool: str = "canonical",
        tone_signature: int = 0,
        total_count: int = 0,
    ):
        self.symbol_hex = symbol_hex
        self.display_text = display_text
        self.status = status
        self.category = category
        self.source_pool = source_pool
        self.tone_signature = tone_signature
        self.total_count = total_count
        # Dict keyed by neighbor_symbol → NeighborEntry (O(1) accumulation)
        self._bucket_maps: Dict[str, Dict[str, NeighborEntry]] = {
            b: {} for b in BUCKET_ORDER
        }

    def add_neighbor(self, position: int, neighbor_symbol: str,
                     neighbor_word: str = "", count: int = 1, tone_sig: int = 0):
        """Add/accumulate a neighbor. O(1) per call."""
        if position == 0 or abs(position) > 6:
            return
        bucket_name = f"{'before' if position < 0 else 'after'}_{abs(position)}"
        bmap = self._bucket_maps[bucket_name]
        existing = bmap.get(neighbor_symbol)
        if existing is not None:
            existing.count += count
        else:
            bmap[neighbor_symbol] = NeighborEntry(
                neighbor_symbol=neighbor_symbol,
                neighbor_word=neighbor_word,
                count=count,
                tone_sig=tone_sig,
            )

    def get_bucket(self, position: int) -> List[NeighborEntry]:
        """Get neighbors at signed position, sorted by count desc."""
        bucket_name = f"{'before' if position < 0 else 'after'}_{abs(position)}"
        entries = list(self._bucket_maps.get(bucket_name, {}).values())
        entries.sort(key=lambda e: e.count, reverse=True)
        return entries

    @property
    def buckets(self) -> Dict[str, List[NeighborEntry]]:
        """Frozen view: dict values as sorted lists. For serialization/iteration."""
        return {
            name: sorted(bmap.values(), key=lambda e: e.count, reverse=True)
            for name, bmap in self._bucket_maps.items()
        }

    def total_co_occurrence_weight(self) -> int:
        """Sum of all neighbor counts. This is NOT total_count."""
        return sum(
            e.count
            for bmap in self._bucket_maps.values()
            for e in bmap.values()
        )

    def to_bytes(self) -> bytes:
        """Serialize. Freezes dicts into sorted lists. CRC32 appended."""
        sym_bytes = self.symbol_hex.encode("utf-8")
        disp_bytes = self.display_text.encode("utf-8")[:128]

        buf = struct.pack(">H", MAGIC_BYTES)
        buf += struct.pack(">B", len(sym_bytes)) + sym_bytes
        buf += struct.pack(">B", len(disp_bytes)) + disp_bytes
        buf += struct.pack(
            ">BBBI",
            STATUS_MAP.get(self.status, 0),
            CATEGORY_MAP.get(self.category, 0),
            POOL_MAP.get(self.source_pool, 0),
            self.tone_signature,
        )
        buf += struct.pack(">I", self.total_count)

        for bucket_name in BUCKET_ORDER:
            entries = sorted(
                self._bucket_maps.get(bucket_name, {}).values(),
                key=lambda e: e.count, reverse=True,
            )
            n = min(len(entries), MAX_BUCKET_ENTRIES)
            buf += struct.pack(">H", n)
            for entry in entries[:n]:
                buf += entry.to_bytes()

        buf += struct.pack(">I", zlib.crc32(buf))
        return buf

    @classmethod
    def from_bytes(cls, data: bytes) -> "BinaryCellV2":
        """Deserialize. Verifies CRC32 before parsing."""
        if len(data) < 6:
            raise ValueError("Data too short")

        # Verify checksum: last 4 bytes are CRC32 of everything before
        stored_crc = struct.unpack(">I", data[-4:])[0]
        computed_crc = zlib.crc32(data[:-4])
        if stored_crc != computed_crc:
            raise ValueError(
                f"CRC32 mismatch: stored={stored_crc:#010x}, "
                f"computed={computed_crc:#010x}"
            )

        pos = 0
        magic = struct.unpack(">H", data[pos:pos + 2])[0]; pos += 2
        if magic != MAGIC_BYTES:
            raise ValueError(f"Bad magic: {magic:#06x}")

        sym_len = data[pos]; pos += 1
        symbol_hex = data[pos:pos + sym_len].decode("utf-8"); pos += sym_len

        disp_len = data[pos]; pos += 1
        display_text = data[pos:pos + disp_len].decode("utf-8"); pos += disp_len

        status_b, cat_b, pool_b, tone_sig = struct.unpack(">BBBI", data[pos:pos + 7]); pos += 7
        total_count = struct.unpack(">I", data[pos:pos + 4])[0]; pos += 4

        cell = cls(
            symbol_hex=symbol_hex, display_text=display_text,
            status=STATUS_REVERSE.get(status_b, "ASSIGNED"),
            category=CATEGORY_REVERSE.get(cat_b, "content"),
            source_pool=POOL_REVERSE.get(pool_b, "canonical"),
            tone_signature=tone_sig, total_count=total_count,
        )

        for bucket_name in BUCKET_ORDER:
            entry_count = struct.unpack(">H", data[pos:pos + 2])[0]; pos += 2
            bmap = {}
            for _ in range(entry_count):
                entry, pos = NeighborEntry.from_bytes(data, pos)
                bmap[entry.neighbor_symbol] = entry
            cell._bucket_maps[bucket_name] = bmap

        return cell


# ── Master Index (bloom-gated hash index) ─────────────────────

class MasterIndex:
    """In-memory hash index with bloom filter for fast anchor lookup.

    Not a B+Tree. It is:
      - bloom filter for O(1) miss checks (3 hash functions)
      - hash map for O(1) hit lookups
      - symbol_hex -> (data_offset, data_length) for mmap reads
    """

    def __init__(self, bloom_size_bytes: int = 1_000_000):
        self.bloom = bytearray(bloom_size_bytes)
        self._bloom_bits = bloom_size_bytes * 8
        self._index: Dict[str, Tuple[int, int]] = {}  # symbol_hex -> (data_offset, data_length)

    def _bloom_hashes(self, key: str) -> List[int]:
        h = hashlib.sha256(key.encode("utf-8")).digest()
        return [
            int.from_bytes(h[0:4], "big") % self._bloom_bits,
            int.from_bytes(h[4:8], "big") % self._bloom_bits,
            int.from_bytes(h[8:12], "big") % self._bloom_bits,
        ]

    def add(self, symbol_hex: str, data_offset: int, data_length: int):
        for bit in self._bloom_hashes(symbol_hex):
            self.bloom[bit // 8] |= (1 << (bit % 8))
        self._index[symbol_hex] = (data_offset, data_length)

    def might_contain(self, symbol_hex: str) -> bool:
        for bit in self._bloom_hashes(symbol_hex):
            if not (self.bloom[bit // 8] & (1 << (bit % 8))):
                return False
        return True

    def get(self, symbol_hex: str) -> Optional[Tuple[int, int]]:
        """Returns (data_offset, data_length) or None."""
        if not self.might_contain(symbol_hex):
            return None
        return self._index.get(symbol_hex)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, symbol_hex: str) -> bool:
        return self.get(symbol_hex) is not None


# ── Corpus Stats (side index — derived, NOT raw truth) ────────

@dataclass
class CorpusTermStats:
    symbol_hex: str
    word: str
    df: int = 0
    total_weight: int = 0
    idf: float = 0.0
    category: str = "content"


class CorpusStats:
    """Corpus-level derived statistics. Separate from raw evidence cells."""

    def __init__(self):
        self.term_stats: Dict[str, CorpusTermStats] = {}
        self.n_docs: int = 0

    def record_document(self, cells: Dict[str, BinaryCellV2]):
        self.n_docs += 1
        seen: set = set()

        for sym, cell in cells.items():
            if sym not in self.term_stats:
                self.term_stats[sym] = CorpusTermStats(
                    symbol_hex=sym, word=cell.display_text, category=cell.category)
            ts = self.term_stats[sym]
            ts.total_weight += cell.total_co_occurrence_weight()
            if sym not in seen:
                ts.df += 1
                seen.add(sym)

            for bmap in cell._bucket_maps.values():
                for entry in bmap.values():
                    nsym = entry.neighbor_symbol
                    if nsym not in self.term_stats:
                        self.term_stats[nsym] = CorpusTermStats(
                            symbol_hex=nsym, word=entry.neighbor_word)
                    nts = self.term_stats[nsym]
                    nts.total_weight += entry.count
                    if nsym not in seen:
                        nts.df += 1
                        seen.add(nsym)

    def compute_idf(self):
        for ts in self.term_stats.values():
            if ts.df > 0 and self.n_docs > 0:
                ts.idf = math.log((self.n_docs + 1.0) / (ts.df + 1.0)) + 1.0
            else:
                ts.idf = 1.0

    def get_idf(self, symbol_hex: str) -> float:
        ts = self.term_stats.get(symbol_hex)
        return ts.idf if ts else 1.0

    def summary(self) -> Dict[str, Any]:
        return {"n_docs": self.n_docs, "n_terms": len(self.term_stats)}


# ── Cell Store (write-once, read-many) ────────────────────────

class CellStore:
    """Write cells to a binary file, read with bloom-gated mmap.

    Offset semantics:
      - data_offset points directly to the cell bytes (past the length prefix)
      - data_length is the cell byte count
      - read does mmap[data_offset : data_offset + data_length]
      - no +4 adjustment needed at read time
    """

    def __init__(self, path: str):
        self.path = path
        self.index = MasterIndex()
        self._mmap = None
        self._file = None

    def write_all(self, cells: Dict[str, BinaryCellV2]):
        """Write all cells to disk in one pass."""
        with open(self.path, "wb") as f:
            f.write(struct.pack(">I", len(cells)))

            for symbol_hex, cell in cells.items():
                cell_bytes = cell.to_bytes()
                # Write length prefix
                f.write(struct.pack(">I", len(cell_bytes)))
                # Record data_offset AFTER the length prefix
                data_offset = f.tell()
                f.write(cell_bytes)
                self.index.add(symbol_hex, data_offset, len(cell_bytes))

        self._save_index()

    def _save_index(self):
        idx_path = self.path + ".idx"
        data = {
            "entries": {
                sym: {"offset": off, "length": ln}
                for sym, (off, ln) in self.index._index.items()
            },
            "bloom_hex": self.index.bloom.hex(),
            "bloom_size": len(self.index.bloom),
        }
        with open(idx_path, "w") as f:
            json.dump(data, f)

    def load_index(self):
        idx_path = self.path + ".idx"
        with open(idx_path) as f:
            data = json.load(f)

        bloom_bytes = bytes.fromhex(data["bloom_hex"])
        self.index = MasterIndex(bloom_size_bytes=len(bloom_bytes))
        self.index.bloom = bytearray(bloom_bytes)
        self.index._bloom_bits = len(bloom_bytes) * 8
        for sym, info in data["entries"].items():
            self.index._index[sym] = (info["offset"], info["length"])

        self._file = open(self.path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def read_cell(self, symbol_hex: str) -> Optional[BinaryCellV2]:
        """Read a single cell by symbol hex. Bloom-gated. CRC-verified."""
        loc = self.index.get(symbol_hex)
        if loc is None:
            return None
        data_offset, data_length = loc
        raw = self._mmap[data_offset: data_offset + data_length]
        return BinaryCellV2.from_bytes(raw)

    def close(self):
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, symbol_hex: str) -> bool:
        return symbol_hex in self.index
