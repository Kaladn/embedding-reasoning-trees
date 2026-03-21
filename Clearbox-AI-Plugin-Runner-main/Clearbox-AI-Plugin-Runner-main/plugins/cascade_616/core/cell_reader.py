"""
Binary cell reader — reads evidence.bin stores.

Self-contained. No external imports beyond stdlib.
This is the read-only subset of binary_cell.py, bundled
so the plugin works without cascade_tokenizer on sys.path.
"""

import hashlib
import json
import mmap
import struct
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

MAGIC_BYTES = 0xB1C3

BUCKET_ORDER = [
    "before_6", "before_5", "before_4", "before_3", "before_2", "before_1",
    "after_1", "after_2", "after_3", "after_4", "after_5", "after_6",
]

STATUS_REVERSE = {0: "ASSIGNED", 1: "TEMP_ASSIGNED", 2: "STRUCTURAL", 3: "AVAILABLE"}
CATEGORY_REVERSE = {0: "content", 1: "function", 2: "structural", 3: "punctuation", 4: "emoji"}
POOL_REVERSE = {0: "canonical", 1: "medical", 2: "structural", 3: "temp_pool"}


@dataclass
class NeighborEntry:
    neighbor_symbol: str
    neighbor_word: str = ""
    count: int = 0
    tone_sig: int = 0

    @classmethod
    def from_bytes(cls, data: bytes, pos: int) -> Tuple["NeighborEntry", int]:
        sym_len = data[pos]; pos += 1
        neighbor_symbol = data[pos:pos + sym_len].decode("utf-8"); pos += sym_len
        word_len = data[pos]; pos += 1
        neighbor_word = data[pos:pos + word_len].decode("utf-8"); pos += word_len
        count, tone = struct.unpack(">IH", data[pos:pos + 6]); pos += 6
        return cls(neighbor_symbol=neighbor_symbol, neighbor_word=neighbor_word,
                   count=count, tone_sig=tone), pos


class BinaryCell:
    """Read-only binary cell. Deserialized from evidence store."""

    __slots__ = (
        "symbol_hex", "display_text", "status", "category",
        "source_pool", "tone_signature", "total_count", "_buckets",
    )

    def __init__(self, symbol_hex, display_text, status, category,
                 source_pool, tone_signature, total_count, buckets):
        self.symbol_hex = symbol_hex
        self.display_text = display_text
        self.status = status
        self.category = category
        self.source_pool = source_pool
        self.tone_signature = tone_signature
        self.total_count = total_count
        self._buckets = buckets

    def get_bucket(self, position: int) -> List[NeighborEntry]:
        bucket_name = f"{'before' if position < 0 else 'after'}_{abs(position)}"
        entries = list(self._buckets.get(bucket_name, {}).values())
        entries.sort(key=lambda e: e.count, reverse=True)
        return entries

    @classmethod
    def from_bytes(cls, data: bytes) -> "BinaryCell":
        if len(data) < 6:
            raise ValueError("Data too short")

        stored_crc = struct.unpack(">I", data[-4:])[0]
        computed_crc = zlib.crc32(data[:-4])
        if stored_crc != computed_crc:
            raise ValueError(f"CRC32 mismatch: {stored_crc:#010x} vs {computed_crc:#010x}")

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

        buckets = {}
        for bucket_name in BUCKET_ORDER:
            entry_count = struct.unpack(">H", data[pos:pos + 2])[0]; pos += 2
            bmap = {}
            for _ in range(entry_count):
                entry, pos = NeighborEntry.from_bytes(data, pos)
                bmap[entry.neighbor_symbol] = entry
            buckets[bucket_name] = bmap

        return cls(
            symbol_hex=symbol_hex, display_text=display_text,
            status=STATUS_REVERSE.get(status_b, "ASSIGNED"),
            category=CATEGORY_REVERSE.get(cat_b, "content"),
            source_pool=POOL_REVERSE.get(pool_b, "canonical"),
            tone_signature=tone_sig, total_count=total_count,
            buckets=buckets,
        )


class MasterIndex:
    def __init__(self, bloom_size_bytes=1_000_000):
        self.bloom = bytearray(bloom_size_bytes)
        self._bloom_bits = bloom_size_bytes * 8
        self._index: Dict[str, Tuple[int, int]] = {}

    def _bloom_hashes(self, key: str) -> List[int]:
        h = hashlib.sha256(key.encode("utf-8")).digest()
        return [
            int.from_bytes(h[0:4], "big") % self._bloom_bits,
            int.from_bytes(h[4:8], "big") % self._bloom_bits,
            int.from_bytes(h[8:12], "big") % self._bloom_bits,
        ]

    def might_contain(self, key: str) -> bool:
        for bit in self._bloom_hashes(key):
            if not (self.bloom[bit // 8] & (1 << (bit % 8))):
                return False
        return True

    def get(self, key: str) -> Optional[Tuple[int, int]]:
        if not self.might_contain(key):
            return None
        return self._index.get(key)

    def __len__(self):
        return len(self._index)


class CellStore:
    """Read-only evidence store. Loads index, mmap-reads cells."""

    def __init__(self, path: str):
        self.path = path
        self.index = MasterIndex()
        self._mmap = None
        self._file = None
        self._word_to_hex: Dict[str, str] = {}
        self._hex_to_word: Dict[str, str] = {}

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

        self._word_to_hex = data.get("word_to_hex", {})
        self._hex_to_word = {v: k for k, v in self._word_to_hex.items()}

        self._file = open(self.path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def resolve_word(self, word: str) -> Optional[str]:
        return self._word_to_hex.get(word.lower())

    def resolve_hex(self, symbol_hex: str) -> Optional[str]:
        return self._hex_to_word.get(symbol_hex)

    def read_cell(self, symbol_hex: str) -> Optional[BinaryCell]:
        loc = self.index.get(symbol_hex)
        if loc is None:
            return None
        offset, length = loc
        raw = self._mmap[offset: offset + length]
        return BinaryCell.from_bytes(raw)

    def close(self):
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()

    def __len__(self):
        return len(self.index)
