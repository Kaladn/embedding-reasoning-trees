"""
Reasoning Engine — deterministic, non-LLM inference over 6-1-6 context maps.

Lexicon-backed from day one:
  - Symbol identity (hex) is primary, token text is display metadata
  - Function words counted as neighbors, never indexed as anchors
  - Category-aware filtering at index time, not just view time
  - Edge keys are (anchor_symbol, offset, neighbor_symbol)

Architecture:
  1. Map text → paragraph-local windowed co-occurrence counts
  2. Pack into 6-1-6 map structure (with category on each neighbor)
  3. Index anchors + edges into SQLite (term_stats, edge_scores)
  4. Query: "What does X do?" → AnswerFrame (subject, predicates, confidence)

Shares the same Lexical Data directory as the cascade tokenizer.
"""

import hashlib
import json
import logging
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from cascade_tokenizer.lexicon_backend import LexiconBackend, SymbolRecord, SymbolStatus

logger = logging.getLogger(__name__)


# ── Suppression policy (shared with bridge patch) ─────────────

SUPPRESSED_ANCHOR_CLASSES: frozenset = frozenset({"function", "structural", "punctuation", "emoji"})
"""Categories that are counted as neighbors but never indexed as anchors."""

# Single symbol for all emoji -- we don't care which one, context handles meaning
EMOJI_SYMBOL = "EMOJI_PLACEHOLDER"
EMOJI_SYMBOL_HEX = "0x00000000FFFF"

# Regex: emoji covers most Unicode emoji ranges
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000200D"             # zero width joiner
    "\U00002B50"             # star
    "\U000023F0-\U000023FA"  # misc technical
    "]+", re.UNICODE
)

# Punctuation / structural characters that get symbols but skip the window
_PUNCT_CHARS = frozenset(".,;:!?\"'`()[]{}/<>@#$%^&*+-=~|\\")
_NUMBER_RE = re.compile(r"^\d+$")


# ── Verb detection (deterministic, no ML) ─────────────────────

VERB_LEXICON: frozenset = frozenset({
    "absorb", "achieve", "act", "add", "affect", "allow", "appear", "apply",
    "become", "begin", "build", "call", "capture", "carry", "cause", "change",
    "come", "compare", "contain", "continue", "contribute", "control", "convert",
    "create", "define", "depend", "develop", "differ", "do", "drive",
    "enable", "enhance", "ensure", "establish", "exceed", "exist", "expand",
    "facilitate", "find", "focus", "form", "function", "generate", "give",
    "grow", "happen", "have", "help", "hold", "impact", "implement", "improve",
    "include", "increase", "indicate", "influence", "involve", "lead", "limit",
    "maintain", "make", "manage", "measure", "mitigate", "modify", "monitor",
    "need", "occur", "offer", "operate", "perform", "play", "prevent", "produce",
    "promote", "protect", "provide", "reach", "reduce", "regulate", "relate",
    "release", "remain", "remove", "represent", "require", "result", "retain",
    "sequester", "serve", "show", "store", "suggest", "support", "sustain",
    "take", "use", "vary", "work",
})

VERB_SUFFIXES = ("ed", "ing", "ify", "ise", "ize", "ate", "en")
NOUN_SUFFIXES = ("tion", "ment", "ness", "ity", "ship", "ism", "ance", "ence")


def is_verb_like(token: str) -> bool:
    """Deterministic verb-likeness check."""
    t = token.lower()
    if t in VERB_LEXICON:
        return True
    if any(t.endswith(s) for s in VERB_SUFFIXES):
        return True
    if any(t.endswith(s) for s in NOUN_SUFFIXES):
        return False
    return False


# ── Data models ───────────────────────────────────────────────

@dataclass
class AnswerFrame:
    """Structured answer — not text-first."""
    subject: str
    subject_symbol: Optional[str]
    predicates: List[Dict[str, Any]]
    confidence: float
    reasoning_trace: Dict[str, Any]
    surface_text: str


@dataclass
class MapReport:
    """Result of mapping a text through the 6-1-6 window."""
    source: str
    window: int
    total_tokens: int
    duration_ms: int
    items: Dict[str, Dict[str, Any]]
    stats: Dict[str, int] = field(default_factory=dict)


# ── Window counting (paragraph-local, same algorithm as bridge) ───

def compute_window_counts(
    token_ids: List[int],
    window: int = 6,
) -> Dict[int, Dict[Tuple[int, int], int]]:
    """Compute co-occurrence counts within +/- window around each token.

    Returns {offset: {(focus_id, ctx_id): count}}.
    Offset negative = before, positive = after.

    Boundary behavior:
      - Position 0 in paragraph: 0-1-N (N = min(window, len-1))
      - Position 2: 2-1-N (ramp up)
      - End of paragraph: N-1-0 (ramp down)
      - Short paragraphs (<13 tokens): partial windows, honest data
    """
    if not token_ids or window <= 0:
        return {}

    counts: Dict[int, Dict[Tuple[int, int], int]] = {}
    length = len(token_ids)

    for idx, focus in enumerate(token_ids):
        # Backward: -window .. -1
        for offset in range(1, window + 1):
            before_pos = idx - offset
            if before_pos < 0:
                break
            pair = (focus, token_ids[before_pos])
            counts.setdefault(-offset, {}).setdefault(pair, 0)
            counts[-offset][pair] += 1

        # Forward: +1 .. +window
        for offset in range(1, window + 1):
            after_pos = idx + offset
            if after_pos >= length:
                break
            pair = (focus, token_ids[after_pos])
            counts.setdefault(offset, {}).setdefault(pair, 0)
            counts[offset][pair] += 1

    return counts


# ── Reasoning Engine ──────────────────────────────────────────

class ReasoningEngine:
    """Deterministic inference engine over 6-1-6 maps, backed by the lexicon.

    Usage::

        engine = ReasoningEngine("Lexical Data", cache_dir="./reasoning_cache")
        report = engine.map_text("Carbon forests sequester carbon dioxide...")
        engine.index_map("doc_001", report)
        answer = engine.query("What does carbon do?")
    """

    def __init__(
        self,
        lexicon_dir: str,
        cache_dir: str = "./reasoning_cache",
        window: int = 6,
        top_k: int = 10,
        load_medical: bool = True,
    ):
        self.window = window
        self.top_k = top_k
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lexicon is the symbol authority
        self.lexicon = LexiconBackend(lexicon_dir, load_medical=load_medical)

        # Tokenization
        self._word_pattern = re.compile(r"\b\w+\b|[^\w\s]")

        # SQLite indexes
        self._term_stats_db = self.cache_dir / "term_stats.db"
        self._edge_scores_db = self.cache_dir / "edge_scores.db"
        self._manifest_db = self.cache_dir / "manifest.db"

        # In-memory symbol lookup: word_lower → (int_id, SymbolRecord)
        self._word_to_int: Dict[str, int] = {}
        self._int_to_word: Dict[int, str] = {}
        self._next_int = 0

        # Stored maps (for get_616 queries)
        self._maps: Dict[str, MapReport] = {}

        self._ensure_schema()

    # ── Schema ─────────────────────────────────────────────

    def _get_db(self, path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(path))
        conn.create_function("ln", 1, lambda x: math.log(x) if x > 0 else 0.0)
        return conn

    def _ensure_schema(self):
        with self._get_db(self._manifest_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS maps_manifest (
                    map_id TEXT PRIMARY KEY,
                    source TEXT,
                    total_tokens INTEGER,
                    indexed_utc TEXT,
                    sha256 TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY, value TEXT
                )
            """)

        with self._get_db(self._term_stats_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS term_stats (
                    term TEXT PRIMARY KEY,
                    symbol TEXT,
                    category TEXT,
                    df INTEGER DEFAULT 0,
                    total_co_occurrence_weight INTEGER DEFAULT 0,
                    idf REAL DEFAULT 0.0
                )
            """)

        with self._get_db(self._edge_scores_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_scores (
                    anchor TEXT,
                    anchor_symbol TEXT,
                    offset INTEGER,
                    neighbor TEXT,
                    neighbor_symbol TEXT,
                    neighbor_category TEXT,
                    raw_count INTEGER,
                    df_edge INTEGER DEFAULT 1,
                    idf REAL DEFAULT 0.0,
                    consensus REAL DEFAULT 0.0,
                    final_rank REAL DEFAULT 0.0,
                    PRIMARY KEY (anchor, offset, neighbor)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_anchor_offset_rank
                ON edge_scores(anchor, offset, final_rank DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_anchor_rank
                ON edge_scores(anchor, final_rank DESC)
            """)

    # ── Tokenization (two-layer) ─────────────────────────
    #
    # Layer 1: full_tokenize() → all tokens including punctuation/emoji
    #   - used for reconstruction and symbol assignment
    #   - every token gets a symbol
    #
    # Layer 2: content_tokenize() → only content words
    #   - punctuation, emoji, bare numbers stripped out
    #   - THIS is what the 6-1-6 window walks over
    #   - "Model training, using PyTorch." → ["model", "training", "using", "pytorch"]
    #   - comma and period don't eat window slots

    def _full_tokenize(self, text: str) -> List[str]:
        """Tokenize preserving everything. For reconstruction/symbol assignment."""
        text = text.strip()
        # Replace emoji with placeholder before word splitting
        text = _EMOJI_RE.sub(f" {EMOJI_SYMBOL} ", text)
        text = text.lower()
        tokens = self._word_pattern.findall(text)
        return [t for t in tokens if t.strip()]

    def _content_tokenize(self, text: str) -> List[str]:
        """Tokenize to content words only. Punctuation, emoji, bare numbers skipped.

        This is the sequence the 6-1-6 window walks over.
        Punctuation does not break sentences or eat window positions.
        """
        full = self._full_tokenize(text)
        content = []
        for t in full:
            # Skip emoji placeholder
            if t == EMOJI_SYMBOL.lower():
                continue
            # Skip bare punctuation
            if len(t) == 1 and t in _PUNCT_CHARS:
                continue
            # Skip multi-char punctuation sequences
            if all(c in _PUNCT_CHARS for c in t):
                continue
            # Skip bare numbers (01, 2025, etc.)
            if _NUMBER_RE.match(t):
                continue
            content.append(t)
        return content

    def _get_int_id(self, word: str) -> int:
        """Get or assign a session-local integer ID for windowing."""
        key = word.lower()
        if key not in self._word_to_int:
            self._word_to_int[key] = self._next_int
            self._int_to_word[self._next_int] = key
            self._next_int += 1
        return self._word_to_int[key]

    # ── Map text → 6-1-6 structure ────────────────────────

    def map_text(self, text: str, source: str = "inline") -> MapReport:
        """Map text into 6-1-6 co-occurrence structure.

        Paragraph boundary = double newline (wall).
        Each paragraph windowed independently.
        Counts merge across paragraphs (sealed universes).

        Tokenization is two-layer:
          - full_tokenize: all tokens get symbols (for reconstruction)
          - content_tokenize: only content words enter the window
          - punctuation, emoji, bare numbers are transparent to counting
        """
        paragraphs = re.split(r"\n\s*\n", text)

        before_map: Dict[str, Dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
        after_map: Dict[str, Dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
        total_tokens = 0
        content_tokens = 0
        skipped_tokens = 0
        t0 = time.time()

        for para in paragraphs:
            # Full tokenize for stats
            full = self._full_tokenize(para)
            total_tokens += len(full)

            # Content-only for windowing
            tokens = self._content_tokenize(para)
            if not tokens:
                continue
            content_tokens += len(tokens)
            skipped_tokens += len(full) - len(tokens)

            token_ids = [self._get_int_id(t) for t in tokens]
            counts = compute_window_counts(token_ids, self.window)

            for offset, pairs in counts.items():
                target = before_map if offset < 0 else after_map
                distance = abs(offset)
                for (focus_id, ctx_id), count in pairs.items():
                    focus_word = self._int_to_word[focus_id]
                    ctx_word = self._int_to_word[ctx_id]
                    target[focus_word][distance][ctx_word] += count

        elapsed_ms = int((time.time() - t0) * 1000)

        # Pack with category metadata (same contract as bridge patch)
        items = self._pack_and_merge(before_map, after_map)

        # Stats
        anchor_count = len(items)
        content_anchor_count = sum(
            1 for v in items.values()
            if v.get("category") not in SUPPRESSED_ANCHOR_CLASSES
        )

        report = MapReport(
            source=source,
            window=self.window,
            total_tokens=total_tokens,
            duration_ms=elapsed_ms,
            items=items,
            stats={
                "total_anchors": anchor_count,
                "content_anchors": content_anchor_count,
                "function_anchors": anchor_count - content_anchor_count,
                "total_tokens": total_tokens,
                "content_tokens": content_tokens,
                "skipped_tokens": skipped_tokens,
            },
        )
        return report

    def _pack_and_merge(
        self,
        before_map: Dict[str, Dict[int, Counter]],
        after_map: Dict[str, Dict[int, Counter]],
    ) -> Dict[str, Dict[str, Any]]:
        """Pack raw counts into the 6-1-6 structure with lexicon metadata."""
        all_words = set(before_map.keys()) | set(after_map.keys())
        merged: Dict[str, Dict[str, Any]] = {}

        for focus_word in all_words:
            rec = self.lexicon.resolve_token(focus_word)

            before_packed = self._pack_side(before_map.get(focus_word, {}))
            after_packed = self._pack_side(after_map.get(focus_word, {}))

            merged[focus_word] = {
                "before": before_packed,
                "after": after_packed,
                "symbol": rec.hex,
                "category": rec.metadata.get("category") if rec.metadata else None,
                "status": rec.status.value,
                "source_pool": rec.source_pool,
                "tone_signature": rec.tone_signature,
            }

            # Category from lexicon payload (check the raw entry)
            raw_entry = (
                self.lexicon._canonical.get(focus_word)
                or self.lexicon._medical.get(focus_word)
                or self.lexicon._structural.get(focus_word)
                or {}
            )
            if raw_entry.get("category"):
                merged[focus_word]["category"] = raw_entry["category"]

        return merged

    def _pack_side(
        self, distance_buckets: Dict[int, Counter]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Pack one side (before or after). All counts survive, sorted desc."""
        packed: Dict[str, List[Dict[str, Any]]] = {}
        for distance, counter in distance_buckets.items():
            items = []
            for token, count in counter.most_common():
                rec = self.lexicon.resolve_token(token)
                raw_entry = (
                    self.lexicon._canonical.get(token)
                    or self.lexicon._medical.get(token)
                    or self.lexicon._structural.get(token)
                    or {}
                )
                items.append({
                    "token": token,
                    "count": int(count),
                    "symbol": rec.hex,
                    "in_lexicon": rec.status != SymbolStatus.TEMP_ASSIGNED,
                    "category": raw_entry.get("category"),
                })
            packed[str(distance)] = items
        return packed

    # ── get_616 view (with suppression) ───────────────────

    def get_616(
        self,
        word: str,
        topk: Optional[int] = None,
        include_suppressed: bool = False,
        map_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return 6-1-6 positional view for a word.

        Raw total is always raw. Visible weight is secondary.
        Filter first, then limit.
        """
        # Find the map
        if map_id and map_id in self._maps:
            report = self._maps[map_id]
        elif self._maps:
            report = list(self._maps.values())[-1]
        else:
            return {
                "word": word, "before": {}, "after": {},
                "total_co_occurrence_weight": 0,
                "view_filtered": not include_suppressed,
                "suppressed_count": 0,
            }

        payload = report.items.get(word.lower())
        if not payload:
            return {
                "word": word, "before": {}, "after": {},
                "total_co_occurrence_weight": 0,
                "view_filtered": not include_suppressed,
                "suppressed_count": 0,
            }

        limit = topk or self.top_k

        def trim(buckets):
            output = {}
            suppressed = 0
            for distance, items in buckets.items():
                if include_suppressed:
                    visible = items
                else:
                    visible = [
                        item for item in items
                        if item.get("category") not in SUPPRESSED_ANCHOR_CLASSES
                    ]
                    suppressed += len(items) - len(visible)
                output[distance] = [
                    [item["token"], item["count"]] for item in visible[:limit]
                ]
            return output, suppressed

        raw_before = payload.get("before", {})
        raw_after = payload.get("after", {})

        # Raw total — always from unfiltered payload (truth)
        raw_weight = 0
        for items in raw_before.values():
            raw_weight += sum(item["count"] for item in items)
        for items in raw_after.values():
            raw_weight += sum(item["count"] for item in items)

        before, sup_b = trim(raw_before)
        after, sup_a = trim(raw_after)
        total_suppressed = sup_b + sup_a

        visible_weight = sum(c for bk in before.values() for _, c in bk)
        visible_weight += sum(c for bk in after.values() for _, c in bk)

        result = {
            "word": word,
            "symbol": payload.get("symbol"),
            "category": payload.get("category"),
            "before": before,
            "after": after,
            "total_co_occurrence_weight": int(raw_weight),
            "view_filtered": not include_suppressed,
            "suppressed_count": total_suppressed,
        }
        if not include_suppressed:
            result["visible_weight"] = int(visible_weight)
        return result

    # ── Indexing into SQLite ──────────────────────────────

    def index_map(self, map_id: str, report: MapReport,
                  recompute: bool = True) -> Dict[str, Any]:
        """Index a map report into term_stats and edge_scores.

        Only content-class symbols become anchors in the index.
        Function words are counted as neighbors but never as anchors.
        Set recompute=False for batch ingestion, then call finalize_indexes().
        """
        self._maps[map_id] = report
        items = report.items

        doc_terms: Set[str] = set()
        term_weights: Dict[str, int] = {}
        edge_counts: Dict[Tuple[str, int, str], int] = {}

        indexed_anchors = 0
        skipped_anchors = 0

        for anchor, anchor_data in items.items():
            # Category gate: function words do not become anchors
            if anchor_data.get("category") in SUPPRESSED_ANCHOR_CLASSES:
                skipped_anchors += 1
                continue

            indexed_anchors += 1
            doc_terms.add(anchor)

            for side, sign in [("before", -1), ("after", 1)]:
                buckets = anchor_data.get(side, {})
                for pos_str, neighbors in buckets.items():
                    offset = sign * int(pos_str)
                    for neighbor in neighbors:
                        token = neighbor["token"]
                        count = neighbor["count"]
                        doc_terms.add(token)
                        term_weights[token] = term_weights.get(token, 0) + count
                        edge_key = (anchor, offset, token)
                        edge_counts[edge_key] = edge_counts.get(edge_key, 0) + count

        # Write term_stats
        with self._get_db(self._term_stats_db) as conn:
            for term in doc_terms:
                rec = self.lexicon.resolve_token(term)
                raw_entry = (
                    self.lexicon._canonical.get(term)
                    or self.lexicon._medical.get(term)
                    or self.lexicon._structural.get(term)
                    or {}
                )
                conn.execute("""
                    INSERT INTO term_stats (term, symbol, category, df, total_co_occurrence_weight, idf)
                    VALUES (?, ?, ?, 1, ?, 0.0)
                    ON CONFLICT(term) DO UPDATE SET
                        df = df + 1,
                        total_co_occurrence_weight = total_co_occurrence_weight + ?
                """, (
                    term, rec.hex, raw_entry.get("category"),
                    term_weights.get(term, 0),
                    term_weights.get(term, 0),
                ))

        # Write edge_scores
        with self._get_db(self._edge_scores_db) as conn:
            for (anchor, offset, neighbor), count in edge_counts.items():
                a_rec = self.lexicon.resolve_token(anchor)
                n_rec = self.lexicon.resolve_token(neighbor)
                n_entry = (
                    self.lexicon._canonical.get(neighbor)
                    or self.lexicon._medical.get(neighbor)
                    or self.lexicon._structural.get(neighbor)
                    or {}
                )
                conn.execute("""
                    INSERT INTO edge_scores
                        (anchor, anchor_symbol, offset, neighbor, neighbor_symbol,
                         neighbor_category, raw_count, df_edge)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(anchor, offset, neighbor) DO UPDATE SET
                        raw_count = raw_count + ?,
                        df_edge = df_edge + 1
                """, (
                    anchor, a_rec.hex, offset, neighbor, n_rec.hex,
                    n_entry.get("category"),
                    count, count,
                ))

        # Write manifest
        content_hash = hashlib.sha256(
            json.dumps(report.items, sort_keys=True).encode()
        ).hexdigest()[:16]

        with self._get_db(self._manifest_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO maps_manifest
                    (map_id, source, total_tokens, indexed_utc, sha256)
                VALUES (?, ?, ?, ?, ?)
            """, (
                map_id, report.source, report.total_tokens,
                datetime.now(timezone.utc).isoformat(), content_hash,
            ))

        if recompute:
            self._recompute_idf()
            self._recompute_edge_ranks()

        return {
            "map_id": map_id,
            "indexed_anchors": indexed_anchors,
            "skipped_function_anchors": skipped_anchors,
            "terms": len(doc_terms),
            "edges": len(edge_counts),
        }

    def finalize_indexes(self):
        """Recompute IDF and edge ranks. Call once after batch indexing."""
        self._recompute_idf()
        self._recompute_edge_ranks()

    def batch_ingest(self, texts: List[str], sources: List[str] = None,
                     show_progress: bool = True) -> Dict[str, Any]:
        """Map and index multiple texts with a single IDF recompute at the end.

        Returns corpus-level stats.
        """
        from tqdm import tqdm

        if sources is None:
            sources = [f"doc_{i:04d}" for i in range(len(texts))]

        total_tok = 0
        total_anchors = 0
        total_func_skipped = 0
        total_edges = 0
        file_stats = []

        iterator = enumerate(zip(texts, sources))
        if show_progress:
            iterator = tqdm(list(iterator), desc="Indexing", unit="doc")

        for i, (text, src) in iterator:
            map_id = f"doc_{i:04d}"
            report = self.map_text(text, source=src)
            result = self.index_map(map_id, report, recompute=False)

            total_tok += report.total_tokens
            total_anchors += result["indexed_anchors"]
            total_func_skipped += result["skipped_function_anchors"]
            total_edges += result["edges"]

            file_stats.append({
                "map_id": map_id,
                "source": src,
                "tokens": report.total_tokens,
                "anchors": result["indexed_anchors"],
                "edges": result["edges"],
            })

            if show_progress:
                iterator.set_postfix({
                    "tok": f"{total_tok:,}",
                    "anchors": f"{total_anchors:,}",
                    "edges": f"{total_edges:,}",
                })

        # Single IDF + rank recompute at the end
        logger.info("Finalizing indexes (IDF + edge ranks)...")
        self.finalize_indexes()

        return {
            "files_indexed": len(texts),
            "total_tokens": total_tok,
            "total_anchors": total_anchors,
            "total_function_skipped": total_func_skipped,
            "total_edges": total_edges,
            "file_stats": file_stats,
        }

    def _recompute_idf(self):
        with self._get_db(self._manifest_db) as conn:
            n_docs = conn.execute("SELECT COUNT(*) FROM maps_manifest").fetchone()[0]
        if n_docs == 0:
            return

        with self._get_db(self._term_stats_db) as conn:
            conn.execute("""
                UPDATE term_stats
                SET idf = ln((? + 1.0) / (df + 1.0)) + 1.0
            """, (n_docs,))

    def _recompute_edge_ranks(self):
        """final_rank = raw_count * idf(neighbor) * log(1 + df_edge)"""
        idf_map = {}
        with self._get_db(self._term_stats_db) as conn:
            for term, idf in conn.execute("SELECT term, idf FROM term_stats"):
                idf_map[term] = idf

        with self._get_db(self._edge_scores_db) as conn:
            edges = conn.execute(
                "SELECT anchor, offset, neighbor, raw_count, df_edge FROM edge_scores"
            ).fetchall()

            for anchor, offset, neighbor, raw_count, df_edge in edges:
                idf = idf_map.get(neighbor, 1.0)
                consensus = math.log(1.0 + df_edge)
                final_rank = raw_count * idf * consensus
                conn.execute("""
                    UPDATE edge_scores
                    SET idf = ?, consensus = ?, final_rank = ?
                    WHERE anchor = ? AND offset = ? AND neighbor = ?
                """, (idf, consensus, final_rank, anchor, offset, neighbor))

    # ── Querying ──────────────────────────────────────────

    def query(self, question: str, mode: str = "what_does_x_do",
              topk: int = 8) -> AnswerFrame:
        """Route a question to the appropriate query handler."""
        if mode == "what_does_x_do":
            return self.query_what_does_x_do(question, topk)
        elif mode == "definition":
            return self.query_definition(question, topk)
        raise ValueError(f"Unknown query mode: {mode}")

    def query_what_does_x_do(self, subject: str, topk: int = 8) -> AnswerFrame:
        """What does X do? — deterministic, no LLM."""
        anchor = self._normalize_anchor(subject)
        if not anchor:
            return AnswerFrame(
                subject=subject, subject_symbol=None,
                predicates=[], confidence=0.0,
                reasoning_trace={"error": f"No anchor found for '{subject}'"},
                surface_text=f"No data found for '{subject}'.",
            )

        anchor_rec = self.lexicon.resolve_token(anchor)

        # Get verb candidates from +1 and +2
        candidates_1 = self._get_top_neighbors(anchor, offset=1, limit=topk)
        candidates_2 = self._get_top_neighbors(anchor, offset=2, limit=topk // 2)

        predicate_candidates = []
        for neighbor, rank, count, df_edge, n_cat in candidates_1:
            if n_cat in SUPPRESSED_ANCHOR_CLASSES:
                continue
            if is_verb_like(neighbor):
                predicate_candidates.append({
                    "token": neighbor, "slot": 1,
                    "rank": rank, "count": count, "df_edge": df_edge,
                    "verb_score": rank,
                })

        for neighbor, rank, count, df_edge, n_cat in candidates_2:
            if n_cat in SUPPRESSED_ANCHOR_CLASSES:
                continue
            if is_verb_like(neighbor):
                predicate_candidates.append({
                    "token": neighbor, "slot": 2,
                    "rank": rank, "count": count, "df_edge": df_edge,
                    "verb_score": rank * 0.8,
                })

        predicate_candidates.sort(key=lambda x: x["verb_score"], reverse=True)
        best_rank = predicate_candidates[0]["rank"] if predicate_candidates else 1.0

        # Build predicate frames
        predicates = []
        for pred in predicate_candidates[:topk]:
            obj_slot = pred["slot"] + 1
            objects = self._get_top_neighbors(anchor, offset=obj_slot, limit=3)
            # Filter function words from objects too
            objects = [o for o in objects if o[4] not in SUPPRESSED_ANCHOR_CLASSES]

            if objects:
                obj_rec = self.lexicon.resolve_token(objects[0][0])
                predicates.append({
                    "verb": pred["token"],
                    "object": objects[0][0],
                    "object_symbol": obj_rec.hex,
                    "confidence": self._compute_confidence(
                        pred["rank"], pred["df_edge"], best_rank
                    ),
                    "evidence": {
                        "predicate_slot": pred["slot"],
                        "object_slot": obj_slot,
                        "verb_rank": pred["rank"],
                        "verb_count": pred["count"],
                        "verb_df": pred["df_edge"],
                        "object_rank": objects[0][1],
                        "object_count": objects[0][2],
                        "best_rank": best_rank,
                    },
                })

        overall_conf = (
            sum(p["confidence"] for p in predicates) / len(predicates)
            if predicates else 0.0
        )

        if predicates:
            frames = [f"{p['verb']} {p['object']}" for p in predicates[:3]]
            surface = f"{anchor.capitalize()}: {', '.join(frames)}."
        else:
            surface = f"No verb-like predicates found for {anchor}."

        return AnswerFrame(
            subject=anchor,
            subject_symbol=anchor_rec.hex,
            predicates=predicates,
            confidence=overall_conf,
            reasoning_trace={
                "anchors_used": [anchor],
                "predicate_candidates_found": len(predicate_candidates),
                "frames_built": len(predicates),
                "query_mode": "what_does_x_do",
            },
            surface_text=surface,
        )

    def query_definition(self, subject: str, topk: int = 8) -> AnswerFrame:
        """Definition-style query: gather top neighbors across all positions."""
        anchor = self._normalize_anchor(subject)
        if not anchor:
            return AnswerFrame(
                subject=subject, subject_symbol=None,
                predicates=[], confidence=0.0,
                reasoning_trace={"error": f"No anchor found for '{subject}'"},
                surface_text=f"No data found for '{subject}'.",
            )

        anchor_rec = self.lexicon.resolve_token(anchor)

        # Gather all neighbors across positions
        all_neighbors = []
        for offset in range(-self.window, self.window + 1):
            if offset == 0:
                continue
            neighbors = self._get_top_neighbors(anchor, offset=offset, limit=topk)
            for n, rank, count, df_edge, n_cat in neighbors:
                if n_cat not in SUPPRESSED_ANCHOR_CLASSES:
                    all_neighbors.append({
                        "token": n, "offset": offset,
                        "rank": rank, "count": count, "df_edge": df_edge,
                    })

        # Dedupe by token, keep highest rank
        seen = {}
        for n in all_neighbors:
            key = n["token"]
            if key not in seen or n["rank"] > seen[key]["rank"]:
                seen[key] = n

        top = sorted(seen.values(), key=lambda x: x["rank"], reverse=True)[:topk]

        predicates = [
            {"token": n["token"], "rank": n["rank"], "count": n["count"],
             "offset": n["offset"], "df_edge": n["df_edge"]}
            for n in top
        ]

        if top:
            words = [n["token"] for n in top[:5]]
            surface = f"{anchor.capitalize()} is associated with: {', '.join(words)}."
            conf = min(top[0]["rank"] / 100.0, 1.0)
        else:
            surface = f"No associations found for {anchor}."
            conf = 0.0

        return AnswerFrame(
            subject=anchor,
            subject_symbol=anchor_rec.hex,
            predicates=predicates,
            confidence=conf,
            reasoning_trace={
                "anchors_used": [anchor],
                "neighbors_found": len(all_neighbors),
                "unique_after_dedup": len(seen),
                "query_mode": "definition",
            },
            surface_text=surface,
        )

    # ── Query helpers ─────────────────────────────────────

    def _normalize_anchor(self, term: str) -> Optional[str]:
        """Find best matching anchor in edge_scores."""
        clean = re.sub(r"[^a-z0-9_ -]+", "", term.lower()).strip()
        singular = clean[:-1] if clean.endswith("s") and len(clean) > 1 else clean

        with self._get_db(self._edge_scores_db) as conn:
            for variant in [clean, clean + "s", singular]:
                row = conn.execute(
                    "SELECT DISTINCT anchor FROM edge_scores WHERE LOWER(anchor) = ? LIMIT 1",
                    (variant,)
                ).fetchone()
                if row:
                    return row[0].lower()

        return None

    def _get_top_neighbors(
        self, anchor: str, offset: int, limit: int = 10
    ) -> List[Tuple[str, float, int, int, Optional[str]]]:
        """Get top neighbors by final_rank. Returns (token, rank, count, df_edge, category)."""
        with self._get_db(self._edge_scores_db) as conn:
            rows = conn.execute("""
                SELECT neighbor, final_rank, raw_count, df_edge, neighbor_category
                FROM edge_scores
                WHERE anchor = ? AND offset = ?
                ORDER BY final_rank DESC, raw_count DESC, neighbor ASC
                LIMIT ?
            """, (anchor, offset, limit)).fetchall()
        return rows

    def _compute_confidence(self, rank: float, df_edge: int, best_rank: float) -> float:
        """Confidence [0-1] from rank + consensus."""
        base = rank / best_rank if best_rank > 0 else 0.0
        if df_edge <= 1:
            cons = 0.3
        elif df_edge == 2:
            cons = 0.6
        else:
            cons = 1.0
        return min(base * (0.6 + 0.4 * cons), 1.0)

    # ── Stats ─────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        with self._get_db(self._manifest_db) as conn:
            n_maps = conn.execute("SELECT COUNT(*) FROM maps_manifest").fetchone()[0]

        with self._get_db(self._term_stats_db) as conn:
            n_terms = conn.execute("SELECT COUNT(*) FROM term_stats").fetchone()[0]

        with self._get_db(self._edge_scores_db) as conn:
            n_edges = conn.execute("SELECT COUNT(*) FROM edge_scores").fetchone()[0]

        return {
            "indexed_maps": n_maps,
            "unique_terms": n_terms,
            "total_edges": n_edges,
            "lexicon": self.lexicon.stats(),
        }
