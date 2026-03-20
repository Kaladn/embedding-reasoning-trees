"""
Parallel ingest with BinaryCell storage backend.

Phase 1: Parallel CPU mapping (multiprocessing, all cores)
Phase 2: Batch lexicon resolution (all unique words resolved ONCE)
Phase 3: Accumulate into BinaryCellV2 objects in memory
Phase 4: Single write_all() to disk + CorpusStats side index
Phase 5: Queries against the cell store

Replaces the 45-minute SQLite upsert loop with in-memory accumulation
and a single sequential write.
"""
import sys, io, os, glob, time, logging, re, math
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("ingest")

PKG_DIR = "C:/Users/mydyi/OneDrive/Documents/Desktop/ml_packages"
LEXICON_DIR = "Lexical Data"
CACHE_DIR = "./reasoning_cache"
WINDOW = 6
WORD_PAT = re.compile(r"\b\w+\b|[^\w\s]")

SUPPRESSED_ANCHOR_CLASSES = frozenset({"function", "structural", "punctuation", "emoji"})

# ── Content-only tokenizer (punctuation/emoji/numbers skip the window) ──

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0000FE00-\U0000FE0F"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF\U0000200D\U00002B50\U000023F0-\U000023FA"
    "]+", re.UNICODE
)
_PUNCT_CHARS = frozenset(".,;:!?\"'`()[]{}/<>@#$%^&*+-=~|\\\u2018\u2019\u201c\u201d\u2014\u2013\u2026")
_NUMBER_RE = re.compile(r"^\d+$")
# Single chars that are not real words (structural letters, stray symbols)
_MIN_WORD_LEN = 2  # Skip all single-character tokens


def _content_tokenize(text):
    """Content words only. Punctuation, emoji, numbers transparent."""
    text = _EMOJI_RE.sub(" ", text)
    tokens = WORD_PAT.findall(text.lower().strip())
    out = []
    for t in tokens:
        if not t.strip():
            continue
        # Skip single characters (punctuation, stray letters, structural)
        if len(t) < _MIN_WORD_LEN:
            continue
        # Skip punctuation sequences
        if all(c in _PUNCT_CHARS for c in t):
            continue
        # Skip bare numbers
        if _NUMBER_RE.match(t):
            continue
        # Skip unicode box-drawing / math symbols
        if any(0x2300 <= ord(c) <= 0x2BFF for c in t):
            continue
        out.append(t)
    return out


def map_one_text(args):
    """Map a single text. Runs in worker process."""
    idx, text, source = args
    paragraphs = re.split(r"\n\s*\n", text)

    # before[anchor][dist][neighbor] = count
    before = defaultdict(lambda: defaultdict(Counter))
    after = defaultdict(lambda: defaultdict(Counter))
    total_tokens = 0
    unique_words = set()

    for para in paragraphs:
        tokens = _content_tokenize(para)
        if not tokens:
            continue
        total_tokens += len(tokens)
        unique_words.update(tokens)

        length = len(tokens)
        for i, focus in enumerate(tokens):
            for offset in range(1, WINDOW + 1):
                bp = i - offset
                if bp < 0:
                    break
                before[focus][offset][tokens[bp]] += 1
            for offset in range(1, WINDOW + 1):
                ap = i + offset
                if ap >= length:
                    break
                after[focus][offset][tokens[ap]] += 1

    # Serialize for pickling
    before_ser = {fw: {d: dict(c) for d, c in dbs.items()} for fw, dbs in before.items()}
    after_ser = {fw: {d: dict(c) for d, c in dbs.items()} for fw, dbs in after.items()}

    return idx, source, total_tokens, before_ser, after_ser, list(unique_words)


def main():
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit files (0=all)")
    args = parser.parse_args()

    # ── Load files ──
    files = sorted(
        glob.glob(os.path.join(PKG_DIR, "*.txt"))
        + glob.glob(os.path.join(PKG_DIR, "*.yaml"))
        + glob.glob(os.path.join(PKG_DIR, "*.py"))
    )
    texts, sources = [], []
    for fp in files:
        try:
            content = open(fp, encoding="utf-8", errors="replace").read().strip()
            if len(content) > 30:
                texts.append(content)
                sources.append(os.path.basename(fp))
        except Exception:
            pass

    if args.limit > 0:
        texts = texts[:args.limit]
        sources = sources[:args.limit]

    n_workers = min(cpu_count(), 16)
    logger.info(f"Loaded {len(texts)} files | {cpu_count()} cores | {n_workers} workers")

    # ════════════════════════════════════════════════════════
    # PHASE 1: Parallel 6-1-6 mapping
    # ════════════════════════════════════════════════════════
    logger.info("Phase 1: Parallel mapping...")
    t0 = time.time()
    work = [(i, texts[i], sources[i]) for i in range(len(texts))]

    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(map_one_text, work, chunksize=10),
            total=len(work), desc=f"Mapping ({n_workers}w)", unit="doc",
        ))
    map_time = time.time() - t0
    logger.info(f"Mapping done in {map_time:.1f}s")

    # ════════════════════════════════════════════════════════
    # PHASE 2: Batch lexicon resolution
    # ════════════════════════════════════════════════════════
    logger.info("Phase 2: Lexicon resolution...")
    t0 = time.time()

    all_unique = set()
    for _, _, _, before_ser, after_ser, doc_words in results:
        all_unique.update(doc_words)
    logger.info(f"  {len(all_unique):,} unique words")

    from cascade_tokenizer.lexicon_backend import LexiconBackend
    lexicon = LexiconBackend(LEXICON_DIR, load_medical=True)

    # word -> (hex, category, status, source_pool, tone_sig)
    word_cache = {}
    for word in tqdm(all_unique, desc="Resolving", unit="word"):
        rec = lexicon.resolve_token(word)
        raw = (lexicon._canonical.get(word)
               or lexicon._medical.get(word)
               or lexicon._structural.get(word) or {})
        cat = raw.get("category", "content") or "content"
        tone = 0
        ts = raw.get("tone_signature", "")
        if isinstance(ts, str) and ts.startswith("TONE_"):
            try:
                tone = int(ts.split("_", 1)[1])
            except ValueError:
                pass
        word_cache[word] = (rec.hex, cat, rec.status.value, rec.source_pool, tone)

    resolve_time = time.time() - t0
    lex_stats = lexicon.stats()
    logger.info(f"Resolved in {resolve_time:.1f}s | "
                f"canonical={lex_stats['canonical_lookups']}, temp={lex_stats['session_temp_assigned']}")

    # ════════════════════════════════════════════════════════
    # PHASE 3: Accumulate into BinaryCellV2 in memory
    # ════════════════════════════════════════════════════════
    logger.info("Phase 3: Accumulating into binary cells...")
    t0 = time.time()

    from cascade_tokenizer.binary_cell import BinaryCellV2, CellStore, CorpusStats

    # Global cell accumulator: symbol_hex -> BinaryCellV2
    global_cells: dict = {}
    corpus_stats = CorpusStats()

    total_tok = 0
    total_anchors = 0
    total_func_skipped = 0
    total_edges = 0

    for idx, source, n_tokens, before_ser, after_ser, _ in tqdm(results, desc="Accumulating", unit="doc"):
        total_tok += n_tokens
        doc_cells: dict = {}  # cells touched by this doc (for corpus stats)

        all_words = set(before_ser.keys()) | set(after_ser.keys())

        for anchor in all_words:
            a_hex, a_cat, a_status, a_pool, a_tone = word_cache.get(anchor, ("", "content", "", "", 0))

            # Category gate
            if a_cat in SUPPRESSED_ANCHOR_CLASSES:
                total_func_skipped += 1
                continue

            total_anchors += 1

            # Get or create cell
            if a_hex not in global_cells:
                global_cells[a_hex] = BinaryCellV2(
                    symbol_hex=a_hex, display_text=anchor,
                    status=a_status, category=a_cat,
                    source_pool=a_pool, tone_signature=a_tone,
                )
            cell = global_cells[a_hex]
            cell.total_count += 1
            doc_cells[a_hex] = cell

            # Accumulate neighbors
            for side, sign in [("before", -1), ("after", 1)]:
                buckets = (before_ser if sign < 0 else after_ser).get(anchor, {})
                for dist, counter in buckets.items():
                    offset = sign * int(dist)
                    for neighbor, count in counter.items():
                        n_hex, n_cat, _, _, n_tone = word_cache.get(neighbor, ("", "content", "", "", 0))
                        cell.add_neighbor(offset, n_hex, neighbor, count=count, tone_sig=n_tone % 65536)
                        total_edges += 1

        # Record doc for corpus stats
        corpus_stats.record_document(doc_cells)

    accum_time = time.time() - t0
    logger.info(f"Accumulated in {accum_time:.1f}s | {len(global_cells):,} cells, {total_edges:,} edges")

    # ════════════════════════════════════════════════════════
    # PHASE 4: Write to disk + compute IDF
    # ════════════════════════════════════════════════════════
    logger.info("Phase 4: Writing binary cells + computing IDF...")
    t0 = time.time()

    os.makedirs(CACHE_DIR, exist_ok=True)
    store_path = os.path.join(CACHE_DIR, "evidence.bin")
    store = CellStore(store_path)
    store.write_all(global_cells)

    corpus_stats.compute_idf()
    write_time = time.time() - t0
    logger.info(f"Written in {write_time:.1f}s | {len(store)} cells on disk")

    # ════════════════════════════════════════════════════════
    # REPORT
    # ════════════════════════════════════════════════════════
    total_time = map_time + resolve_time + accum_time + write_time
    cs = corpus_stats.summary()

    print(f"\n{'='*60}")
    print("CORPUS REPORT")
    print(f"{'='*60}")
    print(f"  Files:             {len(texts)}")
    print(f"  Phase 1 (map):     {map_time:.1f}s")
    print(f"  Phase 2 (resolve): {resolve_time:.1f}s")
    print(f"  Phase 3 (accum):   {accum_time:.1f}s")
    print(f"  Phase 4 (write):   {write_time:.1f}s")
    print(f"  TOTAL:             {total_time:.1f}s")
    print(f"  Total tokens:      {total_tok:,}")
    print(f"  Content anchors:   {total_anchors:,}")
    print(f"  Function skipped:  {total_func_skipped:,}")
    print(f"  Total edges:       {total_edges:,}")
    print(f"  Unique cells:      {len(global_cells):,}")
    print(f"  Corpus terms:      {cs['n_terms']:,}")
    print(f"  Canonical hits:    {lex_stats['canonical_lookups']:,}")
    print(f"  Temp assigned:     {lex_stats['session_temp_assigned']}")

    # ════════════════════════════════════════════════════════
    # VERIFICATION: Read back and query
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("VERIFICATION: Read-back from binary store")
    print(f"{'='*60}")

    store2 = CellStore(store_path)
    store2.load_index()
    print(f"  Loaded index: {len(store2)} cells")

    # Find top anchors by total_count
    top_cells = sorted(global_cells.values(), key=lambda c: c.total_co_occurrence_weight(), reverse=True)

    print(f"\n  Top 15 anchors by co-occurrence weight:")
    for cell in top_cells[:15]:
        # Read back from disk to verify
        read_cell = store2.read_cell(cell.symbol_hex)
        if read_cell:
            print(f"    {read_cell.display_text:25s} weight={read_cell.total_co_occurrence_weight():>8,}  "
                  f"count={read_cell.total_count:>5}  cat={read_cell.category}  "
                  f"idf={corpus_stats.get_idf(cell.symbol_hex):.2f}")

    # Sample positional queries
    print(f"\n  Positional neighbors for top anchors:")
    for cell in top_cells[:5]:
        read_cell = store2.read_cell(cell.symbol_hex)
        if not read_cell:
            continue
        print(f"\n    [{read_cell.display_text}] (sym={read_cell.symbol_hex})")
        for pos in [-1, +1, +2]:
            bucket = read_cell.get_bucket(pos)
            label = f"{'before' if pos < 0 else 'after'}_{abs(pos)}"
            top3 = [(e.neighbor_word, e.count) for e in bucket[:3]]
            if top3:
                print(f"      {label}: {top3}")

    store2.close()


if __name__ == "__main__":
    main()
