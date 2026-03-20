"""
GPU-accelerated parallel ingest of ml_packages through the reasoning engine.

Phase 1: Parallel CPU mapping (multiprocessing, all cores)
Phase 2: Batch lexicon resolution (pre-cache all unique words ONCE)
Phase 3: Batch SQLite writes (single transaction per doc, no per-row commits)
Phase 4: Single IDF/rank recompute at the end

Bottleneck fix: the old code called resolve_token() per neighbor per doc.
New code: resolve ALL unique words once upfront, then index with cached lookups.
"""
import sys, io, os, glob, time, logging, re, sqlite3, hashlib, json, math
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict
from datetime import datetime, timezone

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

# Emoji → single placeholder, punctuation/numbers → skip in window
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0000FE00-\U0000FE0F"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF\U0000200D\U00002B50\U000023F0-\U000023FA"
    "]+", re.UNICODE
)
_PUNCT_CHARS = frozenset(".,;:!?\"'`()[]{}/<>@#$%^&*+-=~|\\")
_NUMBER_RE = re.compile(r"^\d+$")


def _full_tokenize(text):
    """All tokens including punctuation (for symbol assignment)."""
    text = _EMOJI_RE.sub(" emoji_placeholder ", text)
    return [t for t in WORD_PAT.findall(text.lower().strip()) if t.strip()]


def _content_tokenize(text):
    """Content words only. Punctuation, emoji, numbers skip the window."""
    full = _full_tokenize(text)
    out = []
    for t in full:
        if t == "emoji_placeholder":
            continue
        if len(t) == 1 and t in _PUNCT_CHARS:
            continue
        if all(c in _PUNCT_CHARS for c in t):
            continue
        if _NUMBER_RE.match(t):
            continue
        out.append(t)
    return out


def map_one_text(args):
    """Map a single text to raw count dicts. Runs in worker process."""
    idx, text, source = args
    paragraphs = re.split(r"\n\s*\n", text)

    word_to_int = {}
    int_to_word = {}
    next_id = 0

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

        ids = []
        for t in tokens:
            if t not in word_to_int:
                word_to_int[t] = next_id
                int_to_word[next_id] = t
                next_id += 1
            ids.append(word_to_int[t])

        # Window counting
        length = len(ids)
        for i, focus in enumerate(ids):
            for offset in range(1, WINDOW + 1):
                bp = i - offset
                if bp < 0:
                    break
                before[int_to_word[focus]][offset][int_to_word[ids[bp]]] += 1
            for offset in range(1, WINDOW + 1):
                ap = i + offset
                if ap >= length:
                    break
                after[int_to_word[focus]][offset][int_to_word[ids[ap]]] += 1

    # Serialize for pickling
    before_ser = {fw: {d: dict(c) for d, c in dbs.items()} for fw, dbs in before.items()}
    after_ser = {fw: {d: dict(c) for d, c in dbs.items()} for fw, dbs in after.items()}

    return idx, source, total_tokens, before_ser, after_ser, list(unique_words)


def main():
    from tqdm import tqdm

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

    n_workers = min(cpu_count(), 16)
    logger.info(f"Loaded {len(texts)} files | {cpu_count()} cores | {n_workers} workers")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Parallel 6-1-6 mapping (all CPU cores)
    # ══════════════════════════════════════════════════════════════
    logger.info("Phase 1: Parallel mapping...")
    t0 = time.time()
    work = [(i, texts[i], sources[i]) for i in range(len(texts))]

    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(map_one_text, work, chunksize=10),
            total=len(work), desc=f"Mapping ({n_workers}w)", unit="doc"
        ))
    map_time = time.time() - t0
    logger.info(f"Mapping done in {map_time:.1f}s")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Batch lexicon resolution (resolve ALL unique words ONCE)
    # ══════════════════════════════════════════════════════════════
    logger.info("Phase 2: Batch lexicon resolution...")
    t0 = time.time()

    # Collect all unique words across all docs
    all_unique = set()
    for _, _, _, before_ser, after_ser, doc_words in results:
        all_unique.update(doc_words)
    logger.info(f"  {len(all_unique):,} unique words to resolve")

    # Load lexicon and resolve everything once
    from cascade_tokenizer.lexicon_backend import LexiconBackend
    lexicon = LexiconBackend(LEXICON_DIR, load_medical=True)

    # Build cache: word -> (hex, category)
    word_cache = {}
    for word in tqdm(all_unique, desc="Resolving", unit="word"):
        rec = lexicon.resolve_token(word)
        raw = (lexicon._canonical.get(word)
               or lexicon._medical.get(word)
               or lexicon._structural.get(word) or {})
        word_cache[word] = (rec.hex, raw.get("category"), rec.status.value, rec.source_pool)

    resolve_time = time.time() - t0
    lex_stats = lexicon.stats()
    logger.info(f"Resolved in {resolve_time:.1f}s | "
                f"canonical={lex_stats['canonical_lookups']}, "
                f"temp={lex_stats['session_temp_assigned']}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Batch SQLite indexing (single transaction per doc)
    # ══════════════════════════════════════════════════════════════
    logger.info("Phase 3: Batch SQLite indexing...")
    t0 = time.time()

    os.makedirs(CACHE_DIR, exist_ok=True)
    term_db_path = os.path.join(CACHE_DIR, "term_stats.db")
    edge_db_path = os.path.join(CACHE_DIR, "edge_scores.db")
    manifest_db_path = os.path.join(CACHE_DIR, "manifest.db")

    # Create schemas
    for db_path, schema in [
        (manifest_db_path, """
            CREATE TABLE IF NOT EXISTS maps_manifest (
                map_id TEXT PRIMARY KEY, source TEXT,
                total_tokens INTEGER, indexed_utc TEXT, sha256 TEXT);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """),
        (term_db_path, """
            CREATE TABLE IF NOT EXISTS term_stats (
                term TEXT PRIMARY KEY, symbol TEXT, category TEXT,
                df INTEGER DEFAULT 0, total_co_occurrence_weight INTEGER DEFAULT 0,
                idf REAL DEFAULT 0.0);
        """),
        (edge_db_path, """
            CREATE TABLE IF NOT EXISTS edge_scores (
                anchor TEXT, anchor_symbol TEXT, offset INTEGER,
                neighbor TEXT, neighbor_symbol TEXT, neighbor_category TEXT,
                raw_count INTEGER, df_edge INTEGER DEFAULT 1,
                idf REAL DEFAULT 0.0, consensus REAL DEFAULT 0.0,
                final_rank REAL DEFAULT 0.0,
                PRIMARY KEY (anchor, offset, neighbor));
            CREATE INDEX IF NOT EXISTS idx_anchor_rank
                ON edge_scores(anchor, final_rank DESC);
            CREATE INDEX IF NOT EXISTS idx_anchor_offset_rank
                ON edge_scores(anchor, offset, final_rank DESC);
        """),
    ]:
        conn = sqlite3.connect(db_path)
        conn.executescript(schema)
        conn.close()

    total_tok = 0
    total_anchors = 0
    total_func_skipped = 0
    total_edges = 0

    term_conn = sqlite3.connect(term_db_path)
    edge_conn = sqlite3.connect(edge_db_path)
    manifest_conn = sqlite3.connect(manifest_db_path)

    # Use WAL mode for faster writes
    term_conn.execute("PRAGMA journal_mode=WAL")
    edge_conn.execute("PRAGMA journal_mode=WAL")
    term_conn.execute("PRAGMA synchronous=NORMAL")
    edge_conn.execute("PRAGMA synchronous=NORMAL")

    for idx, source, n_tokens, before_ser, after_ser, _ in tqdm(results, desc="Indexing", unit="doc"):
        all_words = set(before_ser.keys()) | set(after_ser.keys())

        doc_terms = set()
        term_weights = {}
        edge_batch = []
        indexed_anchors = 0
        skipped_anchors = 0

        for anchor in all_words:
            a_hex, a_cat, a_status, a_pool = word_cache.get(anchor, ("", None, "", ""))

            # Category gate
            if a_cat in SUPPRESSED_ANCHOR_CLASSES:
                skipped_anchors += 1
                continue

            indexed_anchors += 1
            doc_terms.add(anchor)

            for side, sign in [("before", -1), ("after", 1)]:
                buckets = (before_ser if sign < 0 else after_ser).get(anchor, {})
                for dist, counter in buckets.items():
                    offset = sign * int(dist)
                    for neighbor, count in counter.items():
                        doc_terms.add(neighbor)
                        term_weights[neighbor] = term_weights.get(neighbor, 0) + count
                        n_hex, n_cat, _, _ = word_cache.get(neighbor, ("", None, "", ""))
                        edge_batch.append((anchor, a_hex, offset, neighbor, n_hex, n_cat, count))

        # Batch insert terms
        term_conn.execute("BEGIN")
        for term in doc_terms:
            t_hex, t_cat, _, _ = word_cache.get(term, ("", None, "", ""))
            term_conn.execute("""
                INSERT INTO term_stats (term, symbol, category, df, total_co_occurrence_weight, idf)
                VALUES (?, ?, ?, 1, ?, 0.0)
                ON CONFLICT(term) DO UPDATE SET
                    df = df + 1,
                    total_co_occurrence_weight = total_co_occurrence_weight + ?
            """, (term, t_hex, t_cat, term_weights.get(term, 0), term_weights.get(term, 0)))
        term_conn.execute("COMMIT")

        # Batch insert edges
        edge_conn.execute("BEGIN")
        for anchor, a_hex, offset, neighbor, n_hex, n_cat, count in edge_batch:
            edge_conn.execute("""
                INSERT INTO edge_scores
                    (anchor, anchor_symbol, offset, neighbor, neighbor_symbol,
                     neighbor_category, raw_count, df_edge)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(anchor, offset, neighbor) DO UPDATE SET
                    raw_count = raw_count + ?,
                    df_edge = df_edge + 1
            """, (anchor, a_hex, offset, neighbor, n_hex, n_cat, count, count))
        edge_conn.execute("COMMIT")

        # Manifest
        manifest_conn.execute("""
            INSERT OR REPLACE INTO maps_manifest (map_id, source, total_tokens, indexed_utc, sha256)
            VALUES (?, ?, ?, ?, ?)
        """, (f"doc_{idx:04d}", source, n_tokens,
              datetime.now(timezone.utc).isoformat(), hashlib.md5(source.encode()).hexdigest()[:16]))
        manifest_conn.commit()

        total_tok += n_tokens
        total_anchors += indexed_anchors
        total_func_skipped += skipped_anchors
        total_edges += len(edge_batch)

    index_time = time.time() - t0
    logger.info(f"Indexing done in {index_time:.1f}s")

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: Finalize IDF + edge ranks (single pass)
    # ══════════════════════════════════════════════════════════════
    logger.info("Phase 4: Computing IDF + edge ranks...")
    t0 = time.time()

    n_docs = manifest_conn.execute("SELECT COUNT(*) FROM maps_manifest").fetchone()[0]

    # IDF
    term_conn.create_function("ln", 1, lambda x: math.log(x) if x > 0 else 0.0)
    term_conn.execute(f"UPDATE term_stats SET idf = ln(({n_docs} + 1.0) / (df + 1.0)) + 1.0")
    term_conn.commit()

    # Load IDF into memory
    idf_map = {}
    for term, idf in term_conn.execute("SELECT term, idf FROM term_stats"):
        idf_map[term] = idf

    # Edge ranks: final_rank = raw_count * idf * log(1 + df_edge)
    edges = edge_conn.execute(
        "SELECT anchor, offset, neighbor, raw_count, df_edge FROM edge_scores"
    ).fetchall()

    edge_conn.execute("BEGIN")
    for anchor, offset, neighbor, raw_count, df_edge in edges:
        idf = idf_map.get(neighbor, 1.0)
        consensus = math.log(1.0 + df_edge)
        final_rank = raw_count * idf * consensus
        edge_conn.execute("""
            UPDATE edge_scores SET idf=?, consensus=?, final_rank=?
            WHERE anchor=? AND offset=? AND neighbor=?
        """, (idf, consensus, final_rank, anchor, offset, neighbor))
    edge_conn.execute("COMMIT")

    finalize_time = time.time() - t0
    logger.info(f"Finalized in {finalize_time:.1f}s")

    # Stats
    n_terms = term_conn.execute("SELECT COUNT(*) FROM term_stats").fetchone()[0]
    n_edges = edge_conn.execute("SELECT COUNT(*) FROM edge_scores").fetchone()[0]

    term_conn.close()
    edge_conn.close()
    manifest_conn.close()

    # ══════════════════════════════════════════════════════════════
    # REPORT + QUERIES
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"CORPUS REPORT")
    print(f"{'='*60}")
    print(f"  Files:             {len(texts)}")
    print(f"  Phase 1 (map):     {map_time:.1f}s  ({n_workers} workers)")
    print(f"  Phase 2 (resolve): {resolve_time:.1f}s  ({len(all_unique):,} words)")
    print(f"  Phase 3 (index):   {index_time:.1f}s  (batch SQLite)")
    print(f"  Phase 4 (finalize):{finalize_time:.1f}s")
    print(f"  TOTAL:             {map_time+resolve_time+index_time+finalize_time:.1f}s")
    print(f"  Total tokens:      {total_tok:,}")
    print(f"  Content anchors:   {total_anchors:,}")
    print(f"  Function skipped:  {total_func_skipped:,}")
    print(f"  Total edges:       {total_edges:,}")
    print(f"  Unique terms (DB): {n_terms:,}")
    print(f"  Indexed edges(DB): {n_edges:,}")
    print(f"  Canonical hits:    {lex_stats['canonical_lookups']:,}")
    print(f"  Temp assigned:     {lex_stats['session_temp_assigned']}")

    # Queries using the reasoning engine (reads from the SQLite we just built)
    from cascade_tokenizer.reasoning_engine import ReasoningEngine
    engine = ReasoningEngine(LEXICON_DIR, cache_dir=CACHE_DIR)

    print(f"\n{'='*60}")
    print("WHAT DOES X DO?")
    print(f"{'='*60}")
    for subject in ["model", "token", "security", "deployment", "simulation",
                     "training", "inference", "pipeline", "embedding", "server",
                     "data", "agent", "api", "encryption", "network"]:
        answer = engine.query(subject, mode="what_does_x_do")
        print(f"\n  {subject}: {answer.surface_text}")
        print(f"    conf={answer.confidence:.2f} | sym={answer.subject_symbol}")
        for p in answer.predicates[:3]:
            print(f"      {p['verb']} {p['object']} "
                  f"(conf={p['confidence']:.2f}, n={p['evidence']['verb_count']})")

    print(f"\n{'='*60}")
    print("DEFINITIONS")
    print(f"{'='*60}")
    for subject in ["lexicon", "reasoning", "tokenizer", "python", "memory", "database"]:
        answer = engine.query(subject, mode="definition")
        print(f"\n  {subject}: {answer.surface_text}")
        for p in answer.predicates[:5]:
            print(f"      {p['token']} (rank={p['rank']:.1f}, n={p['count']}, pos={p['offset']})")


if __name__ == "__main__":
    main()
