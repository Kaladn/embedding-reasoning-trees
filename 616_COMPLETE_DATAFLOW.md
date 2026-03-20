# 6-1-6 Complete Data Flow

## The Tree Model

```
                    ANCHOR (root)
                        |
      ┌─────────────────┴─────────────────┐
      |                                   |
  ROOTS (before)                    BRANCHES (after)
      |                                   |
 Position -6 to -1                   Position +1 to +6
      |                                   |
 ┌────┴────┬────┬────┬────┬────┐    ┌────┴────┬────┬────┬────┬────┐
-6   -5   -4   -3   -2   -1      +1   +2   +3   +4   +5   +6
 |    |    |    |    |    |        |    |    |    |    |    |
[Top K neighbors with counts at each position]
```

Every anchor gets 12 fixed positional buckets. Each bucket holds neighbors sorted by count descending. Punctuation, emoji, and numbers are transparent to the window — they get symbols but never occupy a position.

---

## Data Flow

```
PHASE 1: INTAKE
════════════════════════════════════════════════════════

  Text corpus (files on disk)
       ↓
  content_tokenize(text)
    - regex splits words
    - punctuation/emoji/numbers → recognized, skipped in window
    - lowercase normalization
       ↓
  compute_window_counts(tokens)
    - paragraph boundaries → hard walls
    - short paragraphs → 0-1-6, 1-1-6, ..., until full 6-1-6
    - counts co-occurrences at each signed position
       ↓
  MapReport (per document)
    - anchors: dict of anchor → {position → [(neighbor, count), ...]}
    - total_tokens, content_anchors, function_skipped


PHASE 2: LEXICON RESOLUTION
════════════════════════════════════════════════════════

  For every unique word across all documents:
       ↓
  LexiconBackend.resolve_token(word)
    - Canonical (260K+ words) → permanent symbol
    - Medical (735K+ words) → permanent symbol
    - Structural (digits, chars) → permanent symbol
    - None of the above → Temp_Pool assignment
       ↓
  Result: word → SymbolRecord
    - symbol_hex (identity)
    - status (ASSIGNED | TEMP_ASSIGNED)
    - category (content | function | structural)
    - source_pool (canonical | medical | structural | temp)


PHASE 3: BINARY CELL ACCUMULATION
════════════════════════════════════════════════════════

  For each document's MapReport:
       ↓
  For each anchor (content-class only, function words skipped):
       ↓
  BinaryCellV2 (one per anchor symbol):
    - 12 fixed positional buckets (dict keyed by neighbor symbol)
    - O(1) neighbor lookup during accumulation
    - counts grow across documents (same anchor, same neighbor → add)
       ↓
  After all documents processed:
    - freeze dicts → sorted lists (by count desc)
    - write all cells to evidence.bin in one pass
    - build bloom-gated index + word↔hex sidecar


PHASE 4: EVIDENCE STORE
════════════════════════════════════════════════════════

  evidence.bin (binary, write-once/read-many)
    - 4-byte cell count header
    - for each cell: 4-byte length prefix + cell bytes
    - CRC32 checksum per cell (verified on read)

  evidence.bin.idx (JSON sidecar)
    - bloom filter (fast miss check)
    - symbol → (offset, length) for mmap reads
    - word_to_hex reverse lookup (persistent)

  Layout per cell:
  ┌──────────────────────────────────────────┐
  │ symbol_hex (10 bytes)                    │
  │ status (1 byte)                          │
  │ category (1 byte)                        │
  │ source_pool (1 byte)                     │
  │ tone_signature (4 bytes)                 │
  │ total_count (4 bytes)                    │
  │ display_text (length-prefixed, max 128)  │
  │ ─── 12 positional buckets ───            │
  │ before_6: [neighbor_sym, word, count]... │
  │ before_5: ...                            │
  │ ...                                      │
  │ after_6: ...                             │
  │ CRC32 (4 bytes)                          │
  └──────────────────────────────────────────┘


PHASE 5: PREDICTION
════════════════════════════════════════════════════════

  Predictor reads evidence store via mmap.

  predict_next(anchor, k=5)
    → read cell → get bucket at +1 → filter stops → top K

  predict_previous(anchor, k=5)
    → read cell → get bucket at -1 → filter stops → top K

  forward_chain(anchor, steps=6)
    → anchor → top at +1 → follow that word's +1 → repeat
    → skip already-seen words (no loops)

  backward_chain(anchor, steps=6)
    → anchor → top at -1 → follow that word's -1 → repeat

  full_context(anchor, k=5)
    → all 12 buckets, top K each, stopwords filtered


PHASE 6: API (Plugin)
════════════════════════════════════════════════════════

  FastAPI plugin (cascade_616) serves via plugin runner:

  GET  /api/cascade-616/status     → cells count, store path
  GET  /api/cascade-616/panel      → interactive explorer UI
  POST /api/cascade-616/predict    → top K at +1
  POST /api/cascade-616/predict/previous → top K at -1
  POST /api/cascade-616/chain/forward    → forward chain
  POST /api/cascade-616/chain/backward   → backward chain
  POST /api/cascade-616/context    → full 6-1-6 view
```

---

## Counting Rules

**What gets counted:**
- Content words as anchors (only)
- ALL words as neighbors (including function words — raw truth preserved)

**What never gets counted:**
- Punctuation — gets a symbol, transparent to window
- Emoji — single EMOJI_PLACEHOLDER symbol, transparent to window
- Bare numbers — recognized, not counted
- Single characters — recognized, not counted

**What gets filtered at view time (not storage time):**
- Function words suppressed from anchor index and predictions
- SUPPRESSED_VIEW_CLASSES = {"function"} (expandable set)
- Filter first, THEN apply limit (never limit then filter)

**Paragraph boundaries:**
- Hard wall — window never crosses paragraph break
- Short paragraphs: 0-1-6, 1-1-6, 2-1-6... until full 6-1-6 possible
- End of paragraph: countdown 6-1-5, 6-1-4, ...

---

## Identity Rules

- **Symbol hex = node key everywhere** — text is rendering metadata
- **Temp symbols are first-class** — tagged, always stored, never rewritten
- **Promotion creates new canonical entry** — old temp hex stays historical
- **Linkage is a lookup, not a migration** — old maps never falsified

---

## What is NOT in this system

- No SQLite (removed)
- No LLM (deterministic only)
- No training (count-based, not learned)
- No TF-IDF (raw counts, not weighted)
- No answer prose generation (structured predictions only)
- No corpus-merged views (single-document truth, corpus aggregate later)
