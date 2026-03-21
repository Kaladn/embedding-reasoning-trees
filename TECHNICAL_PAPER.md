# Deterministic Positional Co-Occurrence as a Reasoning Substrate: The 6-1-6 Binary Cell Architecture

**Authors:** Lee Mercey, with engineering assistance from Claude (Anthropic)
**Date:** March 2026
**Status:** Working system, corpus-validated

---

## Abstract

We describe a deterministic inference system that derives structured predictions from raw text without training, gradient descent, or language model inference. The system tokenizes a corpus, resolves each word against a persistent symbol lexicon, counts positional co-occurrences in a fixed 6-word-before, 1-anchor, 6-word-after window, and stores the resulting evidence in a binary cell format optimized for memory-mapped reads. Predictions are produced by traversing positional neighborhoods: given any anchor word, the system returns what the corpus says comes before it, after it, and what chains of association follow from it. The system indexed 838 documents (54MB, 7.2M tokens, 50.7M edges) in 21 minutes and serves predictions in under 1 millisecond. It carries no model weights, produces no hallucinations, and every output is traceable to counted evidence.

---

## 1. What It Is

The 6-1-6 engine is a **positional co-occurrence store with a prediction interface**. It answers one fundamental question: given a word, what does the corpus say about its neighborhood?

The architecture has four layers:

### 1.1 Symbol Lexicon

Every word is resolved to a persistent symbol before any counting occurs. The lexicon contains:

- **Canonical pool**: 263,000+ general English words, each with a unique hex address, binary encoding, and metadata
- **Medical pool**: 735,000+ medical/scientific terms
- **Structural pool**: Digits, single characters
- **Spare Slots**: 7.6M+ reserved addresses for future permanent assignment
- **Temp Pool**: 1M runtime-assignable addresses for words not yet in any permanent pool

A word's identity is its **symbol hex address**, not its string representation. This means:
- Two different spellings mapping to the same concept can share a symbol
- A word promoted from temporary to permanent gets a new address; old evidence stays historical
- The system never confuses "bank" (river) with "bank" (financial) if they're assigned different symbols in different pools

The lexicon is not learned. It is a curated, versioned, persistent data structure that existed before this system and will outlast it.

### 1.2 Windowed Counting

Text is split into paragraphs (hard boundaries — the window never crosses a paragraph break). Within each paragraph, every content word becomes an anchor. For each anchor, the system records which words appear at each of 12 signed positions: -6 through -1 (before) and +1 through +6 (after).

Critical design choices:
- **Punctuation, emoji, and bare numbers are transparent** — they receive symbols (so they're recognized) but do not occupy window positions. "Model, training" counts "training" at +1 from "model", not +2.
- **Function words (the, and, of, is, etc.) are counted as neighbors but never as anchors** — they participate in the raw evidence but do not drive predictions or chains.
- **Short paragraphs get shorter windows** — a 5-word paragraph produces 0-1-4, 1-1-4, 2-1-2, etc. The system does not hallucinate context that isn't there.

### 1.3 Binary Cell Store

Each anchor's complete neighborhood is serialized into a single binary cell:

```
[symbol_hex] [status] [category] [source_pool] [tone_signature] [total_count]
[display_text]
[before_6 bucket] [before_5 bucket] ... [before_1 bucket]
[after_1 bucket] [after_2 bucket] ... [after_6 bucket]
[CRC32 checksum]
```

Each bucket contains neighbor entries sorted by count descending. Each entry stores the full neighbor symbol hex (no hash degradation), the display word, and the raw count.

The store is:
- **Write-once, read-many** — built in a single pass, never modified
- **Memory-mapped** — the OS pages cells into RAM on demand
- **Bloom-gated** — a bloom filter provides O(1) miss checks before any disk read
- **CRC-verified** — every cell is checksummed on write and verified on read
- **Self-indexed** — a JSON sidecar stores the bloom filter, offset table, and word↔symbol reverse lookup

### 1.4 Prediction Engine

The prediction engine is a thin read layer over the binary cell store. It provides five operations:

- **predict_next(anchor, k)** — top K neighbors at position +1
- **predict_previous(anchor, k)** — top K neighbors at position -1
- **full_context(anchor, k)** — top K at all 12 positions
- **forward_chain(anchor, steps)** — follow anchor → top at +1 → follow that word's +1 → repeat, skipping already-visited words
- **backward_chain(anchor, steps)** — same, following -1

All operations apply a hardcoded stopword gate: function words are never returned as predictions or chain links, even though they exist in the raw evidence. This is a **view-time filter**, not a storage-time filter. The raw counts remain honest.

---

## 2. What It Does

### 2.1 Corpus-Derived Predictions

From 838 ML training documents (54MB):

```
ai       → driven(2233), symbolis(1707), models(1447), powered(1254)
model    → training(429), performance, predict
security → system(587), measures, features
lexicon  → nlp(70), builder(22), training(13)
```

These are not guesses. They are direct counts: "driven" appeared 2,233 times at position +1 relative to "ai" across the corpus.

### 2.2 Chain Traversal

```
security → system → performance → metrics → accuracy → error → handling
lexicon  → nlp → models → import → json → schema
model    ← ai ← exo ← integrate ← testing ← scalability (backward)
```

Chains follow the strongest unseen neighbor at each step. They represent the corpus's most-trodden paths of association.

### 2.3 Positional Semantics

Position matters. For "ai":
- Position +1 (immediately after): driven, models, powered — **what AI does**
- Position +2: work, system, manager — **what AI operates on**
- Position -1 (immediately before): exo, py, new — **what modifies AI**
- Position -2: py, ai, real — **broader context**

This is not bag-of-words. Position -1 and position +1 carry fundamentally different semantic relationships to the anchor, and the system preserves that distinction.

### 2.4 Claim Verification

If an LLM claims "AI is used for training models," this system can verify:
- "training" appears 700 times at +1 from "ai" ✓
- "models" appears 1,447 times at +1 from "ai" ✓
- Confidence: high (corpus supports the claim)

If an LLM claims "AI is used for cooking":
- "cooking" appears 0 times near "ai" ✗
- Confidence: zero (corpus does not support this)

This is **grounding**, not generation. The system doesn't produce the claim — it checks whether the corpus has evidence for it.

---

## 3. What It Does Not Do

### 3.1 It does not generate natural language

The system returns structured data: lists of (word, count) tuples, chains of associations, positional neighborhoods. It does not produce sentences, paragraphs, or prose. Converting its output to readable text requires either a template or an LLM.

### 3.2 It does not understand meaning

"Bank" at +1 from "river" and "bank" at +1 from "investment" are two different counts in two different cells. The system does not know they are the same word with different meanings. It knows their neighborhoods are different, which is often sufficient, but it has no semantic representation.

### 3.3 It does not learn

There are no weights, no gradients, no loss functions, no training loops. The system counts co-occurrences and stores them. If the corpus is biased, the counts are biased. If the corpus is incomplete, the predictions are incomplete. The system faithfully reflects the corpus, including its flaws.

### 3.4 It does not handle negation

"AI does not cause hallucinations" and "AI causes hallucinations" produce the same co-occurrence counts. The system cannot distinguish positive from negative assertions. This is a fundamental limitation of positional counting.

### 3.5 It does not scale to arbitrary queries

The system answers "what does the corpus say about X's neighborhood?" It cannot answer "why does X cause Y?" or "what would happen if X?" or "compare X and Y." Those require inference capabilities that counting does not provide.

### 3.6 It does not replace an LLM

An LLM synthesizes, reasons (approximately), and generates. This system counts and retrieves. They are complementary: the LLM produces claims, the 6-1-6 engine verifies them against corpus evidence.

---

## 4. Absolute Requirements

### 4.1 Corpus quality determines output quality

The system has no intelligence beyond counting. If the corpus is 838 ML training documents, the predictions are about ML. If the corpus is medical literature, the predictions are about medicine. Garbage in, garbage counts out.

**Requirement:** The corpus must be representative of the domain you want predictions about.

### 4.2 The lexicon must cover the vocabulary

Words not in the lexicon get temporary symbols. Temporary symbols work correctly at runtime but don't carry metadata (category, tone) and create noise in the symbol space if too many accumulate.

**Requirement:** The lexicon should cover at least 70% of the corpus vocabulary by token volume. The current system achieves 66% canonical + 34% temp on ML text, which is acceptable but could be improved with domain-specific lexicon expansion.

### 4.3 The evidence store must be pre-built

There is no on-the-fly indexing. The full corpus must be ingested, windowed, accumulated, and written to the binary cell store before any predictions can be served. On the tested hardware (i9 + RTX 3070), 54MB of text takes 21 minutes.

**Requirement:** Ingestion is a batch process. Plan for it. The prediction path is fast (<1ms), but building the store is not.

### 4.4 Paragraph boundaries must be meaningful

The window never crosses a paragraph break. If the input text has no paragraph breaks (one giant block), the system treats the entire document as one paragraph, which may create false long-range associations. If paragraphs are too short (<13 words), many anchors will never get a full 6-1-6 window.

**Requirement:** Input text should have paragraph-level structure. Single-sentence paragraphs and wall-of-text blocks both degrade quality.

### 4.5 Function word handling must be explicit

The system's usefulness depends entirely on separating content words from function words. If "the" is allowed as an anchor, it dominates every leaderboard and every chain degenerates into "the → the → the." The current system uses a hardcoded stopword list of ~120 words.

**Requirement:** The stopword list must be maintained. New corpora may introduce domain-specific function words that need to be added.

---

## 5. Why Use This Over Existing Clearbox Methods

Clearbox AI Studio currently has two retrieval/reasoning paths:

### 5.1 LakeSpeak (BM25 + dense retrieval)

LakeSpeak chunks documents, builds BM25 and dense vector indexes, and retrieves relevant chunks for a query. It answers "which documents talk about X?" with ranked chunk results.

**What 6-1-6 adds that LakeSpeak doesn't:**
- **Positional structure.** LakeSpeak returns chunks containing the query term. 6-1-6 returns what comes before and after the term, at each specific distance. "Security" in LakeSpeak returns paragraphs mentioning security. "Security" in 6-1-6 returns: +1 is "system"(587), +2 is "measures", -1 is "network." That's a different kind of answer.
- **Cross-document aggregation.** LakeSpeak retrieves from individual chunks. 6-1-6 accumulates across the entire corpus. If "training" appears after "model" in 400 different documents, the count is 400. LakeSpeak would return the top-K chunks; 6-1-6 gives you the aggregate signal.
- **Chain traversal.** LakeSpeak has no mechanism to follow "model → training → data → schema" as a path through the corpus's association space. 6-1-6 does.

**What LakeSpeak does that 6-1-6 doesn't:**
- Returns actual text (readable chunks, not just word lists)
- Handles semantic similarity (dense vectors catch synonyms; 6-1-6 is exact match only)
- Works without pre-building a full index (can ingest one document at a time)

### 5.2 Reasoning Engine (TF-IDF-consensus on 6-1-6 maps)

The existing reasoning engine reads 6-1-6 map JSON files, builds SQLite indexes with TF-IDF-consensus ranking, and answers "what does X do?" by finding verbs at position +1 and objects at +2.

**What the binary cell engine improves:**
- **Speed.** The SQLite reasoning engine took 58 minutes to index 838 files. The binary cell engine does it in 21 minutes. Query-time: SQLite does B-tree lookups (~5ms); binary cells do mmap reads (<1ms).
- **Storage honesty.** The SQLite engine mixes raw counts with derived statistics (IDF, consensus, final_rank) in the same tables. The binary cell engine stores raw evidence only; derived stats are a separate layer. This makes the raw data auditable without rank math contamination.
- **No SQL.** SQLite's upsert loop was the dominant bottleneck (45 minutes of 58). Binary cells write once in a single pass and never update.
- **Portable.** The evidence.bin file is a self-contained binary blob. No database server, no connection strings, no schema migrations. Copy the file, point the config, done.
- **Verb-free prediction.** The old engine only worked for "what does X do?" because it required verb detection heuristics. The binary cell engine returns raw positional neighbors — the consumer decides what to do with them. A verb at +1 answers "what does X do?" but a noun at +1 answers "what kind of X?" and an adjective at -1 answers "how is X described?" The 6-1-6 evidence supports all of these without specialized query modes.

**What the old engine did that the binary cell version currently doesn't:**
- Generated prose answer frames (surface text like "Carbon: sequester soil, store atmosphere")
- Had an answer cache
- Ran as its own FastAPI service on port 5051

These are presentation-layer features, not reasoning capabilities. They can be added to the binary cell engine if needed.

### 5.3 The real argument

The 6-1-6 binary cell engine is not a replacement for LakeSpeak or the LLM. It is a **third signal** that neither of them provides:

| Question | LLM | LakeSpeak | 6-1-6 |
|----------|-----|-----------|-------|
| "What is AI used for?" | Generates an answer (may hallucinate) | Returns chunks mentioning AI | Returns: driven(2233), models(1447), powered(1254) at +1 |
| "Is this claim supported?" | Can't verify against corpus | Returns similar chunks (indirect) | Direct count: does this word appear at this position? Yes/no with exact count |
| "What concepts cluster around security?" | Generates a list (may be generic) | Returns top chunks for "security" | Returns 12-position neighborhood with counts, chains in both directions |
| "What came before this concept historically in the corpus?" | Can't answer from corpus structure | Returns chunks ranked by relevance | backward_chain gives literal corpus-derived association paths |

The argument is not "use this instead." It is: **count-based positional evidence is a signal that generative models and retrieval systems cannot produce on their own, and it costs almost nothing to add.**

---

## 6. Performance Characteristics

### 6.1 Ingestion (batch, one-time)

| Phase | 838 files (54MB) | Bottleneck |
|-------|-------------------|------------|
| Tokenize + window count | 17s (16 workers) | CPU-parallel, fast |
| Lexicon resolution | 8s (64K words) | Dict lookups, fast |
| Cell accumulation | 18 min | Pure Python dict ops, 50M edges |
| Evidence write | 3 min | Sequential disk I/O |
| **Total** | **21 min** | Accumulation dominates |

### 6.2 Query (real-time)

| Operation | Time |
|-----------|------|
| predict_next / predict_previous | <1ms |
| full_context (12 positions) | <1ms |
| forward_chain (6 steps) | <5ms |
| Cold load (index parse + mmap) | ~3s |
| Warm query (any anchor) | <1ms |

### 6.3 Storage

| Component | Size |
|-----------|------|
| evidence.bin | 623 MB |
| evidence.bin.idx | 7.3 MB |
| Total for 64,568 cells | 630 MB |

---

## 7. Limitations and Honest Assessments

1. **Accumulation is too slow.** 18 minutes for 50M edges in Python is not acceptable for large corpora. The dict accumulation is correct but needs to move to Cython, Rust, or GPU. The current system is CPU-bound during this phase despite having an RTX 3070 available.

2. **Stopword handling is a patch.** A hardcoded list of 120 words is pragmatic but not principled. Domain-specific corpora may have domain-specific function words that the list doesn't cover. A proper solution would be frequency-based automatic detection: if a word appears in >80% of documents as an anchor, it's probably a function word.

3. **No subword handling.** "Training" and "trained" are different symbols. "Model" and "models" are different symbols. The system does not lemmatize or stem. This means related forms split their counts instead of pooling them.

4. **No semantic similarity.** "Car" and "automobile" are completely unrelated in this system unless they happen to share neighbors. Dense retrieval handles this; positional counting does not.

5. **Single-corpus scope.** The evidence store reflects one corpus. There is no mechanism to merge stores from different corpora, weight one corpus over another, or incrementally update a store without full rebuild.

6. **The evidence store is large.** 623MB for 54MB of input text is an 11.5x expansion. Larger corpora will produce proportionally larger stores. This is the cost of preserving full positional detail for every anchor-neighbor pair.

---

## 8. Conclusion

The 6-1-6 binary cell engine is a deterministic, auditable, fast positional co-occurrence system. It does not think, learn, or generate. It counts what the corpus says and returns structured evidence. Its value is not as a replacement for language models or retrieval systems, but as a complementary signal that provides:

- Exact corpus-grounded evidence for any anchor word
- Positional structure that bag-of-words retrieval cannot capture
- Chain traversal that reveals corpus-level association paths
- Sub-millisecond query performance from a single binary file

It is honest about what it knows (counted evidence), honest about what it doesn't know (returns empty results, not guesses), and honest about its limitations (no semantics, no negation, no generation).

The system is deployed as a self-contained Clearbox plugin with a bundled binary reader, an interactive explorer UI, and six API endpoints. It requires no external dependencies beyond the Python standard library and FastAPI.

---

*Built with Notepad, terminal, and stubbornness — then rebuilt properly.*
