# Embedding Reasoning Trees in Token-Based AI Generation

Deterministic 6-1-6 prediction engine backed by a symbol-first lexicon and binary cell evidence store.

No LLM. No training. No hallucination. Corpus in, structured predictions out.

## What it does

Takes a text corpus, resolves every word against a persistent lexicon (1M+ symbols), builds positional co-occurrence evidence in a binary cell store, and serves predictions through a FastAPI plugin.

Given an anchor word, the engine returns:
- **Positional neighbors** at each of the 12 positions (-6 to -1, +1 to +6)
- **Forward/backward chains** following top neighbors step by step
- **Full 6-1-6 context** — the complete neighborhood of any anchor

Example from 838 ML training documents (54MB corpus):

```
ai       → driven(2233), symbolis(1707), models(1447), powered(1254)
model    → training(429), performance, predict
security → system → performance → metrics → accuracy → error → handling
lexicon  → nlp → models → import → json → schema (forward chain)
```

## Architecture

```
Corpus (text files)
    ↓
Tokenizer (content-only: punctuation/emoji/numbers transparent to window)
    ↓
Lexicon resolution (Canonical → Medical → Structural → Temp_Pool)
    ↓
6-1-6 window counting (parallel, 16 workers)
    ↓
Binary cell accumulation (dict-per-bucket, O(1) neighbor lookup)
    ↓
Evidence store (evidence.bin + bloom-gated index)
    ↓
Prediction engine (top-K, chains, full context)
    ↓
Plugin API (FastAPI, serves explorer UI)
```

## Key design rules

- **Symbol identity is primary** — hex address is the node key, text is rendering metadata
- **Count everything, index selectively** — function words counted as neighbors, never as anchors
- **Raw evidence stays raw** — no filtering at storage time, only at view time
- **Punctuation/emoji recognized but invisible** — get symbols, never break the 6-1-6 window
- **Temp symbols are first-class** — tagged, always; old maps never rewritten on promotion

## Project structure

```
cascade_tokenizer/           # Core package
    binary_cell.py           # BinaryCellV2 store — 12 fixed positional buckets
    predict.py               # Prediction engine — top-K, chains, full context
    lexicon_backend.py       # Lexicon loader — Canonical, Medical, Structural, Temp_Pool
    reasoning_engine.py      # Windowed map generation + tokenizer

Lexical Data/                # Symbol lexicon (not tracked in git)
    Canonical/               # 260K+ permanent word symbols (A-Z JSON files)
    Medical/                 # 735K+ medical term symbols
    Structural/              # Digits, single characters
    Spare_Slots/             # 7.6M+ available for promotion
    Temp_Pool/               # Runtime/session assignment lane

Clearbox-AI-Plugin-Runner-main/
    plugin_runner.py         # Standalone plugin harness
    plugins/cascade_616/     # Plugin: FastAPI router + explorer UI
        api/router.py        # 7 endpoints (status, predict, chain, context, panel)
        api/panel.html       # Interactive 6-1-6 explorer
        core/engine.py       # Self-contained engine, no other plugin deps

run_ingest.py                # Corpus ingestion script
```

## Quick start

### Ingest a corpus

```bash
python run_ingest.py --corpus /path/to/text/files --output ./reasoning_cache
```

### Run the explorer

```bash
cd Clearbox-AI-Plugin-Runner-main/Clearbox-AI-Plugin-Runner-main
PYTHONPATH="plugins:../../" python plugin_runner.py --port 9090
```

Open `http://localhost:9090/api/cascade-616/panel` — type any word, explore.

### Use the prediction API directly

```python
from cascade_tokenizer.predict import Predictor

p = Predictor("reasoning_cache/evidence.bin")

p.predict_next("ai", k=5)
# [('driven', 2233), ('symbolis', 1707), ('models', 1447), ...]

p.forward_chain("security", steps=6)
# [('system', 587), ('performance', 639), ('metrics', 827), ...]

p.full_context("lexicon", k=3)
# {'before_1': [('evolved', 89), ('cortex', 86), ...], 'after_1': [('nlp', 70), ...], ...}
```

## Requirements

- Python 3.10+
- PyTorch (CUDA optional, used for parallel mapping)
- FastAPI + Uvicorn (for plugin runner)
- tqdm, pandas (for ingestion)

## Performance

838 files (54MB) on i9 + RTX 3070:

| Phase | Time |
|-------|------|
| Parallel mapping (16 workers) | 17s |
| Lexicon resolution (64K words) | 8s |
| Binary cell accumulation (50M edges) | 18 min |
| Evidence store write (623MB) | 3 min |
| **Total ingest** | **~21 min** |
| Prediction query | **<1ms** |

## License

MIT
