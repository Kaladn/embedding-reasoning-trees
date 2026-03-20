"""
Cascade Tokenizer: 6-1-6 prediction engine with lexicon-backed binary cell storage
"""

__version__ = "1.0.0"

from cascade_tokenizer.lexicon_backend import LexiconBackend, SymbolRecord, SymbolStatus
from cascade_tokenizer.reasoning_engine import (
    content_tokenize,
    compute_window_counts,
    map_text,
    MapReport,
    SUPPRESSED_ANCHOR_CLASSES,
)
from cascade_tokenizer.binary_cell import (
    BinaryCellV2,
    NeighborEntry,
    CellStore,
    MasterIndex,
    CorpusStats,
)
from cascade_tokenizer.predict import Predictor, STOP_WORDS

__all__ = [
    # Lexicon
    "LexiconBackend", "SymbolRecord", "SymbolStatus",
    # 6-1-6 mapping
    "content_tokenize", "compute_window_counts", "map_text",
    "MapReport", "SUPPRESSED_ANCHOR_CLASSES",
    # Binary cell storage
    "BinaryCellV2", "NeighborEntry", "CellStore", "MasterIndex", "CorpusStats",
    # Prediction engine
    "Predictor", "STOP_WORDS",
]
