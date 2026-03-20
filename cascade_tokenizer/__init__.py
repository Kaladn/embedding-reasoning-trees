"""
Cascade Tokenizer: Production-ready AI with embedded 6-1-6 reasoning trees
"""

__version__ = "1.0.0"

from cascade_tokenizer.cascade_token import (
    CascadeToken,
    CascadeNode,
    ReasoningCascade,
    NodeType,
    ConstraintType,
)
from cascade_tokenizer.cascade_tokenizer import CascadeTokenizer, CascadeVocabulary
from cascade_tokenizer.lexicon_backend import LexiconBackend, SymbolRecord, SymbolStatus
from cascade_tokenizer.cascade_model import (
    CascadeModel,
    CascadeModelConfig,
    CascadeEmbedding,
    CascadeAttention,
    CascadeTransformerLayer,
)
from cascade_tokenizer.cascade_trainer import (
    CascadeTrainer,
    CascadeDataset,
    CascadeDataPreprocessor,
    TrainingConfig,
)
from cascade_tokenizer.cascade_inference import (
    CascadeInferenceEngine,
    InferenceConfig,
)
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
    # Token structures
    "CascadeToken", "CascadeNode", "ReasoningCascade", "NodeType", "ConstraintType",
    # Tokenizer
    "CascadeTokenizer", "CascadeVocabulary",
    # Lexicon
    "LexiconBackend", "SymbolRecord", "SymbolStatus",
    # Neural model (for future training)
    "CascadeModel", "CascadeModelConfig", "CascadeEmbedding",
    "CascadeAttention", "CascadeTransformerLayer",
    "CascadeTrainer", "CascadeDataset", "CascadeDataPreprocessor", "TrainingConfig",
    "CascadeInferenceEngine", "InferenceConfig",
    # 6-1-6 mapping
    "content_tokenize", "compute_window_counts", "map_text",
    "MapReport", "SUPPRESSED_ANCHOR_CLASSES",
    # Binary cell storage
    "BinaryCellV2", "NeighborEntry", "CellStore", "MasterIndex", "CorpusStats",
    # Prediction engine
    "Predictor", "STOP_WORDS",
]
