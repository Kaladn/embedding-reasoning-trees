"""
CascadeDemo: Production demonstration of cascade-guided tokenizer
End-to-end example using the persistent lexicon backend
"""

import torch
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from cascade_tokenizer import (
    CascadeTokenizer,
    CascadeModel,
    CascadeModelConfig,
    CascadeTrainer,
    CascadeDataset,
    TrainingConfig,
    CascadeInferenceEngine,
    InferenceConfig,
    LexiconBackend,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default lexicon path (relative to project root)
DEFAULT_LEXICON_DIR = str(Path(__file__).parent / "Lexical Data")


class CascadeSystem:
    """Complete cascade-guided generation system backed by the persistent lexicon."""

    def __init__(self, lexicon_dir: str = None, device: str = 'cpu'):
        self.device = device
        self.lexicon_dir = lexicon_dir or DEFAULT_LEXICON_DIR
        self.tokenizer = None
        self.model = None
        self.inference_engine = None

    def build(self, corpus: List[str] = None, vocab_size: int = 30000,
              cascade_definitions: Dict[str, Dict] = None,
              load_medical: bool = True) -> None:
        """Build the system.

        1. Load the lexicon (symbol authority).
        2. If a corpus is provided, pre-resolve all its words so the model's
           vocab_size can be computed from actual data.  Otherwise the model
           uses *vocab_size* as an upper bound.
        3. Create the model and inference engine.
        """
        logger.info("Building Cascade System (lexicon-backed)")

        # Step 1: Tokenizer + lexicon
        self.tokenizer = CascadeTokenizer()
        self.tokenizer.load_lexicon(
            self.lexicon_dir,
            load_medical=load_medical,
            cascade_definitions=cascade_definitions,
        )

        # Step 2: Pre-resolve corpus words so they're in the vocabulary
        if corpus:
            logger.info(f"Pre-resolving {len(corpus)} corpus texts...")
            for text in corpus:
                self.tokenizer.encode(text, add_special_tokens=False)
            actual_vocab = len(self.tokenizer.vocabulary.token_to_id)
            logger.info(f"Vocabulary after corpus scan: {actual_vocab} tokens")
        else:
            actual_vocab = vocab_size

        # Step 3: Model
        logger.info("Creating cascade model...")
        model_config = CascadeModelConfig(
            vocab_size=max(actual_vocab + 1000, vocab_size),  # headroom
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            intermediate_size=512,
            max_position_embeddings=512,
            cascade_embedding_size=64,
            cascade_fusion_layers=2,
        )

        self.model = CascadeModel(model_config, self.tokenizer)
        self.model.to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model created with {param_count:,} parameters")

        # Step 4: Inference engine
        self.inference_engine = CascadeInferenceEngine(
            self.model, self.tokenizer, device=self.device
        )
        logger.info("Inference engine ready")

    def train(self, train_corpus: List[str], eval_corpus: List[str] = None,
              training_config: TrainingConfig = None) -> Dict[str, Any]:
        """Train the cascade model."""
        logger.info("Training Cascade Model")

        if training_config is None:
            training_config = TrainingConfig(
                batch_size=8,
                num_epochs=3,
                learning_rate=1e-4,
                save_steps=500,
                eval_steps=250,
            )

        train_dataset = CascadeDataset(train_corpus, self.tokenizer, max_length=64)
        eval_dataset = CascadeDataset(eval_corpus, self.tokenizer, max_length=64) if eval_corpus else None

        trainer = CascadeTrainer(self.model, self.tokenizer, training_config)
        training_stats = trainer.train(train_dataset, eval_dataset)

        logger.info("Training completed!")
        return training_stats

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text with cascade guidance."""
        if self.inference_engine is None:
            raise ValueError("System not initialized. Call build() first.")
        return self.inference_engine.generate(prompt, **kwargs)

    def save(self, save_dir: str) -> None:
        """Save complete system."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save(str(save_path / "tokenizer.pkl"))
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.config.__dict__
        }, str(save_path / "model.pt"))

        # Also save the temp registry for audit/promotion
        if self.tokenizer.lexicon:
            self.tokenizer.lexicon.save_temp_registry(
                str(save_path / "temp_registry.json")
            )

        logger.info(f"System saved to {save_dir}")

    @classmethod
    def load(cls, model_dir: str, lexicon_dir: str = None, device: str = 'cpu') -> 'CascadeSystem':
        """Load a saved system."""
        system = cls(lexicon_dir=lexicon_dir, device=device)

        system.tokenizer = CascadeTokenizer.load(str(Path(model_dir) / "tokenizer.pkl"))
        # Re-attach lexicon if available
        if lexicon_dir:
            system.tokenizer.load_lexicon(lexicon_dir)

        model_data = torch.load(str(Path(model_dir) / "model.pt"), map_location=device, weights_only=False)
        model_config = CascadeModelConfig(**model_data['model_config'])
        system.model = CascadeModel(model_config, system.tokenizer)
        system.model.load_state_dict(model_data['model_state_dict'])
        system.model.to(device)

        system.inference_engine = CascadeInferenceEngine(
            system.model, system.tokenizer, device=device
        )

        logger.info(f"System loaded from {model_dir}")
        return system


# ------------------------------------------------------------------
# Demo data
# ------------------------------------------------------------------

def create_demo_corpus() -> List[str]:
    return [
        "The artificial intelligence system processes natural language with remarkable accuracy.",
        "Machine learning algorithms learn patterns from large datasets to make predictions.",
        "Neural networks consist of interconnected nodes that simulate brain neurons.",
        "Deep learning models can recognize images, understand speech, and generate text.",
        "Natural language processing enables computers to understand human communication.",
        "Computer vision systems can identify objects and analyze visual scenes.",
        "Robotics combines AI with mechanical engineering to create autonomous machines.",
        "Data science involves extracting insights from complex datasets using statistical methods.",
        "Quantum computing promises to solve problems that are intractable for classical computers.",
        "Blockchain technology provides secure and decentralized transaction recording.",
        "The scientific method involves hypothesis formation, experimentation, and analysis.",
        "Research requires careful observation, data collection, and peer review.",
        "Innovation drives technological progress and societal advancement.",
        "Engineering applies scientific principles to design and build practical solutions.",
        "Mathematics provides the foundation for logical reasoning and problem solving.",
        "Physics explores the fundamental laws governing matter and energy.",
        "Chemistry studies the composition and behavior of atoms and molecules.",
        "Biology investigates living organisms and their interactions with the environment.",
        "Psychology examines human behavior, cognition, and mental processes.",
        "Philosophy questions the nature of reality, knowledge, and existence.",
        "The economy involves the production, distribution, and consumption of goods and services.",
        "Markets facilitate trade between buyers and sellers through price mechanisms.",
        "Finance manages money, investments, and financial risk across time.",
        "Business strategy focuses on competitive advantage and value creation.",
        "Entrepreneurship involves identifying opportunities and creating new ventures.",
        "Leadership requires vision, communication, and the ability to inspire others.",
        "Teamwork combines diverse skills and perspectives to achieve common goals.",
        "Communication enables the exchange of ideas and information between people.",
        "Education develops knowledge, skills, and critical thinking abilities.",
        "Learning is a continuous process of acquiring new understanding and capabilities."
    ]


def create_cascade_definitions() -> Dict[str, Dict]:
    return {
        "intelligence": {
            "central_concept": "intelligence",
            "input_concepts": ["cognition", "reasoning", "learning", "adaptation", "problem", "analysis"],
            "output_concepts": ["artificial", "system", "algorithm", "decision", "solution", "capability"],
            "weights": [0.9, 0.8, 0.9, 0.7, 0.8, 0.7, 0.9, 0.8, 0.7, 0.8, 0.7, 0.6, 0.8],
            "constraints": {
                "central_intelligence": {
                    "semantic": ["cognitive", "analytical", "computational"],
                    "logical": ["requires_input", "produces_output"]
                }
            }
        },
        "learning": {
            "central_concept": "learning",
            "input_concepts": ["data", "experience", "training", "pattern", "feedback", "iteration"],
            "output_concepts": ["knowledge", "skill", "model", "prediction", "adaptation", "improvement"],
            "weights": [0.9, 0.8, 0.9, 0.8, 0.7, 0.8, 0.9, 0.8, 0.9, 0.8, 0.7, 0.8, 0.7],
            "constraints": {}
        },
        "system": {
            "central_concept": "system",
            "input_concepts": ["component", "input", "process", "structure", "design", "architecture"],
            "output_concepts": ["output", "behavior", "function", "performance", "result", "operation"],
            "weights": [0.8, 0.9, 0.8, 0.7, 0.8, 0.7, 0.9, 0.8, 0.8, 0.7, 0.8, 0.7, 0.6],
            "constraints": {}
        },
        "algorithm": {
            "central_concept": "algorithm",
            "input_concepts": ["logic", "steps", "rules", "computation", "method", "procedure"],
            "output_concepts": ["solution", "result", "optimization", "efficiency", "automation", "processing"],
            "weights": [0.9, 0.8, 0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7, 0.8, 0.7, 0.8, 0.6],
            "constraints": {}
        }
    }


# ------------------------------------------------------------------
# Demo runners
# ------------------------------------------------------------------

def run_comprehensive_demo():
    """Full demo: lexicon load → train → generate → save."""
    logger.info("=== COMPREHENSIVE CASCADE DEMO (LEXICON-BACKED) ===")

    corpus = create_demo_corpus()
    cascade_defs = create_cascade_definitions()

    train_corpus = corpus[:24]
    eval_corpus = corpus[24:]

    system = CascadeSystem(device='cpu')
    system.build(corpus=train_corpus, vocab_size=5000, cascade_definitions=cascade_defs)

    # Show lexicon stats
    lex_stats = system.tokenizer.lexicon.stats()
    logger.info(f"Lexicon: {lex_stats['canonical_lookups']} canonical hits, "
                f"{lex_stats['temp_assignments']} temp assignments")

    # Quick training
    training_config = TrainingConfig(
        batch_size=2, num_epochs=1, learning_rate=1e-4,
        save_steps=100, eval_steps=50, output_dir="./demo_output"
    )
    logger.info("Starting training...")
    system.train(train_corpus, eval_corpus, training_config)

    # Generate
    prompts = [
        "The artificial intelligence",
        "Machine learning algorithms",
        "Neural networks can",
    ]

    logger.info("Generating with cascade guidance...")
    for prompt in prompts:
        result = system.generate(prompt, max_length=50, temperature=0.8)
        logger.info(f"  Prompt: '{prompt}'")
        logger.info(f"  Output: '{result['generated_text']}'")
        logger.info(f"  Time:   {result['generation_time']:.3f}s")

        steps = result['generation_log'].get('steps', [])
        guided = sum(1 for s in steps if s.get('cascade_guided', False))
        logger.info(f"  Cascade-guided: {guided}/{len(steps)} steps")

    # Save
    system.save("./demo_cascade_system")

    # Final stats
    lex_stats = system.tokenizer.lexicon.stats()
    gen_stats = system.inference_engine.get_statistics()
    logger.info("\n--- Lexicon Stats ---")
    for k, v in lex_stats.items():
        logger.info(f"  {k}: {v}")
    logger.info("\n--- Generation Stats ---")
    for k, v in gen_stats.items():
        logger.info(f"  {k}: {v}")

    logger.info("\n=== DEMO COMPLETED SUCCESSFULLY ===")


def run_quick_inference_demo():
    """Quick demo: lexicon load → encode/decode → generate (no training)."""
    logger.info("=== QUICK INFERENCE DEMO (LEXICON-BACKED) ===")

    corpus = create_demo_corpus()[:10]
    cascade_defs = create_cascade_definitions()

    system = CascadeSystem(device='cpu')
    system.build(corpus=corpus, vocab_size=1000, cascade_definitions=cascade_defs)

    # Show what happened during corpus resolution
    lex = system.tokenizer.lexicon
    stats = lex.stats()
    logger.info(f"Resolved: {stats['canonical_lookups']} canonical, "
                f"{stats['medical_lookups']} medical, "
                f"{stats['structural_lookups']} structural, "
                f"{stats['temp_assignments']} temp-assigned")

    # Show a few symbol records
    sample_words = ["intelligence", "neural", "the", "quantum"]
    logger.info("\n--- Sample Symbol Records ---")
    for word in sample_words:
        tid = system.tokenizer.vocabulary.get_token_id(word)
        rec = system.tokenizer.vocabulary.get_symbol_record(tid)
        if rec:
            logger.info(f"  {word:20s} | {rec.status.value:14s} | {rec.source_pool:10s} | {rec.hex}")
        else:
            logger.info(f"  {word:20s} | NOT IN VOCAB")

    # Generate
    prompts = ["artificial intelligence", "machine learning", "neural networks"]
    for prompt in prompts:
        result = system.generate(prompt, max_length=30, temperature=0.8)
        logger.info(f"\nPrompt: '{prompt}' -> '{result['generated_text']}'")
        logger.info(f"  Time: {result['generation_time']:.3f}s")

    # Save temp registry
    lex.save_temp_registry("./demo_temp_registry.json")

    logger.info("\n=== QUICK DEMO COMPLETE ===")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cascade System Demo")
    parser.add_argument("--mode", choices=["full", "quick"], default="quick",
                        help="Demo mode: full (with training) or quick (inference only)")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--lexicon-dir", default=None,
                        help="Path to Lexical Data directory (auto-detected if omitted)")

    args = parser.parse_args()

    if args.lexicon_dir:
        global DEFAULT_LEXICON_DIR
        DEFAULT_LEXICON_DIR = args.lexicon_dir

    if args.mode == "full":
        run_comprehensive_demo()
    else:
        run_quick_inference_demo()


if __name__ == "__main__":
    main()
