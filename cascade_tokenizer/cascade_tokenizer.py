"""
CascadeTokenizer: Production tokenizer with embedded 6-1-6 reasoning trees.

Two modes of operation:
  1. Lexicon-backed (production):  load_lexicon() reads the persistent symbol bank.
     Words resolve to deterministic hex-addressed symbols.  Unmatched words get
     temporary symbols from Temp_Pool.
  2. Corpus-backed (legacy/demo):  build_vocabulary() generates a throwaway vocab
     from a text corpus.  Retained for quick tests without the full lexicon.
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict, Counter
import pickle
import json
from pathlib import Path
import logging

from cascade_tokenizer.cascade_token import CascadeToken, ConstraintType, NodeType
from cascade_tokenizer.lexicon_backend import LexiconBackend, SymbolRecord, SymbolStatus


class CascadeVocabulary:
    """Vocabulary management for cascade tokens.

    Token IDs are compact sequential integers suitable for model embedding
    tables.  The lexicon's hex symbol address is preserved on the
    SymbolRecord (the *identity*), but the model only sees small ints.
    """

    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.cascade_tokens: Dict[int, CascadeToken] = {}
        self.symbol_records: Dict[int, SymbolRecord] = {}   # model_id → SymbolRecord
        self.semantic_index: Dict[str, Set[int]] = defaultdict(set)
        self.frequency_map: Dict[int, int] = defaultdict(int)
        self.next_id = 0

        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<MASK>': 4
        }

        for token, token_id in self.special_tokens.items():
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            self.next_id = max(self.next_id, token_id + 1)

    def add_token(self, surface_form: str, cascade_token: CascadeToken = None,
                  symbol_record: SymbolRecord = None) -> int:
        """Add token to vocabulary with a compact sequential ID.

        The SymbolRecord (if provided) is stored alongside for identity /
        metadata, but the returned ID is always a small sequential int.
        """
        if surface_form in self.token_to_id:
            return self.token_to_id[surface_form]

        token_id = self.next_id
        self.next_id += 1

        self.token_to_id[surface_form] = token_id
        self.id_to_token[token_id] = surface_form

        if cascade_token is None:
            cascade_token = CascadeToken(surface_form, token_id)

        self.cascade_tokens[token_id] = cascade_token

        if symbol_record is not None:
            self.symbol_records[token_id] = symbol_record

        # Update semantic index
        if cascade_token.cascade.central_node:
            concept = cascade_token.cascade.central_node.concept
            self.semantic_index[concept].add(token_id)

        return token_id

    def get_token_id(self, surface_form: str) -> int:
        """Get token ID, return UNK if not found"""
        return self.token_to_id.get(surface_form, self.special_tokens['<UNK>'])

    def get_cascade_token(self, token_id: int) -> Optional[CascadeToken]:
        """Get cascade token by ID"""
        return self.cascade_tokens.get(token_id)

    def get_symbol_record(self, token_id: int) -> Optional[SymbolRecord]:
        """Get full symbol record by token ID"""
        return self.symbol_records.get(token_id)

    def find_semantic_neighbors(self, concept: str, max_results: int = 10) -> List[int]:
        """Find tokens with similar semantic concepts"""
        return list(self.semantic_index.get(concept, set()))[:max_results]


class CascadeTokenizer:
    """Production tokenizer with embedded reasoning cascades."""

    def __init__(self, vocab_size: int = 50000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.vocabulary = CascadeVocabulary()
        self.lexicon: Optional[LexiconBackend] = None   # set by load_lexicon()
        self.cascade_templates = {}
        self.constraint_rules = {}
        self.context_window = 512

        # Tokenization patterns
        self.word_pattern = re.compile(r'\b\w+\b|[^\w\s]')
        self.sentence_pattern = re.compile(r'[.!?]+')

        # Logging
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # LEXICON-BACKED MODE (production)
    # ------------------------------------------------------------------

    def load_lexicon(self, lexicon_dir: str, load_medical: bool = True,
                     cascade_definitions: Dict[str, Dict] = None) -> None:
        """Load the persistent lexicon as the vocabulary authority.

        This **replaces** build_vocabulary().  After this call every encode()
        and resolve operation goes through the lexicon backend.
        """
        self.logger.info(f"Loading lexicon from {lexicon_dir} ...")
        self.lexicon = LexiconBackend(lexicon_dir, load_medical=load_medical)
        self._cascade_definitions = cascade_definitions or {}
        self.logger.info("Lexicon loaded and ready.")

    def resolve_and_register(self, word: str, context: str = None,
                             cascade_definitions: Dict[str, Dict] = None) -> int:
        """Resolve a word through the lexicon, register it in the vocabulary,
        attach a cascade, and return its integer token ID.

        This is the single path that feeds the cascade engine.
        """
        if self.lexicon is None:
            raise RuntimeError("No lexicon loaded. Call load_lexicon() first.")

        key = word.lower()

        # Already registered this session?
        if key in self.vocabulary.token_to_id:
            return self.vocabulary.token_to_id[key]

        # Resolve through lexicon (match permanent → else temp assign)
        record = self.lexicon.resolve_token(word, context=context)

        # Build cascade token (ID assigned by vocabulary)
        cascade_token = self._create_cascade_token_from_record(
            record, cascade_definitions or self._cascade_definitions
        )

        # Register in vocabulary (compact sequential ID assigned here)
        token_id = self.vocabulary.add_token(key, cascade_token=cascade_token, symbol_record=record)

        # Update the CascadeToken's own token_id to match the vocabulary's
        cascade_token.token_id = token_id

        self.vocabulary.frequency_map[token_id] = (
            self.vocabulary.frequency_map.get(token_id, 0) + 1
        )

        return token_id

    def _create_cascade_token_from_record(
        self, record: SymbolRecord, cascade_definitions: Dict[str, Dict]
    ) -> CascadeToken:
        """Create a CascadeToken from a SymbolRecord."""
        token = CascadeToken(record.canonical_word, record.to_token_id())
        token.frequency = 1

        # Store the full symbol record as metadata on the token
        token.generation_constraints["symbol_record"] = {
            "hex": record.hex,
            "binary": record.binary,
            "status": record.status.value,
            "source_pool": record.source_pool,
            "tone_signature": record.tone_signature,
            "font_symbol": record.font_symbol,
        }

        # Apply cascade definition if one exists for this word
        word = record.canonical_word
        if word in cascade_definitions:
            self._apply_cascade_definition(token, cascade_definitions[word])
        else:
            self._generate_default_cascade(token)

        return token

    # ------------------------------------------------------------------
    # CORPUS-BACKED MODE (legacy / demo without lexicon)
    # ------------------------------------------------------------------

    def build_vocabulary(self, corpus: List[str], cascade_definitions: Dict[str, Dict] = None) -> None:
        """Build vocabulary from corpus (legacy mode -- no lexicon)."""
        self.logger.info("Building vocabulary from corpus...")

        token_counts = Counter()
        for text in corpus:
            tokens = self._basic_tokenize(text)
            token_counts.update(tokens)

        filtered_tokens = {token: count for token, count in token_counts.most_common(self.vocab_size)
                          if count >= self.min_frequency}

        self.logger.info(f"Filtered vocabulary: {len(filtered_tokens)} tokens")

        for token, frequency in filtered_tokens.items():
            cascade_token = self._create_cascade_token(token, frequency, cascade_definitions)
            token_id = self.vocabulary.add_token(token, cascade_token)
            self.vocabulary.frequency_map[token_id] = frequency

        self.logger.info(f"Vocabulary built: {len(self.vocabulary.token_to_id)} total tokens")

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic word-boundary tokenization."""
        text = text.lower().strip()
        tokens = self.word_pattern.findall(text)
        return [token for token in tokens if token.strip()]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs.

        In lexicon mode each word is resolved on the fly (match → temp assign).
        In legacy mode words must already be in the vocabulary.
        """
        words = self._basic_tokenize(text)
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.vocabulary.special_tokens['<BOS>'])

        for word in words:
            if self.lexicon is not None:
                # Lexicon mode: resolve and register on the fly
                token_id = self.resolve_and_register(word)
            else:
                # Legacy mode
                token_id = self.vocabulary.get_token_id(word)
            token_ids.append(token_id)

        if add_special_tokens:
            token_ids.append(self.vocabulary.special_tokens['<EOS>'])

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        tokens = []

        for token_id in token_ids:
            if skip_special_tokens and token_id in self.vocabulary.special_tokens.values():
                continue

            token = self.vocabulary.id_to_token.get(token_id, '<UNK>')
            tokens.append(token)

        return ' '.join(tokens)

    # ------------------------------------------------------------------
    # Cascade helpers (shared by both modes)
    # ------------------------------------------------------------------

    def _create_cascade_token(self, surface_form: str, frequency: int,
                            cascade_definitions: Dict[str, Dict] = None) -> CascadeToken:
        """Create cascade token with embedded reasoning tree (legacy mode)."""
        token = CascadeToken(surface_form)
        token.frequency = frequency

        if cascade_definitions and surface_form in cascade_definitions:
            cascade_def = cascade_definitions[surface_form]
            self._apply_cascade_definition(token, cascade_def)
        else:
            self._generate_default_cascade(token)

        return token

    def _apply_cascade_definition(self, token: CascadeToken, cascade_def: Dict) -> None:
        """Apply predefined cascade definition to token."""
        input_concepts = cascade_def.get('input_concepts', [])
        central_concept = cascade_def.get('central_concept', token.surface_form)
        output_concepts = cascade_def.get('output_concepts', [])
        weights = cascade_def.get('weights', None)

        token.embed_cascade(input_concepts, central_concept, output_concepts, weights)

        constraints = cascade_def.get('constraints', {})
        for node_id, node_constraints in constraints.items():
            for constraint_type_str, constraint_value in node_constraints.items():
                constraint_type = ConstraintType(constraint_type_str)
                token.add_constraint(node_id, constraint_type, constraint_value)

    def _generate_default_cascade(self, token: CascadeToken) -> None:
        """Generate default cascade for token based on linguistic properties."""
        surface = token.surface_form

        if surface.isalpha():
            if len(surface) <= 3:
                input_concepts = ["syntax", "grammar"]
                output_concepts = ["continuation", "structure"]
            elif surface.endswith('ing'):
                input_concepts = ["action", "process", "ongoing"]
                output_concepts = ["verb", "noun", "modifier"]
            elif surface.endswith('ed'):
                input_concepts = ["completed", "past", "state"]
                output_concepts = ["description", "sequence", "result"]
            else:
                input_concepts = ["meaning", "context"]
                output_concepts = ["association", "continuation"]
        else:
            input_concepts = ["structure", "boundary"]
            output_concepts = ["pause", "emphasis", "separation"]

        central_concept = surface
        token.embed_cascade(input_concepts, central_concept, output_concepts)

    # ------------------------------------------------------------------
    # Generation (tokenizer-level, cascade-guided)
    # ------------------------------------------------------------------

    def guided_generation_step(self, context_ids: List[int], temperature: float = 1.0,
                             top_k: int = 50, constraint_check: bool = True) -> Tuple[int, float, List[str]]:
        """Generate next token using cascade guidance."""
        if not context_ids:
            return self.vocabulary.special_tokens['<BOS>'], 1.0, []

        last_token_id = context_ids[-1]
        last_cascade_token = self.vocabulary.get_cascade_token(last_token_id)

        if not last_cascade_token:
            return self._fallback_generation(context_ids, temperature)

        candidates = last_cascade_token.get_generation_candidates(temperature=temperature)

        if not candidates:
            return self._fallback_generation(context_ids, temperature)

        candidate_token_ids = []
        candidate_weights = []
        constraint_violations = []

        for concept, weight in candidates[:top_k]:
            matching_token_ids = self.vocabulary.find_semantic_neighbors(concept, max_results=5)

            for token_id in matching_token_ids:
                if constraint_check:
                    token = self.vocabulary.id_to_token.get(token_id, '')
                    is_valid, violations = last_cascade_token.validate_constraints(token)
                    if not is_valid:
                        constraint_violations.extend(violations)
                        continue

                candidate_token_ids.append(token_id)
                candidate_weights.append(weight)

        if not candidate_token_ids:
            return self._fallback_generation(context_ids, temperature)

        weights_array = np.array(candidate_weights)
        weights_array = np.exp(weights_array / temperature)
        probabilities = weights_array / np.sum(weights_array)

        selected_idx = np.random.choice(len(candidate_token_ids), p=probabilities)
        selected_token_id = candidate_token_ids[selected_idx]
        selected_probability = probabilities[selected_idx]

        return selected_token_id, selected_probability, constraint_violations

    def _fallback_generation(self, context_ids: List[int], temperature: float) -> Tuple[int, float, List[str]]:
        """Fallback generation when cascade guidance fails."""
        all_token_ids = list(self.vocabulary.frequency_map.keys())
        if not all_token_ids:
            return self.vocabulary.special_tokens['<UNK>'], 0.0, ["fallback_generation", "empty_vocab"]

        frequencies = [self.vocabulary.frequency_map[tid] for tid in all_token_ids]

        frequencies_array = np.array(frequencies, dtype=float)
        frequencies_array = np.exp(frequencies_array / temperature)
        probabilities = frequencies_array / np.sum(frequencies_array)

        selected_idx = np.random.choice(len(all_token_ids), p=probabilities)
        selected_token_id = all_token_ids[selected_idx]
        selected_probability = probabilities[selected_idx]

        return selected_token_id, selected_probability, ["fallback_generation"]

    def generate_text(self, prompt: str = "", max_length: int = 100,
                     temperature: float = 1.0, top_k: int = 50) -> Tuple[str, Dict]:
        """Generate text using cascade-guided sampling."""
        if prompt:
            context_ids = self.encode(prompt, add_special_tokens=False)
        else:
            context_ids = [self.vocabulary.special_tokens['<BOS>']]

        generated_ids = context_ids.copy()
        generation_log = {
            'steps': [],
            'constraint_violations': [],
            'fallback_count': 0
        }

        for step in range(max_length):
            token_id, probability, violations = self.guided_generation_step(
                generated_ids, temperature, top_k
            )

            if token_id == self.vocabulary.special_tokens['<EOS>']:
                break

            generated_ids.append(token_id)

            step_info = {
                'step': step,
                'token_id': token_id,
                'token': self.vocabulary.id_to_token.get(token_id, '<UNK>'),
                'probability': probability,
                'violations': violations
            }
            generation_log['steps'].append(step_info)
            generation_log['constraint_violations'].extend(violations)

            if 'fallback_generation' in violations:
                generation_log['fallback_count'] += 1

        generated_text = self.decode(generated_ids)
        return generated_text, generation_log

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Save tokenizer to file."""
        save_data = {
            'vocabulary': {
                'token_to_id': self.vocabulary.token_to_id,
                'id_to_token': {str(k): v for k, v in self.vocabulary.id_to_token.items()},
                'frequency_map': {str(k): v for k, v in self.vocabulary.frequency_map.items()},
                'semantic_index': {k: list(v) for k, v in self.vocabulary.semantic_index.items()},
                'next_id': self.vocabulary.next_id
            },
            'cascade_tokens': {str(tid): token.serialize() for tid, token in self.vocabulary.cascade_tokens.items()},
            'config': {
                'vocab_size': self.vocab_size,
                'min_frequency': self.min_frequency,
                'context_window': self.context_window
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        self.logger.info(f"Tokenizer saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'CascadeTokenizer':
        """Load tokenizer from file."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        tokenizer = cls(
            vocab_size=save_data['config']['vocab_size'],
            min_frequency=save_data['config']['min_frequency']
        )

        vocab_data = save_data['vocabulary']
        tokenizer.vocabulary.token_to_id = vocab_data['token_to_id']
        tokenizer.vocabulary.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
        tokenizer.vocabulary.frequency_map = defaultdict(int,
            {int(k): v for k, v in vocab_data['frequency_map'].items()})
        tokenizer.vocabulary.semantic_index = defaultdict(set,
            {k: set(v) for k, v in vocab_data['semantic_index'].items()})
        tokenizer.vocabulary.next_id = vocab_data['next_id']

        for tid_str, token_data in save_data['cascade_tokens'].items():
            tid = int(tid_str)
            cascade_token = CascadeToken.deserialize(token_data)
            tokenizer.vocabulary.cascade_tokens[tid] = cascade_token

        tokenizer.context_window = save_data['config']['context_window']

        return tokenizer
