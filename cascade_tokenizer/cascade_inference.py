"""
CascadeInference: Production inference engine for cascade-guided text generation
Optimized runtime with constraint validation and deterministic reasoning paths
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import time
from dataclasses import dataclass, fields
from collections import defaultdict
import logging

from cascade_tokenizer.cascade_model import CascadeModel, CascadeModelConfig
from cascade_tokenizer.cascade_tokenizer import CascadeTokenizer
from cascade_tokenizer.cascade_token import CascadeToken, ConstraintType, NodeType


@dataclass
class InferenceConfig:
    """Configuration for cascade inference"""
    max_length: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    constraint_weight: float = 0.3
    cascade_guidance_strength: float = 0.8
    fallback_threshold: float = 0.1
    beam_size: int = 1
    do_sample: bool = True
    early_stopping: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 3
    min_length: int = 10


# Names of valid InferenceConfig fields, used for safe kwarg filtering
_INFERENCE_CONFIG_FIELDS = {f.name for f in fields(InferenceConfig)}


class CascadeInferenceEngine:
    """High-performance inference engine for cascade-guided generation"""

    def __init__(self, model: CascadeModel, tokenizer: CascadeTokenizer,
                 config: InferenceConfig = None, device: str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()
        self.device = device

        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.generation_stats = {
            'total_generations': 0,
            'cascade_guided_steps': 0,
            'fallback_steps': 0,
            'constraint_violations': 0,
            'avg_generation_time': 0.0
        }

        # Constraint cache for performance
        self.constraint_cache = {}

    def generate(self, prompt: str = "", **kwargs) -> Dict[str, Any]:
        """Main generation interface"""
        start_time = time.time()

        # Override config with kwargs
        generation_config = self._merge_configs(kwargs)

        # Encode prompt
        if prompt:
            input_ids = torch.tensor([self.tokenizer.encode(prompt, add_special_tokens=True)],
                                   dtype=torch.long, device=self.device)
        else:
            input_ids = torch.tensor([[self.tokenizer.vocabulary.special_tokens['<BOS>']]],
                                   dtype=torch.long, device=self.device)

        # Generate based on method
        if generation_config.beam_size > 1:
            generated_ids, generation_log = self._beam_search_generate(input_ids, generation_config)
        else:
            generated_ids, generation_log = self._greedy_or_sample_generate(input_ids, generation_config)

        # Decode result
        generated_text = self.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)

        # Update stats
        generation_time = time.time() - start_time
        self.generation_stats['total_generations'] += 1
        self.generation_stats['avg_generation_time'] = (
            (self.generation_stats['avg_generation_time'] * (self.generation_stats['total_generations'] - 1) +
             generation_time) / self.generation_stats['total_generations']
        )

        return {
            'generated_text': generated_text,
            'generated_ids': generated_ids[0].tolist(),
            'generation_log': generation_log,
            'generation_time': generation_time,
            'prompt': prompt
        }

    def _merge_configs(self, kwargs: Dict[str, Any]) -> InferenceConfig:
        """Merge generation kwargs with default config, ignoring unknown keys."""
        config_dict = self.config.__dict__.copy()
        # Only apply keys that are valid InferenceConfig fields
        for key, value in kwargs.items():
            if key in _INFERENCE_CONFIG_FIELDS:
                config_dict[key] = value
        return InferenceConfig(**config_dict)

    def _greedy_or_sample_generate(self, input_ids: torch.Tensor,
                                 config: InferenceConfig) -> Tuple[torch.Tensor, Dict]:
        """Greedy or sampling generation with cascade guidance"""
        batch_size, seq_len = input_ids.shape
        generated_ids = input_ids.clone()
        generation_log = {
            'steps': [],
            'cascade_decisions': [],
            'constraint_checks': [],
            'reasoning_paths': []
        }

        # Generation loop
        for step in range(config.max_length - seq_len):
            with torch.no_grad():
                # Get cascade tokens for current sequence
                cascade_tokens = self._get_cascade_tokens(generated_ids[0])

                # Forward pass
                outputs = self.model(
                    input_ids=generated_ids,
                    cascade_tokens=cascade_tokens
                )

                # Get next token logits
                next_token_logits = outputs['logits'][:, -1, :]
                cascade_scores = outputs['cascade_scores'][:, -1, :]
                constraint_validity = outputs['constraint_validity'][:, -1, :]

                # Apply cascade guidance
                guided_logits, cascade_info = self._apply_cascade_guidance(
                    next_token_logits, cascade_scores, constraint_validity,
                    cascade_tokens[-1] if cascade_tokens else None, config
                )

                # Apply generation constraints
                guided_logits = self._apply_generation_constraints(
                    guided_logits, generated_ids, config
                )

                # Sample next token
                next_token_id = self._sample_next_token(guided_logits, config)

                # Validate constraints
                constraint_check = self._validate_generation_constraints(
                    next_token_id, cascade_tokens[-1] if cascade_tokens else None
                )

                # Log step
                step_info = {
                    'step': step,
                    'token_id': next_token_id.item(),
                    'token': self.tokenizer.vocabulary.id_to_token.get(next_token_id.item(), '<UNK>'),
                    'cascade_guided': cascade_info['guided'],
                    'constraint_valid': constraint_check['valid'],
                    'reasoning_path': cascade_info.get('reasoning_path', [])
                }
                generation_log['steps'].append(step_info)
                generation_log['cascade_decisions'].append(cascade_info)
                generation_log['constraint_checks'].append(constraint_check)

                # Update stats
                if cascade_info['guided']:
                    self.generation_stats['cascade_guided_steps'] += 1
                else:
                    self.generation_stats['fallback_steps'] += 1

                if not constraint_check['valid']:
                    self.generation_stats['constraint_violations'] += 1

                # Append token
                generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)

                # Check stopping conditions
                if next_token_id.item() == config.eos_token_id and step >= config.min_length:
                    break

        return generated_ids, generation_log

    def _get_cascade_tokens(self, token_ids: torch.Tensor) -> List[CascadeToken]:
        """Get cascade tokens for sequence"""
        cascade_tokens = []
        for token_id in token_ids:
            cascade_token = self.tokenizer.vocabulary.get_cascade_token(token_id.item())
            cascade_tokens.append(cascade_token)
        return cascade_tokens

    def _apply_cascade_guidance(self, logits: torch.Tensor, cascade_scores: torch.Tensor,
                              constraint_validity: torch.Tensor, last_cascade_token: CascadeToken,
                              config: InferenceConfig) -> Tuple[torch.Tensor, Dict]:
        """Apply cascade reasoning guidance to logits"""
        cascade_info = {'guided': False, 'reasoning_path': [], 'confidence': 0.0}

        if last_cascade_token is None:
            return logits, cascade_info

        # Get generation candidates from cascade
        candidates = last_cascade_token.get_generation_candidates(temperature=config.temperature)

        if not candidates:
            return logits, cascade_info

        # Create cascade guidance mask
        cascade_mask = torch.zeros_like(logits)
        reasoning_path = []

        for concept, weight in candidates:
            matching_token_ids = self.tokenizer.vocabulary.find_semantic_neighbors(concept, max_results=10)

            for token_id in matching_token_ids:
                if token_id < logits.shape[-1]:
                    cascade_mask[0, token_id] += weight * config.cascade_guidance_strength
                    reasoning_path.append({
                        'concept': concept,
                        'token_id': token_id,
                        'token': self.tokenizer.vocabulary.id_to_token.get(token_id, '<UNK>'),
                        'weight': weight
                    })

        # Apply cascade guidance
        if torch.sum(cascade_mask) > 0:
            guided_logits = logits + config.constraint_weight * cascade_mask
            cascade_info['guided'] = True
            cascade_info['reasoning_path'] = reasoning_path
            cascade_info['confidence'] = torch.max(cascade_mask).item()
        else:
            guided_logits = logits

        # cascade_scores shape: [batch, 1] -- broadcast-add as a global bias
        guided_logits = guided_logits + config.constraint_weight * cascade_scores.expand_as(guided_logits)

        return guided_logits, cascade_info

    def _apply_generation_constraints(self, logits: torch.Tensor, generated_ids: torch.Tensor,
                                    config: InferenceConfig) -> torch.Tensor:
        """Apply generation-level constraints (repetition penalty, etc.)"""
        if config.repetition_penalty != 1.0:
            for token_id in set(generated_ids[0].tolist()):
                if token_id < logits.shape[-1]:
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= config.repetition_penalty
                    else:
                        logits[0, token_id] *= config.repetition_penalty

        return logits

    def _sample_next_token(self, logits: torch.Tensor, config: InferenceConfig) -> torch.Tensor:
        """Sample next token using specified strategy"""
        # Temperature scaling
        if config.temperature != 1.0:
            logits = logits / config.temperature

        # Top-k filtering
        if config.top_k > 0:
            k = min(config.top_k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
            logits = torch.full_like(logits, -float('inf'))
            logits.scatter_(-1, top_k_indices, top_k_logits)

        # Top-p (nucleus) filtering
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')

        # Sample or take argmax
        if config.do_sample:
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token_id.squeeze(0)

    def _validate_generation_constraints(self, token_id: torch.Tensor,
                                       cascade_token: CascadeToken) -> Dict[str, Any]:
        """Validate generated token against cascade constraints"""
        constraint_check = {'valid': True, 'violations': []}

        if cascade_token is None:
            return constraint_check

        tid = token_id.item() if isinstance(token_id, torch.Tensor) else token_id
        token_text = self.tokenizer.vocabulary.id_to_token.get(tid, '')

        # Use cached validation if available
        cache_key = f"{cascade_token.cascade.cascade_id}_{tid}"
        if cache_key in self.constraint_cache:
            return self.constraint_cache[cache_key]

        # Validate constraints
        is_valid, violations = cascade_token.validate_constraints(token_text)
        constraint_check['valid'] = is_valid
        constraint_check['violations'] = violations

        # Cache result
        self.constraint_cache[cache_key] = constraint_check

        return constraint_check

    def _beam_search_generate(self, input_ids: torch.Tensor,
                            config: InferenceConfig) -> Tuple[torch.Tensor, Dict]:
        """Beam search generation with cascade guidance"""
        batch_size, seq_len = input_ids.shape
        beam_size = config.beam_size

        # Initialize beams
        beams = [(input_ids.clone(), 0.0, [])]

        for step in range(config.max_length - seq_len):
            new_beams = []

            for beam_seq, beam_score, beam_log in beams:
                cascade_tokens = self._get_cascade_tokens(beam_seq[0])

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=beam_seq,
                        cascade_tokens=cascade_tokens
                    )

                    next_token_logits = outputs['logits'][:, -1, :]
                    cascade_scores = outputs['cascade_scores'][:, -1, :]
                    constraint_validity = outputs['constraint_validity'][:, -1, :]

                    guided_logits, cascade_info = self._apply_cascade_guidance(
                        next_token_logits, cascade_scores, constraint_validity,
                        cascade_tokens[-1] if cascade_tokens else None, config
                    )

                    log_probs = F.log_softmax(guided_logits, dim=-1)
                    k = min(beam_size, log_probs.size(-1))
                    top_k_probs, top_k_indices = torch.topk(log_probs, k, dim=-1)

                    for i in range(k):
                        token_id = top_k_indices[0, i]
                        token_prob = top_k_probs[0, i]

                        new_seq = torch.cat([beam_seq, token_id.unsqueeze(0).unsqueeze(0)], dim=-1)
                        new_score = beam_score + token_prob.item()
                        new_log = beam_log + [cascade_info]

                        new_beams.append((new_seq, new_score, new_log))

            # Keep top beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

            # Check stopping condition
            if all(beam[0][0, -1].item() == config.eos_token_id for beam in beams):
                break

        best_beam = beams[0]
        generation_log = {'beam_search_log': best_beam[2], 'final_score': best_beam[1]}

        return best_beam[0], generation_log

    def batch_generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate text for multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results

    def interactive_generate(self, initial_prompt: str = "", **kwargs) -> None:
        """Interactive generation session"""
        print("Cascade Interactive Generation")
        print("Commands: /help, /stats, /config, /quit")
        print("-" * 50)

        current_text = initial_prompt

        while True:
            user_input = input(f"\nPrompt: {current_text}\n> ").strip()

            if user_input == "/quit":
                break
            elif user_input == "/help":
                print("Commands:")
                print("  /help - Show this help")
                print("  /stats - Show generation statistics")
                print("  /config - Show current config")
                print("  /quit - Exit interactive mode")
                continue
            elif user_input == "/stats":
                print("Generation Statistics:")
                for key, value in self.generation_stats.items():
                    print(f"  {key}: {value}")
                continue
            elif user_input == "/config":
                print("Current Configuration:")
                for key, value in self.config.__dict__.items():
                    print(f"  {key}: {value}")
                continue

            result = self.generate(current_text + " " + user_input, **kwargs)
            generated_text = result['generated_text']

            print(f"\nGenerated: {generated_text}")
            print(f"Time: {result['generation_time']:.2f}s")

            if result['generation_log'].get('steps'):
                cascade_steps = [s for s in result['generation_log']['steps'] if s['cascade_guided']]
                if cascade_steps:
                    print(f"Cascade-guided steps: {len(cascade_steps)}/{len(result['generation_log']['steps'])}")

            current_text = generated_text

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return self.generation_stats.copy()

    def clear_cache(self) -> None:
        """Clear constraint validation cache"""
        self.constraint_cache.clear()
        self.logger.info("Constraint cache cleared")
