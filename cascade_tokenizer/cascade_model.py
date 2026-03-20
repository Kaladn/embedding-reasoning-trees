"""
CascadeModel: Neural model integration for cascade-guided generation
Production-ready transformer with embedded reasoning tree guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math

from cascade_tokenizer.cascade_tokenizer import CascadeTokenizer
from cascade_tokenizer.cascade_token import CascadeToken, NodeType


@dataclass
class CascadeModelConfig:
    """Configuration for cascade-guided model"""
    vocab_size: int = 50000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    cascade_embedding_size: int = 256
    cascade_fusion_layers: int = 3
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    temperature: float = 1.0
    top_k: int = 50
    constraint_weight: float = 0.3


class CascadeEmbedding(nn.Module):
    """Embedding layer that incorporates cascade reasoning trees"""

    def __init__(self, config: CascadeModelConfig):
        super().__init__()
        self.config = config

        # Standard token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Cascade-specific embeddings
        self.cascade_node_embeddings = nn.Embedding(1000, config.cascade_embedding_size)
        self.cascade_weight_projection = nn.Linear(1, config.cascade_embedding_size)
        self.cascade_fusion = nn.Linear(config.cascade_embedding_size * 13, config.hidden_size)  # 6+1+6 nodes

        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.cascade_fusion_layers)
        ])

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

        # Concept vocabulary for mapping concepts to IDs
        self.concept_to_id = {}
        self.next_concept_id = 0

    def _get_concept_id(self, concept: str) -> int:
        """Get or create concept ID"""
        if concept not in self.concept_to_id:
            if self.next_concept_id >= 1000:
                # Wrap around if we exceed the embedding table size
                self.concept_to_id[concept] = hash(concept) % 1000
            else:
                self.concept_to_id[concept] = self.next_concept_id
                self.next_concept_id += 1
        return self.concept_to_id[concept]

    def _embed_cascade(self, cascade_token: CascadeToken, device: torch.device) -> torch.Tensor:
        """Embed cascade reasoning tree"""
        cascade_embeddings = []

        # Embed input nodes (6)
        for i in range(6):
            if i < len(cascade_token.cascade.input_nodes) and cascade_token.cascade.input_nodes[i] is not None:
                node = cascade_token.cascade.input_nodes[i]
                concept_id = self._get_concept_id(node.concept)
                concept_emb = self.cascade_node_embeddings(torch.tensor(concept_id, device=device))
                weight_emb = self.cascade_weight_projection(torch.tensor([[node.weight]], device=device))
                node_emb = concept_emb + weight_emb.squeeze(0)
            else:
                node_emb = torch.zeros(self.config.cascade_embedding_size, device=device)
            cascade_embeddings.append(node_emb)

        # Embed central node (1)
        if cascade_token.cascade.central_node is not None:
            node = cascade_token.cascade.central_node
            concept_id = self._get_concept_id(node.concept)
            concept_emb = self.cascade_node_embeddings(torch.tensor(concept_id, device=device))
            weight_emb = self.cascade_weight_projection(torch.tensor([[node.weight]], device=device))
            node_emb = concept_emb + weight_emb.squeeze(0)
        else:
            node_emb = torch.zeros(self.config.cascade_embedding_size, device=device)
        cascade_embeddings.append(node_emb)

        # Embed output nodes (6)
        for i in range(6):
            if i < len(cascade_token.cascade.output_nodes) and cascade_token.cascade.output_nodes[i] is not None:
                node = cascade_token.cascade.output_nodes[i]
                concept_id = self._get_concept_id(node.concept)
                concept_emb = self.cascade_node_embeddings(torch.tensor(concept_id, device=device))
                weight_emb = self.cascade_weight_projection(torch.tensor([[node.weight]], device=device))
                node_emb = concept_emb + weight_emb.squeeze(0)
            else:
                node_emb = torch.zeros(self.config.cascade_embedding_size, device=device)
            cascade_embeddings.append(node_emb)

        # Concatenate and project to hidden size
        cascade_vector = torch.cat(cascade_embeddings, dim=0)
        return self.cascade_fusion(cascade_vector)

    def forward(self, input_ids: torch.Tensor, cascade_tokens: List[CascadeToken] = None,
                position_ids: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with cascade embedding fusion"""
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # Standard embeddings
        token_embeds = self.token_embeddings(input_ids)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = token_embeds + position_embeds

        # Add cascade embeddings if available
        if cascade_tokens is not None:
            cascade_embeds = torch.zeros_like(embeddings)

            for batch_idx in range(batch_size):
                for seq_idx in range(seq_length):
                    if seq_idx < len(cascade_tokens) and cascade_tokens[seq_idx] is not None:
                        cascade_emb = self._embed_cascade(cascade_tokens[seq_idx], device)
                        cascade_embeds[batch_idx, seq_idx] = cascade_emb

            embeddings = embeddings + cascade_embeds

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Apply fusion transformer layers
        for layer in self.fusion_layers:
            embeddings = layer(embeddings)

        return embeddings


class CascadeAttention(nn.Module):
    """Multi-head attention with cascade reasoning guidance"""

    def __init__(self, config: CascadeModelConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_size = config.hidden_size // config.num_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        # Cascade reasoning guidance
        self.cascade_query = nn.Linear(config.cascade_embedding_size, config.hidden_size)
        self.cascade_key = nn.Linear(config.cascade_embedding_size, config.hidden_size)
        self.reasoning_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, cascade_context: torch.Tensor = None,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with cascade-guided attention"""
        batch_size, seq_length, hidden_size = hidden_states.shape

        # Standard attention
        q = self.query(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)

        # Apply cascade reasoning guidance if available
        if cascade_context is not None:
            cascade_q = self.cascade_query(cascade_context)
            cascade_k = self.cascade_key(cascade_context)

            # Compute cascade attention bias
            cascade_scores = torch.matmul(cascade_q.unsqueeze(1), cascade_k.unsqueeze(1).transpose(-1, -2))
            cascade_scores = cascade_scores.expand(-1, self.num_heads, -1, -1)

            # Blend standard and cascade attention
            attention_scores = attention_scores + self.config.constraint_weight * cascade_scores

        if attention_mask is not None:
            attention_scores += attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)

        # Output projection
        output = self.output_projection(context)

        return output


class CascadeTransformerLayer(nn.Module):
    """Transformer layer with cascade reasoning integration"""

    def __init__(self, config: CascadeModelConfig):
        super().__init__()
        self.attention = CascadeAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, cascade_context: torch.Tensor = None,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with residual connections"""
        attention_output = self.attention(hidden_states, cascade_context, attention_mask)
        hidden_states = self.layer_norm1(hidden_states + attention_output)

        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.layer_norm2(hidden_states + ff_output)

        return hidden_states


class CascadeModel(nn.Module):
    """Complete cascade-guided transformer model"""

    def __init__(self, config: CascadeModelConfig, tokenizer: CascadeTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        # Core components
        self.embeddings = CascadeEmbedding(config)
        self.layers = nn.ModuleList([
            CascadeTransformerLayer(config) for _ in range(config.num_layers)
        ])

        # Output head
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Cascade-guided generation components
        self.cascade_scorer = nn.Linear(config.hidden_size, 1)
        self.constraint_validator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # Cascade context projection: hidden_size -> cascade_embedding_size
        self.cascade_context_projection = nn.Linear(config.hidden_size, config.cascade_embedding_size)

        self.init_weights()

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters"""
        return next(self.parameters()).device

    def init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                cascade_tokens: List[CascadeToken] = None, labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass with optional cascade guidance"""
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # Embeddings with cascade fusion
        hidden_states = self.embeddings(input_ids, cascade_tokens)

        # Build cascade context from the embedding output for attention guidance
        cascade_context = self.cascade_context_projection(hidden_states)

        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=device)
        attention_mask = (1.0 - attention_mask.float()) * -10000.0
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, cascade_context, attention_mask)

        # Output processing
        hidden_states = self.output_layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # Cascade-guided scoring
        cascade_scores = self.cascade_scorer(hidden_states)
        constraint_validity = self.constraint_validator(hidden_states)

        outputs = {
            'logits': logits,
            'cascade_scores': cascade_scores,
            'constraint_validity': constraint_validity,
            'hidden_states': hidden_states
        }

        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Standard language modeling loss
            lm_loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size),
                                    shift_labels.view(-1), ignore_index=-100)

            # Cascade consistency loss
            cascade_loss = F.mse_loss(cascade_scores[..., :-1, :],
                                    torch.ones_like(cascade_scores[..., :-1, :]))

            # Constraint validity loss
            constraint_loss = F.binary_cross_entropy(constraint_validity[..., :-1, :].squeeze(-1),
                                                   torch.ones_like(constraint_validity[..., :-1, :].squeeze(-1)))

            total_loss = lm_loss + 0.1 * cascade_loss + 0.1 * constraint_loss
            outputs['loss'] = total_loss
            outputs['lm_loss'] = lm_loss
            outputs['cascade_loss'] = cascade_loss
            outputs['constraint_loss'] = constraint_loss

        return outputs

    def generate_with_cascades(self, input_ids: torch.Tensor, max_length: int = 100,
                             temperature: float = 1.0, top_k: int = 50,
                             do_sample: bool = True) -> Tuple[torch.Tensor, Dict]:
        """Generate text with cascade guidance"""
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        generated_ids = input_ids.clone()
        generation_log = {'steps': [], 'cascade_guidance': [], 'constraint_violations': []}

        with torch.no_grad():
            for step in range(max_length):
                # Get cascade tokens for current sequence
                cascade_tokens = []
                for token_id in generated_ids[0]:
                    cascade_token = self.tokenizer.vocabulary.get_cascade_token(token_id.item())
                    cascade_tokens.append(cascade_token)

                # Forward pass
                outputs = self.forward(generated_ids, cascade_tokens=cascade_tokens)
                logits = outputs['logits'][:, -1, :]
                cascade_scores = outputs['cascade_scores'][:, -1, :]
                constraint_validity = outputs['constraint_validity'][:, -1, :]

                # Apply cascade guidance to logits
                guided_logits = logits + self.config.constraint_weight * cascade_scores.expand_as(logits)

                # Temperature scaling
                if temperature != 1.0:
                    guided_logits = guided_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(guided_logits, min(top_k, guided_logits.size(-1)), dim=-1)
                    guided_logits = torch.full_like(guided_logits, -1e9)
                    guided_logits.scatter_(-1, top_k_indices, top_k_logits)

                # Sample next token
                if do_sample:
                    probs = F.softmax(guided_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_token_id = torch.argmax(guided_logits, dim=-1, keepdim=True)

                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

                # Log generation step
                step_info = {
                    'step': step,
                    'token_id': next_token_id[0, 0].item(),
                    'token': self.tokenizer.vocabulary.id_to_token.get(next_token_id[0, 0].item(), '<UNK>'),
                    'cascade_score': cascade_scores[0, 0].item(),
                    'constraint_validity': constraint_validity[0, 0].item()
                }
                generation_log['steps'].append(step_info)

                # Check for EOS
                if next_token_id[0, 0].item() == self.tokenizer.vocabulary.special_tokens['<EOS>']:
                    break

        return generated_ids, generation_log
