"""
CascadeTrainer: Production training pipeline for cascade-guided models
Handles data preparation, training loops, and cascade optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
import math

from cascade_tokenizer.cascade_model import CascadeModel, CascadeModelConfig
from cascade_tokenizer.cascade_tokenizer import CascadeTokenizer
from cascade_tokenizer.cascade_token import CascadeToken, ConstraintType


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "./cascade_model_output"
    cascade_loss_weight: float = 0.1
    constraint_loss_weight: float = 0.1
    gradient_accumulation_steps: int = 1
    fp16: bool = False  # Default to False; auto-enabled when CUDA is available
    dataloader_num_workers: int = 0  # 0 is safest cross-platform default


def _cascade_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function that handles CascadeToken objects."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    # Keep cascade_tokens as a list-of-lists (not tensorized)
    cascade_tokens = [item['cascade_tokens'] for item in batch]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'cascade_tokens': cascade_tokens,
    }


class CascadeDataset(Dataset):
    """Dataset for cascade-guided training"""

    def __init__(self, texts: List[str], tokenizer: CascadeTokenizer,
                 max_length: int = 512, cascade_definitions: Dict[str, Dict] = None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cascade_definitions = cascade_definitions or {}

        # Pre-tokenize all texts
        self.tokenized_texts = []
        self.cascade_sequences = []

        for text in tqdm(texts, desc="Tokenizing dataset"):
            token_ids = tokenizer.encode(text, add_special_tokens=True)

            # Truncate or pad to max_length
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                token_ids.extend([tokenizer.vocabulary.special_tokens['<PAD>']] * (max_length - len(token_ids)))

            self.tokenized_texts.append(token_ids)

            # Get cascade tokens for this sequence
            cascade_tokens = []
            for token_id in token_ids:
                cascade_token = tokenizer.vocabulary.get_cascade_token(token_id)
                cascade_tokens.append(cascade_token)
            self.cascade_sequences.append(cascade_tokens)

    def __len__(self) -> int:
        return len(self.tokenized_texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        token_ids = torch.tensor(self.tokenized_texts[idx], dtype=torch.long)
        cascade_tokens = self.cascade_sequences[idx]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (token_ids != self.tokenizer.vocabulary.special_tokens['<PAD>']).long()

        # Labels for language modeling (shifted input)
        labels = token_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss

        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'cascade_tokens': cascade_tokens
        }


class CascadeTrainer:
    """Production trainer for cascade models"""

    def __init__(self, model: CascadeModel, tokenizer: CascadeTokenizer,
                 config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = model.device

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Auto-enable fp16 only when CUDA is available
        if self.config.fp16 and not torch.cuda.is_available():
            self.logger.info("FP16 requested but CUDA not available -- disabling mixed precision.")
            self.config.fp16 = False

        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if self.config.fp16 else None

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with weight decay"""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]

        return optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)

    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler"""
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step):
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            return max(0.0, float(num_training_steps - current_step) /
                      float(max(1, num_training_steps - self.config.warmup_steps)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def train(self, train_dataset: CascadeDataset, eval_dataset: CascadeDataset = None) -> Dict[str, Any]:
        """Main training loop"""
        self.logger.info("Starting cascade model training...")

        # Create data loaders with custom collation
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=_cascade_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

        eval_dataloader = None
        if eval_dataset:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.dataloader_num_workers,
                collate_fn=_cascade_collate_fn,
                pin_memory=torch.cuda.is_available(),
            )

        # Calculate total training steps
        num_training_steps = len(train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        num_training_steps = max(num_training_steps, 1)
        self._create_scheduler(num_training_steps)

        # Training metrics
        training_stats = {
            'train_losses': [],
            'eval_losses': [],
            'learning_rates': [],
            'cascade_losses': [],
            'constraint_losses': []
        }

        # Training loop
        self.model.train()
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_cascade_loss = 0.0
            epoch_constraint_loss = 0.0

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

            for step, batch in enumerate(progress_bar):
                # Move tensors to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                cascade_tokens = batch['cascade_tokens']  # list-of-lists, stays on CPU

                # Forward pass with mixed precision
                if self.config.fp16:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            cascade_tokens=cascade_tokens[0],
                            labels=labels
                        )
                        loss = outputs['loss'] / self.config.gradient_accumulation_steps
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        cascade_tokens=cascade_tokens[0],
                        labels=labels
                    )
                    loss = outputs['loss'] / self.config.gradient_accumulation_steps

                # Backward pass
                if self.config.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Accumulate losses
                epoch_loss += outputs['lm_loss'].item()
                epoch_cascade_loss += outputs['cascade_loss'].item()
                epoch_constraint_loss += outputs['constraint_loss'].item()

                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.fp16:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()

                    if self.scheduler:
                        self.scheduler.step()

                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
                        self.logger.info(
                            f"Step {self.global_step}: Loss={loss.item():.4f}, "
                            f"LR={current_lr:.2e}, "
                            f"Cascade Loss={outputs['cascade_loss'].item():.4f}, "
                            f"Constraint Loss={outputs['constraint_loss'].item():.4f}"
                        )

                        training_stats['learning_rates'].append(current_lr)

                    # Evaluation
                    if eval_dataloader and self.global_step % self.config.eval_steps == 0:
                        eval_loss = self.evaluate(eval_dataloader)
                        training_stats['eval_losses'].append(eval_loss)

                        if eval_loss < self.best_loss:
                            self.best_loss = eval_loss
                            self.save_model(f"{self.config.output_dir}/best_model")

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_model(f"{self.config.output_dir}/checkpoint-{self.global_step}")

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'cascade': f"{outputs['cascade_loss'].item():.4f}",
                    'constraint': f"{outputs['constraint_loss'].item():.4f}"
                })

            # End of epoch statistics
            num_batches = max(len(train_dataloader), 1)
            avg_epoch_loss = epoch_loss / num_batches
            avg_cascade_loss = epoch_cascade_loss / num_batches
            avg_constraint_loss = epoch_constraint_loss / num_batches

            training_stats['train_losses'].append(avg_epoch_loss)
            training_stats['cascade_losses'].append(avg_cascade_loss)
            training_stats['constraint_losses'].append(avg_constraint_loss)

            self.logger.info(
                f"Epoch {epoch+1} completed: "
                f"Avg Loss={avg_epoch_loss:.4f}, "
                f"Avg Cascade Loss={avg_cascade_loss:.4f}, "
                f"Avg Constraint Loss={avg_constraint_loss:.4f}"
            )

            # Final evaluation
            if eval_dataloader:
                eval_loss = self.evaluate(eval_dataloader)
                training_stats['eval_losses'].append(eval_loss)

        # Save final model
        self.save_model(f"{self.config.output_dir}/final_model")

        # Save training statistics
        with open(f"{self.config.output_dir}/training_stats.json", 'w') as f:
            json.dump(training_stats, f, indent=2)

        self.logger.info("Training completed!")
        return training_stats

    def evaluate(self, eval_dataloader: DataLoader) -> float:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                cascade_tokens = batch['cascade_tokens']

                if self.config.fp16:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            cascade_tokens=cascade_tokens[0],
                            labels=labels
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        cascade_tokens=cascade_tokens[0],
                        labels=labels
                    )

                total_loss += outputs['loss'].item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.logger.info(f"Evaluation loss: {avg_loss:.4f}")

        self.model.train()
        return avg_loss

    def save_model(self, output_dir: str) -> None:
        """Save model and tokenizer"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.model.config.__dict__
        }, f"{output_dir}/pytorch_model.bin")

        # Save tokenizer
        self.tokenizer.save(f"{output_dir}/tokenizer.pkl")

        # Save config
        with open(f"{output_dir}/config.json", 'w') as f:
            json.dump(self.model.config.__dict__, f, indent=2)

        self.logger.info(f"Model saved to {output_dir}")

    @classmethod
    def load_model(cls, model_dir: str, device: str = 'cpu') -> Tuple[CascadeModel, CascadeTokenizer]:
        """Load trained model and tokenizer"""
        # Load config
        with open(f"{model_dir}/config.json", 'r') as f:
            config_dict = json.load(f)
        config = CascadeModelConfig(**config_dict)

        # Load tokenizer
        tokenizer = CascadeTokenizer.load(f"{model_dir}/tokenizer.pkl")

        # Create model
        model = CascadeModel(config, tokenizer)

        # Load model state
        checkpoint = torch.load(f"{model_dir}/pytorch_model.bin", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.to(device)
        model.eval()

        return model, tokenizer


class CascadeDataPreprocessor:
    """Preprocessor for cascade training data"""

    def __init__(self, tokenizer: CascadeTokenizer):
        self.tokenizer = tokenizer

    def prepare_cascade_definitions(self, corpus: List[str],
                                  domain_knowledge: Dict[str, Any] = None) -> Dict[str, Dict]:
        """Prepare cascade definitions from corpus analysis"""
        cascade_definitions = {}

        # Analyze corpus for semantic relationships
        word_cooccurrence = self._analyze_cooccurrence(corpus)
        semantic_clusters = self._cluster_semantics(word_cooccurrence)

        # Generate cascade definitions
        for word, cluster_info in semantic_clusters.items():
            cascade_def = {
                'central_concept': word,
                'input_concepts': cluster_info.get('input_concepts', []),
                'output_concepts': cluster_info.get('output_concepts', []),
                'weights': cluster_info.get('weights', []),
                'constraints': cluster_info.get('constraints', {})
            }
            cascade_definitions[word] = cascade_def

        return cascade_definitions

    def _analyze_cooccurrence(self, corpus: List[str]) -> Dict[str, Dict[str, float]]:
        """Analyze word co-occurrence patterns"""
        cooccurrence = {}
        window_size = 5

        for text in tqdm(corpus, desc="Analyzing co-occurrence"):
            tokens = self.tokenizer._basic_tokenize(text)

            for i, token in enumerate(tokens):
                if token not in cooccurrence:
                    cooccurrence[token] = {}

                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)

                for j in range(start, end):
                    if i != j:
                        context_token = tokens[j]
                        if context_token not in cooccurrence[token]:
                            cooccurrence[token][context_token] = 0
                        cooccurrence[token][context_token] += 1

        # Normalize co-occurrence scores
        for token in cooccurrence:
            total_count = sum(cooccurrence[token].values())
            if total_count > 0:
                for context_token in cooccurrence[token]:
                    cooccurrence[token][context_token] /= total_count

        return cooccurrence

    def _cluster_semantics(self, cooccurrence: Dict[str, Dict[str, float]]) -> Dict[str, Dict]:
        """Cluster semantic relationships for cascade generation"""
        semantic_clusters = {}

        for word, context_scores in cooccurrence.items():
            sorted_contexts = sorted(context_scores.items(), key=lambda x: x[1], reverse=True)

            top_contexts = sorted_contexts[:12]

            input_concepts = [ctx[0] for ctx in top_contexts[:6]]
            output_concepts = [ctx[0] for ctx in top_contexts[6:12]]

            # Build weights list: inputs + central(1.0) + outputs
            input_weights = [ctx[1] for ctx in top_contexts[:6]]
            output_weights = [ctx[1] for ctx in top_contexts[6:12]]
            weights = input_weights + [1.0] + output_weights

            semantic_clusters[word] = {
                'input_concepts': input_concepts,
                'output_concepts': output_concepts,
                'weights': weights,
                'constraints': {}
            }

        return semantic_clusters
