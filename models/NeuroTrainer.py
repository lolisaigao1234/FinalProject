# models/NeuroTrainer.py
import os
import logging
import time


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from typing import Dict, Tuple, List, Optional
from collections import defaultdict
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import MODELS_DIR, LEARNING_RATE, WEIGHT_DECAY, EPOCHS

logger = logging.getLogger(__name__)

class NLIDataset(Dataset):
    """Dataset class for NLI tasks."""

    def __init__(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
            syntax_features_premise: torch.Tensor,
            syntax_features_hypothesis: torch.Tensor,
            labels: Optional[torch.Tensor] = None
    ):
        """Initialize NLI dataset."""
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.syntax_features_premise = syntax_features_premise
        self.syntax_features_hypothesis = syntax_features_hypothesis
        self.labels = labels

    def __len__(self):
        """Return dataset length."""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Get dataset item."""
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "syntax_features_premise": self.syntax_features_premise[idx],
            "syntax_features_hypothesis": self.syntax_features_hypothesis[idx],
        }

        if self.labels is not None:
            item["labels"] = self.labels[idx]

        return item


class ModelTrainer:
    """A100-optimized trainer class for NLI models."""

    def __init__(
            self,
            model: nn.Module,
            device: torch.device = None,
            learning_rate: float = LEARNING_RATE,
            weight_decay: float = WEIGHT_DECAY,
            save_dir: str = MODELS_DIR,
            use_amp: bool = False,
            grad_accum_steps: int = 1,
            enable_compile: bool = False
    ):
        """Initialize optimized trainer."""
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.grad_accum_steps = grad_accum_steps

        # Enable cuDNN benchmarking
        torch.backends.cudnn.benchmark = True

        # Initialize AMP scaler
        self.scaler = torch.amp.GradScaler(enabled=use_amp)
        # Model compilation for PyTorch 2.0+
        if enable_compile and hasattr(torch, 'compile'):
            self.model = torch.compile(model, backend="cudagraphs")

        self.model.to(self.device)

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler with warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=1000  # Update based on actual steps
        )

        self.criterion = nn.CrossEntropyLoss()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train(
            self,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader] = None,
            epochs: int = EPOCHS,
            save_best: bool = True
    ) -> Dict[str, List[float]]:
        """Optimized training loop with AMP and gradient accumulation."""
        logger.info(f"Training model for {epochs} epochs on {self.device}")
        logger.info(f"Using AMP: {self.use_amp}, Gradient accumulation: {self.grad_accum_steps}x")

        history = defaultdict(list)
        best_val_acc = 0.0
        global_step = 0

        for epoch in range(epochs):
            epoch_start = time.time()
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")):
                # Move batch to device with async transfer
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                # Forward pass with AMP
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                    # Add token_type_ids to forward calls in train method
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch["token_type_ids"],  # Added this parameter
                        syntax_features_premise=batch["syntax_features_premise"],
                        syntax_features_hypothesis=batch["syntax_features_hypothesis"]
                    )

                    loss = self.criterion(outputs, batch["labels"]) / self.grad_accum_steps

                # Backward pass with scaled gradients
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    global_step += 1

                # Update metrics
                train_loss += loss.item() * self.grad_accum_steps
                _, preds = torch.max(outputs, 1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

            # Calculate epoch metrics
            train_loss /= len(train_dataloader)
            train_acc = correct / total
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            if val_dataloader:
                val_loss, val_acc = self.evaluate(val_dataloader)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                # Save best model
                if save_best and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model(os.path.join(self.save_dir, "best_model.pt"))

            # Epoch logging
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]['lr']
            log_msg = (
                f"Epoch {epoch + 1}/{epochs} | "
                f"Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"LR: {lr:.2e}"
            )
            if val_dataloader:
                log_msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            logger.info(log_msg)

        # Save final model
        self.save_model(os.path.join(self.save_dir, "final_model.pt"))
        return history

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Optimized evaluation with AMP."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                    # outputs = self.model(
                    #     input_ids=batch["input_ids"],
                    #     attention_mask=batch["attention_mask"],
                    #     syntax_features_premise=batch["syntax_features_premise"],
                    #     syntax_features_hypothesis=batch["syntax_features_hypothesis"]
                    # )
                    # Add token_type_ids to forward calls in train method
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch["token_type_ids"],  # Added this parameter
                        syntax_features_premise=batch["syntax_features_premise"],
                        syntax_features_hypothesis=batch["syntax_features_hypothesis"]
                    )

                    loss = self.criterion(outputs, batch["labels"])

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

        return val_loss / len(dataloader), correct / total

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """Optimized prediction with AMP."""
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        syntax_features_premise=batch["syntax_features_premise"],
                        syntax_features_hypothesis=batch["syntax_features_hypothesis"]
                    )

                all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())

        return np.array(all_preds)

    def save_model(self, path: str):
        """Save model with AMP state."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model with AMP state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(f"Model loaded from {path}")