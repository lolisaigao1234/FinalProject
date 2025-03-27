# models/trainer.py
import os
import logging
import time
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import MODELS_DIR, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS

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
    """Trainer class for NLI models."""

    def __init__(
            self,
            model: nn.Module,
            device: torch.device = None,
            learning_rate: float = LEARNING_RATE,
            weight_decay: float = WEIGHT_DECAY,
            save_dir: str = MODELS_DIR
    ):
        """Initialize trainer."""
        self.model = model
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
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
        """Train the model."""
        logger.info(f"Training model for {epochs} epochs on {self.device}")

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        best_val_acc = 0.0

        for epoch in range(epochs):
            start_time = time.time()

            # Training
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    syntax_features_premise=batch["syntax_features_premise"],
                    syntax_features_hypothesis=batch["syntax_features_hypothesis"]
                )

                # Calculate loss
                loss = self.criterion(outputs, batch["labels"])

                # Backward pass
                loss.backward()

                # Update weights
                self.optimizer.step()

                # Track loss and predictions
                train_loss += loss.item()

                # Get predictions
                _, preds = torch.max(outputs, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(batch["labels"].cpu().numpy())

            # Calculate training metrics
            train_loss /= len(train_dataloader)
            train_acc = accuracy_score(train_labels, train_preds)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            if val_dataloader is not None:
                val_loss, val_acc = self.evaluate(val_dataloader)

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                # Save best model
                if save_best and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model(os.path.join(self.save_dir, "best_model.pt"))

                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Time: {time.time() - start_time:.2f}s - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Train Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f} - "
                    f"Val Acc: {val_acc:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Time: {time.time() - start_time:.2f}s - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Train Acc: {train_acc:.4f}"
                )

        # Save final model
        self.save_model(os.path.join(self.save_dir, "final_model.pt"))

        return history

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model."""
        self.model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    syntax_features_premise=batch["syntax_features_premise"],
                    syntax_features_hypothesis=batch["syntax_features_hypothesis"]
                )

                # Calculate loss
                loss = self.criterion(outputs, batch["labels"])

                # Track loss and predictions
                val_loss += loss.item()

                # Get predictions
                _, preds = torch.max(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch["labels"].cpu().numpy())

        # Calculate validation metrics
        val_loss /= len(dataloader)
        val_acc = accuracy_score(val_labels, val_preds)

        return val_loss, val_acc

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """Make predictions with the model."""
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    syntax_features_premise=batch["syntax_features_premise"],
                    syntax_features_hypothesis=batch["syntax_features_hypothesis"]
                )

                # Get predictions
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())

        return np.array(all_preds)

    def save_model(self, path: str):
        """Save model to disk."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Model loaded from {path}")
