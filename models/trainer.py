# models/trainer.py
import os
import logging
import time
import glob
from abc import ABC

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from typing import Dict, Tuple, List, Optional
from collections import defaultdict
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup

from utils.common import NLIModel
from config import MODELS_DIR, LEARNING_RATE, WEIGHT_DECAY, EPOCHS

logger = logging.getLogger(__name__)


# Data loading functions for SVM models
def load_parquet_data(dataset_name: str, split: str = 'train',
                      feature_type: str = None,
                      sample_size: Optional[int] = None,
                      cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Load data from parquet files with correct pattern matching."""
    if cache_dir is None:
        cache_dir = 'cache\\parquet'

    # Determine the file pattern based on what we need
    if feature_type:
        pattern = os.path.join(cache_dir, f'{dataset_name}_{split}_{feature_type}*.parquet')
    else:
        # Default to sample files if no specific type
        pattern = os.path.join(cache_dir, f'{dataset_name}_{split}_sample*.parquet')

    parquet_files = glob.glob(pattern)

    if not parquet_files:
        logger.warning(f"No files found for pattern: {pattern}")
        # Try fallback to data files
        pattern = os.path.join(cache_dir, f'{dataset_name}_{split}_data.parquet')
        parquet_files = glob.glob(pattern)

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found for {dataset_name} {split}")

    logger.info(f"Loading data from {len(parquet_files)} files matching: {pattern}")

    dfs = []
    for file in parquet_files:
        logger.info(f"Loading file: {file}")
        df = pd.read_parquet(file)
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(result)} examples with {result.shape[1]} features")

    return result


# SVM Model Implementations
class SVMBaseModel(NLIModel, ABC):
    """Base class for SVM models implementing the NLIModel interface."""

    def __init__(self, kernel: str = 'linear', C: float = 1.0):
        """Initialize SVM model."""
        self.kernel = kernel
        self.C = C
        self.svm = SVC(kernel=kernel, C=C, probability=True)
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the SVM model."""
        logger.info(f"Training SVM with {X.shape[0]} samples and {X.shape[1]} features")
        self.svm.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the SVM model."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet")
        return self.svm.predict(X)


# Feature selection and filtering helpers
def filter_syntactic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only syntactic features columns."""
    # Get columns that have syntactic feature indicators
    syntax_cols = [col for col in df.columns if any(prefix in col for prefix in
                                                    ['premise_const_', 'hypothesis_const_', 'premise_dep_',
                                                     'hypothesis_dep_', 'diff_const_', 'deprel_', 'pos_'])]

    # Always keep the label column
    if 'label' in df.columns:
        syntax_cols.append('label')
    elif 'gold_label' in df.columns:
        syntax_cols.append('gold_label')

    if 'pair_id' in df.columns:
        syntax_cols.append('pair_id')

    logger.info(f"Selected {len(syntax_cols)} syntactic feature columns")
    return df[syntax_cols]


def filter_lexical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only lexical features columns."""
    # Get columns that have lexical feature indicators
    lexical_cols = [col for col in df.columns if any(prefix in col for prefix in
                                                     ['premise_bert_', 'hypothesis_bert_', 'diff_bert_', 'prod_bert_',
                                                      'premise_length', 'hypothesis_length', 'length_diff',
                                                      'length_ratio', 'word_overlap'])]

    # Always keep the label column
    if 'label' in df.columns:
        lexical_cols.append('label')
    elif 'gold_label' in df.columns:
        lexical_cols.append('gold_label')

    if 'pair_id' in df.columns:
        lexical_cols.append('pair_id')

    logger.info(f"Selected {len(lexical_cols)} lexical feature columns")
    return df[lexical_cols]


class SVMWithBagOfWords(NLIModel):
    """SVM model using only bag of words/lexical features."""

    def __init__(self, kernel: str = 'linear', C: float = 1.0):
        """Initialize BoW SVM model."""
        self.kernel = kernel
        self.C = C
        self.svm = SVC(kernel=kernel, C=C, probability=True)
        self.is_trained = False
        self.feature_cols = None

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract lexical features from preprocessed data."""
        logger.info("Extracting lexical features for SVM model")

        # Filter to keep only lexical features
        filtered_data = filter_lexical_features(data)

        # Get feature columns (all except label/ID columns)
        if not self.is_trained:
            self.feature_cols = [col for col in filtered_data.columns
                                 if col not in ['label', 'gold_label', 'pair_id']]
            logger.info(f"Using {len(self.feature_cols)} lexical feature columns")

        # Return features as numpy array
        return filtered_data[self.feature_cols].values

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the SVM model."""
        logger.info(f"Training SVM with {X.shape[0]} samples and {X.shape[1]} lexical features")
        self.svm.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the SVM model."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet")
        return self.svm.predict(X)

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        model_data = {
            'svm': self.svm,
            'kernel': self.kernel,
            'C': self.C,
            'is_trained': self.is_trained,
            'feature_cols': self.feature_cols
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Saved model to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SVMWithBagOfWords':
        """Load model from disk."""
        model_data = joblib.load(filepath)
        instance = cls(kernel=model_data['kernel'], C=model_data['C'])
        instance.svm = model_data['svm']
        instance.is_trained = model_data['is_trained']
        instance.feature_cols = model_data['feature_cols']
        logger.info(f"Loaded model from {filepath}")
        return instance


class SVMWithSyntax(NLIModel):
    """SVM model using only syntactic features."""

    def __init__(self, kernel: str = 'linear', C: float = 1.0):
        """Initialize syntax SVM model."""
        self.kernel = kernel
        self.C = C
        self.svm = SVC(kernel=kernel, C=C, probability=True)
        self.is_trained = False
        self.feature_cols = None

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract syntactic features from preprocessed data."""
        logger.info("Extracting syntactic features for SVM model")

        # Filter to keep only syntactic features
        filtered_data = filter_syntactic_features(data)

        # Remove label and ID columns from consideration
        data_cols = [col for col in filtered_data.columns
                     if col not in ['label', 'gold_label', 'pair_id']]

        if not self.is_trained:
            # During training, store feature columns
            self.feature_cols = data_cols
            logger.info(f"Using {len(self.feature_cols)} syntactic feature columns for training")
            return filtered_data[self.feature_cols].values
        else:
            # During validation/testing, ensure feature consistency
            logger.info(
                f"Ensuring consistent feature set with training data ({len(self.feature_cols)} features expected)")

            # Check for missing columns and add them with zeros
            missing_cols = set(self.feature_cols) - set(data_cols)
            if missing_cols:
                logger.warning(f"Adding {len(missing_cols)} missing columns to validation data")
                for col in missing_cols:
                    filtered_data[col] = 0

            # Ensure we only use the columns that were present during training
            # This handles both missing columns (now added) and any extra columns (will be ignored)
            result = filtered_data[self.feature_cols].values
            logger.info(f"Final feature matrix shape: {result.shape}")
            return result

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the SVM model."""
        logger.info(f"Training SVM with {X.shape[0]} samples and {X.shape[1]} syntactic features")
        self.svm.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the SVM model."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet")

        logger.info(f"Making predictions with SVMWithSyntax model")
        '''
        This line is causing problem.
        '''
        return self.svm.predict(X)

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        model_data = {
            'svm': self.svm,
            'kernel': self.kernel,
            'C': self.C,
            'is_trained': self.is_trained,
            'feature_cols': self.feature_cols
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Saved model to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SVMWithSyntax':
        """Load model from disk."""
        model_data = joblib.load(filepath)
        instance = cls(kernel=model_data['kernel'], C=model_data['C'])
        instance.svm = model_data['svm']
        instance.is_trained = model_data['is_trained']
        instance.feature_cols = model_data['feature_cols']
        logger.info(f"Loaded model from {filepath}")
        return instance


class SVMWithBothFeatures(NLIModel):
    """SVM model using both lexical and syntactic features."""

    def __init__(self, kernel: str = 'linear', C: float = 1.0):
        """Initialize combined SVM model."""
        self.kernel = kernel
        self.C = C
        self.svm = SVC(kernel=kernel, C=C, probability=True)
        self.is_trained = False
        self.feature_cols = None

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract combined features from preprocessed data."""
        logger.info("Extracting combined features for SVM model")

        # Get all feature columns (excluding label/ID columns)
        data_cols = [col for col in data.columns
                     if col not in ['label', 'gold_label', 'pair_id']]

        if not self.is_trained:
            # During training, store feature columns
            self.feature_cols = data_cols
            logger.info(f"Using {len(self.feature_cols)} total feature columns for training")
            return data[self.feature_cols].values
        else:
            # During validation/testing, ensure feature consistency
            logger.info(f"Ensuring consistent feature set with training data")

            # Add missing columns with zeros

            missing_cols = set(self.feature_cols) - set(data_cols)
            if missing_cols:
                logger.warning(f"Adding {len(missing_cols)} missing columns to validation data")
                for col in missing_cols:
                    data[col] = 0

            # Use only the training columns in the same order
            result = data[self.feature_cols].values
            logger.info(f"Final feature matrix shape: {result.shape}")
            return result

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the SVM model."""
        logger.info(f"Training SVM with {X.shape[0]} samples and {X.shape[1]} combined features")
        self.svm.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the SVM model."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet")
        return self.svm.predict(X)

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        model_data = {
            'svm': self.svm,
            'kernel': self.kernel,
            'C': self.C,
            'is_trained': self.is_trained,
            'feature_cols': self.feature_cols
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Saved model to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SVMWithBothFeatures':
        """Load model from disk."""
        model_data = joblib.load(filepath)
        instance = cls(kernel=model_data['kernel'], C=model_data['C'])
        instance.svm = model_data['svm']
        instance.is_trained = model_data['is_trained']
        instance.feature_cols = model_data['feature_cols']
        logger.info(f"Loaded model from {filepath}")
        return instance


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Prepare data for training, handling label conversion."""
    # Get the label column
    if 'gold_label' in df.columns:
        label_col = 'gold_label'
    elif 'label' in df.columns:
        label_col = 'label'
    else:
        raise ValueError("No label column (gold_label or label) found in data")

    # Convert string labels to integers if needed
    labels = df[label_col].values
    if labels.dtype == object:
        label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        labels = np.array([label_map.get(label, -1) for label in labels])

        # Remove examples with unknown labels
        valid_mask = labels != -1
        df = df.loc[valid_mask].reset_index(drop=True)
        labels = labels[valid_mask]

    return df, labels


class SVMTrainer:
    """Trainer for SVM baseline models using preprocessed features."""

    def __init__(self, save_dir: str = os.path.join(MODELS_DIR, 'svm_baselines')):
        """Initialize SVM trainer."""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def run_training(self, args):
        """Run the complete SVM training pipeline based on command line arguments."""
        logger.info(f"Training SVM baselines on {args.dataset} dataset")

        # Load precomputed features
        logger.info("Loading preprocessed features")
        train_data = load_parquet_data(
            args.dataset, 'train',
            feature_type='features_lexical_syntactic'
        )

        # Try loading validation data, otherwise split from train
        try:
            val_data = load_parquet_data(
                args.dataset, 'validation',
                feature_type='features_lexical_syntactic'
            )
            logger.info(f"Loaded separate validation set with {len(val_data)} examples")
        except FileNotFoundError:
            logger.info("No validation features found, will split from train set")
            val_data = None

        # Add to the run_training method in SVMTrainer
        # After loading the train_data

        # Handle any NaN values in training data
        logger.info("Checking for NaN values in training data")
        nan_count_before = train_data.isna().sum().sum()
        if nan_count_before > 0:
            logger.warning(f"Found {nan_count_before} NaN values in training data, filling with 0")
            train_data = train_data.fillna(0)

        # Similarly for val_data
        if val_data is not None:
            nan_count_before = val_data.isna().sum().sum()
            if nan_count_before > 0:
                logger.warning(f"Found {nan_count_before} NaN values in validation data, filling with 0")
                val_data = val_data.fillna(0)

        # Train all three baseline models
        results = self.train_all_models(
            train_data,
            val_data,
            test_size=0.2,
            random_state=42,
            kernel=args.kernel if hasattr(args, 'kernel') else 'linear',
            C=args.C if hasattr(args, 'C') else 1.0
        )

        # Cross-dataset evaluation if requested
        if hasattr(args, 'cross_evaluate') and args.cross_evaluate:
            datasets = {}

            for dataset_name in ['SNLI', 'MNLI', 'ANLI']:
                if dataset_name != args.dataset:  # Skip training dataset
                    try:
                        datasets[dataset_name] = load_parquet_data(
                            dataset_name, 'test',
                            feature_type='features_lexical_syntactic'
                        )
                        logger.info(f"Loaded {len(datasets[dataset_name])} examples from {dataset_name} test set")
                    except FileNotFoundError:
                        logger.warning(f"Could not load test features for {dataset_name}")

            if datasets:
                logger.info("Running cross-dataset evaluation")
                cross_results = self.cross_dataset_evaluation(datasets)
                logger.info("Cross-dataset evaluation complete")

        return results

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare data for training, handling label conversion."""
        # Get the label column
        if 'gold_label' in df.columns:
            label_col = 'gold_label'
        elif 'label' in df.columns:
            label_col = 'label'
        else:
            raise ValueError("No label column (gold_label or label) found in data")

        # Convert string labels to integers if needed
        labels = df[label_col].values
        if labels.dtype == object:
            label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
            labels = np.array([label_map.get(label, -1) for label in labels])

            # Remove examples with unknown labels
            valid_mask = labels != -1
            df = df.loc[valid_mask].reset_index(drop=True)
            labels = labels[valid_mask]

        return df, labels

    def train_model(self, model: NLIModel, train_df: pd.DataFrame,
                    val_df: pd.DataFrame = None, test_size: float = 0.2,
                    random_state: int = 42) -> Dict:
        """Train and evaluate a single SVM model."""
        # Split data if validation set not provided
        if val_df is None:
            train_df, val_df = train_test_split(
                train_df, test_size=test_size, random_state=random_state
            )

        # Prepare data
        train_df, y_train = self.prepare_data(train_df)
        val_df, y_val = self.prepare_data(val_df)

        # Extract features for training only
        logger.info(f"Training {model.__class__.__name__}")
        X_train = model.extract_features(train_df)

        # Train model - this sets self.is_trained = True
        start_time = time.time()
        model.train(X_train, y_train)
        train_time = time.time() - start_time

        # Extract features for validation AFTER training
        # Now self.is_trained = True, so proper feature alignment will occur
        X_val = model.extract_features(val_df)

        # Evaluate model
        start_time = time.time()
        y_pred = model.predict(X_val)
        eval_time = time.time() - start_time

        # The rest remains the same
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='weighted'
        )

        # Save model
        model_path = os.path.join(self.save_dir, f"{model.__class__.__name__}.joblib")
        model.save(model_path)

        # Log results
        logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(f"Training time: {train_time:.2f}s, Evaluation time: {eval_time:.2f}s")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_time': train_time,
            'eval_time': eval_time
        }

    def train_all_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None,
                         test_size: float = 0.2, random_state: int = 42,
                         kernel: str = 'linear', C: float = 1.0) -> Dict:
        """Train all three SVM baseline models."""
        results = {}

        # 1. Bag of Words SVM
        logger.info("Training Bag of Words SVM baseline")
        bow_model = SVMWithBagOfWords(kernel=kernel, C=C)
        results['bow'] = self.train_model(bow_model, train_df, val_df, test_size, random_state)

        # 2. Syntax SVM
        logger.info("Training Syntax-only SVM baseline")
        syntax_model = SVMWithSyntax(kernel=kernel, C=C)
        results['syntax'] = self.train_model(syntax_model, train_df, val_df, test_size, random_state)

        # 3. Combined SVM
        logger.info("Training Combined (BoW + Syntax) SVM baseline")
        combined_model = SVMWithBothFeatures(kernel=kernel, C=C)
        results['combined'] = self.train_model(combined_model, train_df, val_df, test_size, random_state)

        logger.info("All SVM baselines trained successfully")
        return results

    def cross_dataset_evaluation(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """Evaluate trained models on multiple datasets for generalization analysis."""
        results = {}

        # Load trained models
        bow_model = SVMWithBagOfWords.load(os.path.join(self.save_dir, "SVMWithBagOfWords.joblib"))
        syntax_model = SVMWithSyntax.load(os.path.join(self.save_dir, "SVMWithSyntax.joblib"))
        combined_model = SVMWithBothFeatures.load(os.path.join(self.save_dir, "SVMWithBothFeatures.joblib"))

        for dataset_name, df in datasets.items():
            logger.info(f"Evaluating on {dataset_name} dataset")
            df, labels = self.prepare_data(df)

            dataset_results = {}

            # Evaluate BoW model
            bow_features = bow_model.extract_features(df)
            bow_preds = bow_model.predict(bow_features)
            bow_acc = accuracy_score(labels, bow_preds)
            dataset_results['bow'] = {'accuracy': bow_acc}

            # Evaluate Syntax model
            syntax_features = syntax_model.extract_features(df)
            syntax_preds = syntax_model.predict(syntax_features)
            syntax_acc = accuracy_score(labels, syntax_preds)
            dataset_results['syntax'] = {'accuracy': syntax_acc}

            # Evaluate Combined model
            combined_features = combined_model.extract_features(df)
            combined_preds = combined_model.predict(combined_features)
            combined_acc = accuracy_score(labels, combined_preds)
            dataset_results['combined'] = {'accuracy': combined_acc}

            results[dataset_name] = dataset_results
            logger.info(
                f"{dataset_name} results - BoW: {bow_acc:.4f}, Syntax: {syntax_acc:.4f}, Combined: {combined_acc:.4f}")

        return results


'''
This part is not done yet. Missing implementation for the Dataset class and ModelTrainer class.
'''


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
        # self.scaler = torch.amp.GradScaler(device, enabled=use_amp)
        self.scaler = torch.amp.GradScaler(device=device if not device else "cpu", enabled=use_amp)
        # Model compilation for PyTorch 2.0+
        if enable_compile and hasattr(torch, 'compile'):
            self.model = torch.compile(model)

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
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
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
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
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
