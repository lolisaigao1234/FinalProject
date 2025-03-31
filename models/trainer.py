# models/trainer.py
import os
import logging
import time
import glob

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
    """Load data from parquet files with proper pattern matching"""
    cache_dir = cache_dir or 'cache\\parquet'

    # Build the file pattern based on requirements
    pattern = _build_file_pattern(cache_dir, dataset_name, split, feature_type)

    # Find matching files
    parquet_files = glob.glob(pattern)
    if not parquet_files:
        logger.warning(f"No files found for pattern: {pattern}")
        alternative_pattern = os.path.join(cache_dir, f'{dataset_name}_{split}_data.parquet')
        parquet_files = glob.glob(alternative_pattern)

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found for {dataset_name} {split}")

    # Load and concatenate files
    logger.info(f"Loading data from {len(parquet_files)} files matching: {pattern}")
    dfs = [pd.read_parquet(file) for file in parquet_files]
    result = pd.concat(dfs, ignore_index=True)

    # Apply sample size if specified and needed
    if sample_size and len(result) > sample_size:
        result = result.sample(sample_size, random_state=42)

    logger.info(f"Loaded {len(result)} examples with {result.shape[1]} features")
    return result


def _build_file_pattern(cache_dir, dataset_name, split, feature_type=None):
    """Build file pattern for parquet files"""
    if feature_type:
        return os.path.join(cache_dir, f'{dataset_name}_{split}_{feature_type}*.parquet')
    else:
        return os.path.join(cache_dir, f'{dataset_name}_{split}_sample*.parquet')


class FeatureExtractor:
    """Base class for feature extraction"""

    def extract(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features from input data"""
        raise NotImplementedError("Subclasses must implement extract()")

    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Get list of feature column names"""
        raise NotImplementedError("Subclasses must implement get_feature_columns()")


class LexicalFeatureExtractor(FeatureExtractor):
    """Extracts lexical features from data"""

    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Get lexical feature columns"""
        # Filter to keep only lexical features
        filtered_data = filter_lexical_features(data)
        # Get feature columns (all except label/ID columns)
        return [col for col in filtered_data.columns
                if col not in ['label', 'gold_label', 'pair_id']]

    def extract(self, data: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        """Extract lexical features from data"""
        filtered_data = filter_lexical_features(data)
        return _feature_extractor_helper(self, filtered_data, data, feature_cols)
        # if feature_cols is not None:
        #     # Ensure all columns exist in the data
        #     missing_cols = set(feature_cols) - set(filtered_data.columns)
        #     if missing_cols:
        #         logger.warning(f"Adding {len(missing_cols)} missing lexical columns")
        #         filtered_data = filtered_data.copy()  # Create explicit copy
        #         for col in missing_cols:
        #             filtered_data.loc[:, col] = 0
        #     return filtered_data[feature_cols].values
        # else:
        #     cols = self.get_feature_columns(data)
        #     return filtered_data[cols].values


class SyntacticFeatureExtractor(FeatureExtractor):
    """Extracts syntactic features from data"""

    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Get syntactic feature columns"""
        filtered_data = filter_syntactic_features(data)
        return [col for col in filtered_data.columns
                if col not in ['label', 'gold_label', 'pair_id']]

    def extract(self, data: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        """Extract syntactic features from data"""
        filtered_data = filter_syntactic_features(data)
        return _feature_extractor_helper(self, filtered_data, data, feature_cols)
        # if feature_cols is not None:
        #     # Ensure all columns exist in the data
        #     missing_cols = set(feature_cols) - set(filtered_data.columns)
        #     if missing_cols:
        #         logger.warning(f"Adding {len(missing_cols)} missing syntactic columns")
        #         filtered_data = filtered_data.copy()
        #         for col in missing_cols:
        #             filtered_data.loc[:, col] = 0
        #     return filtered_data[feature_cols].values
        # else:
        #     cols = self.get_feature_columns(data)
        #     return filtered_data[cols].values


class CombinedFeatureExtractor(FeatureExtractor):
    """Extracts both lexical and syntactic features"""

    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Get all relevant feature columns"""
        return [col for col in data.columns
                if col not in ['label', 'gold_label', 'pair_id']]

    def extract(self, data: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        """Extract combined features from data"""
        return _feature_extractor_helper(self, data, data, feature_cols)
        # if feature_cols is not None:
        #     # Ensure all columns exist in the data
        #     missing_cols = set(feature_cols) - set(data.columns)
        #     if missing_cols:
        #         logger.warning(f"Adding {len(missing_cols)} missing columns")
        #         data = data.copy()
        #         for col in missing_cols:
        #             data.loc[:, col] = 0
        #     return data[feature_cols].values
        # else:
        #     cols = self.get_feature_columns(data)
        #     return data[cols].values


def _feature_extractor_helper(self, filtered_data: pd.DataFrame, data: pd.DataFrame,
                              feature_cols: List[str]) -> np.ndarray:
    if feature_cols is not None:
        # Ensure all columns exist in the data
        missing_cols = set(feature_cols) - set(filtered_data.columns)
        if missing_cols:
            logger.warning(f"Adding {len(missing_cols)} missing syntactic columns")
            filtered_data = filtered_data.copy()
            for col in missing_cols:
                filtered_data.loc[:, col] = 0
        return filtered_data[feature_cols].values
    else:
        cols = self.get_feature_columns(data)
        return filtered_data[cols].values


class SVMModel(NLIModel):
    """Base class for all SVM models with common functionality"""

    def __init__(self, feature_extractor: FeatureExtractor, kernel: str = 'linear', C: float = 1.0):
        """Initialize SVM model with a feature extractor"""
        self.feature_extractor = feature_extractor
        self.kernel = kernel
        self.C = C
        self.svm = SVC(kernel=kernel, C=C, probability=True)
        self.is_trained = False
        self.feature_cols = None

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features using the feature extractor"""
        logger.info("Enter extract feature base model")
        if not self.is_trained:
            # During training, discover feature columns
            self.feature_cols = self.feature_extractor.get_feature_columns(data)
            logger.info(f"Using {len(self.feature_cols)} feature columns for training")
            return self.feature_extractor.extract(data)
        else:
            # During prediction, use stored feature columns
            logger.info(f"Extracting features with {len(self.feature_cols)} columns")
            logger.info("Not implemented, if entered, should terminate.")
            return np.ndarray(0)
            # return self.feature_extractor.extract(data, self.feature_cols)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the SVM model"""
        logger.info(f"Training SVM with {X.shape[0]} samples and {X.shape[1]} features")
        self.svm.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the SVM model"""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet")
        return self.svm.predict(X)

    def save(self, filepath: str) -> None:
        """Save model to disk"""
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
    def load(cls, filepath: str, feature_extractor: FeatureExtractor) -> 'SVMModel':
        """Load model from disk"""
        model_data = joblib.load(filepath)
        instance = cls(feature_extractor, kernel=model_data['kernel'], C=model_data['C'])
        instance.svm = model_data['svm']
        instance.is_trained = model_data['is_trained']
        instance.feature_cols = model_data['feature_cols']
        logger.info(f"Loaded model from {filepath}")
        return instance


# Feature selection and filtering helpers
def filter_syntactic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only syntactic features columns."""
    # Get columns that have syntactic feature indicators
    syntax_cols = [col for col in df.columns if any(prefix in col for prefix in
                                                    ['premise_const_', 'hypothesis_const_', 'premise_dep_',
                                                     'hypothesis_dep_', 'diff_const_', 'deprel_', 'pos_'])]

    # Always keep the label column
    return _feature_return_helper(df, syntax_cols)
    # if 'label' in df.columns:
    #     syntax_cols.append('label')
    # elif 'gold_label' in df.columns:
    #     syntax_cols.append('gold_label')
    #
    # if 'pair_id' in df.columns:
    #     syntax_cols.append('pair_id')
    #
    # logger.info(f"Selected {len(syntax_cols)} syntactic feature columns")
    # return df[syntax_cols]


def filter_lexical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only lexical features columns."""
    # Get columns that have lexical feature indicators
    lexical_cols = [col for col in df.columns if any(prefix in col for prefix in
                                                     ['premise_bert_', 'hypothesis_bert_', 'diff_bert_', 'prod_bert_',
                                                      'premise_length', 'hypothesis_length', 'length_diff',
                                                      'length_ratio', 'word_overlap'])]

    # Always keep the label column
    return _feature_return_helper(df, lexical_cols)


def _feature_return_helper(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    if 'label' in df.columns:
        feature_cols.append('label')
    elif 'gold_label' in df.columns:
        feature_cols.append('gold_label')

    if 'pair_id' in df.columns:
        feature_cols.append('pair_id')

    logger.info(f"Selected {len(feature_cols)} lexical feature columns")
    return df[feature_cols]


class SVMWithBagOfWords(SVMModel):
    """SVM model using only bag of words/lexical features."""

    def __init__(self, kernel: str = 'linear', C: float = 1.0):
        super().__init__(LexicalFeatureExtractor(), kernel, C)

    @classmethod
    def load(cls, filepath: str) -> 'SVMWithBagOfWords':
        return super().load(filepath, LexicalFeatureExtractor())


class SVMWithSyntax(SVMModel):
    """SVM model using only syntactic features."""

    def __init__(self, kernel: str = 'linear', C: float = 1.0):
        super().__init__(SyntacticFeatureExtractor(), kernel, C)

    @classmethod
    def load(cls, filepath: str) -> 'SVMWithSyntax':
        return super().load(filepath, SyntacticFeatureExtractor())


class SVMWithBothFeatures(SVMModel):
    """SVM model using both lexical and syntactic features."""

    def __init__(self, kernel: str = 'linear', C: float = 1.0):
        super().__init__(CombinedFeatureExtractor(), kernel, C)

    @classmethod
    def load(cls, filepath: str) -> 'SVMWithBothFeatures':
        return super().load(filepath, CombinedFeatureExtractor())


def prepare_labels(labels, label_map=None):
    """Convert string labels to integers with a consistent mapping"""
    if label_map is None:
        label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    if labels.dtype == object:
        return np.array([label_map.get(label, -1) for label in labels])
    return labels


def get_label_column(df: pd.DataFrame) -> Tuple[str, np.ndarray]:
    """Extract label column name and values from dataframe"""
    if 'gold_label' in df.columns:
        return 'gold_label', df['gold_label'].values
    elif 'label' in df.columns:
        return 'label', df['label'].values
    else:
        raise ValueError("No label column (gold_label or label) found in data")


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Clean dataset and extract labels"""
    label_col, labels = get_label_column(df)

    # Convert string labels to integers
    int_labels = prepare_labels(labels)

    # Remove examples with unknown labels
    valid_mask = int_labels != -1
    if not all(valid_mask):
        df = df.loc[valid_mask].reset_index(drop=True)
        int_labels = int_labels[valid_mask]

    return df, int_labels


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

        # Load and prepare datasets
        train_data, val_data = self._load_datasets(args.dataset)

        # Train all models
        results = self._train_all_models(
            train_data,
            val_data,
            kernel=args.kernel if hasattr(args, 'kernel') else 'linear',
            C=args.C if hasattr(args, 'C') else 1.0
        )

        # Cross-dataset evaluation if requested
        if hasattr(args, 'cross_evaluate') and args.cross_evaluate:
            self._run_cross_evaluation(args.dataset)

        return results

    def _load_datasets(self, dataset_name):
        """Load training and validation datasets"""
        # Load training data
        logger.info("Loading preprocessed features")
        train_data = load_parquet_data(
            dataset_name, 'train',
            feature_type='features_lexical_syntactic'
        )

        # Handle NaN values
        train_data = self._handle_nan_values(train_data, "training")

        # Try loading validation data, otherwise return None
        try:
            val_data = load_parquet_data(
                dataset_name, 'validation',
                feature_type='features_lexical_syntactic'
            )
            val_data = self._handle_nan_values(val_data, "validation")
            logger.info(f"Loaded separate validation set with {len(val_data)} examples")
        except FileNotFoundError:
            logger.info("No validation features found, will split from train set")
            val_data = None

        return train_data, val_data

    def _handle_nan_values(self, df, dataset_name):
        """Check for and handle NaN values in dataframe"""
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in {dataset_name} data, filling with 0")
            return df.fillna(0)
        return df

    def _train_all_models(self, train_df, val_df=None, test_size=0.2,
                          random_state=42, kernel='linear', C=1.0):
        """Train all three SVM baseline models"""
        results = {}

        # Define model configurations
        model_configs = [
            ("bow", SVMWithBagOfWords(kernel=kernel, C=C), "Bag of Words"),
            ("syntax", SVMWithSyntax(kernel=kernel, C=C), "Syntax-only"),
            ("combined", SVMWithBothFeatures(kernel=kernel, C=C), "Combined (BoW + Syntax)")
        ]

        # Train each model
        for model_key, model, model_desc in model_configs:
            logger.info(f"Training {model_desc} SVM baseline")
            results[model_key] = self.train_model(model, train_df, val_df, test_size, random_state)

        logger.info("All SVM baselines trained successfully")
        return results

    def train_model(self, model, train_df, val_df=None, test_size=0.2, random_state=42):
        """Train and evaluate a single SVM model"""
        # Split data if validation set not provided
        if val_df is None:
            train_df, val_df = train_test_split(
                train_df, test_size=test_size, random_state=random_state
            )

        # Prepare data
        train_df, y_train = clean_dataset(train_df)
        val_df, y_val = clean_dataset(val_df)

        # Extract features for training only
        logger.info(f"Training {model.__class__.__name__}")
        X_train = model.extract_features(train_df)

        # Train model
        start_time = time.time()
        model.train(X_train, y_train)
        train_time = time.time() - start_time

        # Extract features for validation AFTER training
        X_val = model.extract_features(val_df)

        # Evaluate model
        eval_time, metrics = self._evaluate_model(model, X_val, y_val)

        # Save model
        model_path = os.path.join(self.save_dir, f"{model.__class__.__name__}.joblib")
        model.save(model_path)

        # Log results
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        logger.info(f"Training time: {train_time:.2f}s, Evaluation time: {eval_time:.2f}s")

        return {**metrics, 'train_time': train_time, 'eval_time': eval_time}

    def _evaluate_model(self, model, X_val, y_val):
        """Evaluate model and return metrics"""
        start_time = time.time()
        y_pred = model.predict(X_val)
        eval_time = time.time() - start_time

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='weighted', zero_division=0
        )

        return eval_time, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _run_cross_evaluation(self, source_dataset):
        """Run cross-dataset evaluation"""
        datasets = {}

        # Load test datasets
        for dataset_name in ['SNLI', 'MNLI', 'ANLI']:
            if dataset_name != source_dataset:  # Skip training dataset
                try:
                    datasets[dataset_name] = load_parquet_data(
                        dataset_name, 'test',
                        feature_type='features_lexical_syntactic'
                    )
                    datasets[dataset_name] = self._handle_nan_values(
                        datasets[dataset_name], f"{dataset_name} test"
                    )
                    logger.info(f"Loaded {len(datasets[dataset_name])} examples from {dataset_name}")
                except FileNotFoundError:
                    logger.warning(f"Could not load test features for {dataset_name}")

        if datasets:
            logger.info("Running cross-dataset evaluation")
            results = self.cross_dataset_evaluation(datasets)
            logger.info("Cross-dataset evaluation complete")
            return results
        return {}

    def cross_dataset_evaluation(self, datasets):
        """Evaluate trained models on multiple datasets"""
        results = {}

        # Load trained models
        models = {
            'bow': SVMWithBagOfWords.load(os.path.join(self.save_dir, "SVMWithBagOfWords.joblib")),
            'syntax': SVMWithSyntax.load(os.path.join(self.save_dir, "SVMWithSyntax.joblib")),
            'combined': SVMWithBothFeatures.load(os.path.join(self.save_dir, "SVMWithBothFeatures.joblib"))
        }

        # Evaluate each dataset
        for dataset_name, df in datasets.items():
            logger.info(f"Evaluating on {dataset_name} dataset")
            df, labels = clean_dataset(df)

            # Evaluate each model
            dataset_results = {}
            for model_name, model in models.items():
                features = model.extract_features(df)
                preds = model.predict(features)
                acc = accuracy_score(labels, preds)
                dataset_results[model_name] = {'accuracy': acc}

            # Log results
            results[dataset_name] = dataset_results
            acc_summary = ', '.join([f"{k}: {v['accuracy']:.4f}" for k, v in dataset_results.items()])
            logger.info(f"{dataset_name} results - {acc_summary}")

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
