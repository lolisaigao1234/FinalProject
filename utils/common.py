# utils/common.py
import logging
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Any, Dict
import pandas as pd
import torch
import numpy as np


class NLPBaseComponent:
    """Base class for NLP components with common functionality"""

    def __init__(self, db_handler: Any):
        self.db_handler = db_handler

        # Configure logger with line numbers
        self.logger = logging.getLogger(self.__class__.__name__)
        self._configure_logger()

    def _configure_logger(self):
        """Set up logging format with filename and line numbers"""
        if not self.logger.handlers:  # Avoid duplicate handlers
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False  # Prevent duplicate logs in root logger


class PreprocessorInterface(ABC):
    """Abstract interface for text preprocessing components"""

    @property
    @abstractmethod
    def nlp(self) -> Any:
        """Access to NLP pipeline"""
        pass

    @abstractmethod
    def preprocess_dataset(self, dataset_name: str, split: str,
                           sample_size: Optional[int] = None,
                           force_reprocess: bool = False) -> pd.DataFrame:
        pass

    @abstractmethod
    def prepare_sentence_pairs(self, split_data: pd.DataFrame,
                               dataset_name: str, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def preprocess_dataset_pipeline(self, dataset_name: str,
                                    total_sample_size: int,
                                    force_reprocess: bool) -> None:
        pass


class FeatureExtractorInterface(ABC):
    """Abstract interface for feature extraction components"""

    @abstractmethod
    def extract_features(self, dataset_name: str, split: str,
                         force_recompute: bool = False,
                         sample_size: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def get_embeddings(self, text: List[str]) -> torch.Tensor:
        pass


class NLIModel(ABC):
    """Abstract base class for NLI models."""

    @abstractmethod
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features from input data."""
        pass

    @abstractmethod
    def train(self,
              train_dataset: str,
              train_split: str,
              train_suffix: Optional[str] = None,
              val_dataset: Optional[str] = None,
              val_split: Optional[str] = None,
              val_suffix: Optional[str] = None,
              **kwargs: Any) -> Optional[Dict[str, Any]]:  # Changed signature and return type
        """
        Train the model. Implementations should handle data loading
        based on the provided dataset, split, and suffix information.
        Returns optional dictionary containing training metrics/results.
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass

    @abstractmethod
    def save(self, filepath: str, model_name) -> None:
        """Save model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str, feature_extractor: FeatureExtractorInterface) -> 'NLIModel':
        """Load model from disk."""
        pass

    @abstractmethod
    def evaluate(self,
                 dataset_name: str,
                 split: str,
                 suffix: Optional[str] = None,
                 **kwargs: Any) -> Optional[Dict[str, Any]]:
        """
        Evaluate the model on a given dataset split.
        Returns optional dictionary containing evaluation metrics.
        """
        pass


def get_split_sizes(
        total_sample_size: Optional[int],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
        # test_ratio is implied as 1.0 - train_ratio - val_ratio
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Calculates sample sizes for train, validation, and test splits from a total.
    Ensures that the sum of split sizes equals total_sample_size.
    Prioritizes train, then validation, then test for sample allocation in rounding.
    Returns (train_size, val_size, test_size).
    Returns (None, None, None) if total_sample_size is None or 0.
    """
    if total_sample_size is None or total_sample_size <= 0:
        return None, None, None

    train_size = int(total_sample_size * train_ratio)
    val_size = int(total_sample_size * val_ratio)

    # Ensure train and val are at least 1 if total allows, and they are > 0%
    if train_ratio > 0 and train_size == 0:
        train_size = 1 if total_sample_size >= 1 else 0

    remaining_after_train = total_sample_size - train_size
    if val_ratio > 0 and val_size == 0 and remaining_after_train > 0:
        val_size = 1 if remaining_after_train >= 1 else 0

    # Adjust val_size if it exceeds remaining samples
    val_size = min(val_size, remaining_after_train)

    test_size = total_sample_size - train_size - val_size

    # Final check to prevent negative sizes if ratios are misconfigured or total is very small
    if train_size < 0: train_size = 0
    if val_size < 0: val_size = 0
    if test_size < 0: test_size = 0

    # If sum is not total due to intermediate minimums, adjust test_size (most flexible)
    # This can happen if train_size or val_size was forced to 1.
    current_sum = train_size + val_size + test_size
    if current_sum != total_sample_size and total_sample_size > 0:
        # This simple re-adjustment on test_size might not be perfect for all edge cases
        # but aims to make the sum correct.
        test_size = total_sample_size - train_size - val_size
        if test_size < 0:  # Should not happen if logic above is correct
            logger.warning(
                f"Test size became negative ({test_size}) after adjustment. Resetting. This indicates an issue in splitting logic for total {total_sample_size}.")
            # Fallback: re-evaluate based on remaining, could lead to val getting less if test needs some
            # This part needs careful thought for very small numbers if all splits must exist.
            # For now, the initial calculation is what we mostly rely on.
            # This complex balancing is why fixed N for val/test is sometimes easier for small totals.
            pass

    logger.info(
        f"Calculated split sizes for total {total_sample_size}: train={train_size}, val={val_size}, test={test_size}")
    if train_size + val_size + test_size != total_sample_size and total_sample_size > 0:
        logger.error(
            f"CRITICAL: Split sum ({train_size + val_size + test_size}) does not equal total_sample_size ({total_sample_size})!")
        # This indicates a flaw in the splitting logic for the given ratios and total size.

    return train_size, val_size, test_size
