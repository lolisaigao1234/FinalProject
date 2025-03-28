# utils/common.py
import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Any
import pandas as pd
import torch


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
    def create_train_test_split(self, dataset_name: str,
                                label_column: str = "label",
                                test_size: float = 0.2,
                                random_state: int = 42) -> Dict[str, pd.DataFrame]:
        pass

    @abstractmethod
    def preprocess_dataset_pipeline(self, dataset_name: str,
                                    sample_size: int,
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
